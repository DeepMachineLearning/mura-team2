
# 3.6 - 3.7: trying to use weights from imagenet directly
Methodology:
Collapse the 3-channel pre-trained weight into one by performing operation on the weights of the channel. Then lock the base layers and train 30 epochs to check model performance.
- Comparing different ways of collapsing the weights:
    - configuration:
        - Densenet169, output classes = 1
        - training parameters:
            - optimizer: adam with default parameters
            - epochs: 30
            - batch size: 32
            - input shape: (256, 256, 1)
        - input dagagen parameters:
            - `rotation_range=45`
            - `width_shift_range=0.2`
            - `height_shift_range=0.2`
            - `zoom_range=[1.0, 1.2]`
            - `fill_mode='constant'`
            - `cval=0`
            - `horizontal_flip=True`
            - `vertical_flip=True`
    - Firstly tried collapsing the kernel with [luminosity method](https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/) naively:
    ```python
        a = np.array([0.2126, 0.7152, 0.0722])
        new_conv1_weights = a[0] * conv1_weights[:, :, 0, :] + a[1] * conv1_weights[:, :, 1, :] + a[2] * conv1_weights[:, :, 2, :]
    ```
        - didn't quite work - model just learned to predict everything as negative (0) after 30 epochs.
    
    - Then I realized that since the input image is converted into grayscale with those multipliers, doing the same on the kernel is essentially doing a square on the coefficients for each channel and thus essentially making the green channel weights dominate. So I inverted the coefficient and normalized them:
    ```python
        a = np.array([0.2126, 0.7152, 0.0722])
        a = (1/a) / sum(1/a)
        new_conv1_weights = a[0] * conv1_weights[:, :, 0, :] + a[1] * conv1_weights[:, :, 1, :] + a[2] * conv1_weights[:, :, 2, :]
    ```
        - worked better: study-wise kappa = 0.22 after 30 epochs
        
- I also tested whether it makes a difference to use pre-trained weights vs training from scratch:
    - configuration:
        - Densenet169, output classes = 1
        - training parameters:
            - optimizer: adam with default parameters
            - epochs: 30
            - **batch size: 16** - due to memory requirements for training entire network
            - input shape: `(256, 256, 1)`
        - input dagagen parameters:
            - `rotation_range=45`
            - `width_shift_range=0.2`
            - `height_shift_range=0.2`
            - `zoom_range=[1.0, 1.2]`
            - `fill_mode='constant'`
            - `cval=0`
            - `horizontal_flip=True`
            - `vertical_flip=True`
    - preliminary observation
        - in the first epoch, accuracy is >0.6 when using collapsed ImageNet weights, and 0.55 when initialized weights randomly.
        - without weights, first 10 epochs kappa is almost zero; with weights first 10 epochs brings kappa to 0.19
        - training from scratch tend to converge to predict all as negative (0)

- At the same time I also checked the difference it makes on training when I rotate the image with 360 degrees vs 45 degrees, using the inverse-coefficient collapsed weights (2nd method).
    - configuration:
        - Densenet169, output classes = 1
        - training parameters:
            - optimizer: adam with default parameters
            - epochs: 30
            - batch size: 16
            - input shape: `(256, 256, 1)`
        - input dagagen parameters:
            - `rotation_range=45` or `rotation_range=360` 
            - `width_shift_range=0.2`
            - `height_shift_range=0.2`
            - `zoom_range=[1.0, 1.2]`
            - `fill_mode='constant'`
            - `cval=0`
            - `horizontal_flip=True`
            - `vertical_flip=True`
    - preliminary observation
        - Doesn't observe a significant difference between training set losses and accuracies in the first few epochs
    - After 30 epochs, the model trained with images rotated 360 degrees have a noticably higher kappa (0.23 vs 0.18)

- Lastly, I compare this with turning the image into 3 channel and use the pre-trained imagenet weights:
    - configuration:
        - Densenet169, output classes = 1
        - training parameters:
            - optimizer: adam with default parameters
            - epochs: 30
            - batch size: 16
            - input shape: `(256, 256, 1)` or `(256, 256, 3)`
        - input dagagen parameters:
            - `rotation_range=45`
            - `width_shift_range=0.2`
            - `height_shift_range=0.2`
            - `zoom_range=[1.0, 1.2]`
            - `fill_mode='constant'`
            - `cval=0`
            - `horizontal_flip=True`
            - `vertical_flip=True`
    - didn't see a significant improvement from using 3 channel images