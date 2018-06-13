# Python:
We are using Python 3.6. An easy way to get everything needed for Python 3.6 is to install it via Anaconda: https://www.anaconda.com/download/

# CUDA (Optional):
Note that you can still learn Keras without a GPU, it'll just be slow to train on large images.

If you are using Linux or Windows with supported NVIDIA graphics cards (https://developer.nvidia.com/cuda-gpus), you need to install CUDA to be able to accelerate Tensorflow with your GPU. Instructions are here: [CUDA](https://developer.nvidia.com/cuda-downloads). You'll need version 9.0 for Tensorflow.

# cuDNN (Optional, Linux only)
If you can install cuDNN it would be even better. Instructions are here: [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html). You'll need version 7 for Tensorflow.

# Create a separate conda environment for this project (Optional)
It would be more convenient to use conda to manage your Python environments for different projects. But if you are working on this one only then it doesn't really matter. Read more here: https://conda.io/docs/user-guide/tasks/manage-environments.html

Creating an environment on Mac/Linux is easy, just run this in the terminal:
```bash
conda create --name <name of your environment> python=3.6 <packages you want>
```

for example, you can do 
```bash
conda create --name mura python=3.6 anaconda
```
to install the full suite of Anaconda packages.

In the future, before you run any python code (or jupyter), just switch on the environment in the terminal via
```bash
source activate mura
```

# Tensorflow 
Note: If you are installing the GPU version, please make sure you have CUDA 9 and cuDNN 7 ([How to check](https://medium.com/@changrongko/nv-how-to-check-cuda-and-cudnn-version-e05aa21daf6c))

Install according to the instructions here: [installing with pip](https://www.tensorflow.org/install/install_linux#InstallingNativePip). Note that we need the python 3.6 version. Please do not install the GPU version if you don't have CUDA or cuDNN set up. Also note that don't use `pip3` as instructed, just use `pip`... `pip3` is not installed in your conda environment by default.

Validate your installation by running:
```python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```
in python.

# Keras

```
pip install keras
```
[read more](https://keras.io/#installation)

# Other packages that we need:
```
scikit-learn
pandas
numpy
h5py
```
