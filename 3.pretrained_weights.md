# Exploration into Pre-trained Weights


### [3.1: Body Part Classifier](3_1_body_part_classifier.ipynb)
Trained body part classifier in preparation for ensemble model later.

### [3.6-3.7: Trying to use weights pretrained on ImageNet on single channel input](3_6_collapse_3_channel_weights.md)
Note that the experiments done here are before I started using cyclical learning rates, so sometimes the model would converge to classifying everything as negative.
Discussion:
- By applying the inverted weight from the luminosity method to the kernels corresponding to each channel, model seem to learn well
- Using pre-trained weights makes training much faster
- In image pertubation (to prevent overfit), set the random rotation range to 360 degrees seems to allow model train slightly faster than only rotating within a range of 45 degrees
- There isn't difference between turning input into 3 channel and using 3 channel weights versus using collapsed one-channel weight.
- There also isn't any real advantage when weighting loss with overall sample positive/negative proportion. However, more experiment needed to see if weighting the loss by body part works. 