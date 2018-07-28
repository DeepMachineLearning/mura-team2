# MURA Team 2 Repo

This repo will be mainly designated for collaboratively learning about CV and showcasing our work on the MURA competition.

### Planned folder structure:
```
├── data
│   ├── MNIST_data
│   └── MURA-v1.1
│       ├── train
│       └── valid
├── models
├── setup
├── submission
├── trained_models
└── utils
```

- data: contains the MURA dataset and MNIST dataset. Note that this folder is not uploaded to github
- models: to contain all the model scripts and stored models
- setup: contains the instructions to setup environment and datasets 
- submission: notes, scripts and model files for submission
- trained_models: trained model saved during training 
- utils: stores useful scripts so that we can import to our notebooks.

Note that this structure is preliminary. If it's deemed not flexible enough as we progress we'll update it

## Steps:
- 0: tutorials, get started [(details)](0.tutorial.md)
- 1: exploring around with pre-defined model structure [(details)](1.single_model.md)
- 3: (1) trained a body-part classifier, and (2) explored using collapsed pre-trained ImageNet weights [(details)](3.pretrained_weights.md)
- 4: explored Saxe (2011) [(details)](4_random_weight_test.md)
- 5: Built an ensemble model with DenseNet169 - shared first three dense blocks, and have a separate dense block per body part for the last dense block. [(details)](5.ensemble_with_shared_base_layer.md)
- 6: Model diagnostics with model trained in 5.

## Next steps:
- FN seems to be much larger than FP - maybe we should adjust prediction to a balanced sample?
- More on Cohen's kappa - Can we maximize it given accuracy?
- Streamline model evaluation (done with [utils/mura_metrics.py](utils/mura_metrics.py))
- Generate model performance by body parts - if a body part is significantly different, maybe we should train a separate model (done with [utils/mura_metrics.py](utils/mura_metrics.py))
- extract middle layer activation patterns - is our algo looking at the right place?
- More ways to prevent overfit?
- Try using WGAN to generate more sample for training?
- gather more data (as mentioned in the [forum post](http://forum.aisquaredforum.ca/t/exploratory-model-diagnostics/65/6?u=madcarrot))
- Manually correct the data with color inverted (since kernels are not invarnat to color inversion)
- larger kernel (due to uncertainty around bone plates, we might need larger kernels to get more complicated texture patterns)
- use genetic algorithm to produce best model architecture (with the help from Saxe (2011))

## References
_[1]_ Saxe (2011) _On Random Weights and Unsupervised Feature Learning_