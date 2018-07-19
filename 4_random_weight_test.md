http://www.icml-2011.org/papers/551_icmlpaper.pdf

Saxe found that random CNN weights are usually almost as good as pre-trained weights, and we can separate the predictive power contributed by model structure from the power contributed by training optimization. Hence by using random initialized weights and only train the top layer, it allows for rapid artecture comparison.

First observation: Need to use weighted loss - otherwise network just learns to predict zero for all cases

From comparison between DenseNet and MobileNet it does seem to work - DenseNet reached study-wise Kappa = 0.3 in 30 epochs, while MobileNet only reached 0.2 after 80 epochs. 