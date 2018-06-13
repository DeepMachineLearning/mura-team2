# MURA:
Download from [MURA official website](https://stanfordmlgroup.github.io/competitions/mura/)

Extract to mura-team2/data/MURA-v1.1

# MNIST:
Start python
run the following:
```python
import os
os.chdir('/path/to/your/mura-team2/folder')
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/MNIST_data/', one_hot=True)
```

The following lines should be printed:
```
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting data/MNIST_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting data/MNIST_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting data/MNIST_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting data/MNIST_data/t10k-labels-idx1-ubyte.gz
```

and now there should be the folder `mura-team2/data/MNIST-data`
