from tensorflow.examples.tutorials.mnist import input_data

def read_mnist(path='data\\MNIST_data'):
    mnist = input_data.read_data_sets(path, one_hot = True)
    train_X, train_Y, test_X, test_Y = (
    	mnist.train.images, mnist.train.labels, 
    	mnist.test.images, mnist.test.labels)
    return train_X, train_Y, test_X, test_Y
