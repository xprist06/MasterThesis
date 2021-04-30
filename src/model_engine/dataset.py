from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import numpy as np


class Dataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.height = None
        self.width = None
        self.depth = None
        self.train_x = None
        self.train_y = None
        self.evo_train_x = None
        self.evo_train_y = None
        self.test_x = None
        self.test_y = None
        self.label_names = None
        self.classes_cnt = None

        if dataset == 0:
            self.load_mnist()
        elif dataset == 1:
            self.load_fashion_mnist()
        elif dataset == 2:
            self.load_svhn()
        elif dataset == 3:
            self.load_cifar10()
        elif dataset == 4:
            self.load_cifar100()
        else:
            exit(1)

        self.load_evo_subset()

    def load_mnist(self):
        # Load the MNIST dataset
        print("[INFO] loading MNIST dataset...")
        # Load the data
        (self.train_x, self.train_y), (self.test_x, self.test_y) = mnist.load_data()

        # Reshape data
        self.train_x = self.train_x.reshape(60000, 28, 28, 1)
        self.test_x = self.test_x.reshape(10000, 28, 28, 1)

        # Scale the data to the range [0, 1]
        self.train_x = self.train_x.astype('float32') / 255.0
        self.test_x = self.test_x.astype('float32') / 255.0

        self.height = 28
        self.width = 28
        self.depth = 1
        self.classes_cnt = 10

    def load_fashion_mnist(self):
        # Load the FASHION MNIST dataset
        print("[INFO] loading FASHION MNIST dataset...")
        # Load the data
        (self.train_x, self.train_y), (self.test_x, self.test_y) = fashion_mnist.load_data()

        # Reshape data
        self.train_x = self.train_x.reshape(60000, 28, 28, 1)
        self.test_x = self.test_x.reshape(10000, 28, 28, 1)

        # Scale the data to the range [0, 1]
        self.train_x = self.train_x.astype('float32') / 255.0
        self.test_x = self.test_x.astype('float32') / 255.0

        self.height = 28
        self.width = 28
        self.depth = 1

        label_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.classes_cnt = len(label_names)

    def load_svhn(self):
        # Load SVHN dataset
        print("[INFO] loading SVHN dataset...")
        # Load the data
        # train_raw = loadmat('C:\\Users\\janp\\Documents\\Skola\\DP\\Python\\src\\input\\svhn\\train.mat')
        # test_raw = loadmat('C:\\Users\\janp\\Documents\\Skola\\DP\\Python\\src\\input\\svhn\\test.mat')
        train_raw = loadmat('/home/xprist06/src/input/svhn/train.mat')
        test_raw = loadmat('/home/xprist06/src/input/svhn/test.mat')

        # Load images and labels
        self.train_x = np.array(train_raw['X'])
        self.test_x = np.array(test_raw['X'])
        self.train_y = train_raw['y']
        self.test_y = test_raw['y']

        # Fix the axes of the images
        self.train_x = np.moveaxis(self.train_x, -1, 0)
        self.test_x = np.moveaxis(self.test_x, -1, 0)

        # Scale the data to the range [0, 1]
        self.train_x = self.train_x.astype('float32') / 255.0
        self.test_x = self.test_x.astype('float32') / 255.0
        # self.train_y = self.train_y.astype('int64')
        # self.test_y = self.test_y.astype('int64')

        # Set labels to the range [0, 9]
        for i in range(len(self.train_y)):
            if self.train_y[i] % 10 == 0:
                self.train_y[i] = 0
        for i in range(len(self.test_y)):
            if self.test_y[i] % 10 == 0:
                self.test_y[i] = 0

        self.height = 32
        self.width = 32
        self.depth = 3
        self.classes_cnt = 10

    def load_cifar10(self):
        # Load the CIFAR-10 dataset
        print("[INFO] loading CIFAR-10 dataset...")
        # Load the data
        (self.train_x, self.train_y), (self.test_x, self.test_y) = cifar10.load_data()

        # Scale the data to the range [0, 1]
        self.train_x = self.train_x.astype("float32") / 255.0
        self.test_x = self.test_x.astype("float32") / 255.0

        self.height = 32
        self.width = 32
        self.depth = 3

        # Initialize the label names for the CIFAR-10 dataset
        self.label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.classes_cnt = len(self.label_names)

    def load_cifar100(self):
        # Load the CIFAR-100 dataset
        print("[INFO] loading CIFAR-100 dataset...")
        # Load the data
        (self.train_x, self.train_y), (self.test_x, self.test_y) = cifar100.load_data()

        # Scale the data to the range [0, 1]
        self.train_x = self.train_x.astype("float32") / 255.0
        self.test_x = self.test_x.astype("float32") / 255.0

        self.height = 32
        self.width = 32
        self.depth = 3
        self.classes_cnt = 100

    def load_evo_subset(self):
        train_x, test_x, train_y, test_y = train_test_split(self.train_x, self.train_y, train_size=0.25)
        self.evo_train_x = train_x
        self.evo_train_y = train_y
