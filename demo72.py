import tensorflow as tf
import keras
from keras import datasets
from matplotlib import pyplot as plt

(train_images, train_labels,), (test_images, test_labels) = datasets.fashion_mnist.load_data()
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()