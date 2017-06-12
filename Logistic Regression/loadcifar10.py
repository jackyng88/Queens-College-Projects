# Write function to load the cifar-10 data
# The original code is from http://cs231n.github.io/assignment1/
# The function is in data_utils.py file for reusing.
import pickle
import numpy as np
import os
import random
import matplotlib.pyplot as plt

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f, encoding = 'bytes')
    #text_file = open("cifar10.txt", "w")
    #Z = datadict
    #Z = str(datadict)
    #text_file.write(Z)
    #text_file.close()

    X = datadict[b'data']
    #X = datadict.data

    Y = datadict[b'labels']
    #Y = datadict.labels

    #X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    #X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")




    #X = X.transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,2):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_CIFAR10_data(num_training = 9000, num_val = 1000, num_test = 10000, show_sample=True):
#def get_CIFAR10_data(num_training=49000, num_val=1000, num_test=10000, show_sample=True):
  """
  Load the CIFAR-10 dataset, and divide the sample into training set, validation set and test set
  """

  #cifar10_dir = 'datasets/datasets-cifar-10/cifar-10-batches-py/'
  cifar10_dir = 'datasets\datasets-cifar-10\cifar-10-batches-py'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

  # subsample the data for validation set
  mask = range(num_training, num_training + num_val)
  X_val = X_train[mask]
  y_val = y_train[mask]
  mask = range(num_training)
  X_train = X_train[mask]
  y_train = y_train[mask]
  mask = range(num_test)
  X_test = X_test[mask]
  y_test = y_test[mask]
  return X_train, y_train, X_val, y_val, X_test, y_test


def subset_classes_data(classes):
  # Subset 'plane' and 'car' classes to perform logistic regression
  idxs = np.logical_or(y_train_raw == 0, y_train_raw == 1)
  X_train = X_train_raw[idxs, :]
  y_train = y_train_raw[idxs]
  # validation set
  idxs = np.logical_or(y_val_raw == 0, y_val_raw == 1)
  X_val = X_val_raw[idxs, :]
  y_val = y_val_raw[idxs]
  # test set
  idxs = np.logical_or(y_test_raw == 0, y_test_raw == 1)
  X_test = X_test_raw[idxs, :]
  y_test = y_test_raw[idxs]
  return X_train, y_train, X_val, y_val, X_test, y_test


def visualize_sample(X_train, y_train, classes, samples_per_class=7):
  """visualize some samples in the training datasets """
  num_classes = len(classes)
  for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)  # get all the indexes of cls
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):  # plot the image one by one
      plt_idx = i * num_classes + y + 1  # i*num_classes and y+1 determine the row and column respectively
      plt.subplot(samples_per_class, num_classes, plt_idx)
      plt.imshow(X_train[idx].astype('uint8'))
      plt.axis('off')
      if i == 0:
        plt.title(cls)
  plt.show()


def preprocessing_CIFAR10_data(X_train, y_train, X_val, y_val, X_test, y_test):
  # Preprocessing: reshape the image data into rows
  X_train = np.reshape(X_train, (X_train.shape[0], -1))  # [49000, 3072]
  X_val = np.reshape(X_val, (X_val.shape[0], -1))  # [1000, 3072]
  X_test = np.reshape(X_test, (X_test.shape[0], -1))  # [10000, 3072]

  # Normalize the data: subtract the mean image
  mean_image = np.mean(X_train, axis=0)
  X_train -= mean_image
  X_val -= mean_image
  X_test -= mean_image

  # Add bias dimension and transform into columns
  X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
  X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
  X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T
  return X_train, y_train, X_val, y_val, X_test, y_test


X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = get_CIFAR10_data()

print (X_train_raw.shape)
print (y_train_raw.shape)
'''
# Invoke the above functions to get our data
X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = get_CIFAR10_data()
visualize_sample(X_train_raw, y_train_raw, classes)
subset_classes = ['plane', 'car']
X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = subset_classes_data(subset_classes)
X_train, y_train, X_val, y_val, X_test, y_test = preprocessing_CIFAR10_data(X_train_raw, y_train_raw, X_val_raw,
                                                                            y_val_raw, X_test_raw, y_test_raw)

# As a sanity check, we print out th size of the training and test data dimenstion
print('Train data shape: ', X_train.shape)
print ('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
'''