import os
import numpy as np
import mnist_reader
import cv2

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


X_train, y_train = mnist_reader.load_mnist('.', kind='train')
X_test, y_test = mnist_reader.load_mnist('.', kind='t10k')

makedir("train")
makedir("test")
for i in range(10):
    makedir("train/{}".format(i))
    makedir("test/{}".format(i))
list_train = []
for i in range(len(X_train)):
    label = y_train[i]
    image = X_train[i].reshape((28,28))
    image_path = "train/{}/{}.png".format(label,i)
    cv2.imwrite(image_path,image)
    list_train.append([image_path,label])
with open("train_list.txt","w") as f:
    for i in range(len(list_train)):
        path_label = list_train[i]
        f.write(path_label[0]+","+str(path_label[1])+"\n")

list_test = []
for i in range(len(X_test)):
    label = y_test[i]
    image = X_test[i].reshape((28,28))
    image_path = "test/{}/{}.png".format(label,i)
    cv2.imwrite(image_path,image)
    list_test.append([image_path,label])
with open("test_list.txt","w") as f:
    for i in range(len(list_test)):
        path_label = list_test[i]
        f.write(path_label[0]+","+str(path_label[1])+"\n")