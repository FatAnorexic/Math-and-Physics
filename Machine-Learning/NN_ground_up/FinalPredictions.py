import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import modelNNmain as nn
import nnfs
nnfs.init()

def load_mnist_data(dataset,path):
    labels=os.listdir(os.path.join(path,dataset))

    X=[]
    y=[]

    for label in labels:
        for file in os.listdir(os.path.join(path,dataset,label)):
            image=cv2.imread(os.path.join(path,dataset,label,file),cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
    return np.array(X),np.array(y).astype('uint8')

def create_mnist_data(path):
    X,y=load_mnist_data('train',path)
    X_test, y_test=load_mnist_data('test',path)
    return X,y,X_test,y_test

def final_predictions():
    fashion_mnist_labels = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal',
                            6: 'Shirt',
                            7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}

    image_data=testingT()
    image_data=(image_data.reshape(1,-1).astype(np.float32)-127.5)/127.5
    model=nn.Model().load('fashion_mnist.model')

    confidences=model.predict(image_data)
    predictions=model.output_layer_activation.predictions(confidences)

    prediction=fashion_mnist_labels[predictions[0]]

    print(prediction,'\n\n')
    p=testingP()
    p=(p.reshape(1,-1).astype(np.float32)-127.5)/127.5

    confidences=model.predict(p)
    predictions=model.output_layer_activation.predictions(confidences)

    prediction=fashion_mnist_labels[predictions[0]]
    print(prediction)


def testingT():
    image_data=cv2.imread('real_image/test001.png',cv2.IMREAD_GRAYSCALE)
    image_data=cv2.resize(image_data,(28,28))
    image_data=255-image_data
    return image_data
def testingP():
    image_data=cv2.imread('real_image/test002.png',cv2.IMREAD_GRAYSCALE)
    image_data=cv2.resize(image_data,(28,28))
    image_data=255-image_data
    return image_data
final_predictions()
