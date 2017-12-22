"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import cv2
import keras


K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=32, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    print "Primary caps",primarycaps.shape
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    
    model = models.Model(inputs=x,outputs=out_caps)
    return model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))




def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()





def l1_loss(y_true,y_pred):
    loss = K.sum(K.abs(y_true - y_pred),axis=1)
    return loss
def class_to_label(c):
    if c==0:
        return "Cat"
    elif c==1:
        return "Dog"
    raise Exception("Unrecognized class "+str(c))
        
def load_images(x,y,input_shape,dataset_path):
    output = np.zeros((len(x),input_shape[0],input_shape[1],input_shape[2]))
    for i in range(len(x)):
        try:
            img = cv2.imread(os.path.join(dataset_path,class_to_label(y[i]),x[i]))
        except cv2.error as e:
            print "Error", os.path.join(dataset_path,class_to_label(y[i]),x[i])       
        
        if len(img.shape)>2:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(input_shape[0],input_shape[1]))
        img = img.reshape(input_shape)
    
        output[i] = img
    output = output.astype(np.float32)
    return output 
def load_all_dataset(input_shape,dataset_path):
    labels = {"cat":0,"dog":1}
    X = []
    y = []

    for label_dir in os.listdir(dataset_path):
        for img_file in os.listdir(os.path.join(dataset_path,label_dir)):
            X+=[img_file]
            y+=[labels[label_dir.lower()]]
    print "loading images"
    X = load_images(X,y,input_shape,dataset_path)
    print "sliting dataset"
    x_train,x_test, y_train,y_test = train_test_split(X,y)
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    y_train = np.eye(2)[y_train]
    y_test = np.eye(2)[y_test]
    x_train = x_train.reshape(-1,*input_shape)
    x_test = x_test.reshape(-1,*input_shape)
    return x_train,y_train, x_test,y_test

def load_dataset(input_shape,dataset_path):
    labels = {"cat":0,"dog":1}
    X = []
    y = []

    for label_dir in os.listdir(dataset_path):
        for img_file in os.listdir(os.path.join(dataset_path,label_dir)):
            X.append(img_file)
            y.append(labels[label_dir.lower()])
    x_train,x_test, y_train,y_test = train_test_split(X,y)
    x_test = load_images(x_test,y_test,input_shape,dataset_path)

    x_train = np.array(x_train)
    y_test = np.array(y_test)
    y_train = np.array(y_train)
    return x_train,y_train, x_test,y_test
def generate_indexes(length):
    indexes = range(length)
    return shuffle(indexes)

def generator2(x_train,y_train,batch_size=32):
    while True:
        indexes = generate_indexes(len(x_train))
        print len(x_train)-batch_size
        for i in range(0,(len(x_train)-batch_size),batch_size):
            current_indexes = indexes[i:i+batch_size]
            yield x_train[current_indexes],y_train[current_indexes]

def generator(x_train,y_train,input_shape,dataset_path,batch_size=32):

    while True:
        indexes = generate_indexes(len(x_train))
        currentIndex = 0
        while currentIndex<(len(indexes)-batch_size):
            currentIndex
            currentIndexes = indexes[currentIndex:currentIndex+batch_size]
            img_files = x_train[currentIndexes]
            img_classes = y_train[currentIndexes]
            currentIndex += batch_size
            imgs = load_images(img_files,img_classes,input_shape,dataset_path)
            img_classes = np.eye(2)[img_classes]
            yield imgs,img_classes
if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    # print(args)

    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    input_shape = (48,48,1)
    model = CapsNet(input_shape,2,3)
    model.compile(loss=margin_loss,optimizer=keras.optimizers.Adam(1e-4),metrics=['accuracy'])
    dataset_dir = "/home/mtk/datasets/kagglecatsanddogs_3367a/PetImages"
    # x_train,y_train,x_test, y_test = load_dataset(input_shape, dataset_dir)
    x_train,y_train,x_test, y_test = load_all_dataset(input_shape, dataset_dir)
    # y_test = np.eye(2)[y_test]
    print "x_train", x_train.shape
    print "x_test", x_test.shape
    print "y_train", y_train.shape
    print "y_test", y_test.shape
    # model.fit_generator(generator=generator(x_train, y_train,input_shape,dataset_dir, 32),
    #                     steps_per_epoch=200,
    #                     epochs=30,
    #                     validation_data=[x_test, y_test])
    x_train = x_train.astype(np.float32)/255
    x_test = x_test.astype(np.float32)/255
    model.fit_generator(generator=generator2(x_train, y_train, 32),
                        steps_per_epoch=200,
                        epochs=30,
                        validation_data=[x_test, y_test])
    model.save_weights("result/dog-cats48x48.h5")
    model_json = model.to_json()
    with open("result/dog-cats48x48.json","w+") as f0:
        f0.write(model_json)
    

