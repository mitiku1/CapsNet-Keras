
from __future__ import print_function

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
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from datetime import datetime
import json

K.set_image_data_format('channels_last')

EMOTIONS= {
    0:"anger",
    1:"disgust",
    2:"fear",
    3:"happy",
    4:"surprise",
    5: "sad",
    6: "neutral"
}

datagenerator = ImageDataGenerator(rotation_range=0.3,
                        horizontal_flip=True,width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range= 0.2,
                        zoom_range=0.2,

                        )
class CustomCallBack(callbacks.Callback):
    
    def __init__(self,**kargs):
        callbacks.Callback.__init__(self,**kargs)
        self.last_loss = 1000000000
        self.last_accuracy = 0
        self.current_model_number = 0;
        self.epoch_number = 0
    # def on_train_begin(self,epoch, logs={}):
    #     return
 
    # def on_train_end(self, logs={}):
    #     return
 
    def on_epoch_begin(self,epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_number+=1
        current_loss = logs.get("val_loss")
        current_accuracy  = logs.get("val_acc")
        if (self.last_loss-current_loss) > 0.01:
            current_weights_name = "weights"+str(self.current_model_number)+".h5"
            print("loss improved from "+str(self.last_loss)+" to "+str(current_loss)+", Saving model to "+current_weights_name)
            self.model.save_weights("models/"+current_weights_name);
            self.model.save_weights("models/last_weight.h5")
            self.current_model_number+=1
            self.last_loss = current_loss
            self.last_accuracy = current_accuracy
            with open("log.txt","a+") as logfile:
                logfile.write("________________________________________________________\n")
                logfile.write("EPOCH    =")
                logfile.write(str(epoch)+"\n")
                logfile.write("VAL_ACC  =")
                logfile.write(str(current_accuracy)+"\n")
                logfile.write("VAL_LOSS =")
                logfile.write(str(current_loss)+"\n")
                logfile.write("ACCURACY =")
                logfile.write(str(logs.get("acc"))+"\n")
                logfile.write("LOSS     =")
                logfile.write(str(logs.get("loss"))+"\n")
                logfile.write("********************************************************\n")
            with open("epoch_number.json","w+") as json_file:
                data = {"epoch_number":self.epoch_number}
                json.dump(data,json_file,indent=4)
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

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
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    # Layer 3: Capsule layer. Routing algorithm works here.
    outcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='outcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(outcaps)

    
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








def class_to_label(c):
    return EMOTIONS[c]
def label_to_class(label):
    for emotion in EMOTIONS:
        if EMOTIONS[emotion] == label:
            return emotion
    raise Exception("Unrecognized emotion label:"+str(label))
        
def load_images(x,y,input_shape,dataset_path):
    output = np.zeros((len(x),input_shape[0],input_shape[1],input_shape[2]))
    for i in range(len(x)):
        try:
            img = cv2.imread(os.path.join(dataset_path,class_to_label(y[i]),x[i]))
        except cv2.error as e:
            print("Error", os.path.join(dataset_path,class_to_label(y[i]),x[i]))       
        if img is None:
            print("Couldnot read image from ",os.path.join(dataset_path,class_to_label(y[i]),x[i]))
        if len(img.shape)>2 and self.input_shape[2]==1:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(input_shape[0],input_shape[1]))
        img = img.reshape(input_shape)
        output[i] = img
    output = output.astype(np.float32)
    return output 
def load_all_dataset(input_shape,dataset_path):
    X_train = []
    y_train = []
    for label_dir in os.listdir(os.path.join(dataset_path,"train")):
        for img_file in os.listdir(os.path.join(dataset_path,"train",label_dir)):
            X_train+=[img_file]
            y_train+=[label_to_class(label_dir)]
    print("loading test images")
    # X_train = load_images(X_train,y_train,input_shape,os.path.join(dataset_path,"train"))

    X_test = []
    y_test = []
    for label_dir in os.listdir(os.path.join(dataset_path,"test")):
        for img_file in os.listdir(os.path.join(dataset_path,"test",label_dir)):
            X_test+=[img_file]
            y_test +=[label_to_class(label_dir)]
    X_test = load_images(X_test,y_test,input_shape,os.path.join(dataset_path,"test"))

    

    y_test = np.eye(7)[y_test]

    x_test = X_test.reshape(-1,*input_shape)
    return X_train,y_train, x_test,y_test

def load_dataset(input_shape,dataset_dir):
    X = []
    Y = []
    for em_dir in os.listdir(dataset_dir):
        for im_file in os.listdir(os.path.join(dataset_dir,em_dir)):
            img = cv2.imread(os.path.join(dataset_dir,em_dir,im_file))
            if not (img is None):
                if len(img.shape)>2 and input_shape[2]==1:
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img,(input_shape[0],input_shape[1]))
                X+=[img]
                Y+=[label_to_class(em_dir)]
    x = np.array(X)
    y = np.array(Y)
    x = x.reshape((-1,input_shape[0],input_shape[1],input_shape[2])) 
    return x,y

def generate_indexes(length):
    indexes = range(length)
    return shuffle(indexes)

def generator(x_train,y_train,input_shape,dataset_path,batch_size=32,augmentation=True):

    while True:
        indexes = generate_indexes(len(x_train))
        for i in range(0,(len(x_train)-batch_size),batch_size):
            current_indexes = indexes[i:i+batch_size]
            if augmentation:
                imgs_shape = (len(current_indexes),)+input_shape
                output_images = np.zeros(imgs_shape)
                current_images = load_images(x_train[current_indexes],y_train[current_indexes],input_shape,dataset_path)
                for index in range(len(current_indexes)):
                    output_images[index] = datagenerator.random_transform(current_images[index])
                output_images = output_images.astype(np.float32)/255
                yield output_images,np.eye(7)[y_train[current_indexes]]
            else:
                yield load_images(x_train[current_indexes],y_train,input_shape,dataset_path).astype(np.float32)/255,np.eye(7)[y_train[current_indexes]]


if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.0001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--dataset', default="dataset", type=str,
                        help="Path to dataset")
    parser.add_argument('--resume', default=False, type=bool,
                        help="Resume model from previous state")
    

    args = parser.parse_args()
    input_shape = (96,96,1)
    model = CapsNet(input_shape,7,routings=3)
    model.summary()
    model.compile(loss=margin_loss,optimizer=keras.optimizers.Adam(args.lr),metrics=['accuracy'])
    if os.path.exists("models/last_weight.h5"):
        print("loading model from last checkpoint.")
        model.load_weights("models/last_weight.h5")

    dataset_dir = args.dataset
    # checkpoint = callbacks.ModelCheckpoint('models/weights'+'-{epoch:02d}-acc-{val_acc:0.3f}.h5',
    #                 save_best_only=True, save_weights_only=True, verbose=1)
    # checkpoint2 = callbacks.ModelCheckpoint("models/last_weight.h5")
    customCheckPoint = CustomCallBack()
    if not os.path.exists(dataset_dir):
        print("dataset dir",dataset_dir, "does not exist");
    x_train,y_train = load_dataset(input_shape, os.path.join(dataset_dir,"train"))
    x_test,y_test = load_dataset(input_shape, os.path.join(dataset_dir,"test"))

    x_train = x_train.astype(np.float32)/255
    x_test = x_test.astype(np.float32)/255

    y_train = np.eye(7)[y_train]
    y_test = np.eye(7)[y_test]
    print ("x_train",x_train.shape)
    print ("x_test",x_test.shape)
    print ("y_train",y_train.shape)
    print ("y_test",y_test.shape)
    REMAINING_EPOCHS = args.epochs
    if (args.resume):
        if os.path.exists("epoch_number.json"):
            with open("epoch_number.json","r") as json_file:
                try:
                    data = json.load(json_file)
                    customCheckPoint.epoch_number = data["epoch_number"]
                    REMAINING_EPOCHS -= customCheckPoint.epoch_number
                except:

                    print ("unable to read epoch number from file. resuming epoch from 0.")

                
            print("resuming from previous epoch number:")
            print("previous epoch number",customCheckPoint.epoch_number)
            print("remaining epochs",REMAINING_EPOCHS)
    if REMAINING_EPOCHS < 0:
        REMAINING_EPOCHS =1
    with open("log.txt","a+") as logfile:
        str_date = datetime.now().strftime("%d, %b, %Y %H:%M:%S")
        logfile.write("Starting to train model\n")
        logfile.write(str_date+"\n")


    # model.fit_generator(generator=generator(x_train, y_train,input_shape,dataset_dir+"/train", args.batch_size),
    #                     steps_per_epoch=1000,
    #                     epochs=REMAINING_EPOCHS,
    #                     callbacks=[customCheckPoint],
    #                     validation_data=[x_test, y_test])
    model.fit(x_train,y_train,batch_size = 32,epochs=10,callbacks=[customCheckPoint],validation_data=[x_test,y_test])
    model.save_weights("result/model.h5")
    model_json = model.to_json()
    with open("result/model.json","w+") as f0:
        f0.write(model_json)