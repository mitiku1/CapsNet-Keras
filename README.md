# CapsNet-Keras
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/XifengGuo/CapsNet-Keras/blob/master/LICENSE)

A Keras implementation for emotion recognition of CapsNet in the paper:   
[Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017](https://arxiv.org/abs/1710.09829)   

## Usage

**Step 1.
Install [Keras>=2.0.7](https://github.com/fchollet/keras) 
with [TensorFlow>=1.2](https://github.com/tensorflow/tensorflow) backend.**
```
pip install tensorflow-gpu
pip install keras
```

**Step 2. Clone this repository to local.**
```
git clone https://github.com/XifengGuo/CapsNet-Keras.git capsnet-keras
cd capsnet-keras
```

**Step 3. Train a CapsNet on MNIST**  

Training 
```
python capsulenet.py --dataset dataset-directory --epochs epochs --batch_size  batch_size
```
If --dataset argument is not given then it is assumed that the dataset is found inside folder ./dataset . 

### Dependancies

* tensorflow >= 1.0
* keras >= 2.0
* opencv >= 3.0

