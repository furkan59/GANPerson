from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Reshape
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import UpSampling2D, Conv2D
from tensorflow.python.keras.layers import ELU
from tensorflow.python.keras.layers import Flatten, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.datasets import mnist
import tensorflow as tf
import os
from PIL import Image
from helper import *
import math
import numpy as np


#hyper parametreler
INPUT_DIM_START = 100
START_NN_NEURONCOUNT = 128
WIDTH_OF_IMAGE = 128
HEIGHT_OF_IMAGE = 128
CHANNEL_OF_IMAGE = 3 #RESİMLERİMİZ RENKLİ
CHANNEL_COEFFICIENT = 16
CONVOLUTIONRANGE = 5
UPSAMPLINGRANGE = 2

batch_size = 1
num_epoch = 50
learning_rate = 0.0002
image_path = 'images/'
traindata = "C:/Users/furkam/PycharmProjects/PyhtonTest1/Root/YapayZeka/Dressing/test/resizedtraindata/"

if not os.path.exists(image_path):
    os.mkdir(image_path)
    
def generator(input_dim=INPUT_DIM_START, units=START_NN_NEURONCOUNT, activation='relu'):
    
    model = Sequential()
    model.add(Dense(input_dim=input_dim, units=units))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    denselayerneuroncount = CHANNEL_OF_IMAGE*CHANNEL_COEFFICIENT*2*(WIDTH_OF_IMAGE//4)*(HEIGHT_OF_IMAGE//4)
    model.add(Dense(denselayerneuroncount))
    
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Reshape(((WIDTH_OF_IMAGE//4), (HEIGHT_OF_IMAGE//4), CHANNEL_OF_IMAGE*CHANNEL_COEFFICIENT*2), input_shape=(denselayerneuroncount,)))
    '''
    burada artık elimizde küçük bir matris var çünkü 7*7*128(kanal sayısı) lik bir matris var 
    bunu büyütmek için ise upsampling yapmak gerekli
    
    '''
    model.add(UpSampling2D((UPSAMPLINGRANGE, UPSAMPLINGRANGE)))
    #bu aşamadan sonra matris artık 14*14*128 olacak
    model.add(Conv2D(CHANNEL_COEFFICIENT*CHANNEL_OF_IMAGE, (CONVOLUTIONRANGE, CONVOLUTIONRANGE), padding='same'))
    #bu aşama derinliği yani kanal sayısını etkiler artık matrisin boyutu 14*14*64 oldu
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(UpSampling2D((UPSAMPLINGRANGE, UPSAMPLINGRANGE)))
    #bu aşamadan sonra matrisin boyutu 28*28*64 oldu 
    model.add(Conv2D(CHANNEL_OF_IMAGE, (CONVOLUTIONRANGE, CONVOLUTIONRANGE), padding='same'))
    #bu aşamadan sonra ise matrisin boyutu 28*28*1 oldu
    model.add(Activation('tanh'))
    #tanh olmasının sebebi de 0 ile 1 arasında çıkmasını sağlamak içindir
    print(model.summary())
    return model

def discriminator(input_shape=(WIDTH_OF_IMAGE, HEIGHT_OF_IMAGE, CHANNEL_OF_IMAGE), nb_filter=CHANNEL_COEFFICIENT):
    model = Sequential()
    model.add(Conv2D(nb_filter, 
                         (CONVOLUTIONRANGE, CONVOLUTIONRANGE), 
                         strides=(UPSAMPLINGRANGE, UPSAMPLINGRANGE), 
                         padding='same', input_shape=input_shape))
    
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Conv2D(2*nb_filter, (CONVOLUTIONRANGE, CONVOLUTIONRANGE), strides=(UPSAMPLINGRANGE, UPSAMPLINGRANGE)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(4*nb_filter))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print(model.summary())
    return model



def loadimages():
    import glob
    filelist = glob.glob(traindata + '*.jpg')
    images = np.array([np.array(Image.open(fname)) for fname in filelist])
    return images 

def train():
    #tf.reset_default_graph()
    x_train = loadimages()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], CHANNEL_OF_IMAGE)

    g = generator()
    d= discriminator()

    optimize = Adam(lr=learning_rate, beta_1=0.5)
    d.trainable = True
    d.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=optimize)

    d.trainable = False
    dcgan = Sequential([g, d])
    dcgan.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer=optimize)

    num_batches = x_train.shape[0] // batch_size
    gen_img = np.array([np.random.uniform(-1, 1, INPUT_DIM_START) for _ in range(49)])
    y_d_true = [1] * batch_size
    y_d_gen = [0] * batch_size
    y_g = [1] * batch_size

    for epoch in range(num_epoch):
        for i in range(num_batches):
            x_d_batch = x_train[i*batch_size:(i+1)*batch_size]
            x_g = np.array([np.random.normal(0, 0.5, INPUT_DIM_START) for _ in range(batch_size)])
            x_d_gen = g.predict(x_g)

            d_loss = d.train_on_batch(x_d_batch, y_d_true)
            d_loss = d.train_on_batch(x_d_gen, y_d_gen)

            g_loss = dcgan.train_on_batch(x_g, y_g)
            show_progress(epoch, i, g_loss[0], d_loss[0], g_loss[1], d_loss[1])

        image = combine_images(g.predict(gen_img))
        image = image * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save(image_path + "%03d.png" % (epoch))


if __name__ == '__main__':
    train()

