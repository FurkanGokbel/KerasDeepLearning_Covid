import numpy as np
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras import optimizers
from keras import *
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Dropout, MaxPooling2D, Flatten

girisverisi = np.load("girisverinizxray.npy")
# kalem için 1,0 dedik sınıflandırma işlemi için kullandım
girisverisi = np.reshape(girisverisi, (-1, 224, 224, 3))
cikisverisi = np.array(
    [ [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]
         ])
splitverisi = girisverisi[1:35]
splitverisi = np.append(splitverisi, girisverisi[364:399])
splitverisi = splitverisi.reshape(-1, 224, 224, 3)
splitcikis = np.array(
    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]
     ])


# yapay sinir ağları ardışık olduğu için Sequential le başlıyoruz.
model = Sequential()
# Makalede yazıyor 11X11 e bölücek resimi ve 4X4 olarak ilerleyecek en optimal yol
# filtre ekliyoruz
model.add(Conv2D(50, (3, 3), input_shape=(224, 224, 3), data_format='channels_first'))
model.add(Conv2D(50, 2))
model.add(Conv2D(50, 2))
model.add(Conv2D(50, 2))
model.add(Conv2D(50, 3))
model.add(Conv2D(50, 2))
model.add(Conv2D(50, 2))
model.add(Conv2D(50, 2))
model.add(MaxPooling2D((5, 5), padding='same'))
model.add(Conv2D(50, 2))
model.add(Conv2D(50, 2))
model.add(Conv2D(50, 2))
model.add(MaxPooling2D((3, 3), padding='same'))
model.add(Conv2D(50, 2))

model.add(Flatten())  # resim dosyalarındaki agırlık matrislerini düz hale getirdik
model.add(Dense(1000, activation='relu'))  # dense katmanı ekliyoruz 4096 standart alexnette
# AlexNet
# 2012 yılında evrişimli sinir ağ modellerini ve derin öğrenmenin tekrar popüler hale gelmesini sağlayan ilk çalışmadır. Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton tarafından geliştirilmiştir. Temel olarak LeNet modeline, birbirini takip eden evrişim ve pooling katmanları bulunmasından dolayı, çok benzemektedir. Aktivasyon fonksiyonu olarak ReLU (Rectified Linear Unit), pooling katmanlarında da max-pooling kullanılmaktadır.
model.add(Dropout(0, 3))

model.add(Dense(1000, activation='relu'))
model.add(Dense(2))
# sınıflandırma yapıyosak softmax kullanıcaz regresyon ise sigmoid kullanıcaz
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.00001),
              metrics=['accuracy'])  # kayıp fonksiyonu 2 tane resmimiz olduğu için bu fonksiyonu kullandık
model.summary()

# resmi 255 e bölersek resim kodunu 0 ile 1 arasına sıkıştırmış oluruz buda bizim işlem yükümüzü azaltır.
# batch_size ne kadar buyuk olursa o kadar genellemere açık olur
model.fit(girisverisi / 255, cikisverisi, batch_size=1, epochs=50,
          validation_data=(splitverisi, splitcikis))  # aynı anda kaç resmi yükleyip çalıştırsın

model.save("egitilmisverixray")
# value accu sistem hiç görmemiş resmi ve

# verdiği tepki oranı
# accures eğittiğim modelde aldığım sonuçlar
"""
model = Sequential()
model.add(Conv2D(32, (5,5), activation='relu', input_shape=(50, 50, 3),data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax')) #2 çünkü kanserli kansersiz
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(girisverisi/255,cikisverisi,batch_size=1,epochs=75,validation_data=(splitverisi,splitcikis))#aynı anda kaç resmi yükleyip çalıştırsın
model.save("kerasileuygulama50")
"""
