import numpy as np
import cv2
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from keras import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Dropout, MaxPooling2D, Flatten
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
#yapay sinir ağları ardışık olduğu için Sequential le başlıyoruz.
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
#, input_shape=(50, 50, 3)
# 2012 yılında evrişimli sinir ağ modellerini ve derin öğrenmenin tekrar popüler hale gelmesini sağlayan
model.add(Dropout(0, 3))

model.add(Dense(1000, activation='relu'))
model.add(Dense(2))
# sınıflandırma yapıyosak softmax kullanıcaz regresyon ise sigmoid kullanıcaz
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.00001), metrics=['accuracy'])#kayıp fonksiyonu 2 tane resmimiz olduğu için bu fonksiyonu kullandık



model.load_weights("egitilmisverixray")

"""
Ekran Açma
"""

"""

root = Tk()
root.geometry('550x300+300+150')
root.resizable(width=True, height=True)
def openfn():
    filename = filedialog.askopenfilename(initialdir="/", title="Dosya Seç",
                                               filetypes=(("png files", "*.png"), ("all files", "*.*")))
    return filename

def open_img():
    x = openfn()
    img = Image.open(x)
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()


btn = Button(root, text='Sınıflandırmak İstediğiniz Görsel', command=open_img).pack()
lbl=Label(root,text='Deneme').pack()


mainloop()
"""

def resmiklasordenal(dosyaadi):
    resim = cv2.imread("%s"%dosyaadi)
    return resim

tahmin_sonuc=np.array([])
gercek_cikis=np.array([[1,0]])
#pozitif resimler alınıyor ve tahmin işlemi gerçekleştiriliyor
kactanepozitif=28
for i in range(kactanepozitif):
    dosya = 'test-covid/cp%s.png'%str(i+1)
    #print(dosya)
    klasordenalinmisresim=0
    #string = 'test-covid/cp1.png'
    girisverisi=np.array([])
    klasordenalinmisresim=resmiklasordenal(dosya)
    boyutlandirilmisresim=cv2.resize(klasordenalinmisresim, (224, 224))
    girisverisi=np.append(girisverisi, boyutlandirilmisresim)
    girisverisi=np.reshape(girisverisi, (-1, 224, 224, 3))
    gecici=model.predict(girisverisi)
    if (i==0):tahmin_sonuc=model.predict(girisverisi)
    else:
        tahmin_sonuc=np.append(tahmin_sonuc,model.predict(girisverisi),axis=0)
        gercek_cikis=np.append(gercek_cikis,[[1,0]],axis=0)
    
negatifbaslangic=180
negatifbitis=214
#Negatif resimler alınıyor ve tahmin işlemi gerçekleştiriliyor
for i in range(negatifbaslangic,negatifbitis+1):
    dosya = 'test-covid/%s.png'%str(i)
    #print(i)
    klasordenalinmisresim=0
    #string = 'test-covid/cp1.png'
    girisverisi=np.array([])
    klasordenalinmisresim=resmiklasordenal(dosya)
    boyutlandirilmisresim=cv2.resize(klasordenalinmisresim, (224, 224))
    girisverisi=np.append(girisverisi, boyutlandirilmisresim)
    girisverisi=np.reshape(girisverisi, (-1, 224, 224, 3))
    gecici=model.predict(girisverisi)
    tahmin_sonuc=np.append(tahmin_sonuc,model.predict(girisverisi),axis=0)
    gercek_cikis=np.append(gercek_cikis,[[0,1]],axis=0)



gercek_cikis = [ np.argmin(t) for t in gercek_cikis]
tahmin_sonuc = [ np.argmin(t) for t in tahmin_sonuc ]

conf_mat = confusion_matrix(gercek_cikis, tahmin_sonuc)

total1=sum(sum(conf_mat))
accuracy1=(conf_mat[0,0]+conf_mat[1,1])/total1
print ('Accuracy : ', accuracy1)
specifity1 = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1])
print('Specificity : ', specifity1)
sensivity1 = conf_mat[1,1]/(conf_mat[1,0]+conf_mat[1,1])
print('Sensitivity : ', sensivity1)
skplt.metrics.plot_confusion_matrix(gercek_cikis, tahmin_sonuc,title="DNN-1 Model", normalize=False)


