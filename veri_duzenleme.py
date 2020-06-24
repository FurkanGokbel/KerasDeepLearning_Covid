import cv2
import numpy as np

def resmiklasordenal(dosyaadi):
    resim=cv2.imread("%s"%dosyaadi)      #resimi al
    return resim

girisverisi=np.array([])    #bunun içine atıcaz resimler
for a in range(400):
    klasordenalinmisresim=0
    a=a+1
    a='%d'%a
    a=a.zfill(3)
    string = 'covid19-positive/%s.png'%a
    klasordenalinmisresim=resmiklasordenal(string)
    boyutlandirilmisresim=cv2.resize(klasordenalinmisresim,(224,224))
    girisverisi=np.append(girisverisi,boyutlandirilmisresim)
    print(a)


girisverisi=np.reshape(girisverisi,(-1,224,224,3)) #30 yerine -1 yazdık . Ne kadar varsa al demek
np.save("girisverinizxray",girisverisi)

print(girisverisi)
print(girisverisi.shape)