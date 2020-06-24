import numpy as np
import cv2
from tkinter import filedialog
from tkinter import *

pencere = Tk()
liste = Listbox(bg="white")
liste.pack()
gnulinux_dagitimlari = ["Pardus", "Debian", "Ubuntu", "PclinuxOS", "TruvaLinux", "Gelecek Linux"]
for i in gnulinux_dagitimlari:
    liste.insert(END, i)
def yeni():
    global giris
    pencere2 = Toplevel()
    giris = Entry(pencere2)
    giris.pack()
    btn2 = Button(pencere2, text="tamam",command=ekle)
    btn2.pack()
def ekle():
    liste.insert(END,giris.get())
    giris.delete(0,END)

etiket = Label(text="#################", fg="magenta", bg="light green")
etiket.pack()
btn = Button(text="ekle",bg="orange",fg="navy", command=yeni)
btn.pack()
etiket2 = Label(text="#################", fg="magenta", bg="light green")
etiket2.pack()
mainloop()
