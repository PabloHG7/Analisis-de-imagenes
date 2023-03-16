from tkinter import *
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('imagen.bmp')
cv2.imshow('imagen.bmp', img)

def histograma():
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist, color='gray' )

    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.show()

def Contraccion():

    cmax=250
    cmin=100
    img.astype(dtype=np.int32)
    alto=img.shape[0]
    ancho=img.shape[1]
    try:
        fmax=np.max(img)
        fmin=np.min(img)
        for h in range(alto):
            for w in range(ancho):
                img[h][w][0]=int(round(((cmax-cmin)/(fmax-fmin))*(img[h][w][0]-fmin)+cmin))
                img[h][w][1]=int(round(((cmax-cmin)/(fmax-fmin))*(img[h][w][1]-fmin)+cmin))
                img[h][w][2]=int(round(((cmax-cmin)/(fmax-fmin))*(img[h][w][2]-fmin)+cmin))
        img[img>255]=255
        img[img<0]=0
    except:
        fmax=np.max(img)
        fmin=np.min(img)
        for h in range(alto):
            for w in range(ancho):
                img[h][w]=int(round(((cmax-cmin)/(fmax-fmin))(img[h][w]-fmin)+cmin))
        img[img>255]=255
        img[img<0]=0
    img.astype(dtype=np.uint8)            
            


    cv2.imshow('imagen.bmp', img)

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist, color='gray' )

    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.show()

def desplazar():
    x = img.shape[1] #columnas
    y = img.shape[0] # filas

    M = np.float32([[1, 0, 25], [0, 1, 50]])
    M = np.float32([[1, 0, 0], [0, 1, 3]])

    # Llevamos a cabo la transformación.
    shifted = cv2.warpAffine(img,M, (img.shape[1], img.shape[0]))

    cv2.imshow('img', shifted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def expansion():
    image = cv2.imread('imagen.bmp')
    imageOut = image.copy()
    imageOut.astype(dtype=np.int32)
    alto = image.shape[0]
    ancho = image.shape[1]
    maxi = 10
    mini = 100



    try:
        fmax = np.max(imageOut)
        fmin = np.min(imageOut)
        for h in range(alto):
            for w in range(ancho):
                imageOut[h][w][0] = int(round(((maxi-mini)/(fmax-fmin))*(imageOut[h][w][0]-fmin)+mini))
                imageOut[h][w][1] = int(round(((maxi-mini)/(fmax-fmin))*(imageOut[h][w][1]-fmin)+mini))
                imageOut[h][w][2] = int(round(((maxi-mini)/(fmax-fmin))*(imageOut[h][w][2]-fmin)+mini))
        imageOut[imageOut>255] = 255
        imageOut[imageOut<0] = 0
    except:
        fmax = np.max(imageOut)
        fmin = np.min(imageOut)
        for h in range(alto):
            for w in range(ancho):
                imageOut[h][w] = int(round(((maxi-mini)/(fmax-fmin))*(imageOut[h][w]-fmin)+mini))
        imageOut[imageOut>255] = 255
        imageOut[imageOut<0] = 0


    cv2.imshow('imagen', image.astype(dtype=np.uint8))
    cv2.imshow('imagen de salida', imageOut)
    hist1 = cv2.calcHist([image], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([imageOut], [0], None, [256], [0,256])
    plt.plot(hist2, color='gray')
    plt.xlabel('Intensidad de iluminación');
    plt.ylabel('cantidad de pixeles')
    plt.show()
    plt.plot(hist1, color='gray')
    plt.xlabel('Intensidad de iluminación');
    plt.ylabel('cantidad de pixeles')
    plt.show()

def ecuali():
  img= cv2.imread('imagen.tif', cv2.IMREAD_GRAYSCALE)
  img_equ= cv2.equalizeHist(img)

  cv2.imshow('Imagen sin ecualizar', img)
  cv2.imshow('Imagen equ', img_equ)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def ecualiR():
    img= cv2. imread("Imagen.bmp", cv2.IMREAD_GRAYSCALE)
    width, height =img.shape
    cv2.imshow('Imagen', img)

    x=np.linspace(0,255, num=256, dtype= np.uint8)#
    y=np.zeros(256)#
    y_2=np.zeros(256)#
    img_ecu= np.zeros(img.shape, img.dtype)

    ####        Histograma
    for i in range(width):
        for j in range (height):
            v_pix=img[i,j]#Sacamos el valor del pixel
            y[v_pix]= y[v_pix]+1 #cuantas veces aparece cada pixel
            
    r_min=x[0]
    print('Valor de rmin ',r_min)
    f2=2*(5.5)**2
    print('Valor de rmin ',f2)

    total_pix=width*height
    print('widt',width)
    print('hei',height)
    print('Total pix',total_pix)
    suma=0
    log=0

    for i in range(width):
        for j in range (height):
            for s in range (img[i,j]):
                suma= suma + y[s]#sacamos el valr de la sumatoria
                log=1/(1-suma/total_pix)
            img_ecu[i,j] = r_min+f2*np.log(log)  
            suma=0
            log=0

    for i in range(width):
        for j in range (height):
            v_pix=img_ecu[i,j]#Sacamos el valor del pixel
            y_2[v_pix]= y_2[v_pix]+1 #cuantas veces aparece cada pixel
      
    cv2.imshow("Imagen ecualizada", img_ecu) 
    plt.subplot(1,2,1), plt.bar(x,y)
    plt.subplot(1,2,2), plt.bar(x,y_2)

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Configuración de la raíz
root = Tk()
root.config(bd=15)

Button(root, text="Sacar histograma", command=histograma).pack(side="left", padx=5, pady=5)
Button(root, text="Contracción", command=Contraccion).pack(side="left")
Button(root, text="Desplazamiento", command=desplazar).pack(side="left")
Button(root, text="Expansion", command=expansion).pack(side="left")
Button(root, text="Ecualización", command=ecuali).pack(side="left")
Button(root, text="Ecualización R", command=ecualiR).pack(side="left")

cv2.destroyAllWindows()
