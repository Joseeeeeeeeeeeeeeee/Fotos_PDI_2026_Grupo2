import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage import io

import glob #Añadido para poder leer todas las imágenes de la carpeta

#print("Dimensiones (fila, columnas, canales):", imagenes_rgb[0].shape) # fila: 2296, columnas: 4080, canales: 3
#Hay que cortar jsuto en el marco negro de las fotos. Pasar a 2174 x 2296 (Recorte originl)

#Original:           #Recorte:
#ancho  = 4080       #ancho  = 2174
#alto   = 2296       #alto   = 2296

imagenes = sorted(glob.glob("IMG_20260901_*.jpg"))

imagenes_RGB = []
imagenes_BGR = []

#Lee todas las imagenes en orden de la carpeta y las convierte en RGB. Muestra la original y luego en RGB
for i in imagenes:
    imagen = cv.imread(i)

    if imagen is None:
        print(f'Error al cargar la imagen {i}')

    else:
        imagen_RGB = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)

        roi_RGB = imagen_RGB[0:2296, 646:2820] #Recorte de la imagen para eliminar el marco negro
        roi_BGR = imagen[0:2296, 646:2820]

        imagenes_RGB.append(roi_RGB)
        imagenes_BGR.append(roi_BGR)

for i in imagenes_RGB:
    plt.imshow(i)
    plt.axis("off")
    plt.show()

    gris = cv.cvtColor(i, cv.COLOR_RGB2GRAY)

    R = i[:, :, 0]
    G = i[:, :, 1]
    B = i[:, :, 2]

    plt.imshow(gris, cmap="gray")
    plt.title("Escala de grises")
    plt.axis("off")
    plt.show()

    plt.imshow(R, cmap="gray", vmin=0, vmax=255)
    plt.title("Canal R")
    plt.axis("off")
    plt.show()

    plt.imshow(G, cmap="gray", vmin=0, vmax=255)
    plt.title("Canal G")
    plt.axis("off")
    plt.show()

    plt.imshow(B, cmap="gray", vmin=0, vmax=255)
    plt.title("Canal B")
    plt.axis("off")
    plt.show()


#                   Actualizar github
#git add .
#git commit -m "Añadir nueva funcion"
#git push origin maingit status