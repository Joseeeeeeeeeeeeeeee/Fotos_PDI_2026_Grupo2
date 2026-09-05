import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage import io

import glob #Añadido para poder leer todas las imágenes de la carpeta

def histograma(img):
    """Histograma base de una imagen uint8 en escala de grises."""
    return cv.calcHist([img], [0], None, [256], [0, 256]).ravel()


def histograma_normalizado(img):
    """Histograma dividido por el número total de píxeles."""
    h = histograma(img)
    return h / h.sum()


def mostrar_histograma(img, titulo="", normalizado=False):

    h = histograma_normalizado(img) if normalizado else histograma(img)
    k = np.arange(256)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title(titulo)
    ax[0].axis("off")

    ax[1].plot(k, h)
    ax[1].set_xlim(0, 255)
    ax[1].set_xlabel("Intensidad k")
    ax[1].set_ylabel("p[k]" if normalizado else "h[k]")
    ax[1].set_title(
        "Histograma normalizado"
        if normalizado
        else "Histograma"
    )
    ax[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


#----------------------------------1.2.1-----------------------------------------
# Carguen, visualicen y recorten una región de interés cuando corresponda. En imágenes RGB,
# comparen la escala de grises y los canales de color que resulten útiles para el problema.


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

    gris = cv.cvtColor(i, cv.COLOR_RGB2GRAY)

    R = i[:, :, 0]
    G = i[:, :, 1]
    B = i[:, :, 2]

    #plt.figure(figsize=(18, 5))

    #plt.subplot(1, 5, 1)
    #plt.imshow(i)
    #plt.title("Original RGB")
    #plt.axis("off")

    #plt.subplot(1, 5, 2)
    #plt.imshow(gris, cmap="gray", vmin=0, vmax=255)
    #plt.title("Escala de grises")
    #plt.axis("off")

    #plt.subplot(1, 5, 3)
    #plt.imshow(R, cmap="gray", vmin=0, vmax=255)
    #plt.title("Canal R (rojo)")
    #plt.axis("off")

    #plt.subplot(1, 5, 4)
    #plt.imshow(G, cmap="gray", vmin=0, vmax=255)
    #plt.title("Canal G (verde)")
    #plt.axis("off")

    #plt.subplot(1, 5, 5)
    #plt.imshow(B, cmap="gray", vmin=0, vmax=255)
    #plt.title("Canal B (azul)")
    #plt.axis("off")

    #plt.tight_layout() #Ajusta el tamaño de forma automatica
    #plt.show()

#-------------------Definir que realizar, en este caso se resaltaraán las nervaduras (Venas)---------------------
# Canales útiles: R y G (De forma visual)

#---------------------------------- 1.2.2 ----------------------------------
# Obtengan histogramas de imágenes representativas de cada condición y relacionen su forma con
# el contraste y la visibilidad de la característica de interés.


#---------------------------------- Hoja normal grande ----------------------------------

hoja_normal_grande = cv.imread("IMG_20260901_161426657.jpg")

if hoja_normal_grande is None:
    raise FileNotFoundError("No se pudo leer la imagen")

hoja_normal_grande_RGB = cv.cvtColor(hoja_normal_grande, cv.COLOR_BGR2RGB)

roi_hoja_normal_grande = hoja_normal_grande_RGB[0:2296, 646:2820]

R = roi_hoja_normal_grande[:, :, 0]
G = roi_hoja_normal_grande[:, :, 1]

mostrar_histograma(R, "Hoja normal grande - Canal R")
mostrar_histograma(G, "Hoja normal grande - Canal G")


#----------------------------------Hoja mojada grande ----------------------------------

hoja_mojada_grande = cv.imread("PONER_ARCHIVO_HOJA_MOJADA_GRANDE.jpg")

if hoja_mojada_grande is None:
    raise FileNotFoundError("No se pudo leer la imagen")

hoja_mojada_grande_RGB = cv.cvtColor(hoja_mojada_grande, cv.COLOR_BGR2RGB)

roi_hoja_mojada_grande = hoja_mojada_grande_RGB[0:2296, 646:2820]

R = roi_hoja_mojada_grande[:, :, 0]
G = roi_hoja_mojada_grande[:, :, 1]

mostrar_histograma(R, "Hoja mojada grande - Canal R")
mostrar_histograma(G, "Hoja mojada grande - Canal G")


#---------------------------------- Hoja seca grande ----------------------------------

hoja_seca_grande = cv.imread("PONER_ARCHIVO_HOJA_SECA_GRANDE.jpg")

if hoja_seca_grande is None:
    raise FileNotFoundError("No se pudo leer la imagen")

hoja_seca_grande_RGB = cv.cvtColor(hoja_seca_grande, cv.COLOR_BGR2RGB)

roi_hoja_seca_grande = hoja_seca_grande_RGB[0:2296, 646:2820]

R = roi_hoja_seca_grande[:, :, 0]
G = roi_hoja_seca_grande[:, :, 1]

mostrar_histograma(R, "Hoja seca grande - Canal R")
mostrar_histograma(G, "Hoja seca grande - Canal G")


#---------------------------------- Hoja normal pequeña ----------------------------------

hoja_normal_pequena = cv.imread("PONER_ARCHIVO_HOJA_NORMAL_PEQUENA.jpg")

if hoja_normal_pequena is None:
    raise FileNotFoundError("No se pudo leer la imagen")

hoja_normal_pequena_RGB = cv.cvtColor(hoja_normal_pequena, cv.COLOR_BGR2RGB)

roi_hoja_normal_pequena = hoja_normal_pequena_RGB[0:2296, 646:2820]

R = roi_hoja_normal_pequena[:, :, 0]
G = roi_hoja_normal_pequena[:, :, 1]

mostrar_histograma(R, "Hoja normal pequeña - Canal R")
mostrar_histograma(G, "Hoja normal pequeña - Canal G")


#---------------------------------- Hoja mojada pequeña ----------------------------------

hoja_mojada_pequena = cv.imread("PONER_ARCHIVO_HOJA_MOJADA_PEQUENA.jpg")

if hoja_mojada_pequena is None:
    raise FileNotFoundError("No se pudo leer la imagen")

hoja_mojada_pequena_RGB = cv.cvtColor(hoja_mojada_pequena, cv.COLOR_BGR2RGB)

roi_hoja_mojada_pequena = hoja_mojada_pequena_RGB[0:2296, 646:2820]

R = roi_hoja_mojada_pequena[:, :, 0]
G = roi_hoja_mojada_pequena[:, :, 1]

mostrar_histograma(R, "Hoja mojada pequeña - Canal R")
mostrar_histograma(G, "Hoja mojada pequeña - Canal G")


#---------------------------------- Hoja seca pequeña ----------------------------------

hoja_seca_pequena = cv.imread("PONER_ARCHIVO_HOJA_SECA_PEQUENA.jpg")

if hoja_seca_pequena is None:
    raise FileNotFoundError("No se pudo leer la imagen")

hoja_seca_pequena_RGB = cv.cvtColor(hoja_seca_pequena, cv.COLOR_BGR2RGB)

roi_hoja_seca_pequena = hoja_seca_pequena_RGB[0:2296, 646:2820]

R = roi_hoja_seca_pequena[:, :, 0]
G = roi_hoja_seca_pequena[:, :, 1]

mostrar_histograma(R, "Hoja seca pequeña - Canal R")
mostrar_histograma(G, "Hoja seca pequeña - Canal G")








#                   Actualizar github
#git add .
#git commit -m "Añadir nueva funcion"
#git push origin maingit status