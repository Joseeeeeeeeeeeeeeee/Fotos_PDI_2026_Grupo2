import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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



def procesar_muestra(nombre_archivo, titulo, ancho, alto):

    imagen = cv.imread(nombre_archivo)

    if imagen is None:
        raise FileNotFoundError(
            f"No se pudo leer la imagen: {nombre_archivo}"
        )

    imagen_RGB = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)

    roi_base = imagen_RGB[0:2296, 646:2820]

    ancho_base = 2174
    alto_base = 2296

    x_inicio = (ancho_base - ancho) // 2
    y_inicio = (alto_base - alto) // 2

    roi = roi_base[y_inicio:y_inicio + alto, x_inicio:x_inicio + ancho]

    R = roi[:, :, 0]
    G = roi[:, :, 1]

    hist_R = histograma(R)
    hist_G = histograma(G)

    k = np.arange(256)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # ---------------- Canal R ----------------

    ax[0, 0].imshow(R, cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("Canal R")
    ax[0, 0].axis("off")

    ax[0, 1].plot(k, hist_R)
    ax[0, 1].set_xlim(0, 255)
    ax[0, 1].set_xlabel("Intensidad k")
    ax[0, 1].set_ylabel("h[k]")
    ax[0, 1].set_title("Histograma R")
    ax[0, 1].grid(alpha=0.2)

    # ---------------- Canal G ----------------

    ax[1, 0].imshow(G, cmap="gray", vmin=0, vmax=255)
    ax[1, 0].set_title("Canal G")
    ax[1, 0].axis("off")

    ax[1, 1].plot(k, hist_G)
    ax[1, 1].set_xlim(0, 255)
    ax[1, 1].set_xlabel("Intensidad k")
    ax[1, 1].set_ylabel("h[k]")
    ax[1, 1].set_title("Histograma G")
    ax[1, 1].grid(alpha=0.2)

    fig.suptitle(titulo, fontsize=14)

    plt.tight_layout()
    plt.show()









#------------------------------------------------------------------------------------------
#----------------------------------1.2.1---------------------------------------------------
# Carguen, visualicen y recorten una región de interés cuando corresponda. En imágenes RGB,
# comparen la escala de grises y los canales de color que resulten útiles para el problema.
#------------------------------------------------------------------------------------------

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


#---------------------------------- Plot cada 3 fotos ----------------------------------
mostrar_canales = False #Cambiar a True para mostrar los plt. Dejar en False para no mostrar

if mostrar_canales:
    for i in range(0, len(imagenes_RGB), 3):

        grupo = imagenes_RGB[i:i+3]

        fig, ax = plt.subplots(3, 5, figsize=(16, 10))

        for fila, i in enumerate(grupo):

            gris = cv.cvtColor(i, cv.COLOR_RGB2GRAY)

            R = i[:, :, 0]
            G = i[:, :, 1]
            B = i[:, :, 2]

            ax[fila, 0].imshow(i)
            ax[fila, 0].axis("off")

            ax[fila, 1].imshow(gris, cmap="gray", vmin=0, vmax=255)
            ax[fila, 1].axis("off")

            ax[fila, 2].imshow(R, cmap="gray", vmin=0, vmax=255)
            ax[fila, 2].axis("off")

            ax[fila, 3].imshow(G, cmap="gray", vmin=0, vmax=255)
            ax[fila, 3].axis("off")

            ax[fila, 4].imshow(B, cmap="gray", vmin=0, vmax=255)
            ax[fila, 4].axis("off")

            if fila == 0:
                ax[fila, 0].set_title("Original RGB")
                ax[fila, 1].set_title("Escala de grises")
                ax[fila, 2].set_title("Canal R")
                ax[fila, 3].set_title("Canal G")
                ax[fila, 4].set_title("Canal B")

        plt.tight_layout() #Ajusta el tamaño de forma automatica
        plt.show()

#---------------------------------------------------------------------------------------------
#---------------------------------- 1.2.2 ----------------------------------------------------
# Obtengan histogramas de imágenes representativas de cada condición y relacionen su forma con
# el contraste y la visibilidad de la característica de interés.
#---------------------------------------------------------------------------------------------

ancho = [1861, 1954, 2108, 1288, 1860, 2033, 800, 825, 1372, 1383, 1079, 1285]
alto = [1867, 1452, 1663, 2052, 1150, 1028, 1349, 1256, 591, 744, 950, 669]

muestras = [
    ("IMG_20260901_161412945.jpg", "Hoja normal grande 1"),
    ("IMG_20260901_161426657.jpg", "Hoja normal grande 2"),

    ("IMG_20260901_161504068.jpg", "Hoja mojada grande 1"),
    ("IMG_20260901_161539928.jpg", "Hoja mojada grande 2"),

    ("IMG_20260901_161616155.jpg", "Hoja seca grande 1"),
    ("IMG_20260901_161635904.jpg", "Hoja seca grande 2"),

    ("IMG_20260901_161710879.jpg", "Hoja normal pequeña 1"),
    ("IMG_20260901_161741846.jpg", "Hoja normal pequeña 2"),

    ("IMG_20260901_161811458.jpg", "Hoja mojada pequeña 1"),
    ("IMG_20260901_161851864.jpg", "Hoja mojada pequeña 2"),

    ("IMG_20260901_161918685.jpg", "Hoja seca pequeña 1"),
    ("IMG_20260901_161940183.jpg", "Hoja seca pequeña 2")
]

for i, (archivo, titulo) in enumerate(muestras):
    procesar_muestra(archivo, titulo, ancho[i], alto[i])







#------------------ Actualizar github ------------------

#git status
#git add miniporyecto1.py
#git commit -m "Añadir nueva funcion"
#git push origin main