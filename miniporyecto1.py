import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob #Añadido para poder leer todas las imágenes de la carpeta
from skimage.filters import threshold_multiotsu

def imagenes_CLAHE(nombre_archivo, titulo, ancho, alto, x_0, y_0, gamma=1.5):

    imagen = cv.imread(nombre_archivo)

    if imagen is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {nombre_archivo}")

    imagen_RGB = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)

    # ROI
    x1 = x_0 + ancho
    y1 = y_0 + alto
    roi = imagen_RGB[y_0:y1, x_0:x1]

    # Canales
    R = roi[:, :, 0]

    # Crear figura con 1 fila y 2 columnas para mostrar únicamente los canales CLAHE
    imagen_clahe=aplicar_clahe(R)

    return imagen_clahe


#----------------------------------------------------
#----------------- Segmentaciones  ------------------
#----------------------------------------------------
def segmentacion(image,k):
    
    #elegir_umbral(image,T=128,k=k)
    #umbral_iterativo(image,T0=None,tol=0.5,max_iter=100,k=k)
    #otsu(image,k=k)
    #regiones(image,k=k)
    #multi_otsu(image,k=k)
    k_means(image,4,k=k)
    return None


def elegir_umbral(image,T,k): 
    _, masc=cv.threshold(image,T,255,cv.THRESH_BINARY)

    cv.namedWindow(f'Umbral T={T} - Imagen {k+1}',cv.WINDOW_NORMAL)
    cv.imshow(f'Umbral T={T} - Imagen {k+1}',masc)
    cv.resizeWindow(f'Umbral T={T} - Imagen {k+1}',800,600)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return None

def umbral_iterativo(image,T0,tol,max_iter,k):
    gris=image.astype(np.float32)
    T=float(image.mean() if T0 is None else T0)

    for _ in range(max_iter):
        c0=gris[gris<=T]
        c1=gris[gris>T]

        if len(c0)==0 or len(c1)==0:
            break

        mu0=c0.mean()
        mu1=c1.mean()
        T_nuevo=(mu0+mu1)/2

        if abs(T_nuevo - T) < tol:
            T=T_nuevo
            break

        T=T_nuevo

    mask=np.where(gris>T,255,0).astype(np.uint8)
    
    cv.namedWindow(f'Umbral Iterativo - Imagen {k+1}',cv.WINDOW_NORMAL)
    cv.imshow(f'Umbral Iterativo - Imagen {k+1}',mask)
    cv.resizeWindow(f'Umbral Iterativo - Imagen {k+1}',800,600)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return None

def otsu(image,k):
    T_otsu,mask= cv.threshold(image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    cv.namedWindow(f'Otsu - Imagen {k+1}',cv.WINDOW_NORMAL)
    cv.imshow(f'Otsu - Imagen {k+1}',mask)
    cv.resizeWindow(f'Otsu - Imagen {k+1}',800,600)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return None

def regiones(image,k):
    blue=image[:,:]
    umbrales=[41,69,127]
    clases=np.digitize(blue,umbrales)
    niveles=np.linspace(0,255,4).astype(np.uint8)
    resultado=niveles[clases]

    cv.namedWindow(f'Regiones - Imagen {k+1}',cv.WINDOW_NORMAL)
    cv.imshow(f'Regiones - Imagen {k+1}',resultado)
    cv.resizeWindow(f'Regiones - Imagen {k+1}',800,600)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return None

def multi_otsu(image,k):
    
    umbrales=threshold_multiotsu(image,classes=4)
    clases=np.digitize(image,bins=umbrales)
    niveles=np.linspace(0,255,4).astype(np.uint8)
    resultado=niveles[clases]

    cv.namedWindow(f'Multi-Otsu - Imagen {k+1}',cv.WINDOW_NORMAL)
    cv.imshow(f'Multi-Otsu - Imagen {k+1}',resultado)
    cv.resizeWindow(f'Multi-Otsu - Imagen {k+1}',800,600)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return None

def k_means(image,K,k):
    
    X=image.reshape((-1,1)).astype(np.float32)

    criterio=(cv.TERM_CRITERIA_EPS+cv.TermCriteria_MAX_ITER,100,0.2)
    _,labels,centers=cv.kmeans(X,K,None,criterio,10,cv.KMEANS_PP_CENTERS)

    centers=centers.flatten()
    orden=np.argsort(centers)
    remap=np.zeros(K,dtype=np.uint8)

    for nuevo, viejo in enumerate(orden):
        remap[viejo]=nuevo
    
    labels_ord=remap[labels.flatten()].reshape(image.shape)
    niveles=np.linspace(0,255,K).astype(np.uint8)
    resultado=niveles[labels_ord]

    cv.namedWindow(f'Segmentacion K-means (K={K}) - Imagen {k+1}',cv.WINDOW_NORMAL)
    cv.imshow(f'Segmentacion K-means (K={K}) - Imagen {k+1}',resultado)
    cv.resizeWindow(f'Segmentacion K-means (K={K}) - Imagen {k+1}',800,600)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return None    

#----------------------------------------------------
#----------------- Histograma base ------------------
#----------------------------------------------------
def histograma(img):
    """Histograma base de una imagen uint8 en escala de grises."""
    return cv.calcHist([img], [0], None, [256], [0, 256]).ravel()

#-----------------------------------------------------------
#----------------- Histograma normalizado ------------------
#-----------------------------------------------------------
def histograma_normalizado(img):
    """Histograma dividido por el número total de píxeles."""
    h = histograma(img)
    return h / h.sum()

#--------------------------------------------------------
#------------------ Mostrar histograma ------------------
#--------------------------------------------------------
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

#----------------------------------------------------------------------------------
#------------------ Transformaciones de histograma CLAHE - GAMMA ------------------
#----------------------------------------------------------------------------------
def aplicar_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):

    clahe = cv.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(img)

def aplicar_gamma(img, gamma=1.5):

    r = np.arange(256, dtype=np.float32)

    lut = 255 * (r / 255.0) ** gamma

    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return cv.LUT(img, lut)

#----------------------------------------------------------------
#---------------- Comparación de transformaciones ---------------
#----------------------------------------------------------------
def comparar_transformaciones(nombre_archivo,titulo, ancho, alto, x_0, y_0, gamma=1.5):

    imagen = cv.imread(nombre_archivo)
    
    if imagen is None:
        raise FileNotFoundError(
            f"No se pudo leer la imagen: {nombre_archivo}"
        )

    imagen_RGB = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)

    # ROI
    x1 = x_0 + ancho
    y1 = y_0 + alto
    roi = imagen_RGB[y_0:y1, x_0:x1]

    # Canales
    R = roi[:, :, 0]
    G = roi[:, :, 1]

    canales = [
        ("R", R),
        ("G", G)
    ]
    
    for nombre_canal, canal in canales:

        # Aplicar transformaciones
        imagen_clahe = aplicar_clahe(canal)
        imagen_gamma = aplicar_gamma(canal, gamma)

        # Histogramas
        hist_original = histograma_normalizado(canal)
        hist_clahe = histograma_normalizado(imagen_clahe)
        hist_gamma = histograma_normalizado(imagen_gamma)

        k = np.arange(256)

        # Matriz 3x2
        fig, ax = plt.subplots(3, 2, figsize=(12, 12))

        #------------------------------------------------
        # Fila 1: imagen original + histograma original
        #------------------------------------------------

        ax[0, 0].imshow(canal, cmap="gray", vmin=0, vmax=255)
        ax[0, 0].set_title(f"Canal {nombre_canal} original")
        ax[0, 0].axis("off")

        ax[0, 1].plot(k, hist_original)
        ax[0, 1].set_xlim(0, 255)
        ax[0, 1].set_xlabel("Intensidad k")
        ax[0, 1].set_ylabel("p[k]")
        ax[0, 1].set_title("Histograma original")
        ax[0, 1].grid(alpha=0.2)

        #------------------------------------------------
        # Fila 2: CLAHE + histograma CLAHE
        #------------------------------------------------

        ax[1, 0].imshow(imagen_clahe, cmap="gray", vmin=0, vmax=255)
        ax[1, 0].set_title("CLAHE")
        ax[1, 0].axis("off")

        ax[1, 1].plot(k, hist_clahe)
        ax[1, 1].set_xlim(0, 255)
        ax[1, 1].set_xlabel("Intensidad k")
        ax[1, 1].set_ylabel("p[k]")
        ax[1, 1].set_title("Histograma CLAHE")
        ax[1, 1].grid(alpha=0.2)

        #------------------------------------------------
        # Fila 3: Gamma + histograma Gamma
        #------------------------------------------------

        ax[2, 0].imshow(imagen_gamma, cmap="gray", vmin=0, vmax=255)
        ax[2, 0].set_title(f"Gamma = {gamma}")
        ax[2, 0].axis("off")

        ax[2, 1].plot(k, hist_gamma)
        ax[2, 1].set_xlim(0, 255)
        ax[2, 1].set_xlabel("Intensidad k")
        ax[2, 1].set_ylabel("p[k]")
        ax[2, 1].set_title(f"Histograma Gamma = {gamma}")
        ax[2, 1].grid(alpha=0.2)

        fig.suptitle(
            f"{titulo} - Canal {nombre_canal}",
            fontsize=14
        )

        plt.tight_layout()
        plt.show()

        #elegir_umbral(imagen_clahe,nombre_canal,T=128)
        #umbral_iterativo(imagen_clahe,nombre_canal,T0=None,tol=0.5,max_iter=100)
        #otsu(imagen_clahe,nombre_canal)
        #regiones(imagen_clahe,nombre_canal)
        #multi_otsu(imagen_clahe,nombre_canal)
        #k_means(imagen_clahe,K=4)
    return None

#---------------------------------------------------------------
#------------------ Procesamiento de muestras ------------------
#---------------------------------------------------------------
def procesar_muestra(nombre_archivo, titulo, ancho, alto, x_0, y_0):

    imagen = cv.imread(nombre_archivo)

    if imagen is None:
        raise FileNotFoundError(
            f"No se pudo leer la imagen: {nombre_archivo}"
        )

    imagen_RGB = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)

    #roi_base = imagen_RGB[0:2296, 646:2820]

    x1 = x_0 + ancho
    y1 = y_0 + alto
    roi = imagen_RGB[y_0:y1, x_0:x1]

    R = roi[:, :, 0]
    G = roi[:, :, 1]

    hist_R = histograma_normalizado(R) #histograma(R) a histograma_normalizado(R)
    hist_G = histograma_normalizado(G) #histograma(G) a histograma_normalizado(G)

    k = np.arange(256)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    
    #------------------------------------------
    # ---------------- Canal R ----------------
    #------------------------------------------
    ax[0, 0].imshow(R, cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("Canal R")
    ax[0, 0].axis("off")

    ax[0, 1].plot(k, hist_R)
    ax[0, 1].set_xlim(0, 255)
    ax[0, 1].set_xlabel("Intensidad k")
    ax[0, 1].set_ylabel("p[k]")
    ax[0, 1].set_title("Histograma normalizado R")
    ax[0, 1].grid(alpha=0.2)

    #------------------------------------------
    # ---------------- Canal G ----------------
    #------------------------------------------
    ax[1, 0].imshow(G, cmap="gray", vmin=0, vmax=255)
    ax[1, 0].set_title("Canal G")
    ax[1, 0].axis("off")

    ax[1, 1].plot(k, hist_G)
    ax[1, 1].set_xlim(0, 255)
    ax[1, 1].set_xlabel("Intensidad k")
    ax[1, 1].set_ylabel("p[k]")
    ax[1, 1].set_title("Histograma normalizado G")
    ax[1, 1].grid(alpha=0.2)

    fig.suptitle(titulo, fontsize=14)

    plt.tight_layout()
    plt.show()








#---------------------------------------------------------------        ---------------------------
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

#---------------------------------------------------------------------------------------
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

ancho = [1861, 1954, 2108, 1286, 1860, 1984, 797, 824, 1370, 1383, 1036, 1240]  # w
alto = [1867, 1452, 1652, 2052, 1164, 988, 1343, 1240, 591, 744, 950, 673]     # l 5 - 1150

x_0 = [720, 646, 788, 995, 936, 843, 1230, 1358, 924, 1380, 1036, 1131]
y_0 = [276, 285, 239, 0, 385, 582, 198, 158, 598, 270, 460, 709]

mostrar_histogramas = False #Cambiar a True para mostrar los plt. Dejar en False para no mostrar

muestras = [
    ("IMG_20260901_161412945.jpg", "Hoja normal grande 1"), #1
    ("IMG_20260901_161426657.jpg", "Hoja normal grande 2"), #2

    ("IMG_20260901_161504068.jpg", "Hoja mojada grande 1"), #3
    ("IMG_20260901_161539928.jpg", "Hoja mojada grande 2"), #4

    ("IMG_20260901_161616155.jpg", "Hoja seca grande 1"),   #5
    ("IMG_20260901_161635904.jpg", "Hoja seca grande 2"),   #6

    ("IMG_20260901_161710879.jpg", "Hoja normal pequeña 1"),#7
    ("IMG_20260901_161741846.jpg", "Hoja normal pequeña 2"),#8

    ("IMG_20260901_161811458.jpg", "Hoja mojada pequeña 1"),#9
    ("IMG_20260901_161851864.jpg", "Hoja mojada pequeña 2"),#10 

    ("IMG_20260901_161918685.jpg", "Hoja seca pequeña 1"),  #11
    ("IMG_20260901_161940183.jpg", "Hoja seca pequeña 2")   #12
]
if mostrar_histogramas:
    for i, (archivo, titulo) in enumerate(muestras):
        procesar_muestra(archivo, titulo, ancho[i], alto[i], x_0[i], y_0[i])


#---------------------------------------------------------------------------------------------
#---------------------------------- 1.2.3 ----------------------------------------------------
#Apliquen y comparen al menos dos transformaciones o técnicas de procesamiento de histograma
#vistas en clases. Muestren la imagen y su histograma antes y después del procesamiento
#---------------------------------------------------------------------------------------------

#for i, (archivo, titulo) in enumerate(muestras):
    #comparar_transformaciones(archivo,titulo, ancho[i], alto[i], x_0[i], y_0[i], gamma=1.5)


CLAHEs=[]

for n, (archivo, titulo) in enumerate(muestras):    
    CLAHEs.append(imagenes_CLAHE(archivo, titulo, ancho[n], alto[n], x_0[n], y_0[n], gamma=1.5))

for k in range(len(CLAHEs)):
    segmentacion(CLAHEs[k],k)

#Resultados de las segmentaciones: perfeccion(de 0 a 1)

# Segmentaciones e Imagenes:| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
# elegir umbral (T=128):    |0.9| 1 |0.7| 1 |0.2| 0 | 1 |0.2|0.3|  1 | 0  |  0 | p=
# umbral iterativo:         |0.5| 1 |0.3| 1 |0.2| 0 | 1 | 0 | 0 |0.9 |0.4 |  0 | p=
# otsu:                     |0.5| 1 |0.4| 1 | 0 | 0 | 1 | 0 | 0 |0.9 |0.3 |  0 | p=
# regiones:                 |0.4| 1 |0.3| 1 |0.1| 0 | 1 |0.1|0.2| 1  |  0 |  0 | p=
# multi otsu:               |0.9| 1 |0.5| 1 |0.8| 0 |0.9|0.1| 0 | 1  | 0.4|  0 | p=
# k-means (K=4):            |   |   |   |   |   |   |   |   |   |    |    |    | p=
# k-means (K=8):            |   |   |   |   |   |   |   |   |   |    |    |    | p=


#------------------ Actualizar github ------------------

#git status , despues
#git add miniporyecto1.py 
#git commit -m "Añadir nueva funcion"
#git push origin main
#git fetch origin
#git reset --hard origin/main