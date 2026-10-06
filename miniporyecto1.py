import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob #Añadido para poder leer todas las imágenes de la carpeta
from skimage.filters import threshold_multiotsu

#--------------------------------------------------------------------------------------------------------
#----------------- Histograma base ----------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
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

#--------------------------------------------------------------------------------------------------------------------------------------
#------------------ Transformaciones de histograma CLAHE - GAMMA ------------------
#--------------------------------------------------------------------------------------------------------------------------------------
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

def aplicar_ecualizacion(img):

    return cv.equalizeHist(img)


def aplicar_estiramiento(img):

    return cv.normalize(
        img,
        None,
        0,
        255,
        cv.NORM_MINMAX
    ).astype(np.uint8)

#----------------------------------------------------
#----------------- Imagenes CLAHE  ------------------
#----------------------------------------------------

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

#----------------------------------------------------------------
#---------------- Comparación de transformaciones ---------------
#----------------------------------------------------------------

def comparar_transformaciones(nombre_archivo, titulo, ancho, alto, x_0, y_0, gamma=1.5):

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

    # Canales útiles
    R = roi[:, :, 0]
    G = roi[:, :, 1]

    #------------------------------------------------
    # Transformaciones Canal R
    #------------------------------------------------

    transformaciones_R = [
        ("Original", R),
        ("Ecualizacion", aplicar_ecualizacion(R)),
        ("CLAHE", aplicar_clahe(R)),
        (f"Gamma = {gamma}", aplicar_gamma(R, gamma)),
        ("Estiramiento", aplicar_estiramiento(R))
    ]

    #------------------------------------------------
    # Transformaciones Canal G
    #------------------------------------------------

    transformaciones_G = [
        ("Original", G),
        ("Ecualizacion", aplicar_ecualizacion(G)),
        ("CLAHE", aplicar_clahe(G)),
        (f"Gamma = {gamma}", aplicar_gamma(G, gamma)),
        ("Estiramiento", aplicar_estiramiento(G))
    ]

    k = np.arange(256)

    # 5 filas:
    # Original
    # Ecualizacion
    # CLAHE
    # Gamma
    # Estiramiento
    #
    # 4 columnas:
    # Imagen R | Histograma R | Imagen G | Histograma G

    fig, ax = plt.subplots(5, 4, figsize=(20, 22))

    for fila in range(5):

        nombre_R, imagen_R = transformaciones_R[fila]
        nombre_G, imagen_G = transformaciones_G[fila]

        hist_R = histograma_normalizado(imagen_R)
        hist_G = histograma_normalizado(imagen_G)

        #------------------------------------------------
        # Canal R - Imagen
        #------------------------------------------------

        ax[fila, 0].imshow(
            imagen_R,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        ax[fila, 0].set_title(
            f"R - {nombre_R}"
        )

        ax[fila, 0].axis("off")

        #------------------------------------------------
        # Canal R - Histograma
        #------------------------------------------------

        ax[fila, 1].plot(
            k,
            hist_R
        )

        ax[fila, 1].set_xlim(0, 255)
        ax[fila, 1].set_xlabel("Intensidad k")
        ax[fila, 1].set_ylabel("p[k]")

        ax[fila, 1].grid(alpha=0.2)

        #------------------------------------------------
        # Canal G - Imagen
        #------------------------------------------------

        ax[fila, 2].imshow(
            imagen_G,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        ax[fila, 2].set_title(
            f"G - {nombre_G}"
        )

        ax[fila, 2].axis("off")

        #------------------------------------------------
        # Canal G - Histograma
        #------------------------------------------------

        ax[fila, 3].plot(
            k,
            hist_G
        )

        ax[fila, 3].set_xlim(0, 255)
        ax[fila, 3].set_xlabel("Intensidad k")
        ax[fila, 3].set_ylabel("p[k]")

        ax[fila, 3].grid(alpha=0.2)

    fig.suptitle(
        titulo,
        fontsize=16
    )

    plt.tight_layout()
    plt.show()

    return None
#-------------------------------------------------------------------------------------------------------------------
#------------------ Procesamiento de muestras ------------------
#-------------------------------------------------------------------------------------------------------------------
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

#--------------------------------------------------------------------------------------------------------
#----------------- Segmentaciones  ------------------
#--------------------------------------------------------------------------------------------------------
def segmentar(CLAHEs):
    
    S=[]

    S.append(multi_otsu(CLAHEs[0],k=0))
    S.append(umbral_iterativo(CLAHEs[1],T0=None,tol=0.5,max_iter=100,k=1))
    S.append(elegir_umbral(CLAHEs[2],T=128,k=2))
    S.append(elegir_umbral(CLAHEs[3],T=128,k=3))
    S.append(multi_otsu(CLAHEs[4],k=4))
    S.append(k_means(CLAHEs[5],K=12,k=5))
    S.append(elegir_umbral(CLAHEs[6],T=128,k=6))
    S.append(k_means(CLAHEs[7],K=8,k=7))
    S.append(k_means(CLAHEs[8],K=8,k=8))
    S.append(elegir_umbral(CLAHEs[9],T=128,k=9))
    S.append(k_means(CLAHEs[10],K=12,k=10))
    S.append(k_means(CLAHEs[11],K=12,k=11))
    
    return S

def elegir_umbral(image,T,k): 
    _, masc=cv.threshold(image,T,255,cv.THRESH_BINARY)

    return (masc,"Umbral")

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
    
    return (mask,"Umbral Iterativo")

def otsu(image,k):
    T_otsu,mask= cv.threshold(image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    return (mask,"Otsu")

def regiones(image,k):
    blue=image[:,:]
    umbrales=[41,69,127]
    clases=np.digitize(blue,umbrales)
    niveles=np.linspace(0,255,4).astype(np.uint8)
    resultado=niveles[clases]

    return (resultado,"Regiones")

def multi_otsu(image,k):
    
    umbrales=threshold_multiotsu(image,classes=4)
    clases=np.digitize(image,bins=umbrales)
    niveles=np.linspace(0,255,4).astype(np.uint8)
    resultado=niveles[clases]

    return (resultado,"Multi-Otsu")

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

    return (resultado,"K-Means")

#--------------------------------------------------------------------------------------------------------
#----------------- Morfologia      ----------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

def morfologia(Segmentaciones):
    M=[]

    M.append(cierre(Segmentaciones[0][0],0))
    M.append(cierre(Segmentaciones[1][0],1))
    M.append(cierre(Segmentaciones[2][0],2))
    M.append(cierre(Segmentaciones[3][0],3))
    M.append(cierre(Segmentaciones[4][0],4))
    M.append(erosionar(Segmentaciones[5][0],5))
    M.append(cierre(Segmentaciones[6][0],6)) #Conserva las venas del mismo grosor
    M.append(cierre(Segmentaciones[7][0],7))
    M.append(cierre(Segmentaciones[8][0],8))
    M.append(cierre(Segmentaciones[9][0],9))
    M.append(apertura(Segmentaciones[10][0],10))
    M.append(apertura(Segmentaciones[11][0],11))
    #dilatacion(mask,B,x)
    #apertura(mask,B,x)
    #extraer(mask,B,x)
    
    return M

def dilatacion(image,x):
    mask=image.astype(np.uint8)
    _, mask=cv.threshold(mask,127,255,cv.THRESH_BINARY)

    B=cv.getStructuringElement(cv.MORPH_RECT,(5,5))

    dilatada=cv.dilate(mask,B,iterations=1)

    return (dilatada,"Dilatacion")

def apertura(image,x):
    mask=image.astype(np.uint8)
    _, mask=cv.threshold(mask,127,255,cv.THRESH_BINARY)

    B=cv.getStructuringElement(cv.MORPH_RECT,(5,5))

    abierta=cv.morphologyEx(mask,cv.MORPH_OPEN,B)

    return (abierta,"Apertura")

def erosionar(image,x):
    mask=image.astype(np.uint8)
    _, mask=cv.threshold(mask,127,255,cv.THRESH_BINARY)

    B=cv.getStructuringElement(cv.MORPH_RECT,(5,5))

    erosionada=cv.erode(mask,B,iterations=1)

    return (erosionada,"Erosion")

def cierre(image,x):
    mask=image.astype(np.uint8)
    _, mask=cv.threshold(mask,127,255,cv.THRESH_BINARY)

    B=cv.getStructuringElement(cv.MORPH_RECT,(5,5))

    cerrada=cv.morphologyEx(mask,cv.MORPH_CLOSE,B)

    return (cerrada,"Cierre")

def extraer(image,x):
    mask=image.astype(np.uint8)
    _, mask=cv.threshold(mask,127,255,cv.THRESH_BINARY)

    B=cv.getStructuringElement(cv.MORPH_RECT,(5,5))

    extraccion=cv.erode(mask,B)

    borde=cv.subtract(mask,extraccion)

    return (borde,"Extraccion")

def comparar_morfologias(segmentacion):

    dilatada = dilatacion(segmentacion, 0)[0]
    erosionada = erosionar(segmentacion, 0)[0]
    abierta = apertura(segmentacion, 0)[0]
    cerrada = cierre(segmentacion, 0)[0]

    fig, ax = plt.subplots(1, 5, figsize=(20, 5))

    ax[0].imshow(segmentacion, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("Segmentacion")

    ax[1].imshow(dilatada, cmap="gray", vmin=0, vmax=255)
    ax[1].set_title("Dilatacion")

    ax[2].imshow(erosionada, cmap="gray", vmin=0, vmax=255)
    ax[2].set_title("Erosion")

    ax[3].imshow(abierta, cmap="gray", vmin=0, vmax=255)
    ax[3].set_title("Apertura")

    ax[4].imshow(cerrada, cmap="gray", vmin=0, vmax=255)
    ax[4].set_title("Cierre")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()

#--------------------------------------------------------------------------------------------------------
#----------------- Graficos    ----------------------
#--------------------------------------------------------------------------------------------------------

def grafico(CLAHEs, Segmentaciones, Morfologias):

    for c in range(len(CLAHEs)):
        clahe=CLAHEs[c]
        segmentada,tipo_s=Segmentaciones[c]
        morfologico,tipo_m=Morfologias[c]

        fig, axs=plt.subplots(1,3, figsize=(15,5))

        axs[0].imshow(clahe, cmap='gray', vmin=0, vmax=255)
        axs[0].set_title(f'CLAHE - Imagen {c+1}')
        axs[0].axis('off')

        axs[1].imshow(segmentada, cmap='gray', vmin=0, vmax=255)
        axs[1].set_title(f'Segmentacion: {tipo_s}')
        axs[1].axis('off')


        axs[2].imshow(morfologico, cmap='gray', vmin=0, vmax=255)
        axs[2].set_title(f'Morfologia: {tipo_m}')
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()

def grafico_secuencia_completa(nombre_archivo, titulo,
                               ancho, alto, x_0, y_0,
                               segmentacion, morfologia):

    # ------------------------------------------------
    # Imagen original y ROI
    # ------------------------------------------------
    imagen = cv.imread(nombre_archivo)

    if imagen is None:
        raise FileNotFoundError(
            f"No se pudo leer la imagen: {nombre_archivo}"
        )

    imagen_RGB = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)

    x1 = x_0 + ancho
    y1 = y_0 + alto

    roi = imagen_RGB[y_0:y1, x_0:x1]

    # ------------------------------------------------
    # Realce
    # ------------------------------------------------
    R = roi[:, :, 0]

    realzada = aplicar_clahe(R)

    # ------------------------------------------------
    # Histograma de la imagen realzada
    # ------------------------------------------------
    hist = histograma_normalizado(realzada)
    k = np.arange(256)

    # ------------------------------------------------
    # Resultado final:
    # superponer la mascara final sobre la imagen
    # ------------------------------------------------
    #resultado_final = roi.copy()

    #mascara_final = morfologia > 0

    #resultado_final[mascara_final] = [255, 0, 0]

    # ------------------------------------------------
    # Grafico
    # ------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Original
    ax[0, 0].imshow(roi)
    ax[0, 0].set_title("Imagen original")
    ax[0, 0].axis("off")

    # 2. Realce
    ax[0, 1].imshow(realzada, cmap="gray", vmin=0, vmax=255)
    ax[0, 1].set_title("Realce CLAHE")
    ax[0, 1].axis("off")

    # 3. Histograma
    ax[0, 2].plot(k, hist)
    ax[0, 2].set_xlim(0, 255)
    ax[0, 2].set_xlabel("Intensidad k")
    ax[0, 2].set_ylabel("p[k]")
    ax[0, 2].set_title("Histograma")
    ax[0, 2].grid(alpha=0.2)

    # 4. Mascara inicial
    ax[1, 0].imshow(segmentacion, cmap="gray", vmin=0, vmax=255)
    ax[1, 0].set_title("Mascara inicial")
    ax[1, 0].axis("off")

    # 5. Procesamiento morfologico
    ax[1, 1].imshow(morfologia, cmap="gray", vmin=0, vmax=255)
    ax[1, 1].set_title("Procesamiento morfologico")
    ax[1, 1].axis("off")

    resultado_final = cv.bitwise_and(
        realzada,
        realzada,
        mask=morfologia
    )

    # 6. Resultado final
    ax[1, 2].imshow(
        resultado_final,
        cmap="gray",
        vmin=0,
        vmax=255
    )
    ax[1, 2].set_title("Resultado final")
    ax[1, 2].axis("off")

    fig.suptitle(titulo, fontsize=16)

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

mostrar_transformaciones = True

indices_representativos = [
    1,   # Normal grande
    2,   # Mojada grande
    4,   # Seca grande
    6,   # Normal pequeña
    8,   # Mojada pequeña
    10   # Seca pequeña
]

if mostrar_transformaciones:

    for i in indices_representativos:

        archivo, titulo = muestras[i]

        comparar_transformaciones(
            archivo,
            titulo,
            ancho[i],
            alto[i],
            x_0[i],
            y_0[i],
            gamma=1.5
        )

#---------------------------------------------------------------------------------------------
#---------------------------------- 1.3 ------------------------------------------------------
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
#---------------------------------- 1.3.1 ----------------------------------------------------
#----------------------- Estrategias de segmentacion -----------------------------------------
#---------------------------------------------------------------------------------------------

mostrar_segmentacion = False

if mostrar_segmentacion:

    CLAHEs=[]
    Segmentaciones=[]
    Morfologias=[]

    for n, (archivo, titulo) in enumerate(muestras):    
        CLAHEs.append(
            imagenes_CLAHE(
                archivo,
                titulo,
                ancho[n],
                alto[n],
                x_0[n],
                y_0[n],
                gamma=1.5
            )
        )

    Segmentaciones=segmentar(CLAHEs)


#---------------------------------------------------------------------------------------------
#---------------------------------- 1.3.2 ----------------------------------------------------
# Refinamiento mediante operaciones morfologicas
#---------------------------------------------------------------------------------------------

    #comparar_morfologias(Segmentaciones[x][0]) Compara cada morfologia, con x = 0 hasta x = 11.

    Morfologias=morfologia(Segmentaciones)

#    grafico(
#        CLAHEs,
#        Segmentaciones,
#        Morfologias
#    )
#Resultados de las segmentaciones: perfeccion(de 0 a 1)

# Segmentaciones e Imagenes:| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
# elegir umbral (T=128):    |0.9| 1 |0.7| 1 |0.2| 0 | 1 |0.2|0.3|  1 | 0  |  0 | p=
# umbral iterativo:         |0.5| 1 |0.3| 1 |0.2| 0 | 1 | 0 | 0 |0.9 |0.4 |  0 | p=
# otsu:                     |0.5| 1 |0.4| 1 | 0 | 0 | 1 | 0 | 0 |0.9 |0.3 |  0 | p=
# regiones:                 |0.4| 1 |0.3| 1 |0.1| 0 | 1 |0.1|0.2| 1  |  0 |  0 | p=
# multi otsu:               |0.9| 1 |0.5| 1 |0.8| 0 |0.9|0.1| 0 | 1  | 0.4|  0 | p=
# k-means (K=4):            |0.8| 1 |0.5| 1 |0.5| 0 | 1 |0.1|0.1| 1  | 0.3|  0 | p=
# k-means (K=8):            |0.7| 1 |0.6| 1 |0.7| 0 | 1 |0.5|0.4| 1  | 0.5| 0.2| p=
# k-means (K=12):           |   |   |   |   |   |   |   |   |   |    | 0.8| 0.6| p=

#Resultados de las morfologias: perfeccion(de 0 a 1)

# Morfologias e imagenes:| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
# dilatacion:            |   |   |   |   |   |   |   |   |   |    |    |    | p=
# apertura:              |   |   |   |   |   |   |   |   |   |    |    |    | p=
# erosion:               |   |   |   |   |   |   |   |   |   |    |    |    | p=
# cierre:                |   |   |   |   |   |   |   |   |   |    |    |    | P=
# extraccion:            |   |   |   |   |   |   |   |   |   |    |    |    | p=

#---------------------------------------------------------------------------------------------
#---------------------------------- 1.3.4 ----------------------------------------------------
# Secuencia
#---------------------------------------------------------------------------------------------

#indice = 0 #Escoger cual foto mostrar. indice=0 es la primera foto, indice=11 es la ultima foto

#archivo, titulo = muestras[indice]

#grafico_secuencia_completa(
    #archivo,
    #titulo,
    #ancho[indice],
    #alto[indice],
    #x_0[indice],
    #y_0[indice],
    #Segmentaciones[indice][0],
    #Morfologias[indice][0]
#)

for indice in indices_representativos:

    archivo, titulo=muestras[indice]
    grafico_secuencia_completa(
        archivo,
        titulo,
        ancho[indice],
        alto[indice],
        x_0[indice],
        y_0[indice],
        Segmentaciones[indice][0],
        Morfologias[indice][0])

#---------------------------------------------------------------------------------------------
#---------------------------------- 1.3.5 ----------------------------------------------------
# Ia
#---------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------
#---------------------------------- 1.3.6 ----------------------------------------------------
# Métricas
#---------------------------------------------------------------------------------------------
indice_ia = 1

# Cargar imagen generada por IA en escala de grises
ia = cv.imread(
    "Gemini_Generated_Image_5ieckk5ieckk5iec.jpg", cv.IMREAD_GRAYSCALE)

if ia is None:
    raise FileNotFoundError("No se pudo cargar la imagen generada por IA")

# Convertir la salida de IA a una máscara binaria
_, ia_binaria = cv.threshold(ia, 127, 255, cv.THRESH_BINARY)

# Guardar la máscara binaria sin compresión con pérdida
cv.imwrite(
    "Gemini_Generated_Image_5ieckk5ieckk5iec.png", ia_binaria)

print("Dimensiones máscara IA:", ia_binaria.shape) #1536, 2730
print("Dimensiones máscara método:", Morfologias[indice_ia][0].shape)

# Dimensiones de la imagen original
alto_original = 2296
ancho_original = 4080

# Factores de escala entre la imagen original y la salida de IA
escala_x = ia_binaria.shape[1] / ancho_original
escala_y = ia_binaria.shape[0] / alto_original

# Coordenadas equivalentes del ROI en la imagen generada por IA
x0_ia = round(x_0[indice_ia] * escala_x)
y0_ia = round(y_0[indice_ia] * escala_y)

x1_ia = round((x_0[indice_ia] + ancho[indice_ia]) * escala_x)
y1_ia = round((y_0[indice_ia] + alto[indice_ia]) * escala_y)

# Recortar la misma región utilizada por nuestro método
ia_roi = ia_binaria[
    y0_ia:y1_ia,
    x0_ia:x1_ia
]

# Ajustar exactamente al tamaño de la máscara de nuestro método
ia_roi = cv.resize(
    ia_roi,
    (
        Morfologias[indice_ia][0].shape[1],
        Morfologias[indice_ia][0].shape[0]
    ),
    interpolation=cv.INTER_NEAREST
)

print("Dimensiones IA recortada:", ia_roi.shape)
print("Dimensiones método:", Morfologias[indice_ia][0].shape)

mascara_metodo = Morfologias[indice_ia][0]

superposicion = np.zeros(
    (mascara_metodo.shape[0], mascara_metodo.shape[1], 3),
    dtype=np.uint8
)

metodo_bool = mascara_metodo > 0
ia_bool = ia_roi > 0

# Solo método
superposicion[metodo_bool] = [255, 0, 0]

# Solo IA
superposicion[ia_bool] = [0, 255, 0]

# Coinciden ambos
superposicion[metodo_bool & ia_bool] = [255, 255, 255]

plt.figure(figsize=(7, 7))
plt.imshow(superposicion)
plt.title("Superposición método vs IA")
plt.axis("off")
plt.show()

# ------------------------------------------------
# Métrica global: MAE
# ------------------------------------------------

metodo_float = mascara_metodo.astype(np.float32)
ia_float = ia_roi.astype(np.float32)

mae = np.mean(
    np.abs(metodo_float - ia_float)
)

# ------------------------------------------------
# Métrica entre máscaras: IoU
# ------------------------------------------------

metodo_bool = mascara_metodo > 0
ia_bool = ia_roi > 0

interseccion = np.logical_and(
    metodo_bool,
    ia_bool
).sum()

union = np.logical_or(
    metodo_bool,
    ia_bool
).sum()

iou = interseccion / union if union > 0 else 1.0

# ------------------------------------------------
# Mostrar resultados
# ------------------------------------------------

print("MAE:", mae)
print("IoU:", iou)

#---------------------------------------------------------------------------------------------
#---------------------------------- 1.3.7 ----------------------------------------------------
# Comparacion entre condiciones
#---------------------------------------------------------------------------------------------

fig, ax = plt.subplots(2, 3, figsize=(15, 10))

for j, i in enumerate(indices_representativos):

    fila = j // 3
    columna = j % 3

    mascara = Morfologias[i][0]

    ax[fila, columna].imshow(
        mascara,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    ax[fila, columna].set_title(
        muestras[i][1]
    )

    ax[fila, columna].axis("off")

plt.suptitle(
    "Comparación de segmentación entre condiciones",
    fontsize=16
)

plt.tight_layout()
plt.show()


































#------------------ Actualizar github ------------------

#git status , despues
#git add miniporyecto1.py 
#git commit -m "Añadir nueva funcion"
#git push origin main
#git fetch origin
#git reset --hard origin/main