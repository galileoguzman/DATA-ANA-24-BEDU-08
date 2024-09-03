import cv2
import numpy as np

# Cargar archivo
FILENAME = 'tmp/nube.jpg'
imagen = cv2.imread(FILENAME)

# Convertir a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# aplicar un filtro Gaussiano
imagen_suavizada = cv2.GaussianBlur(imagen_gris, (5, 5), 0)

# Adative umbrella
imagen_binaria = cv2.adaptiveThreshold(
    imagen_suavizada,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
)

# Expandir donde existan posibles textos
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
imagen_dilatada = cv2.dilate(imagen_binaria, kernel, iterations=1)

# Encontrar contornos
contornos, _ = cv2.findContours(
    imagen_dilatada,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# REcorrido de regiones con posible texto
for contorno in contornos:
    x, y, w, h = cv2.boundingRect(contorno)

    # filtrado por area y descartar elementos sin texto
    if w > 30 and h > 10: # Textop con tamano esperado
        relacion_aspecto = w / float(h)
        if 1 < relacion_aspecto < 15: # Tipo de texto buscado/encontrado
            cv2.rectangle(imagen, (x, y), (x + w, y + h), (255, 0, 0), 2)


# Mostrar imagen
cv2.imshow('Imagen with rostros', imagen)

# DEstruir ventanas abiertas
cv2.waitKey(0)
cv2.destroyAllWindows()