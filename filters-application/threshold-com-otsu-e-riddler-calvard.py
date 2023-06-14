import mahotas
import numpy as np
import cv2

# -------------------------------------------------------------------------------

foggy = cv2.imread('foggy.jpg')
night = cv2.imread('night.jpg')
rainy = cv2.imread('rainy.jpg')

# -------------------------------------------------------------------------------

foggy = cv2.cvtColor(foggy, cv2.COLOR_BGR2GRAY) # converte
suave = cv2.GaussianBlur(foggy, (7, 7), 0) # aplica blur
T = mahotas.thresholding.otsu(suave)
temp = foggy.copy()
temp[temp > T] = 255
temp[temp < 255] = 0
temp = cv2.bitwise_not(temp)
T = mahotas.thresholding.rc(suave)
temp2 = foggy.copy()
temp2[temp2 > T] = 255
temp2[temp2 < 255] = 0
temp2 = cv2.bitwise_not(temp2)
resultado_foggy = np.vstack([
np.hstack([foggy, suave]),
np.hstack([temp, temp2])
])

proportion2 = 900.0 / resultado_foggy.shape[1]
new_size2 = (900, int(resultado_foggy.shape[0] * proportion2))
resultado_foggy = cv2.resize(resultado_foggy, new_size2, interpolation = cv2.INTER_AREA)

cv2.imshow("Binarizacao com metodo Otsu e Riddler-Calvard",
resultado_foggy)
cv2.waitKey(0)

# -------------------------------------------------------------------------------

night = cv2.cvtColor(night, cv2.COLOR_BGR2GRAY) # converte
suave = cv2.GaussianBlur(night, (7, 7), 0) # aplica blur
T = mahotas.thresholding.otsu(suave)
temp = night.copy()
temp[temp > T] = 255
temp[temp < 255] = 0
temp = cv2.bitwise_not(temp)
T = mahotas.thresholding.rc(suave)
temp2 = night.copy()
temp2[temp2 > T] = 255
temp2[temp2 < 255] = 0
temp2 = cv2.bitwise_not(temp2)
resultado_night = np.vstack([
np.hstack([night, suave]),
np.hstack([temp, temp2])
])

proportion2 = 900.0 / resultado_night.shape[1]
new_size2 = (900, int(resultado_night.shape[0] * proportion2))
resultado_night = cv2.resize(resultado_night, new_size2, interpolation = cv2.INTER_AREA)

cv2.imshow("Binarizacao com metodo Otsu e Riddler-Calvard",
resultado_night)
cv2.waitKey(0)

# -------------------------------------------------------------------------------

rainy = cv2.cvtColor(rainy, cv2.COLOR_BGR2GRAY) # converte
suave = cv2.GaussianBlur(rainy, (7, 7), 0) # aplica blur
T = mahotas.thresholding.otsu(suave)
temp = rainy.copy()
temp[temp > T] = 255
temp[temp < 255] = 0
temp = cv2.bitwise_not(temp)
T = mahotas.thresholding.rc(suave)
temp2 = rainy.copy()
temp2[temp2 > T] = 255
temp2[temp2 < 255] = 0
temp2 = cv2.bitwise_not(temp2)
resultado_rainy = np.vstack([
np.hstack([rainy, suave]),
np.hstack([temp, temp2])
])

proportion2 = 900.0 / resultado_rainy.shape[1]
new_size2 = (900, int(resultado_rainy.shape[0] * proportion2))
resultado_rainy = cv2.resize(resultado_rainy, new_size2, interpolation = cv2.INTER_AREA)

cv2.imshow("Binarizacao com metodo Otsu e Riddler-Calvard",
resultado_rainy)
cv2.waitKey(0)