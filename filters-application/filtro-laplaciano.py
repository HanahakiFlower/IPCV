import cv2
import numpy as np

# -------------------------------------------------------------------------------

foggy = cv2.imread('foggy.jpg')
night = cv2.imread('night.jpg')
rainy = cv2.imread('rainy.jpg')

# -------------------------------------------------------------------------------

foggy = cv2.cvtColor(foggy, cv2.COLOR_BGR2GRAY)
lap = cv2.Laplacian(foggy, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
resultado_foggy = np.vstack([foggy, lap])

proportion2 = 500.0 / resultado_foggy.shape[1]
new_size2 = (500, int(resultado_foggy.shape[0] * proportion2))
resultado_foggy = cv2.resize(resultado_foggy, new_size2, interpolation = cv2.INTER_AREA)

cv2.imshow("Filtro Laplaciano", resultado_foggy)
cv2.waitKey(0)

# -------------------------------------------------------------------------------

night = cv2.cvtColor(night, cv2.COLOR_BGR2GRAY)
lap = cv2.Laplacian(night, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
resultado_night = np.vstack([night, lap])

proportion2 = 500.0 / resultado_night.shape[1]
new_size2 = (500, int(resultado_night.shape[0] * proportion2))
resultado_night = cv2.resize(resultado_night, new_size2, interpolation = cv2.INTER_AREA)

cv2.imshow("Filtro Laplaciano", resultado_night)
cv2.waitKey(0)

# -------------------------------------------------------------------------------

rainy = cv2.cvtColor(rainy, cv2.COLOR_BGR2GRAY)
lap = cv2.Laplacian(rainy, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
resultado_rainy = np.vstack([rainy, lap])

proportion2 = 500.0 / resultado_rainy.shape[1]
new_size2 = (500, int(resultado_rainy.shape[0] * proportion2))
resultado_rainy = cv2.resize(resultado_rainy, new_size2, interpolation = cv2.INTER_AREA)

cv2.imshow("Filtro Laplaciano", resultado_rainy)
cv2.waitKey(0)