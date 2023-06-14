import cv2
import numpy as np

# -------------------------------------------------------------------------------

foggy = cv2.imread('foggy.jpg')
night = cv2.imread('night.jpg')
rainy = cv2.imread('rainy.jpg')

# -------------------------------------------------------------------------------

foggy = cv2.cvtColor(foggy, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(foggy, (7, 7), 0)
canny1 = cv2.Canny(suave, 20, 120)
canny2 = cv2.Canny(suave, 70, 200)
resultado_foggy = np.vstack([
np.hstack([foggy, suave ]),
np.hstack([canny1, canny2])
])

proportion2 = 900.0 / resultado_foggy.shape[1]
new_size2 = (900, int(resultado_foggy.shape[0] * proportion2))
resultado_foggy = cv2.resize(resultado_foggy, new_size2, interpolation = cv2.INTER_AREA)

cv2.imshow("Detector de Bordas Canny", resultado_foggy)
cv2.waitKey(0)

# -------------------------------------------------------------------------------

night = cv2.cvtColor(night, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(night, (7, 7), 0)
canny1 = cv2.Canny(suave, 20, 120)
canny2 = cv2.Canny(suave, 70, 200)
resultado_night = np.vstack([
np.hstack([night, suave ]),
np.hstack([canny1, canny2])
])

proportion2 = 900.0 / resultado_night.shape[1]
new_size2 = (900, int(resultado_night.shape[0] * proportion2))
resultado_night = cv2.resize(resultado_night, new_size2, interpolation = cv2.INTER_AREA)

cv2.imshow("Detector de Bordas Canny", resultado_night)
cv2.waitKey(0)

# -------------------------------------------------------------------------------

rainy = cv2.cvtColor(rainy, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(rainy, (7, 7), 0)
canny1 = cv2.Canny(suave, 20, 120)
canny2 = cv2.Canny(suave, 70, 200)
resultado_rainy = np.vstack([
np.hstack([rainy, suave ]),
np.hstack([canny1, canny2])
])

proportion2 = 900.0 / resultado_rainy.shape[1]
new_size2 = (900, int(resultado_rainy.shape[0] * proportion2))
resultado_rainy = cv2.resize(resultado_rainy, new_size2, interpolation = cv2.INTER_AREA)

cv2.imshow("Detector de Bordas Canny", resultado_rainy)
cv2.waitKey(0)