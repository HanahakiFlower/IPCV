import numpy as np
import cv2
import mahotas

#Função para facilitar a escrita nas imagem
def escreve(img, texto, cor=(255,0,0)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (10,20), fonte, 0.5, cor, 0,
    cv2.LINE_AA)

# -------------------------------------------------------------------------------

foggy = cv2.imread('foggy.jpg')
proportion1 = 900.0 / foggy.shape[1]
new_size2 = (900, int(foggy.shape[0] * proportion1))
foggy = cv2.resize(foggy, new_size2, interpolation = cv2.INTER_AREA)

night = cv2.imread('night.jpg')
proportion2 = 900.0 / night.shape[1]
new_size2 = (900, int(night.shape[0] * proportion2))
night = cv2.resize(night, new_size2, interpolation = cv2.INTER_AREA)

rainy = cv2.imread('rainy.jpg')
proportion3 = 900.0 / rainy.shape[1]
new_size2 = (900, int(rainy.shape[0] * proportion3))
rainy = cv2.resize(rainy, new_size2, interpolation = cv2.INTER_AREA)

# -------------------------------------------------------------------------------

#Passo 1: Conversão para tons de cinza
img = cv2.cvtColor(night, cv2.COLOR_BGR2GRAY)

#Passo 2: Blur/Suavização da imagem
suave = cv2.blur(img, (7, 7))

#Passo 3: Binarização resultando em pixels brancos e pretos
T = mahotas.thresholding.otsu(suave)
bin = suave.copy()
bin[bin > T] = 255
bin[bin < 255] = 0
bin = cv2.bitwise_not(bin)

#Passo 4: Detecção de bordas com Canny
bordas = cv2.Canny(bin, 70, 150)

#Passo 5: Identificação e contagem dos contornos da imagem
#cv2.RETR_EXTERNAL = conta apenas os contornos externos
# (lx, objetos, lx) = cv2.findContours(bordas.copy() , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts, _ = cv2.findContours(bordas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#A variável lx (lixo) recebe dados que não são utilizados
escreve(img, "Imagem em tons de cinza", 0)
escreve(suave, "Suavizacao com Blur", 0)
escreve(bin, "Binarizacao com Metodo Otsu", 255)
escreve(bordas, "Detector de bordas Canny", 255)

temp = np.vstack([
    np.hstack([img, suave]),
    np.hstack([bin, bordas])])

proportion = 1000.0 / temp.shape[1]
new_size2 = (1000, int(temp.shape[0] * proportion))
resultado_night = cv2.resize(temp, new_size2, interpolation = cv2.INTER_AREA)

cv2.imshow("Quantidade de objetos: " + str(len(cnts)), resultado_night)
cv2.waitKey(0)
imgC2 = night.copy()
cv2.imshow("Imagem Original", night)

cv2.drawContours(imgC2, cnts, -1, (255, 0, 0), 2)
escreve(imgC2, str(len(cnts)) + " objetos encontrados!")
cv2.imshow("Resultado", imgC2)
cv2.waitKey(0)