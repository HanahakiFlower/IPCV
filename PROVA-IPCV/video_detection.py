import cv2

# --------------------------------------------------------------------------
# Função para redimensionar uma imagem
def redim(img, largura):  
    alt = int(img.shape[0]/img.shape[1]*largura)
    img = cv2.resize(img, (largura, alt), interpolation=cv2.INTER_AREA)
    return img

# --------------------------------------------------------------------------
#Função para facilitar a escrita nas imagem
def escreve(img, texto, cor=(255,0,0)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (12,23), fonte, 0.8, cor, 0, cv2.LINE_AA)

# --------------------------------------------------------------------------

# Cria o detector de faces baseado no XML
df = cv2.CascadeClassifier('frontalface.xml')

# Abre um vídeo gravado em disco
camera = cv2.VideoCapture('video.mp4')

# Cria uma contagem para acerto em cada frame
acertos_frame = 0

# --------------------------------------------------------------------------

while True:
    # A função read() retorna se houve sucesso e o próprio frame
    (sucesso, frame) = camera.read()
    
    # -------------------------------------------------------
    # Métodos que possibilitam a quebra do loop:

    # 1. Identifica o final do vídeo
    if not sucesso:  
        break
    
    # 2. Espera que a tecla 's' seja pressionada para sair
    if cv2.waitKey(3) & 0xFF == ord("s"):
        break
    # -------------------------------------------------------

    # Reduz tamanho do frame para acelerar processamento
    frame = redim(frame, 600)

    # Converte para tons de cinza
    frame_pb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta as faces no frame
    faces = df.detectMultiScale(frame_pb, scaleFactor=1.1, minNeighbors=6, 
                                minSize=(62, 64), maxSize=(110, 115), 
                                flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Faz a contagem da acuracia de detecção no atual frame sendo processado
    if len(faces) == 3:
        acertos_frame+=1
    elif len(faces) == 2:
        acertos_frame+=0.66
    elif len(faces) == 1:
        acertos_frame+=0.33

    # Faz um cópia temporaria do frame pra não afetar o frame original
    frame_temp = frame.copy()

    # Desenha retangulos amarelos no frame temporario (colorido)
    for (x, y, lar, alt) in faces:
        cv2.rectangle(frame_temp, (x, y), (x + lar, y + alt), (0, 255, 255), 2)
    
    # Escreve e exibe, frame a frame, a atual precisão da deteccção de faces no video
    escreve(frame_temp, "Acuracia: " + str("{:.1f}".format(0.10810810810810811 * acertos_frame)) + "%")
    # O video possui 925 frames no total
    # 1 frame é 0.10810810810810811% de 100% do video

    # Exibe um frame redimensionado (com perca de qualidade)
    cv2.imshow("Encontrando faces...", redim(frame_temp, 640))


# --------------------------------------------------------------------------
# Fecha streaming
camera.release()
cv2.destroyAllWindows()
