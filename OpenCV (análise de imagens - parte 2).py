import cv2

carregaAlgoritmo = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

imagem = cv2.imread('fotos/image1.jpg')

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

faces = carregaAlgoritmo.detectMultiScale(imagemCinza, scaleFactor=1.6)
# scaleFactor = n>1.1: especifica o quanto da imagem é reduzido em cada escala da imagem
# minNrighbors = n: especifica quanros vizinhos cada retangulos deve ter para retê-lo
# minSize(n,n): tamanho minimo possivel do objeto, onde menores são ignorados

print(faces)

for (x, y, l, a) in faces:
    cv2.rectangle(imagem, (x,y), (x + l, y + a), (0, 255, 0), 2)

cv2.imshow("Faces", imagem)

cv2.waitKey()

