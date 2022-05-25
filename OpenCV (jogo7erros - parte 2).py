#importar as bibliotecas
import cv2
import cv2 as cv
import numpy as np

#criando os metodos
desenhar_imagem = True
ver_imagem_kernel = True

#apresentando as imagens usadas
img1 = cv2.imread('imagens/imgj1.png')
img2 = cv2.imread('imagens/imgj2.png')

#subtraindo as imagens e transformando o resutado na cor cinza
imagem_subtraida = cv.absdiff(img1, img2)
imagem_subtraida_em_cinza = cv.cvtColor(imagem_subtraida, cv.COLOR_BGR2GRAY)

#Limpando o visual e retirando os ruidos
kernel = np.ones((4, 4), np.uint8)
imagem_com_kernel = cv.erode(imagem_subtraida_em_cinza, kernel)

#Impondo condição para visualizar a imagem sem ruidos somente com os erros destacados em branco e fundo preto
if (ver_imagem_kernel):
        cv.imshow('Imagem com kernel aplicado',imagem_com_kernel)

#Retirando os ruidos e retornando somente os contornos
imagem_com_threshold, mask = cv.threshold(imagem_com_kernel, 10, 255, cv.THRESH_BINARY)
contornos, hierarquia = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

#Impondo repetição de análise para que se houver contorno irá haver uma marcação de um retângulo, em todos
for contorno in contornos:
    if(cv.contourArea(contorno) > 5):
        x, y, w, h = cv.boundingRect(contorno)
        if desenhar_imagem:
            cv.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)

#Impondo condição para visualizar a imagem inicial com os erros destacados
if desenhar_imagem:
    cv.imshow('Imagem com Erros Destacados', img2)
    cv.waitKey(0)