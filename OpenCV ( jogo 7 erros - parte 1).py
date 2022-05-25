import cv2
import cv2 as cv

img1 = cv2.imread('imagens/img1.png')
img2 = cv2.imread('imagens/img2.png')

imagem_subtraida = cv.absdiff(img1, img2)
imagem_subtraida_em_cinza = cv.cvtColor(imagem_subtraida, cv.COLOR_BGR2GRAY)

cv.imshow("Imagem Subtraida", imagem_subtraida_em_cinza)
cv.waitKey()
