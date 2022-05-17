import cv2 #importando conteúdo da biblioteca haarcasdes (OpenCV)

carregaAlgoritmo = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
#criando contante e atribuindo o algoritmo de dectecção de objeto em cascata e os parametros exigidos
#(nome da pasta/nome do arquivo xml onde guarda o algoritmo(no caso o de identificar rostos))

imagem = cv2.imread('fotos/image3.jpg')
#criando contante e atribuindo o algoritmo para receber uma imagem e os parametros exigidos
#(pasta/nome e tipo do arquivo)

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
#criando contante e atribuindo o algoritmo para alterar a cor e os parametros exigidos
# (scr (imagem a receber modificação) e code (código de conversão de cores))

faces = carregaAlgoritmo.detectMultiScale(imagemCinza)
#criando contante e atribuindo a detecção de tamanhos diferentes na imagem na constante de detecção de rostos,
# retornando a posição dos objetos em matrizes

print(faces)
#imprimir a contante de identificação dos rostos

for (x, y, l, a) in faces:
    cv2.rectangle(imagem, (x,y), (x + l, y + a), (0, 255, 0), 2)
#estrutura de repetição para os rostos, atribuindo retalgulos nos objetos detectados e os parametros exigidos
#(imagem, eixo x e y da imagem, largura e altura, cor (em RGB), espessura da borda)

cv2.imshow("Faces", imagem)
#algoritmo usado para exibir uma imagem em uma janela, que é ajustada de acordo com o tamanho da mesmo
#com os seus parametros (nome da janela e a imagem)

cv2.waitKey()
#função para exibir a imagem até certo tempo (em milissegundos)