import cv2

webCamera = cv2.VideoCapture(0)
VideoFace = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

while True:
    camera, frame = webCamera.read()

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detecta = VideoFace.detectMultiScale(cinza)

    for (x, y, l, a) in detecta:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 0, 0), 2)

    cv2.imshow("Vídeo WebCamera", frame)

    if cv2.waitKey(1) == ord('s'):
        break

webcamera.release()
cv2.destroyAllWindows()