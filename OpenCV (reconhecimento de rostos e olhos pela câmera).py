import cv2

webcamera = cv2.VideoCapture(0)
VideoFace = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
VideoOlho = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

while True:
    camera, frame = webcamera.read()

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detecta = VideoFace.detectMultiScale(cinza)

    for (x, y, l, a) in detecta:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 0, 0), 2)
        olho = frame[y:y + a, x:x + l]
        olhocinza = cv2.cvtColor(olho, cv2.COLOR_BGR2GRAY)
        reconhecimento = VideoOlho.detectMultiScale(olhocinza)

        for(ox, oy, ol, oa) in reconhecimento:
            cv2.rectangle(olho, (ox, oy), (ox + ol, oy + oa), (0, 0, 255), 2)

    cv2.imshow("Video WebCamera e reconhecimento", frame)

    if cv2.waitKey(1) == ord('s'):
        break

webcamera.release()
cv2.destroyAllWindows()