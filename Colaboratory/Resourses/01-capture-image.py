import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

id= input('enter user id\n')
sampleNum=0
while(1):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        sampleNum = sampleNum +1
        cv2.imwrite('image/user'+str(id) + '.' +str(sampleNum) + '.png', gray[y:y+h, x:x+w] )
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(img, (x, y), (end_cord_x, end_cord_y), color, stroke)
        cv2.waitKey(100)
    cv2.imshow("face",img)
    cv2.waitKey(1)
    if sampleNum > 100:
        break

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()