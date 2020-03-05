import cv2

numberCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numbers = numberCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 10,
        minSize = (20,20)
    )

    for (x,y,w,h) in numbers:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

    cv2.imshow('frame', img)
    k = cv2.waitKey(30)
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()

