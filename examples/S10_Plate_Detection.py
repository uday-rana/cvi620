import cv2

frameWidth = 1000
frameHeight = 500
nPlateCascade = cv2.CascadeClassifier(
    "haarcascades/haarcascade_russian_plate_number.xml"
)
minArea = 600
color = (255, 0, 255)


cap = cv2.VideoCapture("cars_video.mp4")
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 50)
count = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (frameWidth, frameHeight))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 10)

    for x, y, w, h in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(
                img,
                "Number Plate",
                (x, y - 5),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                color,
                2,
            )
            imgRoi = img[y : y + h, x : x + w]
            cv2.imshow("ROI", imgRoi)

    cv2.imshow("Result", img)
    cv2.waitKey(1)
