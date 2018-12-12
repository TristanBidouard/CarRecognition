import cv2
import numpy
import os

font = cv2.FONT_HERSHEY_SIMPLEX

def drawFrameRate(img, frameRate):
	return cv2.putText(img,"FPS: "+str(int(frameRate)),(10,26),font,0.7,(255,0,255),2,cv2.LINE_AA)

def drawCars(img, cars):
	for (x,y,w,h) in cars:
	    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

freq = cv2.getTickFrequency()
frameRateCalc = 1

pathDataset = os.path.dirname(os.path.abspath(__file__)) + "/dataset/"
pathVideo = os.path.dirname(os.path.abspath(__file__)) + "/video/"

webcam = cv2.VideoCapture(pathVideo + "highway.mp4")


while (webcam.isOpened()):

	t1 = cv2.getTickCount()

	img = webcam.read()[1]
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	carCascade = cv2.CascadeClassifier(pathDataset + 'car.xml')

	cars = carCascade.detectMultiScale(gray, 1.1, 5)
	
	print "Found "+str(len(cars))+" car(s)"

	drawCars(img, cars)
	drawFrameRate(img, frameRateCalc)
	cv2.imshow("Car Recognition", img)

	t2 = cv2.getTickCount()
	time1 = (t2-t1)/freq
	frameRateCalc = 1/time1

	if cv2.waitKey(1) == 27:
		break  # esc to quit
	cv2.destroyAllWindows()