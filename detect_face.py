import numpy as np
import cv2



prototxt="deploy.prototxt.txt"
model="res10_300x300_ssd_iter_140000.caffemodel"
confidence=0.5

#loading model
model=cv2.dnn.readNetFromCaffe(prototxt,model)

#loading image
image=cv2.imread("1.jpeg")

#print(image.shape[1])
height=image.shape[0]
width=image.shape[1]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

model.setInput(blob) #feeding blob into the model
detections = model.forward()

#print(detections)
#print(detections[0,0,1,2])
#looping over all the detected faces to check with the confidence and make bounding box
for i in range(0,detections.shape[2]):
	confidence1=detections[0,0,i,2]
	if confidence1>confidence:
		box=detections[0,0,i,3:7]*np.array([width,height,width,height])
		(startX,startY,endX,endY)=box.astype("int")

		y=startY-10 if startY-10>10 else startY+10
		cv2.rectangle(image,(startX,startY),(endX,endY),(0,0,255),2)
		text="{:.2f}%".format(confidence1*100)
		cv2.putText(image,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)


cv2.imshow("Output",image)
cv2.waitKey(0)
