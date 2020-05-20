import face_recognition
import os
import cv2
import numpy as np
import imutils
image_dir = 'dataset1'
num = 1
TP = 0
Threshold = 0.8
protoPath = ("face_detection_model/deploy.prototxt")
modelPath = ("face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

for files in os.listdir(image_dir):
	print(files)
	for file in os.listdir(f'{image_dir}/{files}'):
		image = cv2.imread(f'{image_dir}/{files}/{file}')
		# image = cv2.resize(image, (0, 0), fx=1 / 2, fy=1 / 2)

		size = (300, 300)
		#final_image = cv2.resize(image, size)
		final_image = imutils.resize(image, width=300)
		#final_image = image
		print("")
		print(file)
		(h, w) = final_image.shape[:2]
		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(final_image, 1.0, (h, w),(104.0, 177.0, 123.0), swapRB=False, crop=False)
		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]
			# filter out weak detections
			if confidence > Threshold:
				# compute the (x, y)-coordinates of the bounding box for
				# the face
				TP += 1
				box = detections[0, 0, i, 3 :7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
			cv2.rectangle(final_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
		cv2.imshow("face", final_image)
		filename = "dnnoutput/output" + str(num) + ".jpg"
		cv2.imwrite(filename, final_image)
		num += 1
		cv2.waitKey(0)

print("total images: ", num-1)
print("TP: ", TP)