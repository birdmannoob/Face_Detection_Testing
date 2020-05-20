import cv2
import os
import numpy as np
import imutils
image_dir = 'dataset2'
num = 0
TP = 0
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_alt2.xml')
for files in os.listdir(image_dir):
	print(files)
	for file in os.listdir(f'{image_dir}/{files}'):
		image = cv2.imread(f'{image_dir}/{files}/{file}')
		print("")
		print(file)
		# size = (300, 300)
		# final_image = cv2.resize(image, size)
		final_image = image
		#final_image = imutils.resize(image, width=300)
		gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
		image_array = np.array(gray, "uint8")
		faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.2, minNeighbors=5)

		for (x,y,w,h) in faces:
			cv2.rectangle(final_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
			TP += 1
		cv2.imshow('face', final_image)
		num += 1
		filename = "Haaroutput/output" + str(num) + ".jpg"
		cv2.imwrite(filename, final_image)

		cv2.waitKey(0)
print("total images: ", num)
print("TP: ", TP)
cv2.destroyAllWindows()






