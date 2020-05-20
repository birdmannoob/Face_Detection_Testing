import face_recognition
import os
import cv2
import numpy as np
import imutils
image_dir = 'dataset1'
num = 0
MODEL = "hog"
TP = 0
TPR = 0
for files in os.listdir(image_dir):
	print(files)
	for file in os.listdir(f'{image_dir}/{files}'):
		image = cv2.imread(f'{image_dir}/{files}/{file}')
		size = (300, 300)
		# final_image = cv2.resize(image, size)
		#final_image = image
		final_image = imutils.resize(image, width=300)
		print("")
		print(file)
		rgb_image = final_image[:, :, ::-1]

		image_array = np.array(rgb_image, "uint8")
		# Find all the faces and face encodings in the current frame of video
		locations = face_recognition.face_locations(image_array, model=MODEL)
		encodings = face_recognition.face_encodings(image_array, locations)
		for face_encoding, face_location in zip(encodings, locations):
			face_top_left = (face_location[3], face_location[0])  # original face locations
			face_bottom_right = (face_location[1], face_location[2])  # Note that we shrinked down the frame previously
			# Draw rectangle on face
			cv2.rectangle(final_image, face_top_left, face_bottom_right, [0, 255, 0], 3)
			TP += 1
		cv2.imshow('face', final_image)
		num += 1
		filename = "HoGoutput/output" + str(num) + ".jpg"
		cv2.imwrite(filename, final_image)

		cv2.waitKey(0)

print("total images: ", num)
print("TP: ", TP)
#print("TPR: ", TPR)