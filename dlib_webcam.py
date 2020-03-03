import face_recognition as fr
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-d','--detection_method', required = True,
	            help = 'face detection model to use: either hog or cnn', type = str, default = 'hog')
args = vars(ap.parse_args())

# access webcam camera
video_cap = cv2.VideoCapture(0)
process_video = True

while True:
	# read per fram from webcam stream
	ret, frame = video_cap.read()
	
	if process_video:

		# get face location and face encodings
		face_locations = fr.face_locations(frame, number_of_times_to_upsample = 0, model = args['detection_method'])
		face_encodings = fr.face_encodings(frame, known_face_locations = face_locations)

		for face_location in face_locations:
			# make a bounding box
			top, right, bottom, left = face_location
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

		# show the result 
		cv2.imshow("video", frame)

	if cv2.waitKey(1) == 27:
		break
 
video_cap.release()
cv2.destroyAllWindows()