# importing necessary libraries

import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('imagesBasic/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgElon_test = face_recognition.load_image_file('imagesBasic/Elon test.jpg')
imgElon_test = cv2.cvtColor(imgElon_test, cv2.COLOR_BGR2RGB)

# detecting the face in the images

faceloc = face_recognition.face_locations(imgElon)[0]
encode_elon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255, 0, 255), 2)

faceloc_test = face_recognition.face_locations(imgElon_test)[0]
encode_elon_test = face_recognition.face_encodings(imgElon_test)[0]
cv2.rectangle(imgElon_test, (faceloc_test[3], faceloc_test[0]), (faceloc_test[1], faceloc_test[2]), (255, 0, 255), 2)

# now we have to compare the 2 rectangles of faces and finding the distance between them we are getting the encoding
# above and and compare them using linear SVM

result = face_recognition.compare_faces([encode_elon], encode_elon_test)
facedistance = face_recognition.face_distance([encode_elon], encode_elon_test)    #the lower the distance the better the match is
print(result, facedistance)

cv2.putText(imgElon_test, f'{result} {round(facedistance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)



cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgElon_test)
cv2.waitKey(0)
