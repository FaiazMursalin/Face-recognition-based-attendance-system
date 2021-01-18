# importing necessary libraries

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'image_Attendance'
images = []
class_names = []
mylist = os.listdir(path)
# print(mylist)


for cls in mylist:
    current_image = cv2.imread(f'{path}/{cls}')
    images.append(current_image)
    class_names.append(os.path.splitext(cls)[0])


# print(class_names)


# encoding

def findEncodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        mydatalist = f.readlines()
        # print(mydatalist)
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            time = datetime.now()
            date_time_string = time.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {date_time_string}')


# markAttendance('Elon')

encoded_list_known = findEncodings(images)
print('encoding complete')

# testing image using camera

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_cap = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_cap = cv2.cvtColor(img_cap, cv2.COLOR_BGR2RGB)

    face_current_frame = face_recognition.face_locations(img_cap)
    encode_current_frame = face_recognition.face_encodings(img_cap, face_current_frame)

    # matching

    for encodeface, faceloc in zip(encode_current_frame, face_current_frame):
        matches = face_recognition.compare_faces(encoded_list_known, encodeface)
        face_distance = face_recognition.face_distance(encoded_list_known, encodeface)
        # print(face_distance)
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            name = class_names[match_index]
            # print(name)
            # creating rectangle
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            markAttendance(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)
