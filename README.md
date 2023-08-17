import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# bringing every image and encoding them will take time
# hence we will just import images from the folder directly.

# collecting each image in the list
path='imagesAttendance'
images=[]
classNames=[]

myList=os.listdir(path)
print(myList)

for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)
# splittext is used to separate the root and extension part
# in billgates.jpg =>  billgates is root part and jpg is extension part

# for marking attendace,
# we will create a new function 

def markAttendance(name):
    # we can read and write at same time using r+
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        # print(myDataList)
        nameList=[]
        for line in myDataList:
            # we want to split them using commas 
            # as it is a comma separated file
            entry=line.split(",")
            nameList.append(entry[0])

        # this will prevent multiple entries of the same person in the list.
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# markAttendance('Elon')
def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown=findEncodings(images)
# encoding has been completed till now 

# initializing the webcam
cap=cv2.VideoCapture(0)
# 0 is the user defined id 

# imgS is the smaller version of the image
# (Since we are solving the problem in real time
# ,we want the image to be smaller)


# img is original image, None is pixel size
# 0.25 and 0.25 is the scaling measurements
while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    img=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    # converting everything to rgb

    # we can have multiple people in a frame and hence 
    # we need to define our current frame and endcode it.
    
    faceCurFrame=face_recognition.face_locations(imgS)
    encodeCurFrame=face_recognition.face_encodings(imgS,faceCurFrame)

    # one by one the for loop will grab the encodeFace from 
    # encodeCurFrame and facelocation from faceloc
    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        # the line below does the matching 
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        # the line below finds the distance between the face captured in the webcam 
        # and the faces in our database.
        # The distances will be returned in the form of lists.
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)


        # finding the best match now
        # the minimum value of difference will be 
        # our best match 
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLoc
            # since we have scaled down the image earlier
            # we need to scale up the image as 
            # that rectangle wont be properly visible 
            # if not done
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            # now we are trying to create rectangle around face
            # and also give them image name
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)
    
