# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 22:03:02 2021

@author: Lenovo
"""
#pip install CMake
#pip install dlib
#pip install face-recognition

import cv2
import numpy as np
import face_recognition
import os
import pyttsx3
from pytesseract import pytesseract

#images' folder directory
path = 'Students'
#List for the images and for the names
images = []
studentnames = []
seen = []
#List with all files in the directory
mylist=os.listdir(path)
#print(mylist)

#loop to extract images and names to append it in the lists
for sn in mylist:
    curImg=cv2.imread(f'{path}/{sn}')
    images.append(curImg)
    studentnames.append(os.path.splitext(sn)[0])

#print(studentnames)


def findencoding(images):
    encodinglist = []
    for img in images:
        #change color of image and format for encoding
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #encode faces in the images
        encode = face_recognition.face_encodings(img)[0]
        #add encodings to the list
        encodinglist.append(encode)
    #return the encoded of the known images    
    return encodinglist

encodedlistknown = findencoding(images)
print("Encoding Complete!")
#print(len(encodedlistknown))

from datetime import datetime
def markAttendance(name):
    with open('Attendance_Sheet.csv','r+') as f:
        #read all file content
        myDataList = f.readlines()
        namelist = []
        #loop through the names in excel sheet to see if it is mentioned before
        for line in myDataList:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            time = datetime.now().strftime('%H:%M')
            date = datetime.now().strftime('%D')
            f.writelines(f'\n{name},{date},{time}')
 
def sayName(name):
    if name not in seen:
        seen.append(name)
        text_speech = pyttsx3.init()
        text_speech.say(name)
        text_speech.runAndWait()                     
        
#open camera 
cap = cv2.VideoCapture(0)
while True:
    #take frames 
    success, img = cap.read()
    #resize taken images to save space
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    #convert color to encode
    imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    #save all faces' location captured in one frame in a list
    facescurframe = face_recognition.face_locations(imgs)
    #encode all faces in one frame in a list
    encodescurframe = face_recognition.face_encodings(imgs,facescurframe)
    #loop through both lists of face locations and encodings 
    for encodeface,faceloc in zip(encodescurframe,facescurframe):
        #compare each face with all known faces 
        matches = face_recognition.compare_faces(encodedlistknown,encodeface)
        #get the distance between each face and all known faces
        facedist = face_recognition.face_distance(encodedlistknown,encodeface)
        print(facedist)
        #get the index of the least distance
        matchindex = np.argmin(facedist)
        #if the least distance matches true
        if matches[matchindex]:
            #get the student name
            name = studentnames[matchindex].upper()
            print(name)
            #get locations to make a green triangle with name of recognized face
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,225,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,225,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(225,225,225),2)
            #Save the record of the name in the exel sheet
            markAttendance(name)
            sayName(name)
    #to open window of camera feed        
    cv2.imshow('webcam',img)
    #cv2.waitKey(1)
    #to close and end running press q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()    
            