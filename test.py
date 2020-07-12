# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:54:18 2019

@author: Bharath Gowda
"""


import speech_recognition as sr
import cv2, os

r = sr.Recognizer()

with sr.Microphone() as source:
    print("speak anything")
    audio = r.listen(source)
    try:
        text=r.recognize_google(audio)
        print("u said :{}".format(text))
    except:
        print("cannot recognize")

if text=="photo":
    video=cv2.VideoCapture(0)
    check, frame=video.read()
    print(check)
    print(frame)
    cv2.imshow("capturing", frame)
    cv2.waitKey(0)
    cv2.imwrite('a.jpg',frame)
    video.release()


import cv2
import os
            
def facecrop(image):  
    facedata = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)

    try:
    
        minisize = (img.shape[1],img.shape[0])
        miniframe = cv2.resize(img, minisize)

        faces = cascade.detectMultiScale(miniframe)

        for f in faces:
            x, y, w, h = [ v for v in f ]
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            sub_face = img[y:y+h, x:x+w]

            f_name = image.split('/')
            f_name = f_name[-1]
            
            cv2.imwrite('capture.jpg',sub_face)

    except:
        pass



if __name__ == '__main__':
    file = "a.jpg"
    facecrop(file)


import pyttsx3

def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()
    if y_pos[3]:
        if emotions[3]>0.5 :
            print('happy')
            
            engine = pyttsx3.init()
            rate=engine.getProperty("rate")
            engine.setProperty("rate",100)
            engine.say("Happiness enlightens the day,sharing it would be worth")
            engine.runAndWait()
            os.startfile('new.txt')

    if y_pos[6] :
        if emotions[6]>0.5 :
            print('Neutral')
            
            engine = pyttsx3.init()
            rate=engine.getProperty("rate")
            engine.setProperty("rate",100)
            engine.say("Neither acidic nor dilute,you are attractively neutral")
            engine.runAndWait()
            os.startfile('new.txt')

    if y_pos[0] :
        if emotions[0]>0.5 :
            print('Angry')
            
            engine = pyttsx3.init()
            rate=engine.getProperty("rate")
            engine.setProperty("rate",100)
            engine.say("Deed in anger is always danger,thought up the reason here")
            engine.runAndWait()
            os.startfile('new.txt')

    if y_pos[1] :
        if emotions[1]>0.5 :
            print('Disgusted')
            
            engine = pyttsx3.init()
            rate=engine.getProperty("rate")
            engine.setProperty("rate",100)
            engine.say("cool buddy cool,everything happens to a reason")
            engine.runAndWait()
            os.startfile('new.txt')

    if y_pos[2] :
        if emotions[2]>0.5 :
            print('Fear')
            
            engine = pyttsx3.init()
            rate=engine.getProperty("rate")
            engine.setProperty("rate",100)
            engine.say("brave spirit in you,kicks of fear spirit")
            engine.runAndWait()
            os.startfile('new.txt')

    if y_pos[4] :
        if emotions[4]>0.5 :
            print('Sad')
            
            engine = pyttsx3.init()
            rate=engine.getProperty("rate")
            engine.setProperty("rate",100)
            engine.say("watsupp buddy let me console you,what's the reason")
            engine.runAndWait()
            os.startfile('new.txt')

    if y_pos[5] :
        if emotions[5]>0.5 :
            print('Surprised')
            
            engine = pyttsx3.init()
            rate=engine.getProperty("rate")
            engine.setProperty("rate",100)
            engine.say("Aww!!surpriiiiiised,glad to see you like this")
            engine.runAndWait()
            os.startfile('new.txt')


from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import os

file = 'capture.jpg'
true_image = image.load_img(file)
img = image.load_img(file, grayscale=True, target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48]);

plt.gray()
plt.imshow(true_image)
plt.show()



