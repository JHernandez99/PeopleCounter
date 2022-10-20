import cv2
import numpy as np
#leer video
#video = cv2.VideoCapture('videos/2015_05_12_10_49_54FrontColor.avi',)#gente normal
#video = cv2.VideoCapture('videos/2015_05_12_13_29_08FrontColor.avi',)#sombrero
#video = cv2.VideoCapture('videos/2015_05_12_15_33_24FrontColor.avi',) #senior pelon
video = cv2.VideoCapture('videos/2015_05_12_20_05_32FrontColor.avi',) #noche


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
org = (5, 25)
color = (255, 0, 0)
pasajeros = 0
while(video.isOpened()):
    ret, frame = video.read()
    if(ret == True):

        #=============================================================
        if(cv2.waitKey(30) == ord('s')):
            break
    else:
        break
print("Pasajeros:{}".format(pasajeros))
video.release()
cv2.destroyAllWindows()