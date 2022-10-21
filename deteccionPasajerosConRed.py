import cv2
import numpy as np
from tracker import *



#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
org = (5, 25)
color = (255, 0, 0)
pasajeros = 0


'''
DETECCUION Y CONTEO DE PERSONAS QUE ABORDAN UN AUTOBUS
SE EMPLEA METODO DE IA PARA REALIZAR LA DETECCION BASADO EN
https://pysource.com/2021/12/07/how-to-count-people-from-cctv-cameras-with-opencv-and-deep-learning/
'''

class Segmentacion:
    def __init__(self,no_video):
        print("starting")
        try:
            self.no_video = no_video
            if self.no_video == 1:
                self.video = cv2.VideoCapture('videos/2015_05_12_10_49_54FrontColor.avi', )  # gente normal
            elif self.no_video == 2:
                self.video = cv2.VideoCapture('videos/2015_05_12_13_29_08FrontColor.avi',)#sombrero
            elif self.no_video == 3:
                self.video = cv2.VideoCapture('videos/2015_05_12_15_33_24FrontColor.avi',) #senior pelon
            else:
                self.video = cv2.VideoCapture('videos/2015_05_12_20_05_32FrontColor.avi',) #noche
        except:
            print("error leyendo video")
            raise SystemExit
        print("done")
    def videoSegmentar(self):


        while(self.video.isOpened()):
            ret, frame = self.video.read()
            if (ret == True):
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(frame_gray,1.2,7)

                if faces == ():
                    print('SIN ROSTROS')
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    print('etrna')

                cv2.imshow('frame gray',frame_gray)
                #cv2.imshow('faces',faces)
                cv2.imshow('frame',frame)




                if (cv2.waitKey(30) == ord('s')):
                    break
            else:
                cv2.destroyAllWindows()
                break






    def tracking(self):
        pass
for i in range(0,5):
    sg = Segmentacion(1)
    sg.videoSegmentar()
