'''
DETECCUION Y CONTEO DE PERSONAS QUE ABORDAN UN AUTOBUS
SE EMPLEA METODO DE SEGMENTACION PARA REALIZAR LA DETECCION
'''
import cv2, numpy
from tracker import *

class Segmentacion:
    def __init__(self,no_video):
        print("starting")
        self.no_video = no_video
        if self.no_video == 1:
            self.video = cv2.VideoCapture('videos/2015_05_12_10_49_54FrontColor.avi', )  # gente normal
        elif self.no_video == 2:
            self.video = cv2.VideoCapture('videos/2015_05_12_13_29_08FrontColor.avi',)#sombrero
        elif self.no_video == 3:
            self.video = cv2.VideoCapture('videos/2015_05_12_15_33_24FrontColor.avi',) #senior pelon
        else:
            self.video = cv2.VideoCapture('videos/2015_05_12_20_05_32FrontColor.avi',) #noche
        print("done")
    def videoSegmentar(self):
        pass
    def tracking(self):
        pass

sg = Segmentacion(1)

