'''
DETECCUION Y CONTEO DE PERSONAS QUE ABORDAN UN AUTOBUS
SE EMPLEA METODO DE SEGMENTACION PARA REALIZAR LA DETECCION
'''
import cv2
import numpy as np
from tracker import *
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
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
        bbox = (100,150,180,88)
        gamma=1
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        kernel = np.ones((3, 3), np.uint8)
        lower = np.array([0, 0, 10])
        upper = np.array([359, 359, 110])
        # mostrando los colores seleccionados
        #lw_square = np.full((10, 10, 3), lower, dtype=np.uint8) / 255.0
        #dw_square = np.full((10, 10, 3), upper, dtype=np.uint8) / 255.0
        #plt.subplot(1, 2, 1)
        #plt.imshow(hsv_to_rgb(lw_square))
        #plt.subplot(1, 2, 2)
        #plt.imshow(hsv_to_rgb(dw_square))
        #plt.show()

        while(self.video.isOpened()):
            ret, frame = self.video.read()
            if (ret == True):

                '''
                frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # pasando la imagen a grises
                frame_gris = cv2.LUT(frame_gris,table) #aplicacion de mejora de gamma de imagen

                busqueda = frame_gris[0:238, 80:280]
                _, busqueda = cv2.threshold(busqueda, 200, 250, cv2.THRESH_BINARY)
                #busqueda = cv2.inRange(busqueda, 0, 20)
                #busqueda = cv2.erode(busqueda, kernel, iterations=1)

                contours, _ = cv2.findContours(busqueda, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    # Calculate area and remove small elements
                    area = cv2.contourArea(cnt)
                    if area > 50:
                        # print(cnt[0])
                        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2,offset=(80,0))
                # Show image




                frame = cv2.rectangle(frame,(80,0),(280,238), (0,0,255),3)
                cv2.imshow("IMAGEN", frame_gris)
                cv2.imshow("BUS", frame)
                cv2.imshow("BUSQUEDA", busqueda)
                '''

                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame_hsv = frame_hsv[0:170, 80:280]




                mask = cv2.inRange(frame_hsv, lower, upper)
                #mask = cv2.erode(mask, kernel, iterations=2)
                #mask = cv2.dilate(mask, kernel, iterations=1)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    # Calculate area and remove small elements
                    area = cv2.contourArea(cnt)
                    if area > 250  :
                        # print(cnt[0])
                        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2,offset=(80, 0))
                # Show image
                cv2.imshow('hsv',frame_hsv)
                cv2.imshow('frame',frame)
                cv2.imshow("mask",mask)
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


