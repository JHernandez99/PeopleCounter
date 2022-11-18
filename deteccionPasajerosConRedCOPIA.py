###############################################################################
#
#
#
#
#
###############################################################################
'''
TO DO LIST:
     [x] Conteo de pasajeros
     [] Pruebas en multiples videos
     [x] conteo pasajeros a bordo
     [x] pasajeros que han bajado
'''


import cv2
import os
from datetime import date
import numpy as np
from tracker import *
import signal



#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
    def __init__(self,nombreVideo):
        print("starting")
        #try:
        self.video = cv2.VideoCapture('videos/'+nombreVideo,) #senior pelon
        ret, frame = self.video.read()
        height, width, channels = frame.shape
        self.salida = cv2.VideoWriter('videosPros/'+nombreVideo+'.avi',cv2.VideoWriter_fourcc(*'mp4v'),30.0,(width,height))
        # Load Yolo


        #except ValueError:
        #    print("error leyendo video")
        #    raise SystemExit
        print("done")


    def peopleCounter(self):
        #cv2.cuda.setDevice(0)
        net = cv2.dnn.readNet("yolov3_training_last.weights",  "yolov3_testing.cfg")
        #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        classes = ["person"]
        layer_names = net.getLayerNames()
        #output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        output_layers = net.getUnconnectedOutLayersNames()
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        font = cv2.FONT_HERSHEY_PLAIN
        framesPasados = 0
        trackerPasajeros = EuclideanDistTracker()
        pasajeros = 0
        desfX = 85
        desfY= 103
        cys = []
        cyAnterior = 0
        subiendo = False
        pasajerosSubiendo = 0
        pasajerosBajando = 0
        while(self.video.isOpened()):
            ret, frame = self.video.read()
            if (ret == True):
                image = frame.copy()

                frame = frame[85:200 , 80:290]
                deteccionesPasajeros = []



                # Detecting objects


                height, width, channels = frame.shape
                blob = cv2.dnn.blobFromImage(frame, 0.00392 ,(416, 416), (0, 0, 0), True, crop=False)#(416, 416)
                net.setInput(blob)
                #outs = net.forward()
                outs = net.forward(output_layers)
                # Showing informations on the screen
                class_ids = []
                confidences = []
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.3:
                            # Object detected
                            #print(class_id)
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            if y >10 and w<h:
                                deteccionesPasajeros.append([x,y,w,h])
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)


                boxes_ids = trackerPasajeros.update(deteccionesPasajeros)
                for box_id in boxes_ids:

                    x,y,w,h, id,cy = box_id
                    print(id)
                    cv2.putText(image, str(id),(x+desfY,y-15+desfX),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                    print(cy)
                    if(cy >= 65):
                        2# and cy <= 65):
                        pasajerosSubiendo += 1

                    '''if(cyAnterior<cy):
                        subiendo=True

                    if id > pasajerosSubiendo and subiendo == True:
                        pasajerosSubiendo +=1
                        subiendo = False
                    elif (id > pasajerosSubiendo or id>pasajerosBajando) and subiendo == False:
                        pasajerosBajando +=1
                    '''

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                #print(indexes)

                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        #color = self.colors[class_ids[i]]
                        cv2.rectangle(image, (x+desfY, y+desfX), (x+desfY + w, y+desfX + h), color, 2)
                        cv2.putText(image, label, (x+desfY, y+desfX - 1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)


                pasajeros_totalS = "PASAJEROS S: {}".format(pasajerosSubiendo)
                pasajeros_totalB = "PASAJEROS B: {}".format(pasajerosBajando)

                cv2.putText(image, pasajeros_totalS, (0, 20), font, 2, (0, 255, 0), 2)
                cv2.putText(image, pasajeros_totalB, (0, 45), font, 2, (0, 255, 0), 2)
                cv2.imshow('REGION OF INTEREST',frame)
                cv2.imshow("FINAL IMAGE", image)
                self.salida.write(image)

                if (cv2.waitKey(30) == ord('s')):
                    self.salida.release()
                    cv2.destroyAllWindows()
                    os.kill(os.getpid(), signal.SIGTERM)

                    break
            else:
                self.video.release()
                self.salida.release()
                cv2.destroyAllWindows()
                os.kill(os.getpid(), signal.SIGTERM)
                break




    def videoToImg4lblimg(self,rutaVideo):
        self.video = cv2.VideoCapture('videos/'+rutaVideo, )  #ruta del video para extraer frames
        contador = 0
        num =0
        while(self.video.isOpened()):
            ret, frame = self.video.read()
            if (ret == True):
                contador+=1
                if contador == 40:#guardar frame como imagen
                    num+=1
                    filename = 'images/{}Imagen{}.jpg'.format(rutaVideo,num)
                    print(filename)
                    status = cv2.imwrite(filename,frame)

                    contador=0
                    #cv2.imshow('frame', frame)
                    if status:
                        print('GUARDADO')
                if (cv2.waitKey(30) == ord('s')):
                    break
            else:
                cv2.destroyAllWindows()
                break


'''#sg.videoToImg4lblimg()
carpetaVideos = os.listdir('videos')
print(len(carpetaVideos))
for rutaVideo in carpetaVideos:
    print(rutaVideo)
    #sg.peopleCounter()
    sg.videoToImg4lblimg(rutaVideo)
'''
#2015_05_11_19_52_19FrontColor.avi
#2015_05_11_19_59_19FrontColor.avi
#sg = Segmentacion('2015_05_11_19_59_19FrontColor.avi')
sg = Segmentacion('2015_05_08_08_04_47FrontColor.avi')
sg.peopleCounter()


#algoritmo para obtener frames de los videos
