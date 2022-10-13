import cv2
import numpy as np

video = cv2.VideoCapture('videos/2015_05_12_10_49_54FrontColor.avi',)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
org = (5, 25)
color = (255, 0, 0)
pasajeros = 0
while(video.isOpened()):
    ret, frame = video.read()
    if(ret == True):

        #============================================================
        result = frame.copy()
        imgSinRojo = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 100, 100])
        upper = np.array([1, 255, 255])
        mask = cv2.inRange(imgSinRojo, lower, upper)
        #cv2.imshow('mask', mask)
        imgGris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #imagen en gris
        #cv2.imshow("gris",imgGris)
        result = imgGris + mask


        mpdr = 2 #multiplicador para la imagen de mapa
        #print(result.shape)
        result= result[150:238,80:280]
        mapa2D = np.zeros((88*mpdr,200*mpdr,3),dtype=np.uint8)
        mapa2D[:,:,:] = (255,255,255)

        pasajeros += len((face_cascade.detectMultiScale(result, 1.1, 5)))
        # print("Number of {0} faces!".format(len(faces)))
        cv2.putText(frame, "Number of {0} passengers!".format(pasajeros),
                   org, font, 1, color, 2)
        frame = cv2.rectangle(frame,(80,150),(280,238), (0,0,255),3)
        cv2.imshow("result", result)
        cv2.imshow("BUS", frame)
        cv2.imshow("MAPA2D", mapa2D)



        #=============================================================
        if(cv2.waitKey(30) == ord('s')):
            break
    else:
        break
print("Pasajeros:{}".format(pasajeros))
video.release()
cv2.destroyAllWindows()