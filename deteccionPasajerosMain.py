import cv2
import numpy as np
#leer video
video = cv2.VideoCapture('videos/2015_05_12_10_49_54FrontColor.avi',)#gente normal
#video = cv2.VideoCapture('videos/2015_05_12_13_29_08FrontColor.avi',)#sombrero
#video = cv2.VideoCapture('videos/2015_05_12_15_33_24FrontColor.avi',) #senior pelon
#video = cv2.VideoCapture('videos/2015_05_12_20_05_32FrontColor.avi',) #noche


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
org = (5, 25)
color = (255, 0, 0)
pasajeros = 0
while(video.isOpened()):
    ret, frame = video.read()
    if(ret == True):
        #convert image to gray scale
        frameGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #frameGray=frameGray[:,125:245]
        #aplicando thresholding binario
        _, result = cv2.threshold(frameGray, 70, 120, cv2.THRESH_BINARY_INV)
        #find contours
        contours, _ = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #draw contours on the original image RGB
        for cnt in contours:
            # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > 150:
                # print(cnt[0])
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)#,offset=(130,0))
        # Show image

        cv2.imshow("GRIS", frameGray)
        cv2.imshow("thresholding", result)
        cv2.imshow("IMAGEN", frame)
        #=============================================================
        if(cv2.waitKey(30) == ord('s')):
            break
    else:
        break
print("Pasajeros:{}".format(pasajeros))
video.release()
cv2.destroyAllWindows()



'''
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
        cv2.imshow("mask real",result)

        mpdr = 2 #multiplicador para la imagen de mapa
        #print(result.shape)
        result= result[140:238,80:280]
        result2 = frame[150:238, 80:280]
        mapa2D = np.zeros((88*mpdr,200*mpdr,3),dtype=np.uint8)
        mapa2D[:,:,:] = (255,255,255)

        #pasajeros += len((face_cascade.detectMultiScale(result, 5, 5)))
        #METODO PARA DETECTAR CABEZAS
        #SOLO SE DEJARAN PASAR LOS COLORES MAS OBSCUROS DE LA IMAGEN

        kernel = np.ones((5,5),np.uint8)
        result = cv2.inRange(result,0,20)
        result = cv2.erode(result,kernel,iterations=1)
        _, result = cv2.threshold(result, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > 100:

                #print(cnt[0])
                cv2.drawContours(result2, [cnt], -1, (0, 255, 0), 2)
        # Show image


        # print("Number of {0} faces!".format(len(faces)))
        cv2.putText(frame, "Number of passengers: {0} ".format(pasajeros),
                   org, font, 0.5, color, 2)
        cv2.putText(mapa2D, "Number of passengers: {0} ".format(pasajeros),
                    org, font, 0.5, color, 2)
        frame = cv2.rectangle(frame,(80,150),(280,238), (0,0,255),3)
        cv2.imshow("result", result)
        cv2.imshow("BUS", frame)
        cv2.imshow("MAPA2D", mapa2D)

'''