import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as pit    
import playsound
import time
def buildGauss(frame ,levels):
    pyramid =[frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid
def  reconstructFrame  (pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame =cv2.pyrUp(filteredFrame)
    filteredFrame =filteredFrame[:videoHeight , :videoWidth]
    return filteredFrame

# Webcam Parameters
webcam =None
webcam = cv2.VideoCapture(0)
realWidth= 640
realHeight =480
videoWidth = 320
videoHeight =240
videoChannels =3
videoFrameRate = 15
webcam.set(cv2.CAP_PROP_FRAME_WIDTH,realWidth)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, realHeight)

# # Colour Magnification Parameters
# levels =3
# alpha =170 
# minFrequency =1.0 
# maxfrequency =2.0
# bufferSize= 150
# bufferIndex=0

# # output of splay Parameters
# font=cv2.FONT_HERSHEY_SIMPLEX
# loadingTextLocation =(20, 50)
# bpeTextlocation =(videoWidth // 2+5,30)
# fontScale= 1
# fontColor =(255,255,255)
# lineType= 2
# boxColor =(0, 255,0)
# boxWeight=3

# # Initialize Gonsior Pyramid
# firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
# firstGauss= buildGauss(firstFrame, levels +1)[levels]
# videoGauss=np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
# fourierTransformAvg = np.zeros((bufferSize))

# # Bandpass Filter for Specified Frequencies
# frequencies= (1.0*videoFrameRate) *np.arange(bufferSize) /(1.0*bufferSize)
# mask= (frequencies > minFrequency) & (frequencies <= maxfrequency)

# # Heart Rate Calculation Variables
# bpmCalculationFrequency = 15
# bpmBufferIndex=0
# bpmBufferSize =10
# bpmBuffer=np.zeros((bpmBufferSize))

# i=0
# j=0
# sigma = 0
# iteration = 0
# while  (True):
#     ret,frame = webcam.read()
    
#     if not ret:
#         print("Ignoring empty camera frame.")
#         continue
    
    
#     detectionFrame = frame[int(videoHeight/2):int(realHeight-videoHeight/2), int(videoWidth/2):int(realWidth-videoWidth/2),:]
#     videoGauss[bufferIndex]=buildGauss(detectionFrame, levels+1)[levels]
#     fourierTransform = np.fft.fft(videoGauss, axis = 0)
    
#     fourierTransform[mask == False] =0
    
# # 
# # Grab a Pulse
#     if bufferIndex % bpmCalculationFrequency ==0:
#         i= i+1
#     for buf in range(bufferSize):
#         fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
#     hz= frequencies[np.argmax(fourierTransformAvg)]
#     bpm= 60.0* hz
#     sigma += bpm
#     iteration+=1
#     bpmBuffer[bpmBufferIndex]= bpm
#     bpmBufferIndex = (bpmBufferIndex + i)% bpmBufferSize
#     filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
#     filtered =filtered *alpha
# #leconstruct Resulting Frame
#     filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
# #c12.inshow facel',filteredframe)
#     outputFrame =detectionFrame + filteredFrame
# #cv2.inshow('face2',putputFrame)
#     outputFrame =cv2.convertScaleAbs(outputFrame)
#     cv2.imshow("Gaussan Pyramid",outputFrame)
#     bufferIndex = (bufferIndex + 1) % bufferSize
# #framevideolright/2.realHeight-wideoHeight/2, videowidthyrealwidth videnvidth/2, J=output/case
# #ing cv2.imreod(Abbas1.JPG")
#     h =np.zeros((309,256,3))
#     b,g,r =cv2.split(outputFrame)
#     bins =np.arange(256).reshape(256,1)
#     color =[ (255,0,0),(0,255,0),(0,0,255) ]
#     for item,col in zip([b,g,r],color):
#         hist_item = cv2.calcHist([item],[0],None,[256],[0,255])
#         cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
#         hist= np.int32(np.around(hist_item))
#         pts= np.column_stack((bins,hist))
#         cv2.polylines(h,[pts],False,col)
#     #h-no.fllpud(h)
#     cv2.imshow("colorhist",h)
    
    
    
#     cv2.rectangle(frame, (int(videoWidth/2) , int(videoHeight/2)), (int(realWidth-videoWidth/2), int(realHeight-videoHeight/2)), boxColor, boxWeight)
#     cv2.putText(frame, "Place you face inside the Rectangle", loadingTextLocation, font, fontScale, (255,0,0), lineType)
#     if i > bpmBufferSize:
#         cv2.putText(frame, "Heart Beet: %d" % bpmBuffer.mean(), (20,450), font, fontScale, (0,0,225), lineType)
#         if j == 1:
#             end= time.time()
#             if end - start >= 6.5:
#                 j=0
#         if j == 0:
            
#             start =time.time()
#             j=1
#     else:
#         cv2.putText(frame, "Calculating Heart Rate ...", (20,450), font, fontScale, (0,0,225), lineType)
#     cv2.imshow("Webcam Heart Rate Monitor by : Abbas ", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print(f"your final Bpm is:{sigma/iteration}")
#         break

webcam.release()
cv2.destroyAllWindows()
