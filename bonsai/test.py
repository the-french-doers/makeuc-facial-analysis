import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt
import playsound
import time
def buildGauss(frame, levels):
pyramid = [frame]
for level in range(levels):
frame = cv2.pyrDown(frame)
pyramid.append(frame)
return pyramid
def reconstructFrame(pyramid, index, levels):
filteredFrame = pyramid levels[index]
for level in range( ):
filteredFrame = cv2.pyrUp(filtered Frame)
filteredFrame = filteredFrame[:videoHeight, :videoWidth]
return filteredFrame
#Webcam Parameters
webcam = None
webcam = cv2.VideoCapture(0)
realWidth = 640
realHeight = 480
videoWidth = 320
videoHeight = 240
videoChannels = 3
videoFrameRate = 15
webcam.set(3, realWidth);
webcam.set(4, realHeight);
# Colour Magnification Parameters
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize: 150
bufferIndex = 8
#Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 30)
bpmTextLocation = (videowidth // 2 +5, 38)
fontScale = 1
fontColor (255,255,255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3
#Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))
#Bandpass Filter for Specified Frequencies
frequencies = (1.0 videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask (frequencies > minFrequency) & (frequencies <= maxFrequency)
#Heart Rate Calculation Variables
bpmCalculationFrequency = 15
bpmBufferIndex = 8
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))
i=0
j=0
while (True):
ret, frame = webcam.read()
detectionFrame = frame[videoHeight/2:realHeight-videoHeight/2, videowidth/2:realwidth-videowidth/2, :]
# Construct Gaussian Pyramid
videoGauss[buffer Index] = buildGauss(detection Frame, levels+1)(levels]
fourierTransform = np.fft.fft(videoGauss, akis=0)
#bandpass Filter
fourierTransform[mask = False]= 9
# Beobea Pulse
if bufferIndex % bpmCalculationFrequency == 0:
i= i.i
for buf in range(bufferSize):
fourierTransformAvg[buf] = np.real(fourier Transform[buf]).mean()
bz= frequencies[np.argmax(fourier TransformAvg)]
bpm =60.0 * hz
bpmBuffer[bpmBufferIndex] = bpm
bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize
filtered = np.real(np.fft.ifft(fourier Transform, axis= ))
filtered = filtered *alpha

#Reconstruct Resulting Frame
filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
#cv2.imshow("facel',filteredframe)
outputFrame = detectionFrame + filteredFrame
#cv2,imshow("face2',outputFrame)
outputFrame = cv2.convertScaleAbs(output Frame)
cv2.imshow('Gaussan Pyramid',outputFrame)
bufferIndex = (bufferIndex + 1 ) % bufferSize
#frame[videoHeight/2:realHeight-videoHeight/2, videowidth/2-realwidth-videowidth/2, 1 outputFrame
#ing cv2.Imread('Abbas1.JPG'>
h = np.zeros((300,256,3))
b,g,r = cv2.split(outputFrame)
bins = np.arange(256).reshape(256,1)
color= [(255,0,0),(0,255,0),(0,0,255)]
for item,col in zip([b,g,r],color):
hist_item = cv2.normalize= cv2.calcHist(hist_item,hist_item([item],[0],Mone,[256],[0,255])
,0,255,cv2.NORM_MINMAX)
histanp.int32(np.around(hist_item))
pts= np.column_stack((bins,hist))
cv2.polylines(h,[pts],False,cal)

cv2.imshow("colorhist",h)
cv2.rectangle(frame, (videoWidth/2 , videoHeight/2), (realWidth-videoWidth/2, realHeight-videoHeight/2), boxColor, boxWeight)
cv2.putText(frame, "Place you face inside the Rectangle", loadingTextLocation, font, fontScale, (255,0,0), lineType)
if i > bpmBufferSize:
cv2.putText(frame, "Heart Beet: %d" % bpmBuffer.mean(), (20,450), font, fontScale, (0,0,225), lineType)
if j == 1:
end = time.time()
if end - start >= 6.5:
j=0
if j == 0:
start = time.time()
j= 1
else:
cv2.putText(frame, "Calculating Heart Rate ...", (20,450), font, fontScale, (0,0,225), lineType)
cv2.imshow("Webcam Heart Rate Monitor by : Abbas ", frame)
if cv2.waitkey(1) & 0xFF == ord('q');
break
webcam.release()
cv2.destroyAllWindows()
