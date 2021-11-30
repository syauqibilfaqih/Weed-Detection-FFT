#Import libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from array import *

y=0
x=0

#Load Image
image = cv2.imread("images/Maize1000x714-1000x714.png")
#cv2.imshow("Image", image)

#Green Color Detection
#Color models in HSV
blur = cv2.blur(image,(5,5))
blur0=cv2.medianBlur(blur,5)
blur1= cv2.GaussianBlur(blur0,(5,5),0)
blur2= cv2.bilateralFilter(blur1,9,75,75)
#cv2.imshow("Blured",blur2)
hsv = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)
dark_green = np.array([20 , 2, 20] )
light_green = np.array([100, 300, 300])
mask = cv2.inRange(hsv, dark_green, light_green)
#cv2.imshow("Masked",mask)
result= cv2.bitwise_and(image,image, mask= mask)
#cv2.imshow("Result",result)
edges_detected = cv2.Canny(result,80,300)
#cv2.imshow("Edges Detected", edges_detected)

#====================================================Size per Block=======================================
windowsize_r = 15
windowsize_c = 15

img = edges_detected
y=len(range(0,img.shape[0] - windowsize_r, windowsize_r))
x=len(range(0,img.shape[0] - windowsize_c, windowsize_c))
print(y)
print(x)
print("Jumlah block yang dihasilkan : ",x*y)
pictFilter=[[0 for x in range(x)] for y in range(y)] 
y=0
x=0

for r in range(0,img.shape[0] - windowsize_r, windowsize_r):
    for c in range(0,img.shape[0] - windowsize_c, windowsize_c):
        window = img[r:r+windowsize_r,c:c+windowsize_c]
        hist = np.histogram(window,bins=8)
        pictFilter[y][x]=window
        #cv2.imshow('wind',window)
        #print(np.fft.fft2(window))
        k = cv2.waitKey(0)
        x=x+1
        if k == 27:
            cv2.destroyAllWindows()
    x=0
    y=y+1

#=========================================================PERCOBAAN BLOCKS===================================      
#Pick a picture for 2FFT
#cv2.imshow('Pick',pictFilter[18][18])
#cv2.waitKey(0)

#Freq min
#fft2 = np.fft.fft2(pictFilter[0][3]) #2D-FFT for a block
#plt.imshow(abs(fft2))
#print(sum(sum(abs(fft2))))
#cv2.imshow('sedeng',abs(fft2))
#plt.show() #show the plot
#Freq sedeng
#fft2 = np.fft.fft2(pictFilter[0][0]) #2D-FFT for a block
#plt.imshow(abs(fft2))
#print(sum(sum(abs(fft2))))
#cv2.imshow('min',abs(fft2))
#plt.show() #show the plot
#Freq max
#fft2 = np.fft.fft2(pictFilter[18][18]) #2D-FFT for a block
#plt.imshow(abs(fft2))
#print(sum(sum(abs(fft2))))
#cv2.imshow('max',abs(fft2))
#plt.show() #show the plot

#================================================2D-FFT and Merge algorithm========================================
y=len(range(0,img.shape[0] - windowsize_r, windowsize_r))
x=len(range(0,img.shape[0] - windowsize_c, windowsize_c))
v=0

horizontal=np.hstack((abs(np.fft.fft2(pictFilter[v][0])),abs(np.fft.fft2(pictFilter[v][1]))))
h=2
while h<x-1:
    horizontal=np.hstack((horizontal,abs(np.fft.fft2(pictFilter[v][h]))))
    h=h+1
vertikal=horizontal
v=1
while v<y :
    horizontal=np.hstack((abs(np.fft.fft2(pictFilter[v][0])),abs(np.fft.fft2(pictFilter[v][1]))))
    h=2
    while h<x-1:
        horizontal=np.hstack((horizontal,abs(np.fft.fft2(pictFilter[v][h]))))
        h=h+1
    vertikal=np.vstack((vertikal,horizontal))
    v=v+1

plt.imshow(vertikal)
plt.show()
cv2.waitKey(0)
#=================================================DENSITY FILTERING===========================================================
#density filtering
for r in range(0,img.shape[0] - windowsize_r, windowsize_r):
    for c in range(0,img.shape[0] - windowsize_c, windowsize_c):
        window = img[r:r+windowsize_r,c:c+windowsize_c]
        hist = np.histogram(window,bins=8)
        pictFilter[y][x]=window
        pick_image= pictFilter[y][x]
        hist = cv2.calcHist([pick_image],[0],None,[256],[0,256])
        hist = [val[0] for val in hist]; 
        #Generate a list of indices
        indices = list(range(0, 256));
        #Descending sort-by-key with histogram value as key
        s = [(x,y) for y,x in sorted(zip(hist,indices), reverse=True)]
        #Index of highest peak in histogram
        index_of_highest_peak = s[0][0];
        #Index of second highest peak in histogram
        index_of_second_highest_peak = s[1][0];
        if hist[index_of_second_highest_peak] >= 344 :
            array = np.zeros([50, 50, 3], dtype = np.uint8)
            # setting RGB color values as 255,255,255
            array[:, :] = [255, 255, 255] 
            # displaying the image
            cv2.imshow("image", array)
            
        else : 
            cv2.imshow("window", pictFilter[y][x])
            cv2.waitKey(0)
