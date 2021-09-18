#Import libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

#Load Image
image = cv2.imread("images/Maize1000x714-1000x714.png")
cv2.imshow("Image", image)

#Green Color Detection
#Color models in HSV
blur = cv2.blur(image,(5,5))
blur0=cv2.medianBlur(blur,5)
blur1= cv2.GaussianBlur(blur0,(5,5),0)
blur2= cv2.bilateralFilter(blur1,9,75,75)
cv2.imshow("Blured",blur2)
hsv = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)
dark_green = np.array([20 , 2, 20] )
light_green = np.array([100, 300, 300])
mask = cv2.inRange(hsv, dark_green, light_green)
cv2.imshow("Masked",mask)
result= cv2.bitwise_and(image,image, mask= mask)
cv2.imshow("Result",result)
edges_detected = cv2.Canny(result,80,300)
cv2.imshow("Edges Detected", edges_detected)
#light_green = np.array([120,100,58.8])
#dark_green = np.array([120, 100, 78.5])
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#mask = cv2.inRange(hsv_image, light_green, dark_green)
#result = cv2.bitwise_and(image, image, mask=mask)
#cv2.imshow("Result", result)

cv2.waitKey(0)

#Gray-Scaling


#Edge Detection

