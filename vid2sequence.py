import cv2
import numpy as np
import matplot.pyplot as plt


vidcap = cv2.VideoCapture('Driver Swing_07.06.18.MOV')
success,image = vidcap.read()
count = 0




while success:
  image = np.rot90(image,-1)
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
  
vidcap.release()
