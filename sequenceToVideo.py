import numpy as np
import cv2


fourcc = cv2.VideoWriter_fourcc(*'MP4V')
vidcap = cv2.VideoWriter('recording_2019-02-15-16:33:03-375-800-rgbd.mp4', fourcc, 10.0, (640,480))

for ii in range(375, 800):
    
    img = cv2.imread('/home/cpslab/Lidar-RGB-Recordings/recording_2019-02-27 12:04:55/rgbd-{}.png'.format(ii))
    print(img.shape)
    vidcap.write(img)

vidcap.release()
