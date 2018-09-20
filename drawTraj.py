import cv2
import numpy as np
import matplotlib.pyplot as plt



def main():
    img = np.zeros((256,336,3))

    gt = np.genfromtxt('Drone5.txt')

    green = (0, 255,0)

    #plt.imshow(img)
    #plt.show()

    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output4.mp4',fourcc, 30.0, (336,256))


    cap = cv2.VideoCapture('Drone5.mp4')
    success, frame = cap.read()
    count = 1


    all_frames = []


    while success:
        print(count)


        gt_current_frame = gt[gt[:,0] <= count]
        gt_current_frame = gt_current_frame[gt_current_frame[:,0] > count-20]
        

        if len(gt_current_frame) > 0:
             
            for frames in gt_current_frame:
                
                _, _ , x, y , w , h = frames
                xc = x + w/2
                yc = y + h/2

                xc = int(xc)
                yc = int(yc)

                #img[yc,xc] = green
                
                frame[yc,xc] = green
                
            out.write(frame)
        else:
            out.write(frame) 
        success, frame = cap.read()
        count += 1

    cap.release()
    out.release()


if __name__ == '__main__':
    main()


