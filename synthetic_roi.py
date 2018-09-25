import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from skimage import filters
def arg_parse():
    
    parser = argparse.ArgumentParser(description='Region of interest generation for real - synthetic RGB - IR image pairs')

    parser.add_argument("--seq", dest = 'seq', default = 'Seq2', type = str)

    parser.add_argument("--output", dest = 'output', default = 'output', type = str)

    parser.add_argument("--seq_len", dest = 'seq_len', default = 1220 , type = int)
    return parser.parse_args()

def im2single(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32) / 255

    return im

def normalize(im):
    max_val = np.max(im)
    min_val = np.min(im)
    norm_im = (im - min_val)/(max_val - min_val)

    return norm_im

def main():
    args = arg_parse()

    print(args.seq)

    print(args.output)
    Difference_sum = np.zeros((256,256,3))
    empty_image = np.zeros((256,256,3))
    real_empty = np.zeros((256,256,3))
    for ii in range(1,args.seq_len+1):
        
        fake_ir = cv2.imread("{}/{}_{:06d}_fake_B.png".format(args.seq,args.seq,ii)) 
        fake_ir = im2single(fake_ir)

        real_ir = cv2.imread("{}/{}_{:06d}_real_B.png".format(args.seq,args.seq,ii)) 
        real_ir = im2single(real_ir)

        real_vis = cv2.imread("{}/{}_{:06d}_real_A.png".format(args.seq,args.seq,ii)) 
        real_vis = im2single(real_vis)
        
        
        IR_diff = fake_ir - real_ir
        IR_diff = normalize(abs(IR_diff))
        val = filters.threshold_otsu(IR_diff[:,:,0])
        current_empty = empty_image
        #current_empty = np.zeros((256,256,3))
        current_empty[IR_diff[:,:,0] > val] = 1
        #real_empty[IR_diff[:,:,0] > val] = 255
        IR_diff = IR_diff * current_empty
        #plt.imshow(current_empty)
        #plt.show()
        #plt.imshow(IR_diff)
        #plt.show()
        Difference_sum += IR_diff
        #print(np.max(IR_diff))
        #if ii == 300:
        #plt.imshow(abs(IR_diff))
        #plt.show()


    Difference_sum = normalize(Difference_sum)

    #plt.imshow(Difference_sum)
    #plt.show()

    Difference_sum *= 255
    Difference_sum = Difference_sum.astype(int)
    
    val = filters.threshold_otsu(Difference_sum[:,:,0])
    
    


    real_empty[Difference_sum[:,:,0] > val] = 255
    print(empty_image)
    #plt.imshow(empty_image)
    #plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    closing = cv2.morphologyEx(empty_image, cv2.MORPH_OPEN, kernel)
    #closing = cv2.morphologyEx(empty_image, cv2.MORPH_CLOSE, kernel)
    


    masked_vis = real_vis * closing
    mask_ir = real_ir * closing
    plt.imshow(masked_vis)
    #plt.imshow(real_vis)
    plt.show()

    plt.imshow(mask_ir)
    plt.show()

    fused = cv2.addWeighted(real_vis, 0.5, mask_ir.astype(np.float32),0.5,0)
    plt.imshow(fused)
    plt.show()

if __name__ == "__main__":
    main()


