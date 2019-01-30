import numpy as np
import os
sequences = []
for ii in range(1,31):
    if ii in (12, 14, 16,24):
        continue
    for jj in ['IR','Vis']:
        seq = 'Seq{}-{}'.format(ii,jj)
        sequences.append(seq)

fmt = '%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d'

for seq in sequences:
    curr_dets = np.genfromtxt('/home/gebhardt/fusion_sort/tracking-outputs/sort-camel-results/output-yolov3/{}_det.txt'.format(seq),delimiter=',')
    curr_gt = np.genfromtxt('/home/gebhardt/fusion_sort/CAMEL/annotations/mot-txt/{}.txt'.format(seq),delimiter=',')
    print(curr_dets.shape)
    print(curr_gt.shape)
    os.mkdir(seq)
    os.mkdir('{}/gt'.format(seq))
    np.savetxt('{}/{}.txt'.format(seq,seq),curr_dets,fmt = fmt, delimiter=',')
    np.savetxt('{}/gt/gt.txt'.format(seq),curr_gt,fmt = fmt, delimiter =',')
