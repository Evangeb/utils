import numpy as np
import json


for ii in range(1,31):
    if ii in (12, 14, 16,24):
        continue


    with open('CAMEL/detections/CAMEL-mask-rcnn-R-101-FPN-3x-gn-detections/json/Seq{}-Vis_det.json'.format(ii)) as f:
        data = json.load(f)

    new_data = []

    for datum in data:
        item_class = datum['category_id']
        if item_class == 1 or item_class == 2 or item_class == 3 or item_class == 4 or item_class == 18:
            new_datum = np.zeros(5)
            frame = datum['image_id']
            new_datum[0] = frame
            new_datum[1:] = datum['bbox']
            new_data.append(new_datum)


    new_data = np.asarray(new_data)

    np.savetxt('Seq{}-Vis_det.txt'.format(ii), new_data)



'''
sequences = []
for ii in range(1,31):
  if ii in (12, 14, 16,24):
    continue
  for jj in ['IR','Vis']:
    seq = 'Seq{}-{}_det.txt'.format(ii,jj)
    print(seq)
    left, right = seq.split('-')
    print(left)
    print(right)
    left2,_ = right.split('_')
    print(left2)
    sequences.append(seq)   

print(sequences)
'''
