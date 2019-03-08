from __future__ import division
from __future__ import print_function

import os
import cv2
import json, yaml
import numpy as np
from PIL import Image
from collections import OrderedDict
from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi

""" Converts BU-TIV dataset style annotations to MSCOCO style annotations. Adapted from
    https://github.com/hazirbas/coco-json-converter


    inputs : annotation datapath
    
    outputs : MSCOCO Style JSON annotations
""" 


class GTVIR():
    
    def __init__(self,savepath,annotation_file, data_dir, directory,mode,file):
        self.info = {"year" : 2017,
                     "version" : "1.0",
                     "description" : "GTVIR Benchmark",
                     "contributor" : "Evan Gebhardt, Marilyn Wolf",
                     "date_created" : "2017"
                     }
        self.licenses = [{"id" : 1,
                          "name" : "Attribution-NonCommercial",
                          "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                          }]
        self.type = "instances"

        self.savepath = savepath
        
        self.categories = [{"supercategory": "person","id": 1,"name": "person"},
                           {"supercategory": "vehicle","id": 2,"name": "bicycle"},
                           {"supercategory": "vehicle","id": 3,"name": "car"},
                           {"supercategory": "vehicle","id": 4,"name": "motorcycle"},
                           {"supercategory": "vehicle","id": 5,"name": "aeroplane"},
                           {"supercategory": "vehicle","id": 6,"name": "bus"},
                           {"supercategory": "vehicle","id": 7,"name": "train"},
                           {"supercategory": "vehicle","id": 8,"name": "truck"},
                           {"supercategory": "vehicle","id": 9,"name": "boat"},
                           {"supercategory": "outdoor","id": 10,"name": "traffic light"},
                           {"supercategory": "outdoor","id": 11,"name": "fire hydrant"},
                           {"supercategory": "outdoor","id": 13,"name": "stop sign"},
                           {"supercategory": "outdoor","id": 14,"name": "parking meter"},
                           {"supercategory": "outdoor","id": 15,"name": "bench"},
                           {"supercategory": "animal","id": 16,"name": "bird"},
                           {"supercategory": "animal","id": 17,"name": "cat"},
                           {"supercategory": "animal","id": 18,"name": "dog"},
                           {"supercategory": "animal","id": 19,"name": "horse"},
                           {"supercategory": "animal","id": 20,"name": "sheep"},
                           {"supercategory": "animal","id": 21,"name": "cow"},
                           {"supercategory": "animal","id": 22,"name": "elephant"},
                           {"supercategory": "animal","id": 23,"name": "bear"},
                           {"supercategory": "animal","id": 24,"name": "zebra"},
                           {"supercategory": "animal","id": 25,"name": "giraffe"},
                           {"supercategory": "accessory","id": 27,"name": "backpack"},
                           {"supercategory": "accessory","id": 28,"name": "umbrella"},
                           {"supercategory": "accessory","id": 31,"name": "handbag"},
                           {"supercategory": "accessory","id": 32,"name": "tie"},
                           {"supercategory": "accessory","id": 33,"name": "suitcase"},
                           {"supercategory": "sports","id": 34,"name": "frisbee"},
                           {"supercategory": "sports","id": 35,"name": "skis"},
                           {"supercategory": "sports","id": 36,"name": "snowboard"},
                           {"supercategory": "sports","id": 37,"name": "sports ball"},
                           {"supercategory": "sports","id": 38,"name": "kite"},
                           {"supercategory": "sports","id": 39,"name": "baseball bat"},
                           {"supercategory": "sports","id": 40,"name": "baseball glove"},
                           {"supercategory": "sports","id": 41,"name": "skateboard"},
                           {"supercategory": "sports","id": 42,"name": "surfboard"},
                           {"supercategory": "sports","id": 43,"name": "tennis racket"},
                           {"supercategory": "kitchen","id": 44,"name": "bottle"},
                           {"supercategory": "kitchen","id": 46,"name": "wine glass"},
                           {"supercategory": "kitchen","id": 47,"name": "cup"},
                           {"supercategory": "kitchen","id": 48,"name": "fork"},
                           {"supercategory": "kitchen","id": 49,"name": "knife"},
                           {"supercategory": "kitchen","id": 50,"name": "spoon"},
                           {"supercategory": "kitchen","id": 51,"name": "bowl"},
                           {"supercategory": "food","id": 52,"name": "banana"},
                           {"supercategory": "food","id": 53,"name": "apple"},
                           {"supercategory": "food","id": 54,"name": "sandwich"},
                           {"supercategory": "food","id": 55,"name": "orange"},
                           {"supercategory": "food","id": 56,"name": "broccoli"},
                           {"supercategory": "food","id": 57,"name": "carrot"},
                           {"supercategory": "food","id": 58,"name": "hot dog"},
                           {"supercategory": "food","id": 59,"name": "pizza"},
                           {"supercategory": "food","id": 60,"name": "donut"},
                           {"supercategory": "food","id": 61,"name": "cake"},
                           {"supercategory": "furniture","id": 62,"name": "chair"},
                           {"supercategory": "furniture","id": 63,"name": "couch"},
                           {"supercategory": "furniture","id": 64,"name": "potted plant"},
                           {"supercategory": "furniture","id": 65,"name": "bed"},
                           {"supercategory": "furniture","id": 67,"name": "dining table"},
                           {"supercategory": "furniture","id": 70,"name": "toilet"},
                           {"supercategory": "electronic","id": 72,"name": "tv"},
                           {"supercategory": "electronic","id": 73,"name": "laptop"},
                           {"supercategory": "electronic","id": 74,"name": "mouse"},
                           {"supercategory": "electronic","id": 75,"name": "remote"},
                           {"supercategory": "electronic","id": 76,"name": "keyboard"},
                           {"supercategory": "electronic","id": 77,"name": "cell phone"},
                           {"supercategory": "appliance","id": 78,"name": "microwave"},
                           {"supercategory": "appliance","id": 79,"name": "oven"},
                           {"supercategory": "appliance","id": 80,"name": "toaster"},
                           {"supercategory": "appliance","id": 81,"name": "sink"},
                           {"supercategory": "appliance","id": 82,"name": "refrigerator"},
                           {"supercategory": "indoor","id": 84,"name": "book"},
                           {"supercategory": "indoor","id": 85,"name": "clock"},
                           {"supercategory": "indoor","id": 86,"name": "vase"},
                           {"supercategory": "indoor","id": 87,"name": "scissors"},
                           {"supercategory": "indoor","id": 88,"name": "teddy bear"},
                           {"supercategory": "indoor","id": 89,"name": "hair drier"},
                           {"supercategory": "indoor","id": 90,"name": "toothbrush"}]

        annotations = np.genfromtxt(file, dtype=float)
        
        images = []
        annotation = []

        annotationID = 1
        
        im_list = np.unique(annotations[:,0])

        pathToImages = data_dir + '/' +  directory + '/' + mode

        path, dirs, files = next(os.walk(pathToImages))
        file_count = len(files)

        for im in range(1,file_count+1):
            
            images.append({"id": int(im),
            "width" : 336,
            "height" : 256,
            "file_name" : data_dir + '/' + directory + "/" + mode + "/{:0>6d}.jpg".format(int(im)),
            "license" : 1,
            "flickr_url" : "",
            "coco_url" : "",
            "date_captured" : "2017"})
        
        pathToImages = data_dir + '/' +  directory + '/' + mode
 

        
        for s in annotations:
          
            
            annotation.append({"id" : annotationID,
                               "image_id" :int(s[0]),
                               "category_id" : int(s[2]),
                               "segmentation" : [],
                               "area" : (s[5]*s[6]),
                               "bbox" : [s[3],s[4],s[5],s[6]],
                               "iscrowd" : 0})

            annotationID += 1

        json_data = {"info":self.info,
                     "images" :images,
                     "licenses" :self.licenses,
                     "type" : self.type,
                     "annotations" : annotation,
                     "categories" : self.categories}

        with open(os.path.join(self.savepath, annotation_file +
                                   ".json"), "w") as jsonfile:
          json.dump(json_data,jsonfile, sort_keys = False, indent= 4)


if __name__ == "__main__":

    save_path = "/home/gebhardt/Desktop/Infrared-Datsets/CAMEL_Dataset/Annotation/json"

    data_dir = "/home/gebhardt/Desktop/Infrared-Datsets/CAMEL_Dataset/Annotation"

    data_root = "/home/gebhardt/Desktop/Infrared-Datsets/CAMEL_Dataset"

    data_files = []

    for root, dirs, files in os.walk(data_dir):

        for file in files:
        
            if file.endswith('.txt'):
                data_path = root + '/' + file
                data_files.append(data_path)

    

    for file in data_files:
    
        _, annotation_file = os.path.split(file)
        
        #annotation_file
        annotation_file = annotation_file.strip('.txt')
        directory, mode = annotation_file.rsplit('-')
        if mode == 'Vis':
            mode = 'Visual'
        #print(mode)
        #print(annotation_file)
        GTVIR(save_path,annotation_file,data_root,directory,mode,file)
    
    #GTVIR(save_path,annotation_file,mode,file)
    print("Done")
    
