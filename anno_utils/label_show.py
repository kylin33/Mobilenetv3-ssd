# -*- coding: utf-8 -*-
# @Author  : Anno
# @File    : label_show.py
# @remarks  :

import os
import shutil
import cv2
import numpy as np
import pandas as pd


def read_data():
    data_path = "/home/whf/Temp/11-扫地机/data/train"
    show_save_path = "/home/whf/Temp/11-扫地机/anno_data_show/"
    annotation_file = "/home/whf/Temp/11-扫地机/data/sub-train-annotations-bbox.csv"

    annotations = pd.read_csv(annotation_file)
    class_names = ['BACKGROUND'] + sorted(list(annotations['ClassName'].unique()))
    class_names = ['BACKGROUND'] + ['dataline', 'excrement', 'papercup', 'plasticbag', 'powerstrip', 'slipper', 'socks',
                                    "other"]
    class_dict = {class_name: i for i, class_name in enumerate(class_names)}
    print("class_dict:",class_dict)
    data = []
    for image_id, group in annotations.groupby("ImageID"):
        boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
        labels = np.array([class_dict[name] for name in group["ClassName"]])
        # print(boxes, labels)
        data.append({
            'image_id': image_id,
            'boxes': boxes,
            'labels': labels
        })
    print("datat:",data)
        # print("image_id:",image_id)
        # print("boxes:",boxes)
        # print("labels:",labels)
        # img_dir = data_path+"/"+image_id+".png"
        # print("img_dir:",img_dir)
        # img = cv2.imread(img_dir)
        # h,w,c = img.shape
        # print("img shape:",h,w,c)
        # for box in boxes:
        #     xmin = box[0]
        #     ymin = box[1]
        #     xmax = box[2]
        #     ymax = box[3]
        #
        # #     x1 = int(box[0]*w)
        # #     y1 = int(box[1]*h)
        # #     x2 = int(box[2]*w)
        # #     y2 = int(box[3]*h)
        # #     print("x1,y1:",x1,y1)
        # #     cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1,1)
        # # if not os.path.exists(show_save_path+"/"+str(int(labels[0]))):
        # #     os.mkdir(show_save_path+"/"+str(int(labels[0])))
        # # cv2.imwrite(show_save_path+"/"+str(int(labels[0]))+"/"+image_id+".png",img)
        # # # cv2.imshow("test",img)
        # # # cv2.waitKey(200)
        #
        #
        #     # # ## 生成分类图片与标签
        #     train_data = "/home/whf/Temp/11-扫地机/anno_data/train/data"
        #     train_labels = "/home/whf/Temp/11-扫地机/anno_data/train/labels"
        #     # if not os.path.exists(train_data+"/"+str(int(labels[0]))):
        #     #     os.mkdir(train_data+"/"+str(int(labels[0])))
        #     # cv2.imwrite(train_data+"/"+str(int(labels[0]))+"/"+image_id+".png",img)
        #
        #     # if not os.path.exists(train_labels+"/"+str(int(labels[0]))):
        #     #     os.mkdir(train_labels+"/"+str(int(labels[0])))
        #     with open(train_labels+"/txt_label/"+image_id+".txt","a+") as f:
        #         f.write(str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax)+" "+str(int(labels[0]))+"\n")



def gen_txt_label():
    data_path = "/home/whf/Temp/11-扫地机/data/test"
    show_save_path = "/home/whf/Temp/11-扫地机/anno_data_show/"
    annotation_file = "/home/whf/Temp/11-扫地机/data/sub-test-annotations-bbox.csv"

    annotations = pd.read_csv(annotation_file)
    class_names = ['BACKGROUND'] + sorted(list(annotations['ClassName'].unique()))
    class_names = ['BACKGROUND'] + ['dataline', 'excrement', 'papercup', 'plasticbag', 'powerstrip', 'slipper',
                                    'socks',
                                    "other"]
    class_dict = {class_name: i for i, class_name in enumerate(class_names)}
    data = []
    for image_id, group in annotations.groupby("ImageID"):
        boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
        labels = np.array([class_dict[name] for name in group["ClassName"]])
        # print(boxes, labels)
        # if image_id == "20211213154605705_orig_image":
        #     print(labels)
        data.append({
            'image_id': image_id,
            'boxes': boxes,
            'labels': labels
        })
        print("image_id:",image_id)
        print("boxes:",boxes)
        print("labels:",labels)
        img_dir = data_path+"/"+image_id+".png"
        print("img_dir:",img_dir)
        img = cv2.imread(img_dir)
        h,w,c = img.shape
        print("img shape:",h,w,c)
        for i,box in enumerate(boxes):
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]
            label = labels[i]
            x1 = int(box[0]*w)
            y1 = int(box[1]*h)
            x2 = int(box[2]*w)
            y2 = int(box[3]*h)

            # ## 1、生成txt
            # train_labels = "/home/whf/Temp/11-扫地机/anno_data/test/labels"
            # with open(train_labels+"/txt_label/"+image_id+".txt","a+") as f:
            #     f.write(str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax)+" "+str(int(label))+"\n")

            # ## 2、crop 每个类别
            crop_save_path = "/home/whf/Temp/11-扫地机/anno_data/test/crop"
            if not os.path.exists(crop_save_path+"/"+str(int(label))):
                os.mkdir(crop_save_path+"/"+str(int(label)))
            img_crop = img[y1:y2,x1:x2]
            cv2.imwrite(crop_save_path+"/"+str(int(label))+"/"+image_id+"_"+str(i)+".png",img_crop)


def main():
    read_data()

    # # ## 生成txt形式label
    # gen_txt_label()

if __name__ == '__main__':
    main()