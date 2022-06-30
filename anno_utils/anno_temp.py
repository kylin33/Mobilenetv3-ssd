import os
import glob
from posixpath import split
import cv2
import shutil
import random
import numpy as np
import math
import torch
import pandas as pd
import csv
import shutil


def gen_ar_new_data_txt():
    folder_list = [
        "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/AR-data/20220311AR-1",
        # "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/AR-data/20220311AR-2"
    ]


    label_txt = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/20220311AR-1-2/20220311AR-1-extrement.txt"
    error_label_txt = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/20220311AR-1-2/20220311AR-1-error.txt"

    for folder_dir in folder_list:
        print("folder_dir:",folder_dir)
        sub_folder_path = os.listdir(folder_dir)
        error_num = 0
        for sub_folder in sub_folder_path:
            sub_folder_dir = folder_dir+"/"+sub_folder
            print(sub_folder,sub_folder_dir)

            new_data_img_path = sub_folder_dir+"/rgb"
            new_data_label_path = sub_folder_dir+"/bbox2d"

            label_list = glob.glob(new_data_label_path+"/*.txt")
            label_num = len(label_list)
            for i,label_dir in enumerate(label_list):
                label_name = os.path.split(label_dir)[-1].split(".")[0]
                print(i,label_dir,label_name)
                assert os.path.exists(label_dir)
                img_dir = new_data_img_path+"/"+label_name+".png"
                # assert os.path.exists(img_dir)
                if not os.path.exists(img_dir):
                    with open(error_label_txt,"a+") as f:
                        f.write(label_dir+"\n")
                else:
                    img = cv2.imread(img_dir)
                    h,w,c = img.shape

                    with open(label_dir,"r") as f:
                        annonations = f.readlines()
                    print("annonations:",annonations,len(annonations))
                    if (len(annonations)>2):
                        print("len(annonations)>2")
                        break
                    cls_id = None
                    for annon in annonations:
                        annon = annon.strip()
                        if len(annon) !=0:
                            annon_list = annon.split(",")
                            print("anno_list:",annon_list)
                            if annon_list[0] == "animal_poop":
                                cls_id = "excrement"
                            if annon_list[0] == "chair":
                                cls_id = "chair"
                            # if annon_list[0] == "cup":
                            #     cls_id = "papercup"
                            # if annon_list[0] == "shoes":
                            #     cls_id = "slipper"
                            # if annon_list[0] == "earphones":
                            #     cls_id = "dataline"

                            # cls_list = ["chair","animal_poop"]
                            # if annon_list[0] not in cls_list:
                            #     break
                            if cls_id == None:
                                break

                            x1 = int(annon_list[1])
                            y1 = int(annon_list[2])
                            x2 = int(annon_list[5])
                            y2 = int(annon_list[6])
                            xmin = round(x1/w,5)
                            ymin = round(y1/h,5)
                            xmax = round(x2/w,5)
                            ymax = round(y2/h,5)
                            print("label:",img_dir,xmin,ymin,xmax,ymax,cls_id)

                            with open(label_txt,"a+") as f:
                                f.write(img_dir+" "+str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax)+" "+cls_id+"\n")

def gen_ar_ratio():
    ar_label_dir = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/20220311AR-1-2/20220311AR-1-extrement.txt"

    train_ratio_label = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/excrement-ratio/ar/s7000_train.txt"
    val_ratio_label = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/excrement-ratio/ar/s7000_val.txt"

    with open(ar_label_dir,"r") as f:
        annonations = f.readlines()
    random.shuffle(annonations)
    all_num = len(annonations)
    ratio = 0.38
    for i,annon in enumerate(annonations):
        print(all_num,i,annon)
        if i < ratio*all_num:
            if i<0.9*ratio*all_num:
                with open(train_ratio_label,"a+") as f:
                    f.write(annon)
            else:
                with open(val_ratio_label,"a+") as f:
                    f.write(annon)


def gen_ori_ratio():
    ar_label_dir = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/ori_excrement/excrement.txt"

    train_ratio_label = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/excrement-ratio/ori/extrement_train.txt"
    val_ratio_label = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/excrement-ratio/ori/extrement_val.txt"

    with open(ar_label_dir,"r") as f:
        annonations = f.readlines()
    random.shuffle(annonations)
    all_num = len(annonations)
    ratio = 0.3
    for i,annon in enumerate(annonations):
        print(all_num,i,annon)
        if i < ratio*all_num:
            if i<0.9*ratio*all_num:
                with open(train_ratio_label,"a+") as f:
                    f.write(annon)
            else:
                with open(val_ratio_label,"a+") as f:
                    f.write(annon)

def conacate_old_new():
    old_train = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0304/train_ratio/ori300_ar10s7_train.txt"
    old_val = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0304/train_ratio/ori300_ar10s7_val.txt"

    new_ar_train = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/excrement-ratio/ar/s7000_train.txt"
    new_ar_val = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/excrement-ratio/ar/s7000_val.txt"

    new_ori_train = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/excrement-ratio/ori/extrement_train.txt"
    new_ori_val = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/excrement-ratio/ori/extrement_val.txt"

    all_train = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/train-ratio/cls4_ori400_ars7_train.txt"
    all_val = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/train-ratio/cls4_ori400_ars7_val.txt"

    with open(old_train,"r") as f:
        old_train_lines = f.readlines()
    with open(new_ar_train,"r") as f:
        new_ar_train_lines = f.readlines()
    with open(new_ori_train,"r") as f:
        new_ori_train_lines = f.readlines()
    train_lines = old_train_lines+new_ar_train_lines+new_ori_train_lines
    for annon in train_lines:
        print("train:",annon)
        with open(all_train,"a+") as f:
            f.write(annon)
    
    with open(old_val,"r") as f:
        old_val_lines = f.readlines()
    with open(new_ar_val,"r") as f:
        new_ar_val_lines = f.readlines()
    with open(new_ori_val,"r") as f:
        new_ori_val_lines = f.readlines()
    val_lines = old_val_lines+new_ar_val_lines+new_ori_val_lines
    for val_annon in val_lines:
        print("val:",val_annon)
        with open(all_val,"a+") as f:
            f.write(val_annon)


    # old_train = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0304/ori_cls3/ori300/ori_cls3_300_train.txt"
    # old_val = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0304/ori_cls3/ori300/ori_cls3_300_val.txt"
    # new_ori_train = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/excrement-ratio/ori/extrement_train.txt"
    # new_ori_val = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/excrement-ratio/ori/extrement_val.txt"
    # all_train = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/train-ratio/cls4_ori400_train.txt"
    # all_val = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/train-ratio/cls4_ori400_val.txt"
    # with open(old_train,"r") as f:
    #     old_train_lines = f.readlines()
    # with open(new_ori_train,"r") as f:
    #     new_ori_train_lines = f.readlines()
    # train_lines = old_train_lines+new_ori_train_lines
    # for annon in train_lines:
    #     print("train:",annon)
    #     with open(all_train,"a+") as f:
    #         f.write(annon)
    
    # with open(old_val,"r") as f:
    #     old_val_lines = f.readlines()
    # with open(new_ori_val,"r") as f:
    #     new_ori_val_lines = f.readlines()
    # val_lines = old_val_lines+new_ori_val_lines
    # for val_annon in val_lines:
    #     print("val:",val_annon)
    #     with open(all_val,"a+") as f:
    #         f.write(val_annon)

def label_show():

    label_path = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/train-ratio/cls4_ori400_train.txt"
    with open(label_path,"r") as f:
        annonations = f.readlines()
    for annon in annonations:
        img_dir = annon.split(" ")[0]
        img = cv2.imread(img_dir)
        cv2.imshow("test",img)
        # cv2.waitKey(500)
        keyValue = cv2.waitKey(100)   
        print("0xFF == ord(' '):",0xFF == ord(' '),ord(' '))         
        if keyValue & 0xFF == ord(' '):        
            cv2.waitKey(0)
        elif keyValue & 0xFF == ord('q'):      
            exit(0)

def select_img():
    img_path = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0224/ori_cls3/ori_cls3_img/slipper"
    img_list = glob.glob(img_path+"/*.png")
    print("img num:",img_list)
    for img_dir in img_list:
        print(img_dir)


def move_img_to_cls():
    txt_label = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/train-ratio/cls4_ori400_train.txt"
    save_path = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/cls_test"

    # with open(txt_label,"r") as f:
    #     annonations = f.readlines()
    # for annon in annonations:
    #     annon_split = annon.split(" ")


    with open(txt_label,"r") as f:
        annonations = f.readlines()
    for annons in annonations:
        annons_split = annons.strip().split(" ")
        print("annons_split:",annons_split,"\n",annons_split[1:],len(annons_split[1:])/5,"\n")
        img_name = os.path.split(annons_split[0])[-1]
        img = cv2.imread(annons_split[0])
        h,w,c = img.shape
        cls_num = 0
        box_num = int(len(annons_split[1:])/5)
        for i in range(box_num):
            print("cls:",annons_split[i*5+5])
            img_dir = annons_split[0].strip()
            cls_name = annons_split[i*5+5].strip()
            x1 = annons_split[i*5+1].strip()
            y1 = annons_split[i*5+2].strip()
            x2 = annons_split[i*5+3].strip()
            y2 = annons_split[i*5+4].strip()
            minx1 = int(float(x1)*w)
            miny1 = int(float(y1)*h)
            maxx2 = int(float(x2)*w)
            maxy2 = int(float(y2)*h)
            cv2.rectangle(img, (minx1, miny1), (maxx2, maxy2), (0, 0, 255), 2, 1)
            cv2.putText(img, cls_name, (minx1, miny1 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            save_dir = save_path+"/"+cls_name
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # shutil.copy(img_dir,save_dir)

            cv2.imwrite(save_dir+"/"+img_name,img)


if __name__=="__main__":
    print("start .....")

    # # ## ar 新数据生成txt
    # gen_ar_new_data_txt()

    # # ## 虚拟label按比例划分
    # gen_ar_ratio()

    # # ## ori label按比例划分
    # gen_ori_ratio()

    # # ## 合并新增类与原始数据
    # conacate_old_new()


    # label_show()


    # # ## slect img
    # select_img()

    # # ## 拷贝图片到相应类别文件夹        
    # move_img_to_cls()
