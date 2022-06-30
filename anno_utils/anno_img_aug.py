# -*- coding: utf-8 -*-
# @Author  : Anno
# @File    : anno_img_aug.py
# @remarks  :

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

def cal_iou_xyxy(box1,box2):
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    #计算两个框的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    #计算相交部分的坐标
    xmin = max(x1min,x2min)
    ymin = max(y1min,y2min)
    xmax = min(x1max,x2max)
    ymax = min(y1max,y2max)

    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    #计算iou
    iou = intersection / union
    return iou

def cutMix():
    labels_path = "/home/whf/Temp/11-扫地机/anno_data/train/labels/txt_label"
    crop_object_path = "/home/whf/Temp/11-扫地机/anno_data/train/crop"
    img_path = "/home/whf/Temp/11-扫地机/data/train"
    img_aug_path = "/home/whf/Temp/11-扫地机/anno_data/train/img_aug"

    labels_list = glob.glob(labels_path+"/*.txt")
    for i,label in enumerate(labels_list):
        label_name = os.path.split(label)[-1].split(".")[0]
        img_dir = img_path+"/"+label_name+".png"
        assert os.path.exists(img_dir)
        print(i,label,label_name)
        with open(label,"r") as f:
            annonations = f.readlines()

        id_list = []
        if len(annonations) >1:
            # print("annotation:", annonations)
            for anno in annonations:
                anno_list = anno.strip().split(" ")
                id_list.append(int(anno_list[-1]))
                print("anno_list:",anno_list)
                w = 640
                h = 480
                xmin = anno_list[0]
                ymin = anno_list[1]
                xmax = anno_list[2]
                ymax = anno_list[3]
                label = anno_list[-1]
                x1 = int(float(anno_list[0]) * w)
                y1 = int(float(anno_list[1])* h)
                x2 = int(float(anno_list[2]) * w)
                y2 = int(float(anno_list[3]) * h)

        else:
            for anno in annonations:
                anno_list = anno.strip().split(" ")
                id_list.append(int(anno_list[-1]))
                print("anno_list:",anno_list)
                w = 640
                h = 480
                xmin = anno_list[0]
                ymin = anno_list[1]
                xmax = anno_list[2]
                ymax = anno_list[3]
                label = anno_list[-1]
                x1 = int(float(anno_list[0]) * w)
                y1 = int(float(anno_list[1])* h)
                x2 = int(float(anno_list[2]) * w)
                y2 = int(float(anno_list[3]) * h)

def copy_paste(im, labels, segments, p=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=im, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

def random_perspective(im, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

def load_mosaic():
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    # # ## train data
    # labels_path = "/home/whf/Temp/11-扫地机/anno_data/train/labels/txt_label"
    # crop_object_path = "/home/whf/Temp/11-扫地机/anno_data/train/crop"
    # img_path = "/home/whf/Temp/11-扫地机/data/train"
    # img_aug_save_path = "/home/whf/Temp/11-扫地机/anno_data/train/img_aug/augImg"
    # img_label_aug_save_path = "/home/whf/Temp/11-扫地机/anno_data/train/img_aug/augImgLables"

    # ## test data
    labels_path = "/home/whf/Temp/11-扫地机/anno_data/test/labels/txt_label"
    crop_object_path = "/home/whf/Temp/11-扫地机/anno_data/test/crop"
    img_path = "/home/whf/Temp/11-扫地机/data/test"
    img_aug_save_path = "/home/whf/Temp/11-扫地机/anno_data/test/img_aug/augImg"
    img_label_aug_save_path = "/home/whf/Temp/11-扫地机/anno_data/test/img_aug/augImgLables"

    labels_list = glob.glob(labels_path+"/*.txt")
    img_num = len(labels_list)
    for label_i,label in enumerate(labels_list):
        img_label_name = os.path.split(label)[-1].split(".")[0]
        img_dir = img_path+"/"+img_label_name+".png"
        assert os.path.exists(img_dir)
        print(label_i,label,img_label_name)

        # print("index:",index,self.indices)
        labels4, segments4 = [], []
        img_size = 640
        s = img_size
        indices = img_num
        index = label_i
        mosaic_border = [-img_size // 2, -img_size // 2]
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in mosaic_border]  # mosaic center x, y
        indices = [index] + random.choices(range(0,indices), k=3)  # 3 additional image indices
        # print("indices:",indices)
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            # img, _, (h, w) = load_image(self, index)
            print("index:",index)
            txt_label_dir = labels_list[index]

            label_name = os.path.split(txt_label_dir)[-1].split(".")[0]
            indices_img_dir = img_path + "/" + label_name + ".png"
            path = indices_img_dir
            im = cv2.imread(path)  # BGR
            assert im is not None, 'Image Not Found ' + path
            h0, w0 = im.shape[:2]  # orig hw
            r = img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)),interpolation=cv2.INTER_AREA)
            img = im
            h = h0
            w = w0

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
        # cv2.imwrite("./temp/"+str(label_i)+".png",img4)

            # ## read label
            with open(txt_label_dir, "r") as f:
                annonations = f.readlines()
            id_list = []
            for anno in annonations:
                anno_list = anno.strip().split(" ")
                id_list.append(int(anno_list[-1]))
                print("anno_list:", anno_list)
                w = 640
                h = 480
                xmin = anno_list[0]
                ymin = anno_list[1]
                xmax = anno_list[2]
                ymax = anno_list[3]
                label = anno_list[-1]
                x1 = int(float(anno_list[0]) * w)
                y1 = int(float(anno_list[1]) * h)
                x2 = int(float(anno_list[2]) * w)
                y2 = int(float(anno_list[3]) * h)

                box_label = [int(label), x1+padw, y1+padh, x2+padw, y2+padh]

                segments = []
                labels4.append(box_label)
                segments4.extend(segments)


        # Concat/clip labels
        labels4 = np.asarray(labels4)
        labels4 = labels4.astype(np.float)
        print("labels41:",type(labels4),labels4)
        ## labels4 = np.concatenate(labels41)
        # print("labels4..:",labels4.shape,labels4)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        print("labels4:",labels4)

        # cv2.imwrite("./mosaic/"+str(index)+"_noAug.jpg",img4)
        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=0.0)

        # cv2.imwrite("./mosaic/"+str(index)+"_copy_paste.jpg",img4)
        img4, labels4 = random_perspective(img4, labels4, segments4,
                                           degrees=0,
                                           translate=0.1,
                                           scale=0.5,
                                           shear=0,
                                           perspective=0.0,
                                           border=mosaic_border)  # border to remove

        for l4 in labels4:
            print("l4:", l4)
            obj_id = int(l4[0])
            x1 = int(l4[1])
            y1 = int(l4[2])
            x2 = int(l4[3])
            y2 = int(l4[4])
            # img4 = cv2.putText(img4, str(obj_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            # cv2.rectangle(img4, (x1, y1), (x2, y2), (0, 255, 0), 1, 1)
            with open(img_label_aug_save_path+"/"+"img_aug_mos_te_"+str(label_i)+".txt","a+") as f:
                f.write(str(x1)+" "+str(y1)+" "+str(x2)+" "+str(y2)+" "+str(obj_id)+"\n")
        cv2.imwrite(img_aug_save_path+"/"+"img_aug_mos_te_"+str(label_i)+".png",img4)


def Mosaic():
    load_mosaic()


def gen_train_val():
    # ## train
    train_label_path = "/home/whf/Temp/11-扫地机/anno_data/train/labels/txt_label"
    train_img_path = "/home/whf/Temp/11-扫地机/data/train"
    train_aug_label_path = "/home/whf/Temp/11-扫地机/anno_data/train/img_aug/augImgLables"
    train_aug_img_path = "/home/whf/Temp/11-扫地机/anno_data/train/img_aug/augImg"
    train_txt = "/home/whf/Temp/11-扫地机/anno_data/train.txt"

    # ## test
    train_label_path = "/home/whf/Temp/11-扫地机/anno_data/test/labels/txt_label"
    train_img_path = "/home/whf/Temp/11-扫地机/data/test"
    train_aug_label_path = "/home/whf/Temp/11-扫地机/anno_data/test/img_aug/augImgLables"
    train_aug_img_path = "/home/whf/Temp/11-扫地机/anno_data/test/img_aug/augImg"
    train_txt = "/home/whf/Temp/11-扫地机/anno_data/test.txt"

    train_label_list = glob.glob(train_label_path+"/*.txt")
    print(train_label_list)
    for train_label in train_label_list:
        label_name = os.path.split(train_label)[-1].split(".")[0]
        train_img_dir = train_img_path+"/"+label_name+".png"
        assert os.path.exists(train_img_dir)
        with open(train_txt,"a+") as f:
            f.write(train_img_dir+" "+train_label+"\n")

    train_aug_label_list = glob.glob(train_aug_label_path+"/*.txt")
    for train_aug_label in train_aug_label_list:
        label_aug_name = os.path.split(train_aug_label)[-1].split(".")[0]
        train_aug_img_dir = train_aug_img_path+"/"+label_aug_name+".png"
        assert os.path.exists(train_aug_img_dir)
        with open(train_txt,"a+") as f:
            f.write(train_aug_img_dir+" "+train_aug_label+"\n")

# ##对背景增广并将label写入csv/txt
def cut_background():
    back_img_dir = "/home/whf/Temp/11-扫地机/anno_data/原始图/20220214背景2/orig"
    img_crop_dir = "/home/whf/Temp/11-扫地机/anno_data/train/crop/"
    aug_save_dir = "/home/whf/Temp/11-扫地机/anno_data/原始图/20220214背景2_aug/"

    all_crop_img_list = []
    for i in range(1,8):
        class_crop_img_list = glob.glob(img_crop_dir+str(i)+"/*.png")
        all_crop_img_list.append(class_crop_img_list)

    f_csv = open(aug_save_dir+"/back_aug_img_label.csv",'w',encoding='utf-8')
    csv_writer = csv.writer(f_csv)
    csv_writer.writerow(["ImageID","XMin","YMin","XMax","YMax","ClassName"])

    class_name = ['dataline', 'excrement', 'papercup', 'plasticbag', 'powerstrip', 'slipper','socks']

    back_img_list = glob.glob(back_img_dir+"/*.png")
    for bi,back_img_dir in enumerate(back_img_list):
        img_name = os.path.split(back_img_dir)[-1].split(".")[0]
        print("back_img_dir:",back_img_dir,img_name)
        back_img = cv2.imread(back_img_dir)
        bh,bw,bc = back_img.shape
        print("bh,bw,bc:",bh,bw,bc)
        for i in range(0,7):
            back_img_copy = back_img.copy()
            crop_img_list = all_crop_img_list[i]
            crop_img = random.choice(crop_img_list)
            crop_img_arrary = cv2.imread(crop_img)
            ch,cw,cc = crop_img_arrary.shape
            print("ch,cw,cc:",ch,cw,cc)
            ClassName = class_name[i]

            rx1 = random.randint(0,bw-cw)
            if (bh-ch) > int(0.5*bh):
                ry1 = random.randint(int(0.5*bh),bh-ch)
            elif (bh-ch) > int(0.3*bh):
                ry1 = random.randint(int(0.3*bh),bh-ch)
            else:
                ry1 = random.randint(0,bh-ch)
            back_img_copy[ry1:ry1+ch,rx1:rx1+cw] = crop_img_arrary
            x1 = rx1/bw
            y1 = ry1/bh
            if (rx1+cw)>bw:
                x2 = bw/bw
            else:
                x2 = (rx1+cw)/bw
            y2 = (ry1+ch)/bh

            # cv2.rectangle(back_img_copy, (rx1, ry1), (rx1+cw, ry1+ch), [255,255,0], 1, cv2.LINE_AA)
            # cv2.imshow("test.png",back_img_copy)
            # cv2.waitKey(100)
            cv2.imwrite(aug_save_dir+"/back_aug_img/"+img_name+"_backAug_"+str(i)+".png",back_img_copy)
            with open(aug_save_dir+"/back_aug_img_label.txt","a+") as f:
                f.write(aug_save_dir+"/back_aug_img/"+img_name+"_backAug_"+str(i)+".png"+" "+str(rx1)+" "+str(rx1+cw)+" "+str(ry1)+" "+str(ry1+ch)+" "+ClassName+"\n")

            ImageID = img_name+"_backAug_"+str(i)
            csv_writer.writerow([ImageID,str(x1),str(y1),str(x2),str(y2),ClassName])

    f_csv.close()


def merge_csv():
    aug_csv = "/home/whf/Temp/11-扫地机/anno_data/原始图/20220214背景2_aug/back_aug_img_label.csv"
    ori_train_csv_label = "/home/whf/Temp/11-扫地机/data/sub-train-annotations-bbox.csv"
    ori_test_csv_label = "/home/whf/Temp/11-扫地机/data/sub-test-annotations-bbox.csv"

    new_train_csv_label_1 = "/home/whf/Temp/11-扫地机/anno_data/原始图/20220211背景_aug/sub-train0214-annotations-bbox.csv"
    new_test_csv_label_1 = "/home/whf/Temp/11-扫地机/anno_data/原始图/20220211背景_aug/sub-test0214-annotations-bbox.csv"

    new_train_csv_label = "/home/whf/Temp/11-扫地机/anno_data/原始图/20220214背景2_aug/sub-train02141832-annotations-bbox.csv"
    new_test_csv_label = "/home/whf/Temp/11-扫地机/anno_data/原始图/20220214背景2_aug/sub-test02141832-annotations-bbox.csv"

    # with open(new_train_csv_label,'w',encoding='utf-8') as f_train_csv:
    #     csv_writer = csv.writer(f_train_csv)
    #     csv_writer.writerow([ImageID,str(x1),str(y1),str(x2),str(y2),ClassName])

    total = len(open(aug_csv).readlines())
    print('The total lines is ',total)

    data = []
    i = 0
    # ## 增广背景图片划分train/test 存放在sub-train02141832-annotations-bbox/sub-test02141832-annotations-bbox
    with open(aug_csv) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        #header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader:            # 将csv 文件中的数据保存到data中
            i+=1
            # data.append(row[5])           # 选择某一列加入到data数组中
            if i < total*0.9:
                print(i,row)
                with open(new_train_csv_label,'a',encoding='utf-8') as f_train_csv:
                    csv_writer = csv.writer(f_train_csv)
                    csv_writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5]])
            else:
                print(i,row)
                with open(new_test_csv_label,'a',encoding='utf-8') as f_train_csv:
                    csv_writer = csv.writer(f_train_csv)
                    csv_writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5]])

    # ## ori train/test 写到sub-train02141832-annotations-bbox/sub-test02141832-annotations-bbox
    with open(ori_train_csv_label) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        #header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader:            # 将csv 文件中的数据保存到data中
            print(i,row)
            with open(new_train_csv_label,'a',encoding='utf-8') as f_train_csv:
                csv_writer = csv.writer(f_train_csv)
                csv_writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5]])
    with open(ori_test_csv_label) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        #header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader:            # 将csv 文件中的数据保存到data中
            print(i,row)
            with open(new_test_csv_label,'a',encoding='utf-8') as f_train_csv:
                csv_writer = csv.writer(f_train_csv)
                csv_writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5]])

    # ## new_train_csv_label_1 train/test 写到sub-train02141832-annotations-bbox/sub-test02141832-annotations-bbox
    with open(new_train_csv_label_1) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        #header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader:            # 将csv 文件中的数据保存到data中
            print(i,row)
            with open(new_train_csv_label,'a',encoding='utf-8') as f_train_csv:
                csv_writer = csv.writer(f_train_csv)
                csv_writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5]])

    with open(new_test_csv_label_1) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        #header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader:            # 将csv 文件中的数据保存到data中
            print(i,row)
            with open(new_test_csv_label,'a',encoding='utf-8') as f_train_csv:
                csv_writer = csv.writer(f_train_csv)
                csv_writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5]])


def move_aug2train():
    if 1:
        aug_img_path = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/20220211背景_aug/back_aug_img"
        aug_img_label = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/20220211背景_aug/back_aug_img_label.csv"
        train_aug_img_label = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/20220211背景_aug/sub-train0214-annotations-bbox.csv"
        test_aug_img_label = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/20220211背景_aug/sub-test0214-annotations-bbox.csv"
    else:
        aug_img_path = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/20220214背景2_aug/back_aug_img"
        aug_img_label = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/20220214背景2_aug/back_aug_img_label.csv"
        train_aug_img_label = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/20220214背景2_aug/sub-train02141832-annotations-bbox.csv"
        test_aug_img_label = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/20220214背景2_aug/sub-test02141832-annotations-bbox.csv"

    train_img_path = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/train"
    test_img_path = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/test"

    data = []
    with open(aug_img_label) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        #header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader:            # 将csv 文件中的数据保存到data中
            data.append(row[0])           # 选择某一列加入到data数组中
    print("data:",data)
    all_num = len(data)
    i = 0
    # ## 拷贝tain aug img 到 data
    with open(train_aug_img_label) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        #header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader:            # 将csv 文件中的数据保存到data中
            # data.append(row[0])           # 选择某一列加入到data数组中
            # print(row[0])
            if (row[0] in data) and (row[0]!="ImageID"):
                i+=1
                aug_img_dir = aug_img_path+"/"+row[0]+".png"
                print(all_num,i,aug_img_dir)
                shutil.copyfile(aug_img_dir,train_img_path+"/"+row[0]+".png")
    # ## 拷贝test aug img 到 data
    with open(test_aug_img_label) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        #header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader:            # 将csv 文件中的数据保存到data中
            # data.append(row[0])           # 选择某一列加入到data数组中
            if (row[0] in data) and (row[0]!="ImageID"):
                i+=1
                aug_img_dir = aug_img_path+"/"+row[0]+".png"
                print(all_num,i,aug_img_dir)
                shutil.copyfile(aug_img_dir,test_img_path+"/"+row[0]+".png")

 
def split_train_val_test_csv():
    ori_train_csv = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/sub-train-annotations-bbox.csv"
    ori_test_csv = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/sub-test-annotations-bbox.csv"

    aug1_csv = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/20220211背景_aug/back_aug_img_label.csv"
    aug2_csv = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/20220214背景2_aug/back_aug_img_label.csv"

    new_train_csv = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/trainVal/train0215.csv"
    new_val_csv = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/trainVal/val0215.csv"
    new_test_csv = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/trainVal/test0215.csv"

    all_img_dir = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/trainVal/"

    data = []
    i = 0
    # #读取所有csv
    with open(ori_train_csv) as csvfile:
        csv_reader = csv.reader(csvfile)  
        #header = next(csv_reader)      
        for row in csv_reader:        
            i+=1
            print(i,row)
            data.append(row)
    with open(ori_test_csv) as csvfile:
        csv_reader = csv.reader(csvfile)  
        #header = next(csv_reader)      
        for row in csv_reader:        
            i+=1
            print(i,row)
            data.append(row)
    with open(aug1_csv) as csvfile:
        csv_reader = csv.reader(csvfile)  
        #header = next(csv_reader)      
        for row in csv_reader:        
            i+=1
            print(i,row)
            data.append(row)
    with open(aug2_csv) as csvfile:
        csv_reader = csv.reader(csvfile)  
        #header = next(csv_reader)      
        for row in csv_reader:        
            i+=1
            print(i,row)
            data.append(row)

    all_num = len(data)
    # print("ori data:",data)
    random.shuffle(data)
    print("shuffle data:",data)

    with open(new_train_csv,'a',encoding='utf-8') as f_train_csv:
        csv_writer = csv.writer(f_train_csv)
        csv_writer.writerow(["ImageID","XMin","YMin","XMax","YMax","ClassName"])
    with open(new_val_csv,'a',encoding='utf-8') as f_val_csv:
        csv_writer = csv.writer(f_val_csv)
        csv_writer.writerow(["ImageID","XMin","YMin","XMax","YMax","ClassName"])
    with open(new_test_csv,'a',encoding='utf-8') as f_test_csv:
        csv_writer = csv.writer(f_test_csv)
        csv_writer.writerow(["ImageID","XMin","YMin","XMax","YMax","ClassName"])

    for i,row in enumerate(data):
        print(all_num,i,row)
        if i < 0.8*all_num:
            with open(new_train_csv,'a',encoding='utf-8') as f_train_csv:
                if (row[0]!="ImageID"):
                    csv_writer = csv.writer(f_train_csv)
                    csv_writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5]])
                    shutil.copy(all_img_dir+"/temp/"+row[0]+".png",all_img_dir+"/trainval")
        elif (i>0.8*all_num) and (i<0.9*all_num):
            with open(new_val_csv,'a',encoding='utf-8') as f_train_csv:
                if (row[0]!="ImageID"):
                    csv_writer = csv.writer(f_train_csv)
                    csv_writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5]])
                    shutil.copy(all_img_dir+"/temp/"+row[0]+".png",all_img_dir+"/trainval")
        else:
            with open(new_test_csv,'a',encoding='utf-8') as f_train_csv:
                if (row[0]!="ImageID"):
                    csv_writer = csv.writer(f_train_csv)
                    csv_writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5]]) 
                    shutil.copy(all_img_dir+"/temp/"+row[0]+".png",all_img_dir+"/test")           
    

def move_img():
    train_dir = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/train"    
    test_dir = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/test"    

    aug1_img = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/20220211背景_aug/back_aug_img"
    aug2_img = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/20220214背景2_aug/back_aug_img"

    save_path = "/home/supernode/anno/MobileSSD/MobileNetV3-SSD/data/trainVal/temp"

    img_list = []
    train_list = glob.glob(train_dir+"/*.png")
    test_list = glob.glob(test_dir+"/*.png")
    aug1_list = glob.glob(aug1_img+"/*.png")
    aug2_list = glob.glob(aug2_img+"/*.png")

    img_list = train_list+test_list+aug1_list+aug2_list
    all_num = len(img_list)
    for i,img_dir in enumerate(img_list):
        print(all_num,i,img_dir)
        shutil.copy(img_dir,save_path)


def trans_csv2txt():
    train_csv_path = "/home/whf/Temp/11-sweeper/data/sub-train-annotations-bbox.csv"
    test_csv_path = "/home/whf/Temp/11-sweeper/data/sub-test-annotations-bbox.csv"
    train_img_path = "/home/whf/Temp/11-sweeper/data/train"
    test_img_path = "/home/whf/Temp/11-sweeper/data/test"

    trainValTest_save_path = "/home/whf/Temp/11-sweeper/data/txt_label/trainValTest.txt"
    train_txt_save_path = "/home/whf/Temp/11-sweeper/data/txt_label/train.txt"
    val_txt_save_path = "/home/whf/Temp/11-sweeper/data/txt_label/val.txt"
    test_save_txt_path = "/home/whf/Temp/11-sweeper/data/txt_label/test.txt"

    data = []
    i = 0
    # #读取所有csv
    with open(train_csv_path) as csvfile:
        csv_reader = csv.reader(csvfile)  
        #header = next(csv_reader)      
        for row in csv_reader:        
            i+=1
            if row[0]!= "ImageID":
                train_img_name = row[0]
                train_img_dir = train_img_path+"/"+train_img_name+".png"
                assert os.path.exists(train_img_dir)
                print(i,row)
                row[0] = train_img_dir
                data.append(row)

    with open(test_csv_path) as csvfile:
        csv_reader = csv.reader(csvfile)  
        #header = next(csv_reader)      
        for row in csv_reader:        
            i+=1
            if row[0]!= "ImageID":
                test_img_name = row[0]
                test_img_dir = test_img_path+"/"+test_img_name+".png"
                assert os.path.exists(test_img_dir)
                print(i,row)
                row[0] = test_img_dir
                data.append(row)

    # #################################################
    # ## 合并同一图片的box 到同一键下
    dict = {}
    for i,annons in enumerate(data):
        print("annons:",annons)
        # annon_list = annons.strip().split(" ")
        dict.setdefault(annons[0],[]).append(annons)
    # print("annons:",i,annons,annon_list)
    # ## 遍历所有键
    for key in dict.keys():
        img_str = key
        print("key:",key,dict[key],"\n")
        # if len(dict[key])>1:
        #     print("key:",dict[key],"\n")
        #     print("*********************************************")
        boxes = []
        labels = []
        img_id = key
        f1 = open(trainValTest_save_path,"a+")
        f1.write(img_str)
        # 遍历每个键下的所有label，即同一张图的所有框   
        for annon in dict[key]:
            print("annons_split:",annon)
            img = annon[0]
            x1 = annon[1]
            y1 = annon[2]
            x2 = annon[3]
            y2 = annon[4]
            cls_name = annon[5].strip()
            f1.write(" "+x1+" "+y1+" "+x2+" "+y2+" "+cls_name)
        f1.write("\n")
        
        f1.close()

    
    with open(trainValTest_save_path,"r") as f:
        annonations = f.readlines()
    # ## 划分train/val/test
    all_num = len(annonations)
    random.shuffle(annonations)
    for i,row in enumerate(annonations):
        print(all_num,i,row)

        # img = cv2.imread(row[0])
        # h,w,c = img.shape
        # print("shape:",w,h,c)
        # cls = row[5]
        # x1 = int(float(row[1])*w)
        # y1 = int(float(row[2])*h)
        # x2 = int(float(row[3])*w)
        # y2 = int(float(row[4])*h)
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, 1)
        # cv2.imshow("test",img)
        # cv2.waitKey(500)

        if i<0.8*all_num:
            with open(train_txt_save_path,"a+") as f:
                f.write(row)
        elif i<0.9*all_num:
            with open(val_txt_save_path,"a+") as f:
                f.write(row)
        else:
            with open(test_save_txt_path,"a+") as f:
                f.write(row)
    
    # ##############################################################################
    # all_num = len(data)
    # random.shuffle(data)
    # for i,row in enumerate(data):
    #     print(all_num,i,row)

    #     # img = cv2.imread(row[0])
    #     # h,w,c = img.shape
    #     # print("shape:",w,h,c)
    #     # cls = row[5]
    #     # x1 = int(float(row[1])*w)
    #     # y1 = int(float(row[2])*h)
    #     # x2 = int(float(row[3])*w)
    #     # y2 = int(float(row[4])*h)
    #     # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, 1)
    #     # cv2.imshow("test",img)
    #     # cv2.waitKey(500)

    #     if i<0.8*all_num:
    #         with open(train_txt_save_path,"a+") as f:
    #             f.write(row[0]+" "+row[1]+" "+row[2]+" "+row[3]+" "+row[4]+" "+row[5]+"\n")
    #     elif i<0.9*all_num:
    #         with open(val_txt_save_path,"a+") as f:
    #             f.write(row[0]+" "+row[1]+" "+row[2]+" "+row[3]+" "+row[4]+" "+row[5]+"\n")
    #     else:
    #         with open(test_save_txt_path,"a+") as f:
    #             f.write(row[0]+" "+row[1]+" "+row[2]+" "+row[3]+" "+row[4]+" "+row[5]+"\n")


def gen_ar_new_data_txt():
    new_data_img_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/20220218AR-3/rgb"
    new_data_label_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/20220218AR-3/bbox2d"

    label_txt = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0312/20220311AR-1-2/20220311AR-1-2.txt"


    label_list = glob.glob(new_data_label_path+"/*.txt")
    label_num = len(label_list)
    for i,label_dir in enumerate(label_list):
        label_name = os.path.split(label_dir)[-1].split(".")[0]
        print(i,label_dir,label_name)

        img_dir = new_data_img_path+"/"+label_name+".png"
        img = cv2.imread(img_dir)
        h,w,c = img.shape

        with open(label_dir,"r") as f:
            annonations = f.readlines()
        print(annonations)
        cls_id = None
        for annon in annonations:
            annon = annon.strip()
            if len(annon) !=0:
                annon_list = annon.split(",")
                print("anno_list:",annon_list)
                if annon_list[0] == "animal_poop":
                    cls_id = "excrement"
                if annon_list[0] == "cup":
                    cls_id = "papercup"
                if annon_list[0] == "shoes":
                    cls_id = "slipper"
                if annon_list[0] == "earphones":
                    cls_id = "dataline"

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

def gen_ar_train_val():
    txt_label_1 = "/home/whf/Temp/11-sweeper/anno_data/AR-data/20220218AR-1/img_label.txt"
    txt_label_2 = "/home/whf/Temp/11-sweeper/anno_data/AR-data/20220218AR-2/img_label.txt"
    txt_label_3 = "/home/whf/Temp/11-sweeper/anno_data/AR-data/20220218AR-3/img_label.txt"

    train_txt_save_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/trainval/vr_1-3_all/vr_train0218.txt"
    val_txt_save_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/trainval/vr_1-3_all/vr_val0218.txt"


    # txt_label_1 = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/vr_cls3/vr_cls3.txt"
    # train_txt_save_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/vr_cls3/vr_cls3_train.txt"
    # val_txt_save_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/vr_cls3/vr_cls3_val.txt"

    with open(txt_label_1,"r") as f:
        annotations1 = f.readlines()
    with open(txt_label_2,"r") as f:
        annotations2 = f.readlines()
    with open(txt_label_3,"r") as f:
        annotations3 = f.readlines()

    annons_list = annotations1+annotations2+annotations3

    # annons_list = annotations1

    all_num = len(annons_list)
    random.shuffle(annons_list)
    for i,annons in enumerate(annons_list):
        if i<0.9*all_num:
            with open(train_txt_save_path,"a+") as f:
                f.write(annons)
        else:
            with open(val_txt_save_path,"a+") as f:
                f.write(annons)

def gen_train_val_test():
    ori_train_txt = "/home/whf/Temp/11-sweeper/data/txt_label/train.txt"
    ori_val_txt = "/home/whf/Temp/11-sweeper/data/txt_label/val.txt"

    vr_train_txt = "/home/whf/Temp/11-sweeper/anno_data/AR-data/trainval/vr_1-3/vr_train0218.txt"
    vr_val_txt = "/home/whf/Temp/11-sweeper/anno_data/AR-data/trainval/vr_1-3/vr_val0218.txt"

    new_train_txt = "/home/whf/Temp/11-sweeper/anno_data/AR-data/trainval/ori_vr_4cls_train0218.txt"
    new_val_txt = "/home/whf/Temp/11-sweeper/anno_data/AR-data/trainval/ori_vr_4cls_val0218.txt"


    # ori_train_txt = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori/ori_4cls_train.txt"
    # ori_val_txt = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori/ori_4cls_val.txt"
    # new_train_txt = "/home/whf/Temp/11-sweeper/anno_data/AR-data/trainval/ori_vr_4cls/ori_vr_4cls_train0219.txt"
    # new_val_txt = "/home/whf/Temp/11-sweeper/anno_data/AR-data/trainval/ori_vr_4cls/ori_vr_4cls_val0219.txt"


    cls4_name = ["excrement","papercup","slipper","dataline"]
    # ## 4个类别ori+vr存入新train txt
    with open(ori_train_txt,"r") as f:
        annonations1 = f.readlines()
    for annon in annonations1:
        annon_split = annon.strip().split(" ")
        if annon_split[-1] in cls4_name:
            with open(new_train_txt,"a+") as f:
                f.write(annon)
    with open(vr_train_txt,"r") as f:
        annonations2 = f.readlines()
    for annon in annonations2:
        annon_split = annon.strip().split(" ")
        if annon_split[-1] in cls4_name:
            with open(new_train_txt,"a+") as f:
                f.write(annon)
    
    # ## 4个类别ori+vr存入新val txt
    with open(ori_val_txt,"r") as f:
        annonations1 = f.readlines()
    for annon in annonations1:
        annon_split = annon.strip().split(" ")
        if annon_split[-1] in cls4_name:
            with open(new_val_txt,"a+") as f:
                f.write(annon)
    with open(vr_val_txt,"r") as f:
        annonations2 = f.readlines()
    for annon in annonations2:
        annon_split = annon.strip().split(" ")
        if annon_split[-1] in cls4_name:
            with open(new_val_txt,"a+") as f:
                f.write(annon)

def cls_img_split():
    # # ## 1\真实图分类存储，并提取其中4个类别
    # # train_txt_path = "/home/whf/Temp/11-sweeper/data/txt_label"
    # # tval_txt_path = "/home/whf/Temp/11-sweeper/data/txt_label"
    # # test_txt_path = "/home/whf/Temp/11-sweeper/data/txt_label"

    # save_img_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori/img"

    # ori_4cls_txt = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori/ori_4_cls.txt"

    # txt_label_list = [
    #     "/home/whf/Temp/11-sweeper/data/txt_label/train.txt",
    #     "/home/whf/Temp/11-sweeper/data/txt_label/val.txt",
    #     "/home/whf/Temp/11-sweeper/data/txt_label/test.txt",
    # ]


    # save_img_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori/imgcrop_train"

    # ori_4cls_txt = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori/ori_4cls_train.txt"

    # txt_label_list = [
    #     "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori/ori_4cls_train.txt",
    #     "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori/ori_4cls_val.txt",
    #     "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori/ori_4cls_test.txt",
    # ]


    # # save_img_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/ori_cls4_split"
    # # txt_label_list = [
    # #     "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/ori_cls4_train.txt",
    # # ]


    # cls4_name = ["excrement","papercup","slipper","dataline"]
    # # cls4_name = ["papercup","slipper","powerstrip"]
    # for txt_label in txt_label_list:
    #     with open(txt_label,"r") as f:
    #         annonations = f.readlines()
    #     for annon in annonations:
    #         annon_lsit = annon.strip().split(" ")
    #         print("annon_lsit:",annon_lsit)
    #         img_dir = annon_lsit[0]
    #         cls_name = annon_lsit[-1]
    #         cls_save_path = save_img_path+"/"+cls_name
    #         if not os.path.exists(cls_save_path):
    #             os.makedirs(cls_save_path)
    #         shutil.copy(img_dir,cls_save_path)

    #         # if cls_name in cls4_name:
    #         #     with open(ori_4cls_txt,"a+") as f:
    #         #         f.write(annon)


    # ## 2\4个雷被数据划分train/val/test
    # ori_4cls_txt = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori/ori_4_cls.txt"
    # ori_4_cls_train = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori/ori_4cls_train.txt"
    # ori_4_cls_val = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori/ori_4cls_val.txt"
    # ori_4_cls_test = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori/ori_4cls_test.txt"
    # test_img_save_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori/test_4cls_img" 

    ## 3个雷被数据划分train/val/test
    ori_4cls_txt = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/ori_cls3/ori_cls3.txt"
    ori_4_cls_train = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/ori_cls3/ori_cls3_train.txt"
    ori_4_cls_val = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/ori_cls3/ori_cls3_val.txt"
    ori_4_cls_test = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/ori_cls3/ori_cls3_test.txt"
    test_img_save_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori/test_4cls_img" 

    with open(ori_4cls_txt,"r") as f:
        annonations = f.readlines()
    all_num = len(annonations)
    random.shuffle(annonations)
    for i,row in enumerate(annonations):
        print(all_num,i,row)
        img_dir = row.strip().split( )[0]
        if i<0.8*all_num:
            with open(ori_4_cls_train,"a+") as f:
                f.write(row)
        elif i<0.9*all_num:
            with open(ori_4_cls_val,"a+") as f:
                f.write(row)
        else:
            with open(ori_4_cls_test,"a+") as f:
                f.write(row)
            # shutil.copy(img_dir,test_img_save_path)



def gen_num_cls():
    img_label_dir = "/home/whf/Temp/11-sweeper/data/txt_label/trainValTest.txt"

    cls4_label_save_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/ori_cls4_all.txt"

    cls4_name = ["excrement","papercup","slipper","dataline"]
    with open(img_label_dir,"r") as f:
        annonations = f.readlines()
    n = 0
    for annons in annonations:
        annons_split = annons.strip().split(" ")
        print("annons_split:",annons_split,"\n",annons_split[1:],len(annons_split[1:])/5,"\n")
        # if int(len(annons_split[1:])/5)>1:
        #     n+=1
        cls_num = 0
        box_num = int(len(annons_split[1:])/5)
        for i in range(box_num):
            print("cls:",n,annons_split[i*5+5])
            img_dir = annons_split[0].strip()
            cls_name = annons_split[i*5+5].strip()
            x1 = annons_split[i*5+1].strip()
            y1 = annons_split[i*5+2].strip()
            x2 = annons_split[i*5+3].strip()
            y2 = annons_split[i*5+4].strip()

            if cls_num==0:
                if cls_name in cls4_name:
                    cls_num+=1
                    with open(cls4_label_save_path,"a+") as f:
                        f.write(img_dir+" "+x1+" "+y1+" "+x2+" "+y2+" "+cls_name)
            if cls_num>=1:
                if cls_name in cls4_name:
                    cls_num+=1
                    with open(cls4_label_save_path,"a+") as f:
                        f.write(" "+x1+" "+y1+" "+x2+" "+y2+" "+cls_name)
            if (i == (box_num-1)) and (cls_num!=0):
                with open(cls4_label_save_path,"a+") as f:
                    f.write("\n")

def cls_label_show():
    cls4_label_save_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/ori_cls4_all.txt"
    img_save_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/oti_cls4_all"

    with open(cls4_label_save_path,"r") as f:
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
        cv2.imwrite(img_save_path+"/"+img_name,img)
        # cv2.imshow("test",img)
        # cv2.waitKey(500)


def gen_ori_train_val_test():
    ori_cls4_labe_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/ori_cls4_all.txt"

    ori_cls4_train = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/ori_cls4_train.txt"
    ori_cls4_val = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/ori_cls4_val.txt"
    ori_cls4_test = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/ori_cls4_test.txt"

    test_img_save_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/cls4_test_img"

    with open(ori_cls4_labe_path,"r") as f:
        annonations = f.readlines()
    random.shuffle(annonations)
    all_num = len(annonations)
    for i,row in enumerate(annonations):
        img_dir = row.strip().split( )[0]
        if i<0.8*all_num:
            with open(ori_cls4_train,"a+") as f:
                f.write(row)
        elif i<0.9*all_num:
            with open(ori_cls4_val,"a+") as f:
                f.write(row)
        else:
            with open(ori_cls4_test,"a+") as f:
                f.write(row)
            shutil.copy(img_dir,test_img_save_path)


def split_data_ratio():
    # # # 1\按比例划分真实数据
    # # ori_train_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/ori_cls4_train.txt"
    # # ori_val_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/ori_cls4_val.txt"

    # # train_ratio_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/分比例实验/ori/ori_5000/cls4_6s5_train.txt"
    # # val_ratio_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/分比例实验/ori/ori_5000/cls4_6s5_val.txt"

    # # ori_train_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/ori_cls4_train.txt"
    # # ori_val_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/ori_cls4_val.txt"

    # # train_ratio_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/分比例实验/ori/ori_5000/cls4_6s5_train.txt"
    # # val_ratio_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/分比例实验/ori/ori_5000/cls4_6s5_val.txt"


    # # ## 0223 3cls
    # # ori_train_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/ori_cls3/ori_cls3_train.txt"
    # # ori_val_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/ori_cls3/ori_cls3_val.txt"

    # # train_ratio_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/train_ratio-0223/ori1000_vr1000/ori_cls3_3s1_train.txt"
    # # val_ratio_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/train_ratio-0223/ori1000_vr1000/ori_cls3_3s1_val.txt"


    # ori_label_list = [
    #     "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/ori_cls4_train.txt",
    #     "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/ori_cls4_val.txt"
    # ]

    # with open(ori_train_label,"r") as f:
    #     annonations1 = f.readlines()
    # with open(ori_val_label,"r") as f:
    #     annonations2 = f.readlines()
    # annonations = annonations1+annonations2
    # random.shuffle(annonations)
    # all_num = len(annonations)
    # ratio = 1/3
    # print("num:",all_num,0.9*ratio*all_num)
    # for i , annon in enumerate(annonations):
    #     if i < ratio*all_num:
    #         if i<0.9*ratio*all_num:
    #             with open(train_ratio_label,"a+") as f:
    #                 f.write(annon)
    #         else:
    #             with open(val_ratio_label,"a+") as f:
    #                 f.write(annon)


    # ## 2\按比例划分虚拟数据
    # # ori_train_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/trainval/vr_1-3_all/vr_train0218.txt"
    # # ori_val_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/trainval/vr_1-3_all/vr_val0218.txt"

    # # train_ratio_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/分比例实验/vr/vr_1000/vr_cls4_4s1_train.txt"
    # # val_ratio_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/分比例实验/vr/vr_1000/vr_cls4_4s1_val.txt"

    # # ## 0223
    # ori_train_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/vr_cls3/vr_cls3_train.txt"
    # ori_val_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/vr_cls3/vr_cls3_val.txt"

    # train_ratio_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/train_ratio-0223/ori1000_vr3000/vr_cls3_3s3_train.txt"
    # val_ratio_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/train_ratio-0223/ori1000_vr3000/vr_cls3_3s3_val.txt"

    # ori_label_list = [
    #     "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/ori_cls4_train.txt",
    #     "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/ori_cls4_val.txt"
    # ]

    # with open(ori_train_label,"r") as f:
    #     annonations1 = f.readlines()
    # with open(ori_val_label,"r") as f:
    #     annonations2 = f.readlines()
    # annonations = annonations1+annonations2
    # random.shuffle(annonations)
    # all_num = len(annonations)
    # ratio = 3/3
    # print("num:",all_num,0.9*ratio*all_num)
    # for i , annon in enumerate(annonations):
    #     if i < ratio*all_num:
    #         if i<0.9*ratio*all_num:
    #             with open(train_ratio_label,"a+") as f:
    #                 f.write(annon)
    #         else:
    #             with open(val_ratio_label,"a+") as f:
    #                 f.write(annon)


    # ## 3\合并不同比例真是虚拟+真是
    # ori_ratio_train_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/分比例实验/ori/ori_2000/ori_cls4_6s2_train.txt"
    # ori_ratio_val_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/分比例实验/ori/ori_2000/ori_cls4_6s2_val.txt"

    # vr_ratio_train_label = "//home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/分比例实验/vr/vr_4000/vr_cls4_4s4_train.txt"
    # vr_ratio_val_label = "//home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/分比例实验/vr/vr_4000/vr_cls4_4s4_val.txt"

    # vr_ratio_train_label = "//home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/分比例实验/vr/vr_4000/vr_train0218.txt"
    # vr_ratio_val_label = "//home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/分比例实验/vr/vr_4000/vr_val0218.txt"

    # ori_vr_cat_train_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/分比例实验/test_ori_vr/test_ori2000_vr4000/ori2000_vr4000_train.txt"
    # ori_vr_cat_val_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/ori0221/4_cls_label/分比例实验/test_ori_vr/test_ori2000_vr4000/ori2000_vr4000_val.txt"


    ori_ratio_train_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/train_ratio-0223/ori2000_vr1000/ori_cls3_3s2_train.txt"
    ori_ratio_val_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/train_ratio-0223/ori2000_vr1000/ori_cls3_3s2_val.txt"

    vr_ratio_train_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/train_ratio-0223/ori2000_vr1000/vr_cls3_3s1_train.txt"
    vr_ratio_val_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/train_ratio-0223/ori2000_vr1000/vr_cls3_3s1_val.txt"

    ori_vr_cat_train_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/train_ratio-0223/ori2000_vr1000/ori2000_vr1000_train.txt"
    ori_vr_cat_val_label = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/train_ratio-0223/ori2000_vr1000/ori2000_vr1000_val.txt"


    with open(ori_ratio_train_label,"r") as f:
        train_annotations1 = f.readlines()
    with open(vr_ratio_train_label,"r") as f:
        train_annotationa2 = f.readlines()
    train_annotations = train_annotations1+train_annotationa2
    for annons in train_annotations:
        with open(ori_vr_cat_train_label,"a+") as f:
            f.write(annons)

    

    with open(ori_ratio_val_label,"r") as f:
        val_annotations1 = f.readlines()
    with open(vr_ratio_val_label,"r") as f:
        val_annotationa2 = f.readlines()
    val_annotations = val_annotations1+val_annotationa2
    for annons in val_annotations:
        with open(ori_vr_cat_val_label,"a+") as f:
            f.write(annons)


def label_show():
    label_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/ori_cls3/ori_cls3.txt"
    show_img_save_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test_02221744/ori_cls3/ori_cls3_img-show"

    with open(label_path,"r") as f:
        annonations = f.readlines()
    for annon in annonations:
        annons_split = annon.strip().split(" ")
        img_name = os.path.split(annons_split[0])[-1]
        img = cv2.imread(annons_split[0])
        h,w,c = img.shape
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
        cv2.imwrite(show_img_save_path+"/"+img_name,img)

        # cv2.imshow("test",img)
        # # cv2.waitKey(500)
        # keyValue = cv2.waitKey(500)   
        # print("0xFF == ord(' '):",0xFF == ord(' '),ord(' '))         
        # if keyValue & 0xFF == ord(' '):        
        #     cv2.waitKey(0)
        # elif keyValue & 0xFF == ord('q'):      
        #     exit(0)

def ar_label_show():
    label_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/20220218AR-3/bbox2d"
    img_path = "/home/whf/Temp/11-sweeper/anno_data/AR-data/20220218AR-3/rgb"

    img_path = "/run/user/1000/gvfs/smb-share:server=192.168.3.172,share=data/扫地机采集数据/20220303AR-1/rgb"
    label_path = "/run/user/1000/gvfs/smb-share:server=192.168.3.172,share=data/扫地机采集数据/20220303AR-1//bbox2d"

    label_list = glob.glob(label_path+"/*.txt")
    label_num = len(label_list)
    for i,label_dir in enumerate(label_list):
        label_name = os.path.split(label_dir)[-1].split(".")[0]
        print(i,label_dir,label_name)

        img = cv2.imread(img_path+"/"+label_name+".png")

        with open(label_dir,"r") as f:
            annonations = f.readlines()
        print(annonations)
        for annon in annonations:
            annon = annon.strip()
            if len(annon) !=0:
                annon_list = annon.split(",")
                print("anno_list:",annon_list)
                cls_id = annon_list[0]
                x1 = int(annon_list[1])
                y1 = int(annon_list[2])
                x2 = int(annon_list[5])
                y2 = int(annon_list[6])
                
                # x1 = int(annon_list[1])
                # y1 = int(annon_list[3])
                # x2 = int(annon_list[5])
                # y2 = int(annon_list[7])

                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1,1)
        cv2.imshow("test",img)
        # cv2.waitKey(500)
        keyValue = cv2.waitKey(100)   
        print("0xFF == ord(' '):",0xFF == ord(' '),ord(' '))         
        if keyValue & 0xFF == ord(' '):        
            cv2.waitKey(0)
        elif keyValue & 0xFF == ord('q'):      
            exit(0)

def gen_vr_txt_label():
    print("..........")
    # # ## 1\ ar 原始生成label转[img,xmin,ymin,xmax,ymax,cls,...]
    # new_data_img_path = "/home/supernode/anno/AR-data/20220304AR-4/rgb"
    # new_data_label_path = "/home/supernode/anno/AR-data/20220304AR-4/bbox2d"
    # label_txt = "/home/supernode/anno/projects/SweeprData/AR-data/trainval/test0304/vr_cls3/20220303AR-4.txt"

    # label_list = glob.glob(new_data_label_path+"/*.txt")
    # label_num = len(label_list)

    # for i,label_dir in enumerate(label_list):
    #     label_name = os.path.split(label_dir)[-1].split(".")[0]
    #     print(i,label_dir,label_name)

    #     img_dir = new_data_img_path+"/"+label_name+".png"
    #     img = cv2.imread(img_dir)
    #     h,w,c = img.shape

    #     with open(label_dir,"r") as f:
    #         annonations = f.readlines()
    #     print(annonations)
    #     cls_id = None
    #     for annon in annonations:
    #         annon = annon.strip()
    #         if len(annon) !=0:
    #             annon_list = annon.split(",")
    #             print("anno_list:",annon_list)

    #             if annon_list[0] == "powerstrip":
    #                     cls_id = "powerstrip"
    #             if annon_list[0] == "cup":
    #                 cls_id = "papercup"
    #             if annon_list[0] == "slippers":
    #                 cls_id = "slipper"

    #             x1 = int(annon_list[1])
    #             y1 = int(annon_list[2])
    #             x2 = int(annon_list[5])
    #             y2 = int(annon_list[6])
    #             xmin = round(x1/w,5)
    #             ymin = round(y1/h,5)
    #             xmax = round(x2/w,5)
    #             ymax = round(y2/h,5)
    #             print("label:",img_dir,xmin,ymin,xmax,ymax,cls_id)

    #             with open(label_txt,"a+") as f:
    #                 f.write(img_dir+" "+str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax)+" "+cls_id+"\n")

    
    # # ## 2\ ar 提取本批数据中的部分图片
    # all_label_txt = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test0304/20220303AR-1.txt"
    # new_label_txt = "/home/whf/Temp/11-sweeper/anno_data/AR-data/test0304/20220303AR-1_s3000.txt"

    # with open(all_label_txt,"r") as f:
    #     annonations = f.readlines()
    # random.shuffle(annonations)
    # all_num = len(annonations)
    # for i,annon in enumerate(annonations):
    #     print(i,annon)
    #     if i <=3000:
    #         with open(new_label_txt,"a+") as f:
    #             f.write(annon)



    # # ## 3\ ar 合并各部分生成虚拟数据
    # annotations = []
    # vr_label_list = ["/home/supernode/anno/projects/SweeprData/AR-data/trainval/test0304/vr_cls3/20220303AR-1.txt",
    #                 "/home/supernode/anno/projects/SweeprData/AR-data/trainval/test0304/vr_cls3/20220303AR-2.txt",
    #                 "/home/supernode/anno/projects/SweeprData/AR-data/trainval/test0304/vr_cls3/20220303AR-3.txt",
    #                 "/home/supernode/anno/projects/SweeprData/AR-data/trainval/test0304/vr_cls3/20220303AR-4.txt"
    # ]
    # concat_label_txt = "/home/supernode/anno/projects/SweeprData/AR-data/trainval/test0304/vr_cls3/ar_concat0304_cls3.txt"
    # for vr_label in vr_label_list:
    #     with open(vr_label,"r") as f:
    #         annons = f.readlines()
    #     annotations +=annons
    # random.shuffle(annotations)
    # all_num = len(annotations)
    # for i,annon in enumerate(annotations):
    #     print(i,annon)
    #     with open(concat_label_txt,"a+") as f:
    #         f.write(annon)
    

    # # ## 4\ ar 划分虚拟数据到不同比例
    # concat_label_txt = "/home/supernode/anno/projects/SweeprData/AR-data/trainval/test0304/vr_cls3/ar_concat0304_cls3.txt"
    # concat_label_split = "/home/supernode/anno/projects/SweeprData/AR-data/trainval/test0304/vr_cls3/ratio/ar_concat0304_cls3_10s7.txt"
    # with open(concat_label_txt,"r") as f:
    #     annonations = f.readlines()
    # random.shuffle(annonations)
    # all_num = len(annonations)
    # for i,annon in enumerate(annonations):
    #     print(i,annon)
    #     if i<= 7*all_num/10:
    #         with open(concat_label_split,"a+") as f:
    #             f.write(annon)


    # ## 4\ori300真实+部分虚拟
    ori300_label_train_txt = "/home/supernode/anno/projects/SweeprData/AR-data/trainval/test0304/ori_cls3/ori300/ori_cls3_300_train.txt"
    ori300_label_val_txt = "/home/supernode/anno/projects/SweeprData/AR-data/trainval/test0304/ori_cls3/ori300/ori_cls3_300_val.txt"

    ar_ratio_label = "/home/supernode/anno/projects/SweeprData/AR-data/trainval/test0304/vr_cls3/ratio/ar_concat0304_cls3_10s7.txt"

    save_ori_ar_ratio_train = "/home/supernode/anno/projects/SweeprData/AR-data/trainval/test0304/train_ratio/ori300_ar10s7_train.txt"
    save_ori_ar_ratio_val = "/home/supernode/anno/projects/SweeprData/AR-data/trainval/test0304/train_ratio/ori300_ar10s7_val.txt"

    with open(ar_ratio_label,"r") as f:
        annonations = f.readlines()
    random.shuffle(annonations)
    all_num = len(annonations)
    # ## 划分ar train/val
    for i,annon in enumerate(annonations):
        print(i,annon)
        if i <= 0.9*all_num:
            with open(save_ori_ar_ratio_train,"a+") as f:
                f.write(annon)
        else:
            with open(save_ori_ar_ratio_val,"a+") as f:
                f.write(annon)
    # #concat   (ar train)+(ori train)
    with open(ori300_label_train_txt,"r") as f:
        ori_train_annonations = f.readlines()
    for i,annon in enumerate(ori_train_annonations):
        print("ori train:",i,annon)
        with open(save_ori_ar_ratio_train,"a+") as f:
            f.write(annon)
    # #concat   (ar val)+(ori val)
    with open(ori300_label_val_txt,"r") as f:
        ori_val_annonations = f.readlines()
    for i,annon in enumerate(ori_val_annonations):
        print("ori val:",i,annon)
        with open(save_ori_ar_ratio_val,"a+") as f:
            f.write(annon)


def train_label_show():
    label_path = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0317/test0317.txt"

    with open(label_path,"r") as f:
        annonations = f.readlines()
    for i,annon in enumerate(annonations):
        print(i,annon)
        annon_split = annon.split(" ")
        img_dir = annon_split[0]
        img = cv2.imread(img_dir)
        h,w,c = img.shape

        x1 = int(float(annon_split[1])*w)
        y1 = int(float(annon_split[2])*h)
        x2 = int(float(annon_split[3])*w)
        y2 = int(float(annon_split[4])*h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, 1)
        cv2.imshow("test",img)
        cv2.waitKey(100)


def gen_ar_new_data_label():
    folder_list = [
    "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/AR-data/20220316AR-3",
    # "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/AR-data/20220311AR-2"
    ]


    label_txt = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/AR-data/20220316AR-3/20220316AR-3.txt"
    error_label_txt = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/AR-data/20220316AR-3/20220316AR-3-error.txt"

    for folder_dir in folder_list:
        print("folder_dir:",folder_dir)
        sub_folder_path = os.listdir(folder_dir)
        error_num = 0
        for sub_folder in sub_folder_path:
            sub_folder_dir = folder_dir+"/"+sub_folder
            print("sub_folder:",sub_folder,sub_folder_dir)

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
                            if annon_list[0] == "cup":
                                cls_id = "papercup"
                            if annon_list[0] == "shoes":
                                cls_id = "slipper"
                            if annon_list[0] == "slippers":
                                cls_id = "slipper"
                            if annon_list[0] == "powerstrip":
                                cls_id = "powerstrip"
                            # if annon_list[0] == "earphones":
                            #     cls_id = "dataline"

                            # cls_list = ["chair","animal_poop"]
                            # if annon_list[0] not in cls_list:
                            #     break
                            if cls_id == None:
                                print("mo cls_id ........")
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


def concat_label():
    label_list = [
        "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/AR-data/20220315AR/20220315AR.txt",
        "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/AR-data/20220316AR-1/20220316AR-1.txt",
        "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/AR-data/20220316AR-2/20220316AR-2.txt",
        "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/AR-data/20220316AR-3/20220316AR-3.txt",
    ]

    new_label_dir = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0317/test0317.txt"

    for label_dir in label_list:
        print(label_dir)
        with open(label_dir,"r") as f:
            annonations = f.readlines()
        lines_num = len(annonations)
        for i,row in enumerate(annonations):
            print(lines_num,i,row)
            with open(new_label_dir,"a+") as f:
                f.write(row)


def select_ratio():
    new_label_dir = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0317/test0317.txt"

    ratio_txt_train_label = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0317/ar_ratio/test0317_7s1_train.txt"
    ratio_txt_val_label = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0317/ar_ratio/test0317_7s1_val.txt"
    select_ratio = 1/7

    # ## 合并同一img对应label到同一键下
    # ## 每个img地址存到list
    dict = {}
    img_list = []
    with open(new_label_dir,"r") as f:
        annonations = f.readlines()
    lines_num = len(annonations)
    for i,row in enumerate(annonations):
        row_split = row.strip().split(" ")
        # print(lines_num,i,row,"\n",row_split)
        # ## 相同img对应的标签合并 如[img1 box1,img1 box2]合并为{img1:["img1 box1","img1 box2"]}
        dict.setdefault(row_split[0],[]).append(row)
        img_list.append(row_split[0])
    img_num = len(img_list)

    # ##合并img_list中相同的img
    img_list_new = list(set(img_list))
    print(img_num,lines_num,len(img_list_new))

    # ## 取一定比例张图片
    img_ratio_lsit = []
    ratio = select_ratio        
    random.shuffle(img_list_new)
    img_list_new_num = len(img_list_new)
    for i,img in enumerate(img_list_new):
        if i <= img_list_new_num*ratio:
            img_ratio_lsit.append(img)
    
    # ##取出选定比例中img对应的所有label
    img_ratio_num = len(img_ratio_lsit)
    for i,ratio_img in enumerate(img_ratio_lsit):
        print("ratio_img:",img_ratio_num,i,ratio_img) 
        if i < img_ratio_num*0.9:
            # 遍历每个键下的所有label，即同一张图的所有框   
            for annon in dict[ratio_img]:
                # anno_split = annon.strip().split(" ")
                with open(ratio_txt_train_label,"a+") as f:
                    f.write(annon)
        else:
            for annon in dict[ratio_img]:
                # anno_split = annon.strip().split(" ")
                with open(ratio_txt_val_label,"a+") as f:
                    f.write(annon)



    # data = []
    # # ## 遍历所有键
    # for key in dict.keys():
    #     # print(key,dict[key],"\n")
    #     boxes = []
    #     labels = []
    #     img_id = key
    #     # 遍历每个键下的所有label，即同一张图的所有框   
    #     for annon in dict[key]:
    #         anno_split = annon.strip().split(" ")
    #         # print("annon:",annon,anno_split)
    #         cls_name = anno_split[-1]
    #         cls_id = class_dict[cls_name]
    #         # print("cls_id:",cls_name,class_dict[cls_name])
    #         boxes.append([float(i) for i in anno_split[1:-1]])
    #         labels.append(int(cls_id))


    # for i in range(10):
    #     img_list.append(1)
    # img_list_new = list(set(img_list))
    # print(img_list_new)


def concat_ori_ar():
    before_label_train = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0304/train_ratio_0317_oneBoxRow/ori300_ar10s1_train.txt"
    before_label_val = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0304/train_ratio_0317_oneBoxRow/ori300_ar10s1_val.txt"

    label_0317_train = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0317/ar_ratio/test0317_7s1_train.txt"
    label_0317_val = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0317/ar_ratio/test0317_7s1_val.txt"

    new_label_train = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0317/ori_ar_ratio/ori300_ar2000_train.txt"
    new_label_val = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0317/ori_ar_ratio/ori300_ar2000_val.txt"

    with open(before_label_train,"r") as f:
        train_annonations1 = f.readlines()
    with open(label_0317_train,"r") as f:
        train_annonations2 = f.readlines()
    annonations = train_annonations1+train_annonations2
    print("annonations:",len(annonations))
    for annon in annonations:
        with open(new_label_train,"a+") as f:
            f.write(annon)
    
    with open(before_label_val,"r") as f:
        val_annonations1 = f.readlines()
    with open(label_0317_val,"r") as f:
        val_annonations2 = f.readlines()
    val_annonations = val_annonations1+val_annonations2
    print("val_annonations:",len(val_annonations))
    for val_annon in val_annonations:
        with open(new_label_val,"a+") as f:
            f.write(val_annon)


def modify_before_label():
    label_path = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0304/train_ratio"

    new_label_path = "/media/supernode/d8ae89f8-e157-4adf-9925-e77d142006fb/qlin/AnnoData/AnnoProjects-All/1-Sweeper/anno/projects/SweeprData/AR-data/trainval/test0304/train_ratio_0317_oneBoxRow/"


    for txt_label in os.listdir(label_path):
        txt_label_dir = label_path+"/"+txt_label
        new_label_dir = new_label_path+"/"+txt_label
        txt_name = os.path.split(txt_label)[-1].split(".")[0]
        print(txt_label_dir,txt_name)
        with open(txt_label_dir,"r") as f:
            annonations = f.readlines()
        for anno in annonations:
            annons_split = anno.strip().split(" ")

            box_num = int(len(annons_split[1:])/5)
            for i in range(box_num):
                print("cls:",annons_split[i*5+5])
                img_dir = annons_split[0].strip()
                cls_name = annons_split[i*5+5].strip()
                x1 = annons_split[i*5+1].strip()
                y1 = annons_split[i*5+2].strip()
                x2 = annons_split[i*5+3].strip()
                y2 = annons_split[i*5+4].strip()
                with open(new_label_dir,"a+") as f:
                    f.write(img_dir+" "+x1+" "+y1+" "+x2+" "+y2+" "+cls_name+"\n")



if __name__=="__main__":
    print("start .....")
    # # ## 1、多个目标贴到同一张图上
    # cutMix()


    # # ## 2、mosaic 数据增强
    # Mosaic()


    # # ## 3.1、cutMix+background
    # cut_background()

    # # ## 3.2、合并csv(ori+aug)
    # merge_csv()

    # # ## 移动增广图片分别到test/train
    # move_aug2train()


    # # ## 生程训练txt
    # gen_train_val()


    # # ## move img
    # move_img()

    # # # ## 生成train、val、test csv (ori+全部增广)
    # split_train_val_test_csv()

    # #######################################################################

    # # ## 1.1生成原始数据集csv对应的txt
    # trans_csv2txt()

    # # ## 1.1.1 所有数据label(格式：dir,box1,cls1,box2,cls2.....) 提前出4个类
    # gen_num_cls()

    # # ## 1.1.2 真实数据show出boxs
    # cls_label_show()

    # # ## 1.1.3 ori labe[dir,bx1,cls1,box2,cls2,....] 划分train/val/test
    # gen_ori_train_val_test()

    # # ## 1.2新增AR数据生成txt
    # # 注意这种写法只针对于一个label file只存有一个bbox
    # gen_ar_new_data_txt()

    # # ## 1.3新增数据后train/val划分
    # gen_ar_train_val()

    # # ## 1.4合并生成train/val/test
    # gen_train_val_test()


    # # ## 1.5 真实数据各类别统计
    # cls_img_split()


    # # ## 1.6 按比例split数据
    # split_data_ratio()


    # # ## 真实数据label show [img,box1,cls1,box2,cls2,....]
    # label_show()

    # # ## ar数据标注框显示
    # ar_label_show()

    # # ## 训练格式label显示[img,x1,y1,x2,y2,cls.....] (小数形式)
    # train_label_show()

    # # # ## vr数据txt label生成
    # gen_vr_txt_label()



# ##################################################################

    # # ## 2.1新增AR数据label生成  格式 [img1 box1,img1 box2,img2 box2 ........]
    # gen_ar_new_data_label()

    # # ## 2.2合并新增数据label [img1 box1,img1 box2,img2 box2 ........]
    # concat_label()

    # # ## 2.3.0修改原始训练label
    # modify_before_label()

    # # ## 2.3从concat后测label中随机取一定比例  
    # # ## 注意现在的label每个box写一行 [img1 box1,img1 box2,img2 box2 ........]
    # select_ratio()

    # ## 2.4新选取比例ar数据与ori_label合并
    concat_ori_ar()

