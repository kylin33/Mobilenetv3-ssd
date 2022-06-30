import numpy as np
import pathlib
import random
import math
import cv2
import pandas as pd
import os

# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, image_id):
    # loads 1 image from dataset, returns img, original hw, resized hw
    # image_file = self.root / self.dataset_type / f"{image_id}.png"
    # image_file = str(self.root)+"/trainval/"+image_id+".png"
    image_file = image_id
    img = cv2.imread(str(image_file))
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

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


class OpenImagesDataset:

    def __init__(self, root,
                 transform=None, target_transform=None,
                 dataset_type="train", balance_data=False, augment = False):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()
        self.img_size = 640
        self.mosaic_border = [-self.img_size// 2, -self.img_size // 2]
        self.augment = augment

        self.data, self.class_names, self.class_dict = self._read_data()
        self.balance_data = balance_data
        self.min_image_num = -1
        if self.balance_data:
            self.data = self._balance_data()
        self.ids = [info['image_id'] for info in self.data]

        self.class_stat = None

        n = len(self.data)
        self.n = n
        self.indices = range(n)

    def _getitem(self, index):
        image_info = self.data[index]
        # print("image_info:",image_info)
        image = self._read_image(image_info['image_id'])
        x = image_info['boxes']
        boxes = x.copy()
        boxes[:, 0] *= image.shape[1]
        boxes[:, 1] *= image.shape[0]
        boxes[:, 2] *= image.shape[1]
        boxes[:, 3] *= image.shape[0]
        labels = image_info['labels']
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        # print(image.shape)
        return image_info['image_id'], image, boxes, labels

    # def __getitem__(self, index):
    #     mosaic = 1 and random.random() < 0.5
    #     if mosaic :
    #         image , boxes, labels = load_mosaic(self, index)    
    #     else:
    #         _, image, boxes, labels = self._getitem(index)
    #     return image, boxes, labels

    def __getitem__(self, index):
        # mosaic = 1 and random.random() < 0.5
        # # cutmix = 1 and random.random() < 0.3
        # if mosaic :
        #     image , boxes, labels = self.load_mosaic(index)
        # # elif cutmix:
        # #     _, image , boxes, labels = self.cutMix(index)
        # else:
        #     _, image, boxes, labels = self._getitem(index)
        
        # _, image , boxes, labels = self.cutMix(index) 
        # image , boxes, labels = self.load_mosaic(index)
        _, image, boxes, labels = self._getitem(index)
        return image, boxes, labels

    def get_annotation(self, index):
        """To conform the eval_ssd implementation that is based on the VOC dataset."""
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def get_image(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        if self.transform:
            image, _ = self.transform(image)
        return image

    def _read_data(self):
        # ## 最原始csv格式
        # annotation_file = f"{self.root}/sub-{self.dataset_type}-annotations-bbox.csv"
        # # annotation_file = f"{self.root}/{self.dataset_type}.csv"
        # print("annotation_file:",annotation_file)
        # annotations = pd.read_csv(annotation_file)
        # class_names = ['BACKGROUND'] + sorted(list(annotations['ClassName'].unique()))
        # class_names = ['BACKGROUND'] +['dataline','excrement', 'papercup', 'plasticbag', 'powerstrip', 'slipper', 'socks',  "other"]
        # class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        # data = []
        # for image_id, group in annotations.groupby("ImageID"):
        #     boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
        #     labels = np.array([class_dict[name] for name in group["ClassName"]])
        #     # print(boxes, labels)
        #     data.append({
        #         'image_id': image_id,
        #         'boxes': boxes,
        #         'labels': labels
        #     })


        annotation_file = str(self.root)+"/"+self.dataset_type+".txt"
        class_names = ['BACKGROUND'] +['dataline', 'papercup', 'powerstrip',  'slipper', 'socks', 'chargingBase']
        # class_names = ['BACKGROUND'] +['dataline','excrement', 'papercup', 'slipper']
        # class_names = ['BACKGROUND'] +['excrement','powerstrip','papercup', 'slipper']
        # ## 0317
        # class_names = ['BACKGROUND'] +['powerstrip','papercup', 'slipper']
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}

        # # #************1 、先前格式每行只对应一个box,如一张图两个框会写为两行*******************#
        # ## 读取所有txt_label存入字典，相同img name的label合并到同一键
        # dict = {}
        # data = []
        # with open(annotation_file,"r") as f:
        #     annonations = f.readlines()
        # # random.shuffle(annonations)
        # for i,annons in enumerate(annonations):
        #     annon_list = annons.strip().split(" ")
        #     # ## 相同img对应的标签合并 如[img1 box1,img1 box2]合并为{img1:["img1 box1","img1 box2"]}
        #     dict.setdefault(annon_list[0],[]).append(annons)
        #     print("annons:",i,annons,"*****",annon_list)
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
        #         boxes.append([float(i) for i in anno_split[1:-2]])
        #         labels.append(int(cls_id))
        #     boxes_arrary = np.array(boxes)
        #     labels_arrary = np.array(labels)
        #     data.append({
        #         'image_id': img_id,
        #         'boxes': boxes_arrary,
        #         'labels': labels_arrary
        #     })

        
        # # #***********2、格式一张图多个物体写为一行*****************#
        data = []
        with open(annotation_file,"r") as f:
            annonations = f.readlines()
        data = []
        random.shuffle(annonations)
        for annon in annonations:
            annons_split = annon.strip().split(" ")
            img_id = annons_split[0].strip()
            # print("annon:",annon,anno_split)
            box_num = int(len(annons_split[1:])/5)
            boxes = []
            labels = []
            for i in range(box_num):
                # print("cls:",annons_split[i*5+5])
                img_dir = annons_split[0].strip()
                cls_name = annons_split[i*5+5].strip()
                try:
                    cls_id = class_dict[cls_name]
                except:
                    print(cls_name)
                    exit(0)
                x1 = float(annons_split[i*5+1].strip())
                y1 = float(annons_split[i*5+2].strip())
                x2 = float(annons_split[i*5+3].strip())
                y2 = float(annons_split[i*5+4].strip())
                boxes.append([x1,y1,x2,y2])
                labels.append(int(cls_id))  
            boxes_arrary = np.array(boxes)
            labels_arrary = np.array(labels)
            data.append({
                'image_id': img_id,
                'boxes': boxes_arrary,
                'labels': labels_arrary
            })

        return data, class_names, class_dict

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.class_stat is None:
            self.class_stat = {name: 0 for name in self.class_names[1:]}
            for example in self.data:
                for class_index in example['labels']:
                    class_name = self.class_names[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                   f"Number of Images: {len(self.data)}",
                   f"Minimum Number of Images for a Class: {self.min_image_num}",
                   "Label Distribution:"]
        for class_name, num in self.class_stat.items():
            content.append(f"\t{class_name}: {num}")
        return "\n".join(content)

    def _read_image(self, image_id):
        # image_file = self.root / self.dataset_type / f"{image_id}.png"
        # image_file = str(self.root)+ "/trainval/"+image_id+".png"
        image_file = image_id
        # print("image_file:",image_file)
        if not os.path.isfile(image_file):
            print("image_file:",image_file)
        image = cv2.imread(str(image_file))
        # image = cv2.resize(image, (224,224))
        # print(image.shape)
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _balance_data(self):
        label_image_indexes = [set() for _ in range(len(self.class_names))]
        for i, image in enumerate(self.data):
            for label_id in image['labels']:
                label_image_indexes[label_id].add(i)
        label_stat = [len(s) for s in label_image_indexes]
        self.min_image_num = min(label_stat[1:])
        sample_image_indexes = set()
        for image_indexes in label_image_indexes[1:]:
            image_indexes = np.array(list(image_indexes))
            sub = np.random.permutation(image_indexes)[:self.min_image_num]
            sample_image_indexes.update(sub)
        sample_data = [self.data[i] for i in sample_image_indexes]
        return sample_data
    

    def cutMix(self, index):
        indices = [index] + [self.indices[random.randint(0, self.n - 1)] for _ in range(3)]  # 3 additional image indices
        img_0 = None
        img_crop = None
        crop_w = None
        crop_h = None
        box0 = None
        labels0 = None
        box0_num = None
        image_info_0 = None
        for i, index in enumerate(indices):
            # Load image
            image_info = self.data[index]
            
            img, _, (h, w) = load_image(self, image_info['image_id'])
            # h,w,c = img.shape
            # print("h,w:",h,w )
            if i==0:
                img_0 = img
                x = image_info['boxes']
                box0 = x.copy()
                labels0 = image_info['labels']
                x1_0 = int(round(box0[0][0]*w,2))
                y1_0 = int(round(box0[0][1]*h,2))
                x2_0 = int(round(box0[0][2]*w,2))
                y2_0 = int(round(box0[0][3]*h,2))
                w_0 = x2_0-x1_0
                h_0 = y2_0-y1_0
                img_w0 = w
                img_h0 = h
                image_info_0 = image_info

            if i == 2:
                # Labels
                x = image_info['boxes']
                box = x.copy()
                # print("box info:",box)
                tag = image_info['labels']
                x1 = int(round(box[0][0]*w,2))
                y1 = int(round(box[0][1]*h,2))
                x2 = int(round(box[0][2]*w,2))
                y2 = int(round(box[0][3]*h,2))
                crop_w = x2-x1
                crop_h = y2-y1
                img_crop = img[y1:y2,x1:x2]

                # print("box[][]:",crop_w , x2,x1,crop_h , y2,y1,box[0][0],w,box[0][0]*w,int(round(box[0][0]*w,2)),box[0][1],h,box[0][1]*h,int(round(box[0][1]*h,2)))
                # print("box[]:",box[0][2],w,box[0][2]*w,int(round(box[0][2]*w,2)),box[0][3],h,box[0][3]*h,int(round(box[0][3]*h,2)))
                # # print("x1,x2,y1,y2:",x1,x2,y1,y2)
                # # cv2.imwrite("test.png",img_crop)
                # # print("img_crop_box:",img_crop_box)
                # print("img id:",image_info['image_id'])

                # print("w-crop_w:",img_w0,img_h0,x1,x2,y1,y2,crop_h,img_h0-crop_h,crop_w,img_w0-crop_w)
                rx1 = random.randint(0,img_w0-crop_w)
                ry1 = random.randint(0,img_h0-crop_h)
        img_0[ry1:ry1+crop_h,rx1:rx1+crop_w] = img_crop
        # cv2.imwrite("test_paste.png",img_0)    

        # boxes = box0
        # labels = labels0
        # print("boxes labels:",boxes,boxes.shape,boxes.shape[0],labels,labels.shape)
        # print("box0,box:",box0.shape,box.shape)
        # print("labels0,tag:",labels0,tag)
    
        box1 = np.array([[rx1/w,ry1/h,(rx1+crop_w)/w,(ry1+crop_h)/h]])
        # print("type:",box0,box1)
        boxes_c = np.concatenate((box0,box1),axis=0)[:box0.shape[0]+1]
        labels = np.concatenate((labels0,tag),axis=0)[:labels0.shape[0]+1]
        # print("boxes after:",np.concatenate((box0,box),axis=0),"\n",boxes)
        # print("labels:",labels)

        # for i in boxes_c:
        #     # cv2.rectangle(img4, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), [255,255,0], 1, cv2.LINE_AA)
        #     cv2.rectangle(img_0, (int(i[0]*w), int(i[1]*h)), (int(i[2]*w), int(i[3]*h)), [255,255,0], 1, cv2.LINE_AA)
        # cv2.imwrite("/home/whf/Temp/11-扫地机/projects/temp/"+image_info['image_id']+".png", img_0)


        if self.transform:
            # print("self.transform:")
            image, boxes_mix, labels = self.transform(img_0, boxes_c, labels)
        if self.target_transform:
            boxes_mix, labels = self.target_transform(boxes_c, labels)

        return image_info_0, image, boxes_mix, labels
    

    def load_mosaic(self, index):
        # loads images in a 4-mosaic

        labels4 = []
        tag4 = []
        s = self.img_size

        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + [self.indices[random.randint(0, self.n - 1)] for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            image_info = self.data[index]
            img, _, (h, w) = load_image(self, image_info['image_id'])

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

            # if i == 2:
            # Labels
            x = image_info['boxes']
            tag = image_info['labels']

            # x = self.labels[index]
            labels = x.copy()
            if x.size > 0:  # Normalized xywh to pixel xyxy format
                labels[:, 0] = (w * (x[:, 0]) + padw)/w
                labels[:, 1] = (h * ( x[:, 1]) + padh)/h
                labels[:, 2] = (w * (x[:, 2]) + padw)/w
                labels[:, 3] = (h * (x[:, 3]) + padh)/h

            # print(labels)
            labels4.append(labels)
            tag4.append(tag)
        # print("labels4:",labels4)
        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            tag4=  np.concatenate(tag4,0)
            np.clip(labels4[:, 0:], 0, 2 * s, out=labels4[:, 0:])  # use with random_perspective
            # img4, labels4 = replicate(img4, labels4)  # replicate

        # # Augment
        # img4, labels4 = random_perspective(img4, labels4,
        #                                    degrees=self.hyp['degrees'],
        #                                    translate=self.hyp['translate'],
        #                                    scale=self.hyp['scale'],
        #                                    shear=self.hyp['shear'],
        #                                    perspective=self.hyp['perspective'],
        #                                    border=self.mosaic_border)  # border to remove


            # # cv2.imwrite("./mosaic/"+str(index)+"_noAug.jpg",img4)
            # # Augment
            # img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=0.0)

            # # cv2.imwrite("./mosaic/"+str(index)+"_copy_paste.jpg",img4)
            # img4, labels4 = random_perspective(img4, labels4, segments4,
            #                                    degrees=0,
            #                                    translate=0.1,
            #                                    scale=0.5,
            #                                    shear=0,
            #                                    perspective=0.0,
            #                                    border=mosaic_border)  # border to remove



        # for i in labels4:
        #     # cv2.rectangle(img4, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), [255,255,0], 1, cv2.LINE_AA)
        #     cv2.rectangle(img4, (int(i[0]*w), int(i[1]*h)), (int(i[2]*w), int(i[3]*h)), [255,255,0], 1, cv2.LINE_AA)
        # cv2.imwrite("mosica.png", img4)

        if self.transform:
            img4, labels, tag = self.transform(img4, labels, tag)
        if self.target_transform:
            labels, tag = self.target_transform(labels, tag)

        # print(labels)
        return img4, labels, tag


