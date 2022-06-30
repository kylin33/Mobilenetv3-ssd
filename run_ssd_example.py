from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer

import argparse
import time
import cv2
import sys
import os
import time
import glob
import numpy as np

import torch
from thop  import profile
import time
import onnx
from onnxsim import simplify


from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite,create_mobilenetv3_ssd_lite_predictor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', nargs='+', type=str, default='/home/supernode/test/MobileNetV3-SSD-New/models/test0610/new_model_cls7_ori700/best.pt', help='model.pt path(s)')
    # parser.add_argument('--model_path', nargs='+', type=str, default='models/pt_model/last.pt', help='model.pt path(s)')
    parser.add_argument('--net_type', type=str, default='mb3-ssd-lite', help='source')  # file/folder, 0 for webcam
    parser.add_argument("--mode", type=str,default="test", help="Select test or conversion mode")
    parser.add_argument('--label_path', type=str, default='/home/supernode/test/MobileNetV3-SSD/models/pt_model/open-images-model-labels.txt', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--image_path', type=str, default='/home/supernode/asr/zhy/test_image/20211215180933487_orig_image.png', help='source')  # file/folder, 0 for webcam    
    opt = parser.parse_args()
    print(opt)

class_names = [name.strip() for name in open(opt.label_path).readlines()]

from gpu_mem_track import MemTracker
# gpu_tracker = MemTracker(path='log/')
# gpu_tracker.track()

if opt.net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif opt.net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif opt.net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif opt.net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif opt.net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
elif opt.net_type == 'mb3-ssd-lite':
    net = create_mobilenetv3_ssd_lite(len(class_names), is_test=True)

else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)


net.load(opt.model_path)
# net = net.to(torch.device('cuda:0'))
net = net.to(torch.device("cpu"))

# gpu_tracker.track()

if opt.net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif opt.net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif opt.net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif opt.net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif opt.net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
elif opt.net_type == 'mb3-ssd-lite':
    predictor = create_mobilenetv3_ssd_lite_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)


if opt.mode == "test":
    # test image
    time_start = time.time()
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    root = os.getcwd()
    path = root+"/test_image/"+time_now

    if not os.path.exists(path):
        os.makedirs(path)
        print('文件夹创建完成  '+path)
    i = 0

    img_paths = glob.glob("/home/supernode/asr/zhy/valimg11/"+"*.png")
    for img_path in img_paths:
        orig_image = cv2.imread(img_path)
        # f = open(path+"/"+os.path.basename(img_path).replace(".png",".txt"),"w")

        boxes, labels, probs = predictor.predict(orig_image, 10, 0.4)
        flag = 0

        for i in range(boxes.size(0)):
            if(probs[i]<0.2):
                continue
            box = boxes[i, :]

            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
            #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            if(labels[i] == 2):
                flag = 1
            else:
                flag = 0
            cv2.putText(orig_image, label,
                        (box[0] + 20, box[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
            box = np.array(box)
            # f.write(" ".join(str(i) for i in box)+" "+label + "\n")
        # print(labels, probs, boxes)
        newimg_path = path+ "/"+os.path.basename(img_path)

        if(flag==1):
            continue
        # cv2.imwrite(newimg_path, orig_image)
        print(f"Found {len(probs)} objects. The output image is {newimg_path}")
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print("*******************",time_sum)

elif opt.mode == "conversion":
    # create onnx model
    image = torch.randn(1, 3, 320, 320).to(torch.device('cuda:0')) #.cuda()

    # flop, param = profile(net, inputs=(image,))
    # print("flop:",flop)
    # print("param:", param)
    f = opt.model_path.replace('.pt', '.onnx')
    torch.onnx.export(net, image, f, verbose=False, opset_version=12, input_names=['images'],
                            output_names=['output'])

    onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)

    sim_f = f.replace('.onnx', '_sim.onnx')
    model_simp, check_ = simplify(onnx_model)
    assert check_, "simplified onnx model could not be validated"
    onnx.save(model_simp, sim_f)
