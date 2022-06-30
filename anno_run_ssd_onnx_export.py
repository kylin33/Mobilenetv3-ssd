from glob import glob
import glob
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer

import argparse
import cv2
import sys

import torch
import onnx
from onnxsim import simplify


from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite,create_mobilenetv3_ssd_lite_predictor

import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', nargs='+', type=str, default='/home/whf/Temp/11-扫地机/projects/anno_model/new_model_mos_backaug_0215/best.pt', help='model.pt path(s)')
    parser.add_argument('--net_type', type=str, default='mb3-ssd-lite', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--label_path', type=str, default='/home/whf/Temp/11-扫地机/projects/MobileNetV3-SSD/models/new_model/open-images-model-labels.txt', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--image_path', type=str, default='/home/whf/Temp/11-扫地机/projects/test_imgs/test', help='source')  # file/folder, 0 for webcam    
    parser.add_argument('--image_save_path', type=str, default="/home/whf/Temp/11-扫地机/projects/test_imgs/test_result", help='source')  # file/folder, 0 for webcam    
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
net = net.to(torch.device('cuda:0'))

# # gpu_tracker.track()

# if opt.net_type == 'vgg16-ssd':
#     predictor = create_vgg_ssd_predictor(net, candidate_size=200)
# elif opt.net_type == 'mb1-ssd':
#     predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
# elif opt.net_type == 'mb1-ssd-lite':
#     predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
# elif opt.net_type == 'mb2-ssd-lite':
#     predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
# elif opt.net_type == 'sq-ssd-lite':
#     predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
# elif opt.net_type == 'mb3-ssd-lite':
#     predictor = create_mobilenetv3_ssd_lite_predictor(net, candidate_size=200)
# else:
#     predictor = create_vgg_ssd_predictor(net, candidate_size=200)


# img_save_path = opt.image_save_path
# img_list = glob.glob(opt.image_path+"/*.png")
# for img_dir in img_list:
#     print(img_dir)
#     img_name = os.path.split(img_dir)[-1].split(".")[0]
#     # test image
#     orig_image = cv2.imread(img_dir)
#     print(orig_image.shape)
#     # image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
#     boxes, labels, probs = predictor.predict(orig_image, 10, 0.5)
#     print("boxes:",boxes,labels)

#     for i in range(boxes.size(0)):
#         box = boxes[i, :]
#         cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)
#         #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
#         label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
#         cv2.putText(orig_image, label,(int(box[0]) + 20, int(box[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)  # line type

#     path = img_save_path+"/"+img_name+".jpg"
#     cv2.imwrite(path, orig_image)
#     print(f"Found {len(probs)} objects. The output image is {path}")


## create onnx model
image = torch.randn(1, 3, 300, 300).to(torch.device('cuda:0')) #.cuda()
f = opt.model_path.replace('.pt', '.onnx')
print("f:",f)
torch.onnx.export(net, image, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['output'])

onnx_model = onnx.load(f)
onnx.checker.check_model(onnx_model)

sim_f = f.replace('.onnx', '_sim.onnx')
print("sim_f:",sim_f)
model_simp, check_ = simplify(onnx_model)
assert check_, "simplified onnx model could not be validated"
onnx.save(model_simp, sim_f)
