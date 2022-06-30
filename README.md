# MobileNetV3-SSD
MobileNetV3-SSD implementation in PyTorch 

**目的**
Object Detection 
应用于目标检测


**使用MobileNetV3-SSD实现目标检测**

**Support Export ONNX**

代码参考（严重参考以下代码）

**一 SSD部分**

[A PyTorch Implementation of Single Shot MultiBox Detector ](https://github.com/amdegroot/ssd.pytorch)

**二 MobileNetV3 部分**

[1 mobilenetv3 with pytorch，provide pre-train model](https://github.com/xiaolai-sqlai/mobilenetv3) 


[2 MobileNetV3 in pytorch and ImageNet pretrained models ](https://github.com/kuan-wang/pytorch-mobilenet-v3)


[3 Implementing Searching for MobileNetV3 paper using Pytorch ](https://github.com/leaderj1001/MobileNetV3-Pytorch)

[4 Google MobileNetV3-ssd using Tensorflow](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssdlite_mobilenet_v3_small_320x320_coco.config)

**训练过程**

**首次训练**

python3 anno_train_ssd.py --dataset_type open_images --datasets /home/supernode/Downloads/clean_bot --net mb3-ssd-lite  \
--model_save_folder models/test0610/new_model_cls7_ori700  \
--scheduler multi-step --lr 0.01 --t_max  50 --validation_epochs 1 --num_epochs 200 --base_net_lr 0.01 --batch_size 32


**测试图片**

python run_ssd_example.py  

**转化onnx模型**

python run_ssd_example.py --mode conversion
