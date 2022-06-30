import argparse
import os
import logging
import sys
import itertools

import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from  vision.utils.misc import fitness

from test import ModelEMA

from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite

from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite, create_mobilenetv3_ssd_lite_predictor
from vision.utils import box_utils, measurements
import cv2

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")


parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")

parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=2, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/new_model_check/',
                    help='Directory for saving checkpoint models')

# parser.add_argument('--checkpoint_folder', default='models/new_model_check_cut_mos/',
#                     help='Directory for saving checkpoint models')
                    


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        # print(boxes, labels)
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        # print("locations:",type(locations))
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes.float())  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def Mean_average_precision(rec,prec):
    if rec is None or prec is None:
        return np.nan
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec>t) == 0:
            p = 0
        else:
            p = np.max(np.nan_to_num(prec)[rec>=t])
        ap += p/11
    return ap

def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        # print(boxes[0])
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        # for si, pred in enumerate(locations):
        #     nl = len(labels)
        #     # tcls = labels[:,0]

        #     if nl:
        #         detected = []
        #         tcls_tensor = labels
                
        #         for cls  in torch.unique(tcls_tensor):
        #             ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
        #             # pi = (cls == pred)

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_case


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


if __name__ == '__main__':
    timer = Timer()

    logging.info(args)
    if args.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    elif args.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb3-ssd-lite':
        create_net = lambda num: create_mobilenetv3_ssd_lite(num)
        config = mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    datasets = []
    last = "models/new_model_0212/last.pt"
    best = "models/new_model_0212/best.pt"

    # last = "models/new_model_0212_cut_mos/last.pt"
    # best = "models/new_model_0212_cut_mos/best.pt"

    # for dataset_path in args.datasets:
    #     if args.dataset_type == 'voc':
    #         dataset = VOCDataset(dataset_path, transform=train_transform,
    #                              target_transform=target_transform)
    #         label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
    #         store_labels(label_file, dataset.class_names)
    #         num_classes = len(dataset.class_names)
    #     elif args.dataset_type == 'open_images':
    #         dataset = OpenImagesDataset(dataset_path,
    #              transform=train_transform, target_transform=target_transform,
    #              dataset_type="train", balance_data=args.balance_data)
    #         label_file = os.path.join(args.checkpoint_folder, "open-images-model-labels.txt")
    #         store_labels(label_file, dataset.class_names)
    #         logging.info(dataset)
    #         num_classes = len(dataset.class_names)

    #     else:
    #         raise ValueError(f"Dataset tpye {args.dataset_type} is not supported.")
    #     datasets.append(dataset)
    # logging.info(f"Stored labels into file {label_file}.")
    # train_dataset = ConcatDataset(datasets)
    # logging.info("Train dataset size: {}".format(len(train_dataset)))
    # train_loader = DataLoader(train_dataset, args.batch_size,
    #                           num_workers=args.num_workers,
    #                           shuffle=True,
    #                           drop_last=True)
    logging.info("Prepare Validation datasets.")
    for dataset_path in args.datasets:
        
        if args.dataset_type == "voc":
            val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                    target_transform=target_transform, is_test=True)
            num_classes = len(val_dataset.class_names)
        elif args.dataset_type == 'open_images':
            val_dataset = OpenImagesDataset(dataset_path,
                                            transform=test_transform, target_transform=target_transform,
                                            dataset_type="test")
            num_classes = len(val_dataset.class_names)
            logging.info(val_dataset)
        logging.info("validation dataset size: {}".format(len(val_dataset)))

        val_loader = DataLoader(val_dataset, args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=False,
                                drop_last=True)
        logging.info("Build network.")
        net = create_net(num_classes)
        #print(net)

        # ema = ModelEMA(net)

        min_loss = -10000.0
        last_epoch = -1

        base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
        extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
        if args.freeze_base_net:
            logging.info("Freeze base net.")
            freeze_net_layers(net.base_net)
            params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                    net.regression_headers.parameters(), net.classification_headers.parameters())
            params = [
                {'params': itertools.chain(
                    net.source_layer_add_ons.parameters(),
                    net.extras.parameters()
                ), 'lr': extra_layers_lr},
                {'params': itertools.chain(
                    net.regression_headers.parameters(),
                    net.classification_headers.parameters()
                )}
            ]
        elif args.freeze_net:
            freeze_net_layers(net.base_net)
            freeze_net_layers(net.source_layer_add_ons)
            freeze_net_layers(net.extras)
            params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
            logging.info("Freeze all the layers except prediction heads.")
        else:
            params = [
                {'params': net.base_net.parameters(), 'lr': base_net_lr},
                {'params': itertools.chain(
                    net.source_layer_add_ons.parameters(),
                    net.extras.parameters()
                ), 'lr': extra_layers_lr},
                {'params': itertools.chain(
                    net.regression_headers.parameters(),
                    net.classification_headers.parameters()
                )}
            ]

        timer.start("Load Model")
        if args.resume:
            logging.info(f"Resume from the model {args.resume}")
            net.load(args.resume)
        elif args.base_net:
            logging.info(f"Init from base net {args.base_net}")
            net.init_from_base_net(args.base_net)
        elif args.pretrained_ssd:
            logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
            net.init_from_pretrained_ssd(args.pretrained_ssd)
        logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

        net.to(DEVICE)

        criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                                center_variance=0.1, size_variance=0.2, device=DEVICE)
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                    + f"Extra Layers learning rate: {extra_layers_lr}.")

        if args.scheduler == 'multi-step':
            logging.info("Uses MultiStepLR scheduler.")
            milestones = [int(v.strip()) for v in args.milestones.split(",")]
            scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                        gamma=0.1, last_epoch=last_epoch)
        elif args.scheduler == 'cosine':
            logging.info("Uses CosineAnnealingLR scheduler.")
            scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
        else:
            logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
            parser.print_help(sys.stderr)
            sys.exit(1)

        logging.info(f"Start training from epoch {last_epoch + 1}.")

        #sys.exit(0)#test

        # best_fitness= 10
        # for epoch in range(last_epoch + 1, args.num_epochs):
        #     scheduler.step()
        #     train(train_loader, net, criterion, optimizer,
        #           device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
            

        # val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
    
        net.eval()
        running_loss = 0.0
        running_regression_loss = 0.0
        running_classification_loss = 0.0
        num = 0
        for _, data in enumerate(val_loader):
            images, boxes, labels = data
            img_copy = images.numpy()
            import numpy as np, cv2
            image_copy_cv = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
            img_shape = img_copy.shape
            h = img_shape[-2]
            w = img_shape[-1]
            c = img_shape[0]
            images = images.to(DEVICE)
            boxes = boxes.to(DEVICE)
            labels = labels.to(DEVICE)
            # print(boxes[0])
            num += 1

            with torch.no_grad():
                confidence, locations = net(images)
                # regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
                pos_mask = labels > 0
                predicted_locations = locations[pos_mask, :].reshape(-1, 4)
                print("confidence locations:",confidence, "\n",locations)
                print("predicted_locations:",predicted_locations)

                for show_label in predicted_locations:
                    x1 = int(max(0,show_label[0])*w)
                    y1 = int(max(0,show_label[1])*h)
                    x2 = int(max(0,show_label[2])*w)
                    y2 = int(max(0,show_label[3])*h)
                    cv2.rectangle(image_copy_cv, (x1, y1), (x2, y2), [255,255,0], 1, cv2.LINE_AA)
            cv2.imshow("test.png",image_copy_cv)
            cv2.waitKey(100)



        # nms_method = "hard"
        # if args.net == 'mb3-ssd-lite':
        #     predictor = create_mobilenetv3_ssd_lite_predictor(net, nms_method=nms_method, device=DEVICE)
        # else:
        #     logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        #     parser.print_help(sys.stderr)
        #     sys.exit(1)
        
        # true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(val_dataset)
        # class_names = [name.strip() for name in open(args.label_file).readlines()]
        # results = []
        # for i in range(len(val_dataset)):
        #     print("process image", i)
        #     timer.start("Load Image")
        #     image = val_dataset.get_image(i)
        #     print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        #     timer.start("Predict")
        #     boxes, labels, probs = predictor.predict(image)
        #     print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        #     indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        #     results.append(torch.cat([
        #         indexes.reshape(-1, 1),
        #         labels.reshape(-1, 1).float(),
        #         probs.reshape(-1, 1),
        #         boxes + 1.0  # matlab's indexes start from 1
        #     ], dim=1))
        # results = torch.cat(results)
        # for class_index, class_name in enumerate(class_names):
        #     if class_index == 0: continue  # ignore background
        #     prediction_path = f"./det_test_{class_name}.txt"
        #     with open(prediction_path, "w") as f:
        #         sub = results[results[:, 1] == class_index, :]
        #         for i in range(sub.size(0)):
        #             prob_box = sub[i, 2:].numpy()
        #             image_id = val_dataset.ids[int(sub[i, 0])]
        #             print(
        #                 image_id + " " + " ".join([str(v) for v in prob_box]),
        #                 file=f
        #             )
        # aps = []
        # print("\n\nAverage Precision Per-class:")
        # for class_index, class_name in enumerate(class_names):
        #     if class_index == 0:
        #         continue
        #     prediction_path =  f"./det_test_{class_name}.txt"
        #     ap = compute_average_precision_per_class(
        #         true_case_stat[class_index],
        #         all_gb_boxes[class_index],
        #         all_difficult_cases[class_index],
        #         prediction_path,
        #         args.iou_threshold,
        #         args.use_2007_metric
        #     )
        #     aps.append(ap)
        #     print(f"{class_name}: {ap}")

        # print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")






    
