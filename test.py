from copy import deepcopy 
import torch
import torch.nn as nn

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

def test(scores, boxes):
    boxes = boxes[0]
    scores = scores[0]
    if not prob_threshold:
        prob_threshold = self.filter_threshold
    # this version of nms is slower on GPU, so we move data to CPU.
    boxes = boxes.to(cpu_device)
    scores = scores.to(cpu_device)
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, scores.size(1)):
        probs = scores[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.size(0) == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
        box_probs = box_utils.nms(box_probs, self.nms_method,
                                    score_threshold=prob_threshold,
                                    iou_threshold=self.iou_threshold,
                                    sigma=self.sigma,
                                    top_k=top_k,
                                    candidate_size=self.candidate_size)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.size(0))
    if not picked_box_probs:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    picked_box_probs = torch.cat(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]