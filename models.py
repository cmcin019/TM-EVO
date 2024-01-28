# Imports
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import YolosImageProcessor, YolosForObjectDetection

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights 

def get_deter():
	# TODO: DETR-RESNET50
	processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
	model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval()
	id2label = model.config.id2label
	return processor, model, id2label

def get_yolo():
	# TODO: YOLO-RESNET50
	processor = YolosImageProcessor.from_pretrained('hustvl/yolos-tiny')
	model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny').eval()
	id2label = model.config.id2label
	return processor, model, id2label

def get_FasterRCNN():
	# TODO: FASTER CNN - RESNET50
	# Step 1: Initialize model with the best available weights
	weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
	model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
	model.eval()

	processor = lambda images, return_tensors: torch.Tensor(weights.transforms()(images)).unsqueeze(0)

	def list_tuples_to_dict(l):
		d = {'scores': [], 'labels': [], 'boxes': []}
		for i in l:
			d['scores'] += [i[0]]
			d['labels'] += [i[1]]
			d['boxes'] += [i[2]]
		return [d]
	
	processor.post_process_object_detection = lambda *x, target_sizes, threshold: list_tuples_to_dict([(s, l, b) for (s, l, b) in zip(x[0][0]['scores'], x[0][0]['labels'], x[0][0]['boxes']) if s >= threshold])
	id2label = weights.meta["categories"]
	return processor, model, id2label

def get_RetinaNet():
	# TODO: RETINA NET - RESNET50
	# Step 1: Initialize model with the best available weights
	weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
	model = retinanet_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
	model.eval()

	processor = lambda images, return_tensors: torch.Tensor(weights.transforms()(images)).unsqueeze(0)

	def list_tuples_to_dict(l):
		d = {'scores': [], 'labels': [], 'boxes': []}
		for i in l:
			d['scores'] += [i[0]]
			d['labels'] += [i[1]]
			d['boxes'] += [i[2]]
		return [d]
	
	processor.post_process_object_detection = lambda *x, target_sizes, threshold: list_tuples_to_dict([(s, l, b) for (s, l, b) in zip(x[0][0]['scores'], x[0][0]['labels'], x[0][0]['boxes']) if s >= threshold])
	id2label = weights.meta["categories"]
	return processor, model, id2label

def get_MaskRCNN():
	# TODO: MASK RCNN- RESNET50
	# Step 1: Initialize model with the best available weights
	weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
	model = maskrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.9)
	model.eval()

	processor = lambda images, return_tensors: torch.Tensor(weights.transforms()(images)).unsqueeze(0)

	def list_tuples_to_dict(l):
		d = {'scores': [], 'labels': [], 'boxes': []}
		for i in l:
			d['scores'] += [i[0]]
			d['labels'] += [i[1]]
			d['boxes'] += [i[2]]
		return [d]
	
	processor.post_process_object_detection = lambda *x, target_sizes, threshold: list_tuples_to_dict([(s, l, b) for (s, l, b) in zip(x[0][0]['scores'], x[0][0]['labels'], x[0][0]['boxes']) if s >= threshold])
	id2label = weights.meta["categories"]
	return processor, model, id2label
