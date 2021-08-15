import numpy as np
import pandas as pd
from collections import Counter
import sys
import torch

try: from evaluation.IOU import *
except: from TCTracker.evaluation.IOU import *

class Evaluation:
	def __init__(self, inp_dataset, outputs, iou_threshold = None, get_mean_average =False):
		self.classes_list = inp_dataset.classes
		self.image_id_list = inp_dataset.image_id_list
		self.inputs = self.get_input_dicts(inp_dataset)
		self.outputs = self.get_output_dicts(outputs)
		if get_mean_average == False:
			if iou_threshold: self.metrics = self.log_metrics_for_one_threshold(iou_threshold)
			else: self.metrics = self.log_metrics_for_one_threshold()

	def get_input_dicts(self, inp_dataset):
		name_id_list = inp_dataset.image_id_list
		true_boxes = []
		#get ground truth boxes
		for image_info in inp_dataset.dataset_dicts:
			for i in range(len(image_info['annotations'])):
				box = image_info['annotations'][i]['bbox']
				true_boxes.append([image_info['image_id'], image_info['annotations'][i]['category_id'],
									box[0], box[1], box[0] + box[2],box[1] + box[3]])
		return true_boxes

	def get_output_dicts(self, outputs):
		pred_boxes = []
		for image_id, output in outputs.items():
			if len(output['instances']) > 0:
				for i in range(len(output['instances'])):
					pred_bbox = output['instances'].get_fields()['pred_boxes'].tensor[i]
					x1, y1, x2, y2 = get_xy_coordinates(pred_bbox)
					pred_score = output['instances'].get_fields()['scores'][i].item()
					pred_classes = output['instances'].get_fields()['pred_classes'][i].item()
					pred_boxes.append([image_id, pred_classes, pred_score, x1, y1, x2, y2])

		#ordering prediction boxes in descending order of confidence scores
		return pred_boxes

	def log_metrics_for_one_threshold(self, iou_threshold = 0.5):
		'''
		returns metrics calculated given a particular threshold
		Arguments:
			iou_threshold(type: float) : IOU threshold to classify prediction
							as TPor FP; default value is 0.5
		Returns:
			metrics(type: dictionary): consists of the following metrics for specified threshold:-
				Average Precision(type: float)
				Average Recall(type: float)
				Average Specificity(type: float)
				F1 Score(type: float)
				Confusion Matrix(type: List): confusion matrix in terms of TP, FP, TN and FN
		'''
		#list of required prediction boxes and gt
		avg_p = []
		avg_rec = []
		confusion_matrix ={}
		df_list = []
		id_list = []
		print('='*95)
		count = 0
		for class_id, class_name in self.classes_list.items():
			try:
				true_boxes = [x for x in self.inputs if x[1] == class_id]
				num_gt = len(true_boxes)
				num_gt_boxes_per_image = Counter([gt[0] for gt in true_boxes])
				num_gt_boxes_per_image = {key: torch.zeros(val) for key, val in num_gt_boxes_per_image.items()}
				#get list of boxes of one class and arrange them in descending order of confidence scores
				pred_boxes = [x for x in self.outputs if x[1] == class_id]
				if len(pred_boxes) > 0:
					pred_boxes = torch.tensor(pred_boxes)
					pred_boxes = pred_boxes[pred_boxes[:, 2].argsort(descending = True)]
					#get list of ground truth boxes belonging to one class

					tp = torch.zeros(len(pred_boxes))
					fp = torch.zeros(len(pred_boxes))
					all_ious = []
					log_for_df = []
					boxes_list = []
					best_iou = 0.0
					
					for pred_idx, detection in enumerate(pred_boxes):
						id_list.append(detection[0].item())
						gt_boxes = [x for x in true_boxes if x[0] == detection[0]]
						if gt_boxes == []:
							log_for_df.append([detection[0].item(), 0.0, 0.0])
							boxes_list.append([np.nan, detection[-4:]])

						for gt_idx, gt_box in enumerate(gt_boxes):
							iou = intersection_over_union(torch.tensor(gt_box[-4:]), detection[-4:]).item()
							log_for_df.append([detection[0].item(), detection[2].item(), iou])
							boxes_list.append([gt_box[-4:], detection[-4:]])
							if iou > best_iou:
								best_iou = iou
								best_iou_idx = gt_idx
						
						if best_iou > iou_threshold:
							if (int(detection[0].item()) in num_gt_boxes_per_image.keys()) and (num_gt_boxes_per_image[int(detection[0].item())][best_iou_idx] == 0):
								tp[pred_idx] = 1
								num_gt_boxes_per_image[int(detection[0].item())][best_iou_idx] = 1
							else:
								fp[pred_idx] = 1
						else:
							fp[pred_idx] = 1

					epsilon = 1e-6 #for numerical stability
					num_tp = len([x for x in tp if x == 1])
					num_fp = len([x for x in fp if x == 1])
					num_fn = num_gt - len([x for x in tp if x == 1])
					num_tn = len([x for x in self.image_id_list.values() if (x not in np.array(true_boxes)[:, 0].tolist()) and (x not in pred_boxes[:, 0])])
					precision = num_tp / (num_tp + num_fp + epsilon)
					recall = num_tp / (num_tp + num_fn + epsilon)
					specificity = num_tn / (num_tn + num_fp + epsilon)
					accuracy = (num_tp + num_tn) / (num_tp + num_tn + num_fp + num_fn + epsilon)
					f_score = (2 * precision * recall) / (precision + recall + epsilon)
					c_m = {'TP': num_tp, 'FP': num_fp, 'TN': num_tn, 'FN': num_fn, 'precision': precision, 'recall': recall, 'specificity': specificity, 'accuracy': specificity, 'f_score': f_score, 'GT': num_gt, 'Pred': len(pred_boxes)}
					confusion_matrix[class_name] = c_m
				else:
					c_m = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'accuracy': 0.0, 'f_score': 0.0, 'GT': num_gt, 'Pred': len(pred_boxes)}
			except Exception as e:
				sys.exit(e)
				
			print(f'For class \'{class_name}\'')
			print(f'\tNumber of pred_boxes: {len(pred_boxes)}, Number of true_boxes: {len(true_boxes)}')
			print(f'confusion_matrix: {confusion_matrix[class_name]}')
			print('='*95)
		return confusion_matrix

	# def log_metrics_for_all_thresholds(self, iou_threshold = 0.5):
	# 	iou_thresholds = [x/100 for x in range(45, 96, 5)]
	# 	metrics = []
	# 	for iou in iou_thresholds:
	# 		metrics.append(log_metrics_for_one_threshold(iou_threshold))
	# 	AP_list = [x['AP'] for x in metrics]
	# 	AR_list = [x['AR'] for x in metrics]

	# 	mAP = sum(AP_list) / len(AP_list)
	# 	mAR = 2 * torch.trapz(AR_list, iou_thresholds).item()

	# 	return {'mAP[.5:.05:.95]': mAP, 'mAR[.5:.05:.95]': mAR}