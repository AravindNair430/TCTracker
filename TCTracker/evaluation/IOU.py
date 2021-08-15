import torch

#get xy coordinates from box of type torch.Tensor
def get_xy_coordinates(box):
	box_x1 = box[..., 0:1]
	box_y1 = box[..., 1:2]
	box_x2 = box[..., 2:3]
	box_y2 = box[..., 3:4]

	return box_x1, box_y1, box_x2, box_y2

#calculate IoU
def intersection_over_union(box1, box2):
	box1_x1, box1_y1, box1_x2, box1_y2 = get_xy_coordinates(box1)
	box2_x1, box2_y1, box2_x2, box2_y2 = get_xy_coordinates(box2)

	x1 = torch.max(box1_x1, box2_x1)
	y1 = torch.max(box1_y1, box2_y1)
	x2 = torch.min(box1_x2, box2_x2)
	y2 = torch.min(box1_y2, box2_y2)

	intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
	box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
	box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
	return intersection / (box1_area + box2_area - intersection + 1e-6)