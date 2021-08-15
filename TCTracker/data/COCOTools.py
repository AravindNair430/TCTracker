import cv2
import json
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil


def get_lists(path, years_list):
	'''
		parameters:
			path: path to directory containing the folders corresponding to each year
				  (for each year, the corresponding cyclone folder is present with two folders, namely 'Masks' and 'Raw_images')
			years_list: list of years ffor which the dataaset is to be generated

		return values:
			mask_list: list of segmentation masks in folder 'Masks'
			image_list: list of satellite images in folder 'Raw_images'
			green_list: list of images with segmentation marked in green color {in range (0, 70, 0)-(0, 255, 0) in BGR}
	'''
	years = list(map(lambda x: str(x), years_list))
	
	cyclones = []
	cyclones1 = []

	for i in years:
		if os.path.exists(path + '/' + i + '/.DS_Store'):
			os.remove(path + '/' + i + '/.DS_Store')
		c = os.listdir(path + '/' + i)
		c.sort()
		c = list(map(lambda x: i + '/'+ x, c))
		cyclones.extend(c)

	green_list = []
	mask_list = []
	image_list = []

	for cyclone in cyclones:
		masks = os.listdir(path + '/' + cyclone + '/Masks')
		images = os.listdir(path + '/' + cyclone + '/Raw_images')
		masks = list(filter(lambda x: '.png' in x, masks))
		images = list(filter(lambda x: '.png' in x, images))
		masks.sort()
		images.sort()
		mask_list.append(list(map(lambda x: cyclone + '/Masks/' + x, masks)))
		image_list.append(list(map(lambda x: cyclone + '/Raw_images/' + x, images)))
		green_list1 = []
		for j, mask in enumerate(masks):
			image = cv2.imread(path + '/' + cyclone + '/Masks/' + mask)
			mask1 = cv2.inRange(image, (0, 70, 0), (0, 255, 0))
			image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			green_result = cv2.bitwise_or(image1, mask1, mask=mask1)
		
			if cv2.countNonZero(green_result) > 0:
				green_list1.append(cyclone + '/Masks/' + mask)
		green_list.append(green_list1)
	return mask_list, image_list, green_list


def print_distribution(y_train, y_val, y_test, color_set):
	print('========================================================================')
	print('train size :', y_train.shape[0], ', val size :', y_val.shape[0], ', test size :', y_test.shape[0], ', total :', y_train.shape[0] + y_val.shape[0] + y_test.shape[0])
	print('========================================================================')
	print('train distribution\n', len(y_train[y_train == 'green']), len(y_train[y_train == 'blank']))
	print('========================================================================')
	print('val distribution\n', len(y_val[y_val == 'green']), len(y_val[y_val == 'blank']))
	print('========================================================================')
	print('test distribution\n', len(y_test[y_test == 'green']), len(y_test[y_test == 'blank']))
	print('========================================================================')
	print('color ditribution : ', len(y_train[y_train == 'green']) + len(y_val[y_val == 'green']) + len(y_test[y_test == 'green']),
		len(y_train[y_train == 'blank']) + len(y_val[y_val == 'blank']) + len(y_test[y_test == 'blank']))

	print('\n========================================================================')
	print('train percentage:', (len(y_train[y_train == 'green']) / y_train.shape[0]), (len(y_train[y_train == 'blank']) / y_train.shape[0]))
	print('val percentage:', (len(y_val[y_val == 'green']) / y_val.shape[0]), (len(y_val[y_val == 'blank']) / y_val.shape[0]))
	print('test percentage:', (len(y_test[y_test == 'green']) / y_test.shape[0]), (len(y_test[y_test == 'blank']) / y_test.shape[0]))
	print('========================================================================')
	print('\n========================================================================')
	print('data distribution\n', len(color_set[color_set == 'green'])/color_set.shape[0], len(color_set[color_set == 'blank'])/color_set.shape[0])


def train_test_val_split(path, mask_list, image_list):
	train_size = 511
	val_size = 170
	test_size = 171

	mask_list = [s for sub in mask_list for s in sub]
	mask_list.sort()
	image_list = [sa for sub_arr in image_list for sa in sub_arr]
	image_list.sort()

	
	green_list = []
	blank_list = []
	data = []

	for i, j in zip(mask_list, image_list):
		image = cv2.imread(path + '/' + i)
		mask1 = cv2.inRange(image, (0, 70, 0), (0, 255, 0))
		image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		green_result = cv2.bitwise_or(image1, mask1, mask=mask1)
		
		if cv2.countNonZero(green_result) > 0:
			data.append([(j, i), 'green'])

		else:
			data.append([(j, i), 'blank'])

	df = pd.DataFrame(data, columns = ('name', 'color'))

	x, x_test, y, y_test = train_test_split(df['name'], df['color'], test_size=0.2, train_size=0.8, stratify = df['color'])
	x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.25, train_size =0.75, stratify = y)

	dst_mask = '/Users/aravind/Desktop/test_dataset/train/Masks'
	dst_image = '/Users/aravind/Desktop/test_dataset/train/Raw_images'
	if not os.path.exists(dst_mask):
		os.makedirs(dst_mask)
	if not os.path.exists(dst_image):
		os.makedirs(dst_image)
	x_train.sort_values(inplace =True)
	for i in x_train:
		shutil.copy2(path + '/' + i[0], dst_image)
		shutil.copy2(path + '/' + i[1], dst_mask)

	dst_mask = '/Users/aravind/Desktop/test_dataset/val/Masks'
	dst_image = '/Users/aravind/Desktop/test_dataset/val/Raw_images'
	if not os.path.exists(dst_mask):
		os.makedirs(dst_mask)
	if not os.path.exists(dst_image):
		os.makedirs(dst_image)
	x_val.sort_values(inplace =True)
	for i in x_val:
		shutil.copy2(path + '/' + i[0], dst_image)
		shutil.copy2(path + '/' + i[1], dst_mask)
	
	dst_mask = '/Users/aravind/Desktop/test_dataset/test/Masks'
	dst_image = '/Users/aravind/Desktop/test_dataset/test/Raw_images'
	if not os.path.exists(dst_mask):
		os.makedirs(dst_mask)
	if not os.path.exists(dst_image):
		os.makedirs(dst_image)
	x_test.sort_values(inplace =True)
	for i in x_test:
		shutil.copy2(path + '/' + i[0], dst_image)
		shutil.copy2(path + '/' + i[1], dst_mask)

def get_mask_definition(src_mask, src_image, set_name):
	masks = {}
	mask_list = os.listdir(src_mask)
	image_list = os.listdir(src_image)
	image_list.sort()
	mask_list.sort()
	green_list = []

	for i, j in zip(image_list, mask_list):
		image = cv2.imread(src_mask + '/' + j)
		mask1 = cv2.inRange(image, (0, 70, 0), (0, 255, 0))
		image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		green_result = cv2.bitwise_or(image1, mask1, mask=mask1)
		color = ""
		category = ""

		if cv2.countNonZero(green_result) > 0:
			color = "(0, 255, 0)"
			category = "tropical_cyclone"
			green_list.append((i, j))


		if color != "":
			mask_dict = {"mask" : "Masks/" + j, "color_categories" :{color : {"category" : category, "super_category" : "clouds"}}}
			masks["Raw_images/" + i] = mask_dict
	mask_def_dict = {"masks" : masks, "super_categories" : {"clouds" : ["early_disturbances", "tropical_cyclone"]}}

	with open( src_mask.replace('/Masks', '') + '/mask_definitions.json', 'w') as f:
		json.dump(mask_def_dict, f, indent=4)


def get_annotaions(src_mask, src_image, set_name, json_utils = None):
	mask_list = os.listdir(src_mask)
	mask_list = list(filter(lambda x: '.png' in x, mask_list))
	image_list = os.listdir(src_image)
	image_list = list(filter(lambda x: '.png' in x, image_list))
	image_list.sort()
	mask_list.sort()
	count = 0

	if not json_utils:
		json_utils = {"info": {"description": set_name + "Dataset","url": "no-url/datasets.com","version": 1, "year": "2019", 
		"contributor": "ABC", "date_created": "12/01/2021" }, "licenses": [{"url": "no-url/licences.com","id": 0,"name": set_name}],
		"images" : [], "annotations" : [], "categories": [{"supercategory" : "clouds", "id": 1, "name": "early_disturbances"},
		{"supercategory": "clouds", "id": 2, "name": "tropical_cyclone"}]}

	images = []
	annotations = []

	for mask_name, image_name in zip(mask_list, image_list):
		l = []
		image_info = {"license": 0,"file_name": "","width": 1339,"height": 688,"id": 0}
		annotation_info = {"segmentation": [],"iscrowd": 0,"image_id": 0,"category_id": 0,"id": 0,"bbox": [],"area": 0.0}
		#print(mask_name)
		mask = cv2.imread(src_mask + '/' + mask_name)
		mask1 = cv2.inRange(mask, (0, 70, 0), (0, 255, 0))
		mask2 = cv2.inRange(mask, (0, 0, 70), (0, 0, 255))
		image1 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
		green_result = cv2.countNonZero(cv2.bitwise_or(image1, mask1, mask=mask1))
	
		if  green_result > 0:
			annotation_info["category_id"] = 2
			contours, heirarchy = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		if (green_result != 0):
			boxes = []
			area = 0
			for c in contours:
				a = cv2.contourArea(c)
				if a > 2:
					(x, y, w, h) = cv2.boundingRect(c)
					boxes.append([x,y, x+w,y+h])
					area += a
					l.append(c.flatten().tolist())

			boxes = np.asarray(boxes)
			left, top = np.min(boxes, axis=0)[:2]
			right, bottom = np.max(boxes, axis=0)[2:]

			image_info["file_name"] = image_name
			image_info["id"] = count

			annotation_info["segmentation"] = l
			annotation_info["image_id"] = count
			annotation_info["id"] = count
			annotation_info["bbox"] = [x, y, w, h]
			annotation_info["area"] = area
			images.append(image_info)
			annotations.append(annotation_info)
			count += 1

	json_utils["images"] = images
	json_utils["annotations"] = annotations

	print(type(json_utils))
	with open(src_mask.replace('Masks', '') + set_name + '_coco_instances.json', 'w') as f:
		json.dump(json_utils, f)

	with open(src_mask.replace('Masks', '') + set_name + '_coco_instances(1).json', 'w') as f:
		json.dump(json_utils, f, indent = 4)