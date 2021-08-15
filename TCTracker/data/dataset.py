from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
import os

class COCOInputDataset:
	def __init__(self, dataset_name, dataset_json_path, dataset_images_path):
		self.dataset_name = dataset_name
		self.dataset_json_path = dataset_json_path
		self.dataset_images_path = dataset_images_path
		self.dataset_dicts = DatasetCatalog.get(dataset_name)
		self.dataset_metadata = MetadataCatalog.get(dataset_name)
		self.image_id_list = self.extract_image_id_list(self.dataset_dicts, self.dataset_images_path)
		self.classes = self.get_classes()

	def extract_image_id_list(self, dataset_dicts, dataset_images_path):
		image_id = dict()
		for image_info in dataset_dicts:
			image_id[image_info['file_name'].replace(dataset_images_path + '/', '')] = image_info['image_id']

		list_images = os.listdir(dataset_images_path)
		list_images.sort()
		list_images = [i for i in list_images if i[-4:] == '.png']
		next_id = len(image_id)
		for i in list_images:
			if i not in list(image_id.keys()):
				image_id[i] = len(image_id)

		return image_id

	def get_classes(self):
		classes = {}
		for i, j in zip(list(self.dataset_metadata.thing_dataset_id_to_contiguous_id.values()),
							self.dataset_metadata.thing_classes):
			if i not in list(classes.keys()):
				classes[i] = j

		return classes

	def get_image_name(self, id):
		image_list = list(self.image_id_list.keys())
		id_list = list(self.image_id_list.values())
		return image_list[id_list.index(id)]

	def get_image_list(self):
		return list(self.image_id_list)

class ImageDataset:
	def __init__(self, dataset_name, dataset_images_path):
		self.dataset_name = dataset_name
		self.dataset_images_path = dataset_images_path
		self.image_id_list = self.get_images_list()

	def get_images_list(self):
		d = {}
		list_images = os.listdir(self.dataset_images_path)
		list_images.sort()
		list_images = [i for i in list_images if i[-4:] == '.png']
		for i, j in enumerate(list_images):
			d[i] = j
		return d

	def get_image_name(self, id):
		return self.image_id_list[id]