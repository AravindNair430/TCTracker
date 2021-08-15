import torch, torchvision
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2

import matplotlib.pyplot as plt

class Classifier:
	def __init__(self, cnn_model_path):
		self.cnn_model_path = cnn_model_path
		self.cnn_model = self.get_cnn_model()

	def get_cnn_model(self):
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		return torch.load(self.cnn_model_path, map_location=device)

	def get_cnn_prediction(self, image, show_image = False):
		image_transform = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
		image = image_transform(image).float()
		image = Variable(image, requires_grad=False)
		image = image.unsqueeze(0)

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		with torch.no_grad():
			output = self.cnn_model(image.to(device))
		_, preds = torch.max(output, 1)
		_, indices = torch.sort(output, descending = True)

		prob = nn.functional.softmax(output, dim = 1) *100
		prob, indices_prob = torch.sort(prob, descending = True)
		preds = preds.to('cpu')
		prob = prob.to('cpu')
		indices_prob = indices_prob.to('cpu')
		indices = indices.to('cpu')

		class_names = ['TC', 'Not_TC']
		pred_confidence_list = []
		for i in range(len(indices)):
			d={}
			#print('class_names: ', class_names, preds, 'prob: ', prob)
			d[class_names[indices[i][0]]] = prob[i][0]
			#d[class_names[indices[i][1]]] = prob[i][1]
			pred_confidence_list.append(d)
		if show_image:
			ax = plt.subplot(1, 1, 1)
			ax.axis('off')
			ax.set_title('predicted: {}'.format(class_names[preds[0]]))
			imshow(image[0].cpu())
			for Class, score in pred_confidence_list[0].items():
				print('Class :', Class, '-> confidence score :{:.4f}'.format(score.item()))
			print(class_names[preds[0]])
		return class_names[preds[0]], [pred[class_names[preds[0]]] for pred in pred_confidence_list][0].item()

	def pred_on_cropped_image(self,image_id, im, output, show_image = False):
		pred_boxes = {}
		best_score = 0.0
		#print(len(output['instances'].to('cpu')), output['instances'].get_fields()['pred_boxes'])
		if len(output['instances'].to('cpu')) > 1:
			bbox_pred = output['instances'].to('cpu').get_fields()['pred_boxes'].tensor.numpy().tolist()
			for i, bbox in enumerate(bbox_pred):
				count_tc = 0
				idx = 0
				img = im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				if show_image:
					plt.tick_params(left = False, right  = False, labelleft = False, labelbottom = False)
					plt.imshow(img.get_image()[:, :, ::-1])
					plt.show()
				img_pil = Image.fromarray(img)
				cnn_pred_class, cnn_score = self.get_cnn_prediction(img_pil)
				if cnn_pred_class == 'TC':
					if cnn_score > best_score:
						best_score= cnn_score
						pred_boxes[image_id] = [bbox, i]
				#print(pred_boxes.keys())

			if len(list(pred_boxes.keys())) == 0:
				pred_boxes[image_id] = [[], -1]

		else:
			pred_boxes[image_id] = [[], -1]
		return pred_boxes