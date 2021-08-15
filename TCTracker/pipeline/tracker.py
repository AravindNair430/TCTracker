try: import pipeline.detector as dt
except: import TCTracker.pipeline.detector as dt

try: import pipeline.classifier as cl
except: import TCTracker.pipeline.classifier as cl
try: from TCTracker.data.dataset import COCOInputDataset
except: from TCTracker.data.dataset import COCOInputDataset

try: from data.dataset import ImageDataset
except: from TCTracker.data.dataset import ImageDataset

from detectron2.structures import boxes, Instances
import cv2
import torch

class Tracker:
    def __init__(self, detectron2_detector=None, cnn_classifier=None):
        assert detectron2_detector, 'provide detector object'
        assert cnn_classifier, 'provide classifier object'
        self.detectron2_detector = detectron2_detector
        self.cnn_classifier = cnn_classifier

    def wind_speed_filter(self, inp_dataset, all_outputs, df):
        for idx, output in all_outputs.items():
            if len(output['instances'].to('cpu')) > 0:
                image_shape = cv2.imread(inp_dataset.dataset_images_path + '/' + inp_dataset.get_image_name(idx)).shape[:-1]
                wind_speed = df[df['Image Name'] == inp_dataset.get_image_name(idx)[:-14]]['Wind Speed'].item()
                if not wind_speed >= 34:
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    pred_bboxes = boxes.Boxes(torch.Tensor([]).to(device))
                    scores = torch.Tensor([]).to(device)
                    pred_class = torch.Tensor([]).to(device).type(torch.int64)
                    pred_masks = torch.Tensor([]).to(device).type(torch.uint8)

                    obj = Instances(image_size = image_shape)
                    obj.set('pred_classes', pred_class)
                    obj.set('scores', scores)
                    obj.set('pred_masks', pred_masks)
                    obj.set('pred_boxes', pred_bboxes)
                    all_outputs[idx] = {'instances' : obj}
        return all_outputs

    def get_req_outputs(self, output, idx, pred_box, image_shape):
        if idx != -1:
            pred_box = [pred_box]
            req_score = [output.get_fields()['scores'].numpy().tolist()[idx]]
            req_class = [output.get_fields()['pred_classes'].numpy().tolist()[idx]]
            req_mask = [output.get_fields()['pred_masks'].numpy().tolist()[idx]]

        else:
            pred_box = []
            req_mask = []
            req_class = []
            req_score = []

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pred_bboxes = boxes.Boxes(torch.Tensor(pred_box).to(device))
        scores = torch.Tensor(req_score).to(device)
        pred_class = torch.Tensor(req_class).to(device).type(torch.int64)
        pred_masks = torch.Tensor(req_mask).to(device).type(torch.uint8)

        obj = Instances(image_size = image_shape)
        obj.set('pred_classes', pred_class)
        obj.set('scores', scores)
        obj.set('pred_masks', pred_masks)
        obj.set('pred_boxes', pred_bboxes)
        return {'instances' : obj}

    def get_predictions(self, df, inp_dataset = None):
        assert inp_dataset, 'Provide inp_dataset (Object of type InputDataset)'

        all_outputs = self.detectron2_detector.get_predictions_of_all_images(inp_dataset)
        all_outputs = self.wind_speed_filter(inp_dataset, all_outputs, df)
        image_id_list = inp_dataset.image_id_list
        count = 1
        for image_id, output in all_outputs.items():
            if (count % 50 == 0) or (count == len(image_id_list)) or (count == 1):
                string_a = str(count) + ' image ' if count == 1 else str(count) + ' images '
                print('Predictions for ' + string_a + 'made')
            if len(output['instances'].to('cpu')) > 1:
                best_pred_box = []
                best_score = 0.0
                image_name = inp_dataset.get_image_name(image_id)
                image = cv2.imread(inp_dataset.dataset_images_path + '/' + image_name)
                pred_boxes_and_idx = self.cnn_classifier.pred_on_cropped_image(image_id, image, output)
                all_outputs[image_id] = self.get_req_outputs(output['instances'].to('cpu'), pred_boxes_and_idx[image_id][1], pred_boxes_and_idx[image_id][0], image.shape[:-1])
            count += 1
        return all_outputs