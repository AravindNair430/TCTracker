try: from datasets.dataset import COCOInputDataset
except: from Cyclotron.datasets.dataset import COCOInputDataset

try: from datasets.dataset import ImageDataset
except: from Cyclotron.datasets.dataset import ImageDataset

from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import boxes, Instances
from detectron2.utils.visualizer import Visualizer, ColorMode

import torch
import cv2, os, time
import matplotlib.pyplot as plt

class Detector:
    def __init__(self, NUM_CLASSES = 1, NUM_WORKERS = 2,
                load_model_flag = True, cfg_file_path = None, input_dataset = None,
                model_name = None, output_dir = './Outputs/', parameters = None):
        assert model_name, "model_name required"
        assert parameters['THRESHOLD'], 'model test threshold required in parameters'
        self.model_name = model_name
        self.parameters = parameters

        if load_model_flag:
            assert self.model_name, "provide model architecture name of the saved detectron2 model Eg. mask_rcnn_R_50_FPN_1x"
            self.predictor = self.load_model(NUM_CLASSES = NUM_CLASSES, cfg_file_path = cfg_file_path)
        
        else:
            assert input_dataset, "provide input dataset object for training"
            self.predictor = self.train_model(NUM_CLASSES = NUM_CLASSES, NUM_WORKERS = NUM_WORKERS,
                                            inp_dataset = input_dataset, output_dir = output_dir)

    #trains a detectron2 Mask R-CNN model
    def train_model(self, NUM_CLASSES, NUM_WORKERS, inp_dataset, output_dir):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/' + self.model_name + '.yaml'))
        print(self.model_name)
        cfg.DATASETS.TRAIN = (inp_dataset.dataset_name,)
        cfg.DATASETS.TEST = ()

        cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/' + self.model_name + '.yaml')
        if self.parameters:
            if 'device' in self.parameters.keys():
                cfg.MODEL.DEVICE = self.parameters['device']
            else:
                if not torch.cuda.is_available(): cfg.MODEL.DEVICE = 'cpu'

            if 'IMS_PER_BATCH' in self.parameters.keys():
                cfg.SOLVER.IMS_PER_BATCH = self.parameters['IMS_PER_BATCH']
                print('images per batch', self.parameters['IMS_PER_BATCH'])
            else:
                cfg.SOLVER.IMS_PER_BATCH = 13
            if 'BASE_LR' in self.parameters.keys():
                cfg.SOLVER.BASE_LR = self.parameters['BASE_LR']
                print('base lr', self.parameters['BASE_LR'])
            if 'MAX_ITER' in self.parameters.keys():
                cfg.SOLVER.MAX_ITER = self.parameters['MAX_ITER']
                print('max iter', self.parameters['MAX_ITER'])
            if 'BATCH_SIZE_PER_IMAGE' in self.parameters.keys():
                cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.parameters['BATCH_SIZE_PER_IMAGE']
                print('batch size per image', self.parameters['BATCH_SIZE_PER_IMAGE'])
        
        else:
            if not torch.cuda.is_available(): cfg.MODEL.DEVICE = 'cpu'
            cfg.SOLVER.IMS_PER_BATCH = 13
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/' + self.model_name + '.yaml')

        cfg.OUTPUT_DIR = output_dir
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume = False)
        trainer.train()

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.parameters['THRESHOLD']
        predictor = DefaultPredictor(cfg)

        cfg.dump(stream=open(cfg.OUTPUT_DIR + '/model_final_cfg.yaml', 'w'))
        return predictor

    #loads saved detectron2 model
    def load_model(self, NUM_CLASSES, cfg_file_path):
        assert cfg_file_path, 'provide cfg_file_path'
        cfg = get_cfg()
        if self.parameters and 'device' in self.parameters.keys():
            cfg.MODEL.DEVICE = self.parameters['device']
        else:
            if not torch.cuda.is_available(): cfg.MODEL.DEVICE = 'cpu'
        
        cfg.merge_from_file(cfg_file_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.parameters['THRESHOLD']
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
        predictor = DefaultPredictor(cfg)
        return predictor

    def visualize_predictions(self, image, outputs, metadata = None, instance_mode = ColorMode.IMAGE):
        v = Visualizer(image[:, :, ::-1],
                        metadata = metadata,
                        scale=0.75,
                        instance_mode = instance_mode)
        out = v.draw_instance_predictions(outputs)

        plt.tick_params(left = False, right  = False,
                        labelleft = False, labelbottom = False)
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()
        return out.get_image()[:, :, ::-1]

    #saves visualised prediction images
    def save_prediction_images(self, inp_dataset = None, save_path = None, outputs = None, sem_seg_color = None, is_seg_color = False):
        assert save_path, 'Provide save_path to save the images'
        assert inp_dataset, 'Provide inp_dataset'
        assert outputs, 'Provide outputs'
        for i, output in outputs.items():
            image = cv2.imread(inp_dataset.dataset_images_path + '/' + inp_dataset.get_image_name(i))
            if is_seg_color:
                metadata = inp_dataset.dataset_metadata
                metadata.get('thing_colors') = [sem_seg_color]
                image_result = self.visualize_predictions(image = image,
                                                    outputs = output['instances'].to('cpu'),
                                                    metadata = inp_dataset.dataset_metdata,
                                                    instance_mode = ColorMode.SEGMENTATION)
            image_result = self.visualize_predictions(image, output['instances'].to('cpu'))
            cv2.imwrite(save_path + '/' + inp_dataset.get_image_name(i), image_result)

    def get_predictions_of_all_images(self, inp_dataset = None, visualize = False):
        assert inp_dataset, 'Provide inp_dataset'
        start = time.time()
        image_id_list = inp_dataset.image_id_list
        outputs_in_Instance_class = {}
        count = 1
        for image_name, image_id in image_id_list.items():
            d = {}
            if (count % 50 == 0) or (count == len(image_id_list)) or (count == 1):
                string_a = str(count) + ' image ' if count == 1 else str(count) + ' images '
                print('Predictions for ' + string_a + 'made')
            image = cv2.imread(inp_dataset.dataset_images_path + '/' + image_name)
            o = self.predictor(image)
            
            if not o:
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
                o = {'instances': obj}

            d = o['instances'].to('cpu')
            outputs_in_Instance_class[image_id] = {'instances': d}
            if visualize: image_result = self.visualize_predictions(image, o)
            count += 1
        print(f'Got all pedictions in {time.time() - start}')
        return outputs_in_Instance_class