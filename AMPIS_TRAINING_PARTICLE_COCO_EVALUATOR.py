import pycocotools
import torch,torchvision
import detectron2
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
setup_logger()
import detectron2.utils.events
import tensorflow as tf
import tensorboard

import ampis
import schedule

# LOADING DATA AND MODEL TRAINING (MODULE IMPORTS)

import cv2
import numpy as np
import os
from pathlib import Path
import pickle
import sys
import datetime

## detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog, build_detection_test_loader
)
from detectron2.engine import DefaultTrainer, DefaultPredictor

from ampis import data_utils, visualize

# LOADING DATA

EXPERIMENT_NAME = 'particle' # can be particle or satellite
root = Path('examples','powder') # path to folder containing labels
json_path_train = Path(root, 'data','via_2.0.8/', f'via_powder_{EXPERIMENT_NAME}_masks_training.json')  # path to training data
json_path_val = Path(root,'data','via_2.0.8/', f'via_powder_{EXPERIMENT_NAME}_masks_validation.json')  # path to validation data

assert json_path_train.is_file(), 'training file not found!'
assert json_path_val.is_file(), 'validation file not found!'

# REGISTRATION
DatasetCatalog.clear()  # resets catalog, helps prevent errors from running cells multiple times

# store names of datasets that will be registered for easier access later
dataset_train = f'{EXPERIMENT_NAME}_Train'
dataset_valid = f'{EXPERIMENT_NAME}_Val'

# register the training dataset
DatasetCatalog.register(dataset_train,
                        lambda f = json_path_train: data_utils.get_ddicts(label_fmt='via2',  # annotations generated from vgg image annotator
                                                                          im_root=f,  # path to the training data json file
                                                                          dataset_class='Train'))  # indicates this is training data

# register the validation dataset. Same exact process as above
DatasetCatalog.register(dataset_valid,
                        lambda f = json_path_val: data_utils.get_ddicts(label_fmt='via2',  # annotations generated from vgg image annotator
                                                                        im_root=f,  # path to validation data json file
                                                                        dataset_class='Validation'))  # indicates this is validation data
print(f'Registered Datasets: {list(DatasetCatalog.data.keys())}')

## There is also a metadata catalog, which stores the class names.
for d in [dataset_train, dataset_valid]:
    MetadataCatalog.get(d).set(**{'thing_classes': [EXPERIMENT_NAME]})

## VISUALIZE TRAINING IMAGES
np.random.seed(42960)
for i in np.random.choice(DatasetCatalog.get(dataset_train), 3, replace=False):
    visualize.display_ddicts(i, None, dataset_train, suppress_labels=True)

## VISUALIZE VALIDATION IMAGES
for i in DatasetCatalog.get(dataset_valid):
    visualize.display_ddicts(i, None, dataset_valid, suppress_labels=True)

## MODEL CONFIGURATION

cfg = get_cfg() # initialize cfg object
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))  # load default parameters for Mask R-CNN # ResNet50 feature extractor
#cfg.MODEL.ROI_HEADDS.SCORE_THRESH_TEST = 0.7 # Set threshold for this model
cfg.INPUT.MASK_FORMAT = 'polygon'  # masks generated in VGG image annotator are polygons
cfg.DATASETS.TRAIN = (dataset_train,)  # dataset used for training model
cfg.DATASETS.TEST = (dataset_train, dataset_valid)  # we will look at the predictions on both sets after training
cfg.SOLVER.IMS_PER_BATCH = 1 # number of images per batch (across all machines)
cfg.SOLVER.CHECKPOINT_PERIOD = 400# number of iterations after which to save model checkpoints
cfg.MODEL.DEVICE='cpu'  # 'cpu' to force model to run on cpu, 'cuda' if you have a compatible gpu
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # Since we are training separate models for particles and satellites there is only one class output
cfg.TEST.DETECTIONS_PER_IMAGE = 400 if EXPERIMENT_NAME == 'particle' else 150  # maximum number of instances that can be detected in an image (this is fixed in mask r-cnn)
cfg.SOLVER.MAX_ITER = 150  # maximum number of iterations to run during training
# Increasing this may improve the training results, but will take longer to run (especially without a gpu!)


cfg.TEST.EVAL_PERIOD = 50
#detectron2.utils.events.TensorboardXWriter('logs')







#cfg.SOLVER.BASE_LR = 0.0025 #Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones. Such decay can happen simultaneoulsy with other changes to the learning rate from outside this scheduler.
#cfg.SOLVER.STEPS = [] #do not decay learning rate

# model weights will be downloaded if they are not present
weights_path = Path('models','model_final_f10217.pkl')
if weights_path.is_file():
    print('Using locally stored weights: {}'.format(weights_path))
else:
    weights_path = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    print('Weights not found, weights will be downloaded from source: {}'.format(weights_path))
cfg.MODEL.WEIGHTs = str(weights_path)
cfg.OUTPUT_DIR = str(Path(f'{EXPERIMENT_NAME}_output'))
# make the output directory
os.makedirs(Path(cfg.OUTPUT_DIR), exist_ok=True)




## TRAINING OF THE MODEL

# note this cell generates a huge wall of text
trainer = DefaultTrainer(cfg)  # create trainer object from cfg



print('Está entrenando no parado')

trainer.resume_or_load(resume=False)  # start training from iteration 0
print("AQUÍ EMPIEZA A ENTRENAR")
trainer.train()  # train the model!

print('EL ENTRENAMIENTO LLEGA HASTA AQUÍ')
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H: %H:%M:%S"))

## VISUALIZING MODEL PREDICTIONS

# load the weights of the model we want to use
model_checkpoints = sorted(Path(cfg.OUTPUT_DIR).glob('*.pth'))  # paths to weights saved during training
cfg.DATASETS.TEST = (dataset_train, dataset_valid)  # predictor requires this field to not be empty
cfg.MODEL.WEIGHTS = str(model_checkpoints[-1])  # use the last model checkpoint saved during training. If you want to see the performance of other checkpoints you can select a different index from model_checkpoints.
predictor = DefaultPredictor(cfg)  # create predictor object


evaluator = COCOEvaluator('particle_Val',output_dir=str(Path(f'{EXPERIMENT_NAME}_AP_Eval_output')))
val_loader =  build_detection_test_loader(cfg,'particle_Val')
result = inference_on_dataset(trainer.model,val_loader,evaluator)

print("SE IMPRIME RESULT_VAL (LARGE DATASETS)")
print(result)


## FOR A SINGLE IMAGE


img_path = Path(root, 'data','images_png','Sc1Tile_001-005-000_0-000.png')
img = cv2.imread(str(img_path))
outs = predictor(img)
data_utils.format_outputs(img_path, dataset='test', pred=outs)
visualize.display_ddicts(ddict=outs,  # predictions to display
                                 outpath=None, dataset='Test',  # don't save figure
                                 gt=False,  # specifies format as model predictions
                                img_path=img_path)  # path to image


## IMAGEN APARTE
print("CARGAMOS IMAGEN NUESTRA 1")
img_path = Path('PRUEBA_LABELS_png/4.png')
img = cv2.imread(str(img_path))
outs = predictor(img)
data_utils.format_outputs(img_path, dataset='test', pred=outs)
visualize.display_ddicts(ddict=outs,  # predictions to display
                                 outpath=None, dataset='Test',  # don't save figure
                                 gt=False,  # specifies format as model predictions
                                img_path=img_path)  # path to image

print("CARGAMOS IMAGEN NUESTRA 2")
img_path = Path('PRUEBA_LABELS_png/9.png')
img = cv2.imread(str(img_path))
outs = predictor(img)
data_utils.format_outputs(img_path, dataset='test', pred=outs)
visualize.display_ddicts(ddict=outs,  # predictions to display
                                 outpath=None, dataset='Test',  # don't save figure
                                 gt=False,  # specifies format as model predictions
                                img_path=img_path)  # path to image



## FOR ALL IMAGES IN TRAIN AND VALIDATION SETS AND SAVE IN DISK

results = []
for ds in cfg.DATASETS.TEST:
    print(f'Dataset: {ds}')
    for dd in DatasetCatalog.get(ds):
        print(f'\tFile: {dd["file_name"]}')
        img = cv2.imread(dd['file_name'])  # load image
        outs = predictor(img)  # run inference on image

        # format results for visualization and store for later
        # note the use of format_outputs(), which ensures that the data is stored correctly for later
        results.append(data_utils.format_outputs(dd['file_name'], ds, outs))

        # visualize results
        visualize.display_ddicts(outs, None, ds, gt=False, img_path=dd['file_name'])


# save to disk
prediction_save_path = Path(f'{EXPERIMENT_NAME}-results.pickle')
with open(prediction_save_path, 'wb') as f:
    pickle.dump(results, f)

print('SE HAN GUARDADO LOS DATOS!!')


