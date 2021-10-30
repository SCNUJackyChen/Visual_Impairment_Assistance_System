# %cd /content/InsightFace_Pytorch
import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank


conf = get_config(False)
mtcnn = MTCNN()
print('mtcnn loaded')

learner = face_learner(conf, True)
learner.threshold = conf.threshold
if conf.device.type == 'cpu':
    learner.load_state(conf, 'mobilefacenet.pth', True, True)
else:
    learner.load_state(conf, 'final.pth', True, True)
learner.model.eval()
print('learner loaded')

img = Image.open('123.jpg')
img = mtcnn.align(img)

enc = learner.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
print(enc)