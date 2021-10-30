import cv2
from PIL import Image
import numpy as np
import os
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
from datetime import datetime

from sound import PorcupineDemo
import pvporcupine
import soundfile
from pvrecorder import PvRecorder

from pathlib import Path
data_path = Path('data')
save_path = data_path/'facebank'/'unk'
if not save_path.exists():
    save_path.mkdir()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    parser.add_argument(
        '--keywords',
        nargs='+',
        help='List of default keywords for detection. Available keywords: %s' % ', '.join(sorted(pvporcupine.KEYWORDS)),
        choices=sorted(pvporcupine.KEYWORDS),
        metavar='')
    parser.add_argument(
        '--keyword_paths',
        nargs='+',
        help="Absolute paths to keyword model files. If not set it will be populated from `--keywords` argument",
        default=
        ['sdk/detect__en_windows_2021-11-08-utc_v1_9_0.ppn', 
        'sdk/finish__en_windows_2021-11-08-utc_v1_9_0.ppn'])

    parser.add_argument('--library_path', help='Absolute path to dynamic library.', default=pvporcupine.LIBRARY_PATH)

    parser.add_argument(
        '--model_path',
        help='Absolute path to the file containing model parameters.',
        default=pvporcupine.MODEL_PATH)

    parser.add_argument(
        '--sensitivities',
        nargs='+',
        help="Sensitivities for detecting keywords. Each value should be a number within [0, 1]. A higher " +
             "sensitivity results in fewer misses at the cost of increasing the false alarm rate. If not set 0.5 " +
             "will be used.",
        type=float,
        default=None)

    parser.add_argument('--audio_device_index', help='Index of input audio device.', type=int, default=-1)

    parser.add_argument('--output_path', help='Absolute path to recorded audio for debugging.', default=None)

    parser.add_argument('--show_audio_devices', action='store_true')



    args = parser.parse_args()

    
    if args.keyword_paths is None:
        if args.keywords is None:
            raise ValueError("Either `--keywords` or `--keyword_paths` must be set.")

        keyword_paths = [pvporcupine.KEYWORD_PATHS[x] for x in args.keywords]
    else:
        keyword_paths = args.keyword_paths

    if args.sensitivities is None:
        args.sensitivities = [0.5] * len(keyword_paths)

    if len(keyword_paths) != len(args.sensitivities):
        raise ValueError('Number of keywords does not match the number of sensitivities.')

    # wake = PorcupineDemo(
    #     library_path=args.library_path,
    #     model_path=args.model_path,
    #     keyword_paths=keyword_paths,
    #     sensitivities=args.sensitivities,
    #     output_path=args.output_path,
    #     input_device_index=args.audio_device_index)

    # wake.start()


    
    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'ir_se50.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    
    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    # inital camera
    cap = cv2.VideoCapture(0)
    cap.set(3,1280) # 1280 640
    cap.set(4,720) # 720  480
    cap.set(5,30)
    if args.save:
        video_writer = cv2.VideoWriter(conf.data_path/'recording.avi', cv2.VideoWriter_fourcc(*'XVID'), 6, (1280,720))
        # frame rate 6 due to my laptop is quite slow...

    THRESHOLD = 30000
    area = lambda x1, y1, x2, y2: abs(x1 - x2) * abs(y1 - y2)
    is_snap = False
    is_update = False
    # detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    while cap.isOpened():
        
        isSuccess,frame = cap.read()
        if isSuccess:            
            # print(wake.keyword)
            # if wake.keyword == 'detect ':
            try:
                # image = Image.fromarray(frame)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                c1, c2, c3 = [clahe.apply(frame[:,:,i]) for i in range(3)]
                # c1, c2, c3 = [cv2.equalizeHist(frame[:,:,i]) for i in range(3)]
                frame = cv2.merge((c1,c2,c3))
                image = Image.fromarray(cv2.merge((c1,c2,c3)))
                bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice 

                # bboxes = detector.detectMultiScale(cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                # faces = []
                # for idx, bbox in enumerate(bboxes):
                #     (x, y, w, h) = bbox
                #     bboxes[idx] = [x, y, x + w, y + h]
                #     bboxes[idx] = bboxes[idx] + [-20,-20,20,20]
                #     faces.append(Image.fromarray(frame[x:x+w, y:y+h]).resize((112,112)))
                #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # # print(bboxes)
                # print(faces)

                # if not has_snap:
                results, score = learner.infer(conf, faces, targets, args.tta)

                #     unk_ids = [i for i, x in enumerate(results.tolist()) if x == -1] # unknown id
                #     unk_id_areas = {i : area(*bboxes[i].tolist())  for i in unk_ids }
                #     nearest_unk_id = max(unk_id_areas, key = lambda x: unk_id_areas[x])
                #     print(unk_id_areas[nearest_unk_id])
                #     if (unk_id_areas[nearest_unk_id] >= THRESHOLD):
                #         warped_face = np.array(faces[nearest_unk_id])
                #         cv2.imwrite(str(save_path/'{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-"))), warped_face)
                #         has_snap = True

                

                for idx,bbox in enumerate(bboxes):
                    if args.score:
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    else:
                        frame = draw_box_name(bbox, names[results[idx] + 1], frame)
            except Exception as e:
                # import traceback
                # traceback.print_exc()
                print(e)
                # print('detect error')    
            # elif wake.keyword == 'finish ':
            #     if not is_update:
            #         # auto-speaker
            #         # say name
            #         # change file name
            #         name = 'jacky'
            #         if name in os.listdir('D:/NUS/project/ITSS/InsightFace_Pytorch/data/facebank/'):
            #             os.removedirs('D:/NUS/project/ITSS/InsightFace_Pytorch/data/facebank/'+name)
            #         os.rename('D:/NUS/project/ITSS/InsightFace_Pytorch/data/facebank/unk/', 'D:/NUS/project/ITSS/InsightFace_Pytorch/data/facebank/'+name)
            #         targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
            #         is_update = True

            cv2.imshow('face Capture', frame)

        if args.save:
            video_writer.write(frame)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break
        
    cap.release()
    if args.save:
        video_writer.release()
    cv2.destroyAllWindows()    