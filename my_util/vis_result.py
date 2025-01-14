import sys
sys.path.append('/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/mmcv-master')
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import json
import os
import numpy as np
import cv2


def load_file(path):
    anno = {}
    for i in range(len(path)):
        f = json.load(open(path[i], 'r'))
        for info in f:
            im_name = info['name']
            label = info['defect_name']
            bbox = info['bbox']
            if im_name not in anno.keys():
                anno[im_name] = [[bbox, label]]
            else:
                anno[im_name].append([bbox, label])

    return anno

def load_test_file(path):
    anno = {}
    for i in range(len(path)):
        f = json.load(open(path[i], 'r'))
        for info in f:
            im_name = info['name']
            label = info['category']
            bbox = info['bbox']
            score = info['score']
            if im_name not in anno.keys():
                anno[im_name] = [[bbox, label,score]]
            else:
                anno[im_name].append([bbox, label,score])
    return anno

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def two_size(data):
    data = str(data)
    data = data.split('.')
    new_data = data[0]+'.'+ data[1][:2]
    return float(new_data)

def vis():
    cfg = [mmcv.Config.fromfile(config_) for config_ in config]
    for cfg_ in cfg:
        cfg_.model.pretrained = None

    # construct the model and load checkpoint
    model = [build_detector(cfg_.model, test_cfg=cfg_.test_cfg) for cfg_ in cfg]
    _ = [load_checkpoint(model[i], model_path[i]) for i in range(len(model))]
    # test a single image
    imgs = os.listdir(pic_path)
    from tqdm import tqdm
    for im in tqdm(imgs):
        img = pic_path+im
        img = mmcv.imread(img)
        for i in range(len(model)):
            result = inference_detector(model[i], img, cfg[i])
            re,image = show_result(img,result,dataset='cloths',show = False,score_thr = 0.5)
            if im in anno.keys():
                bboxes = anno[im]
                for bbox in bboxes:
                    box = bbox[0]
                    label = bbox[1]
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # cv2.putText(image, '%d' % cat[label], (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
                    cv2.putText(image, '%d' % label, (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
            name = config[i].split('_')[-1].split('.')[0]
            cv2.namedWindow(str(name),0)
            cv2.resizeWindow(str(name),1920,1080)
            cv2.imshow(str(name),image)
        cv2.waitKey(0)



def result():
    cfg = mmcv.Config.fromfile(config2make_json)
    cfg.model.pretrained = None

    # construct the model and load checkpoint
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, model2make_json)
    # test a single image
    imgs = os.listdir(pic_path)
    meta = []
    from tqdm import tqdm
    for im in tqdm(imgs):
        img = pic_path + im
        img = mmcv.imread(img)
        result = inference_detector(model, img)
        re,img = show_result(img, result, dataset='cloths', show=False,score_thr = 0.5)
        if len(re):
            for box in re:
                anno = {}
                anno['name'] = im
                anno['category'] = int(box[5])
                anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
                anno['score'] = float(box[4])
                meta.append(anno)
    with open(json_path, 'w') as fp:
        json.dump(meta, fp, cls=MyEncoder,indent=4, separators=(',', ': '))

if __name__ == "__main__":
    cat = {
        '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
        '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
        '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20, '云织': 20,
        '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
    }

    json_file = [
        "/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train2_20190828/Annotations/anno_train.json",
        "/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train1_20190818/Annotations/anno_train.json",
        '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/Aug_json.json']

    json_test_file = [
        '/home/remo/Desktop/cloth_flaw_detection/Results/result_niu.json',
    ]


    # pic_path = '/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_testA_20190818/'
    # pic_path = '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/Train_image/'
    # pic_path = '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/Val_image/'
    pic_path = '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/Val_balance_image/'

    config = [
              # '/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/cloth_config/cloth_faster_rcnn_r101_fpn_1x_4.py',
              # '/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/cloth_config/cloth_faster_rcnn_r101_fpn_1x_10.py',
              # '/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/cloth_config/cloth_faster_rcnn_r101_fpn_1x_10_test.py',
              # '/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/cloth_config/cloth_faster_rcnn_r101_fpn_1x_9.py',
              # '/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/cloth_config/cloth_faster_rcnn_r101_fpn_1x_10.py',
              # '/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/cloth_config/cloth_faster_rcnn_r101_fpn_1x_10.py',
              '/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/cloth_config/cloth_faster_rcnn_r101_fpn_1x_10_alldata_50epoch.py',
              '/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/cloth_config/cloth_faster_rcnn_r101_fpn_1x_24.py',
              # '/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/cloth_config/cloth_faster_rcnn_r50_fpn_1x_17.py',
              # '/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/cloth_config/cloth_faster_rcnn_r101_fpn_1x_15.py',
              # '/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/cloth_config/cloth_faster_rcnn_r101_fpn_1x_10_alldata.py',
              # '/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/cloth_config/cloth_faster_rcnn_r101_fpn_1x_13.py',
              # '/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/cloth_config/cascade_rcnn_r101_fpn_1x_1.py',
              # '/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/cloth_config/cloth_faster_rcnn_r101_fpn_1x_9_iou03.py',
              # '/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/cloth_config/cloth_faster_rcnn_r101_fpn_1x_8_big.py',
              # './cloth_config/cloth_faster_rcnn_r101_fpn_1x_4_test.py',
              # './cloth_config/cloth_faster_rcnn_r101_fpn_1x_5.py'
              ]
    model_path = [
                  # '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_4/latest.pth',
                  # '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_10/latest.pth',
                  # '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_10/latest.pth',
                  # '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_10/latest.pth',
                  # '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_15/latest.pth',
                  '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_10_alldata_50epoch/latest.pth',
                  '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_24/latest.pth',
                  # '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r50_fpn_1x_17/latest.pth',
                  # '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_12/latest.pth',
                  # '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_13/latest.pth',
                  # '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/cascade_rcnn_r101_fpn_1x_1/latest.pth',
                  # '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_9/latest.pth',
                  # '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_8/latest.pth',
                  # '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_4/latest.pth',
                  # '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_5/latest.pth'
                  ]

    model2make_json = "/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_10_traindata/latest.pth"
    config2make_json = '/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/cloth_config/cloth_faster_rcnn_r101_fpn_1x_10_traindata.py'
    # json_path = '/home/remo/Desktop/cloth_flaw_detection/Results/result_10_traindata.json'
    json_path = '/home/remo/Desktop/cloth_flaw_detection/Results/result_10_traindata.json'

    anno = {}
    # result()
    # anno = load_file(json_file)
    anno = load_test_file(json_test_file)
    vis()


