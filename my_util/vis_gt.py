# encoding:utf/8
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import os
import json
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import xml.dom.minidom
from xml.dom.minidom import Document
from tqdm import tqdm
from easydict import EasyDict as edict
import os.path as osp
import math
from tqdm import tqdm

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio
import getpass  # 获取用户名
import random

USER = getpass.getuser()


class Config:
    def __init__(self):
        self.json_paths = ['']  # train json
        self.val_json_paths = ['']     # val json

        self.allimg_path = ''   # 训练图片集
        self.val_img_path = ''  # 验证图片集
        self.add_num = 0        # add_aug_data 扩增数据数量

        self.result_json = '' # 模型对val 的输出结果
        self.divide_json  = ['']

class DataAnalyze:
    '''
    bbox 分析类，
        1. 每一类的bbox 尺寸统计
        2.
    '''

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.category = {
            '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
            '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
            '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20, '云织': 20,
            '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
        }
        self.reverse_category = {
            1:'破洞', 2:'水渍',3: '三丝', 4:'结头', 5:'花板跳', 6:'百脚', 7:'毛粒',
            8:'粗经', 9:'松经', 10:'断经', 11:'吊经', 12:'粗维', 13:'纬缩', 14:'浆斑', 15:'整经结', 16:'星跳',
            17:'断氨纶', 18:'稀密档', 19:'磨痕', 20:'死皱'
        }
        self.num_classes = 20  # 前景类别

        self.all_instance, self.cla_instance, self.img_instance = self._create_data_dict(cfg.json_paths, cfg.allimg_path)
        # if hasattr(cfg, 'val_json_paths') and not  cfg.val_json_paths == '' :
        #     self.val_all_instance, self.val_cla_instance, self.val_img_instance = self._create_data_dict(cfg.val_json_paths, cfg.val_img_path)
        # else:
        #     self.val_all_instance, self.val_cla_instance, self.val_img_instance = (None, None, None)

        '''
        all_instance 
        [
            {'bbox': [2000.66, 326.38, 2029.87, 355.59],
             'defect_name': '结头',
             'name': 'd6718a7129af0ecf0827157752.jpg',
             'abs_path' : 'xxx/xxx.jpg',
             'classes': 1
             'w':1,
             'h':1,
             'area':1,
             }
        ]

        cla_instance 
            {'1':[], '2':[] }
        '''

        self.num_data = len(self.all_instance)

    def _create_data_dict(self, json_path, data_file, flag_ins_list=False):
        '''
        flag_ins_list: True 传入的json_path 为 [instance1, instance2, ] 不用json 文件读取
        :return:
            instance:
                {'bbox': [2000.66, 326.38, 2029.87, 355.59],
                 'defect_name': '结头',
                 'name': 'd6718a7129af0ecf0827157752.jpg',
                 'abs_path' : 'xxx/xxx.jpg',
                 'w':1,
                 'h':1,
                 'area':1,
                 'im_w':1
                 'im_h':2
                 }

        all_instance
            [instance1, instance2, instance3]

        cla_instance
            {'1':[instance, instances2], '2'[instance, ]}

        img_instance
            {'xx1.jpg': [instance]  'xxx.jpg':[instance, instance]}

        '''
        if flag_ins_list:
            json_path = [json_path ]# 为 [ [ins1, ins2] ] 2维数组

        all_instance = []
        key_classes = list(range(1, self.num_classes + 1))  # 1 ... num_classes

        cla_instance = edict({str(k): [] for k in key_classes})  # key 必须是字符串
        img_instance = edict()
        if isinstance(json_path, str):
            json_path = [json_path]

        if isinstance(json_path, list):
            for path in json_path:
                if flag_ins_list:
                    gt_list = path
                else:
                    gt_list = json.load(open(path, 'r'))

                for instance in gt_list:
                    instance = edict(instance)
                    instance.classes = int(self.category[instance.defect_name])  # add classes int
                    w, h = compute_wh(instance.bbox)
                    instance.w = round(w, 2)  # add w
                    instance.h = round(h, 2)  # add h
                    instance.area = round(w * h, 2)  # add area
                    instance.abs_path = osp.join(data_file, instance.name)  # add 绝对路径
                    all_instance.append(instance)  # 所有instance

                    cla_instance[str(instance.classes)].append(instance)  # 每类的instance

                    if instance.name not in img_instance.keys():  # 每张图片的instance
                        img_instance[instance.name] = [instance]
                    else:
                        img_instance[instance.name].append(instance)

        return all_instance, cla_instance, img_instance

    # def load_coco_format(self):
    #     all_instance = []
    #     key_classes = list(range(1, self.num_classes + 1))  # 1 ... num_classes
    #
    #     cla_instance = edict({str(k): [] for k in key_classes})  # key 必须是字符串
    #     img_instance = edict()
    #     if isinstance(self.cfg.json_paths, str):
    #         self.cfg.json_paths = [self.cfg.json_paths]
    #     if isinstance(self.cfg.json_paths, list):
    #         for path in self.cfg.json_paths:
    #             gt_list = json.load(open(path, 'r'))

    def ana_classes(self):
        ws_all = []
        hs_all = []
        for cla_name, bboxes_list in self.cla_instance.items():  # str '1': [edict()]
            ws = []
            hs = []
            for instance in bboxes_list:
                ws.append(instance.w)
                hs.append(instance.h)
                ws_all.append(instance.w)
                hs_all.append(instance.h)
            # plt.title(cla_name, fontsize='large',fontweight = 'bold')
            # plt.scatter(ws, hs, marker='x', label=cla_name, s=30)
        plt.scatter(ws_all, hs_all, marker='x', s=30)

        plt.grid(True)
        plt.show()

    def draw_cls(self):
        myfont = matplotlib.font_manager.FontProperties(fname='/home/remo/Desktop/simkai_downcc/simkai.ttf')
        cls = [i for i in range(1,self.num_classes+1)]
        cls_each = [len(self.cla_instance[str(i)]) for i in range(1,self.num_classes+1)]
        plt.xticks(range(1, len(cls) + 1), cls, font_properties=myfont, rotation=0)
        plt.bar(cls, cls_each, color='rgb')
        plt.legend()
        plt.grid(True)
        plt.show()

    def add_aug_data(self, add_num=500, aug_save_path=None, json_file_path=None):
        '''
        1. 设定补充的数据量
        2. 低于这些类的才需要补充
        3. 补充增广函数
            1. 每张图片增广多少张
        :return:
        '''
        if aug_save_path is None or json_file_path is None:
            raise NameError

        if not osp.exists(aug_save_path):
            os.makedirs(aug_save_path)

        transformer = Transformer()

        aug_json_list = []

        for cla_name, bboxes_list in self.cla_instance.items():  # str '1': [edict()]
            cla_num = len(bboxes_list)
            # 按需增广
            # if cla_num >= add_num:
            #     continue
            # 补充数据
            # cla_add_num = add_num - cla_num  #

            # 每张图进行增广
            cla_add_num = cla_num

            each_num = int(math.ceil(cla_add_num / cla_num * 1.0))  # 向上取整 每张图片都进行增广
            # 每张图进行增广扩充
            for instance in tqdm(bboxes_list, desc='cla %s ;process %d: ' % (cla_name, each_num)):  # img_box edict
                # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area

                img = cv2.imread(instance.abs_path)
                try:  # 检测图片是否可用
                    h, w, c = img.shape
                    img_info = edict({'img_h':h, "img_w":w})
                except:
                    print("%s is wrong " % instance.abs_path)
                import copy
                for i in range(each_num):  # 循环多次进行增广保存
                    img_ins = copy.deepcopy(self.img_instance[instance.name])
                    aug_img, img_info_tmp = transformer.aug_img(img, img_ins, img_info = img_info) # list
                    if img_info_tmp is not None:
                        aug_name = '%s_aug%d.jpg' % (
                            osp.splitext(instance.name)[0], i) # 6598413.jpg -> 6598413_aug0.jpg, 6598413_aug1.jpg
                        aug_abs_path = osp.join(aug_save_path, aug_name)
                        for ins in img_info_tmp:
                            ins.name = aug_name
                            ins.abs_path = aug_abs_path
                            aug_json_list.append(ins)

                        cv2.imwrite(aug_abs_path, aug_img)
                    else:
                        continue
        #
        # # 保存aug_json 文件
        random.shuffle(aug_json_list)
        with open(json_file_path, 'w') as f:
            json.dump(aug_json_list, f, indent=4, separators=(',', ': '), cls=MyEncoder)

    def vis_gt(self, flag_show_raw_img=False):
        '''
        可视化gt， 按键盘d 向前， 按a 向后
        :return:
        '''
        transformer = Transformer()
        cur_node = 0
        set_img_name = list(self.img_instance.keys())  # 所有图片的名称的 list
        while True:
            img_name = set_img_name[cur_node]
            instances_list = self.img_instance[img_name]
            # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area

            cv2.namedWindow('img', 0)
            cv2.resizeWindow('img', 1920, 1080)
            print('num gt: ', len(instances_list))
            print('img_name: %s ' % (img_name))

            ins_init = instances_list[0]
            img = cv2.imread(ins_init.abs_path)
            img_aug, _ = transformer(img, ins_init)
            for instance in instances_list:
                box = instance.bbox
                cv2.rectangle(img_aug, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(img_aug, '%d' % instance.classes, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (0, 255, 0), 1)

            k = cv2.waitKey(0)
            if k == ord('d'):
                cur_node += 1
            if k == ord('a'):
                cur_node -= 1
                if cur_node <= 0:
                    cur_node = 0
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            cv2.imshow('img', img_aug)
            if flag_show_raw_img:
                cv2.imshow('raw', img)

    def visRes_gt(self, gt_img_ins, res_img_ins, gtline=0.8,resline=0.8):
        '''
        1. 可视化gt 和 result 效果

        :param gt_img_ins: gt 的img_instance
        :param res_img_ins:  result 的img_instance
        :return:
        '''
        empty_ins = [
            edict(
                {'abs_path': '',
                 'area': 1,
                 'bbox': [0,0,0,0],
                 'classes': -1,
                 'defect_name': '',
                 'h': 0,
                 'name': '',
                 'w': 0,
                 'score':0}
            )
        ]

        cur_node = 0
        set_img_name = list(gt_img_ins.keys())  # 所有图片的名称的 list
        while True:
            img_name = set_img_name[cur_node]
            gt_ins_list = gt_img_ins[img_name] # gt instance 列表
            if img_name in res_img_ins.keys():
                res_ins_list = res_img_ins[img_name] # result
                res_num = len(res_ins_list)
            else:
                res_ins_list = empty_ins
                res_num = 0
            # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area

            cv2.namedWindow('img', 0)
            cv2.resizeWindow('img', 1920, 1080)


            ins_init = gt_ins_list[0]
            img = cv2.imread(ins_init.abs_path)

            for gt_ins in gt_ins_list :
                gt_box = gt_ins.bbox

                # 绘制gt
                cv2.rectangle(img, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (0, 0, 255), 2)
                cv2.putText(img, '%d' % gt_ins.classes, (int(gt_box[0]), int(gt_box[1])), cv2.FONT_HERSHEY_COMPLEX,gtline,
                            (0, 0, 255), 1)

            for res_ins in res_ins_list:
                res_box = res_ins.bbox

                # 绘制result
                cv2.rectangle(img, (int(res_box[0]), int(res_box[1])), (int(res_box[2]), int(res_box[3])), (0, 255, 0), 2)
                cv2.putText(img, '%d %0.3f' % (res_ins.classes, res_ins.score), (int(res_box[2]), int(res_box[1])), cv2.FONT_HERSHEY_COMPLEX, resline,
                            (0, 255, 0), 1)

            k = cv2.waitKey(0)
            if k == ord('d'):
                cur_node += 1
            if k == ord('a'):
                cur_node -= 1
                if cur_node <= 0:
                    cur_node = 0
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            print('img_name: %s ' % (img_name))
            print('num gt : ', len(gt_ins_list))
            print('num res: ', res_num)
            cv2.imshow('img', img)
        pass

    def val_analyze(self):
        '''
        1. val 作为gt的 instance
        2. 模型在val 上的输出 instace
        3. 可视化比较
        4. 指标比较
        :return:q
        '''
        if not (hasattr(self.cfg, 'result_json') and self.cfg.result_json != ''):
            raise(" no result_json ")

        res_ins_list = self.load_res_json(self.cfg.result_json)
        valr_all_instance, valr_cla_instance, valr_img_instance = self._create_data_dict(res_ins_list, self.cfg.val_img_path, flag_ins_list=True)
        self.val_all_instance, self.val_cla_instance, self.val_img_instance = self.load_coco_format(self.cfg.val_json_paths,
                                                                                                    self.cfg.val_img_path)
        self.visRes_gt(self.val_img_instance, valr_img_instance)

    def load_res_json(self, path):
        '''
        结果json 转换为 raw json 方式，
        增加 defect_name
        :param path:
        :return:
        '''

        raw_ins_list = []
        for i in range(len(path)):
            ins_list = json.load(open(path[i], 'r'))
            for instance in ins_list:
                instance = edict(instance)
                instance.defect_name = self.reverse_category[instance.category]

                raw_ins_list.append(instance)

        return raw_ins_list

    def load_coco_format(self, json_path, data_file):
        all_instance = []
        key_classes = list(range(1, self.num_classes + 1))  # 1 ... num_classes

        cla_instance = edict({str(k): [] for k in key_classes})  # key 必须是字符串
        img_instance = edict()
        if isinstance(json_path, str):
            json_path = [json_path]

        if isinstance(json_path, list):
            for path in json_path:
                gt_dict = json.load(open(path, 'r'))
                im_anno = edict()
                bbox_anno = edict()
                for anno in gt_dict['images']:
                    anno = edict(anno)
                    im_name = anno['file_name']
                    image_id = anno['id']
                    width = anno['width']
                    height = anno['height']
                    if str(image_id) not in im_anno.keys():
                        im_anno[str(image_id)] = {'im_name':im_name,'width':width,'height':height}
                for anno in gt_dict['annotations']:
                    image_id = anno['image_id']
                    category_id = anno['category_id']
                    bbox = anno['bbox']
                    bbox[2] = bbox[0]+bbox[2]
                    bbox[3] = bbox[1]+bbox[3]
                    area = anno['area']
                    if str(image_id) not in bbox_anno.keys():
                        bbox_anno[str(image_id)] = [{'category_id':category_id,'bbox':bbox,'area':area}]
                    else:
                        bbox_anno[str(image_id)].append({'category_id': category_id, 'bbox': bbox, 'area': area})

                for image_id in im_anno.keys():
                    im_anno_temp = im_anno[image_id]
                    bbox_anno_temp = bbox_anno[image_id]
                    for bbox in bbox_anno_temp:
                        instance = edict()
                        instance.imgid = image_id
                        instance.name = im_anno_temp['im_name']
                        instance.im_w = im_anno_temp['width']
                        instance.im_h = im_anno_temp['height']
                        instance.area = bbox['area']
                        instance.bbox = bbox['bbox']
                        instance.w = bbox['bbox'][2] - bbox['bbox'][0]
                        instance.h = bbox['bbox'][3] - bbox['bbox'][1]
                        instance.classes = bbox['category_id']
                        instance.abs_path = osp.join(data_file, im_anno_temp['im_name'])  # add 绝对路径
                        instance.defect_name = self.reverse_category[bbox['category_id']]
                        all_instance.append(instance)
                        cla_instance[str(bbox['category_id'])].append(instance)  # 每类的instance

                        if instance.name not in img_instance.keys():  # 每张图片的instance
                            img_instance[instance.name] = [instance]
                        else:
                            img_instance[instance.name].append(instance)
                return all_instance, cla_instance, img_instance

    def divide_trainval(self, ratio=0.2, del_json=None, del_path=None, train_json='', val_json=''):
        import random

        train_ins_list = []
        val_ins_list = []
        if del_json is None:
            divide_jsons = self.cfg.divide_json
        if del_path is None:
            del_path = self.cfg.allimg_path

        if isinstance(divide_jsons, str):
            divide_jsons = [divide_jsons]
        if not isinstance(divide_jsons, list):
            raise ("divide_jsons error !!")

        # for divide_json in divide_jsons:
        all_instance, cla_instance, img_instance = self._create_data_dict(divide_jsons, del_path)
        all_ins_keys = set(img_instance.keys())
        num_ins = len(list(all_ins_keys))
        num_val = int(num_ins  * ratio)
        print("total num : " ,num_ins)
        print("val   num : " ,num_val)
        print("train num : " ,num_ins - num_val)

        val_ins_keys = random.sample(all_ins_keys, num_val)
        train_ins_keys = set(all_ins_keys) - set(val_ins_keys)
        for ins_key in all_ins_keys:
            if ins_key in val_ins_keys:
                val_ins_list += (img_instance[ins_key])
            elif ins_key in train_ins_keys:
                train_ins_list += (img_instance[ins_key])

        with open(train_json, 'w') as f:
            json.dump(train_ins_list, f, indent=4, separators=(',', ': '), cls=MyEncoder)
            print('train json save : ', train_json)
        with open(val_json, 'w') as f:
            json.dump(val_ins_list, f, indent=4, separators=(',', ': '), cls=MyEncoder)
            print('val json save : ', val_json)


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


class Transformer:
    def __init__(self):
        self.aug_img_seq = iaa.Sequential([
            # iaa.Fliplr(0.8),
            # iaa.Flipud(0.8),
            iaa.Invert(1.0),
            # iaa.Crop(percent=0.1)
        ], random_order=True)
        # pass

    def __call__(self, imgBGR, instance=None):
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

        # imgRGB = self.aug_img_seq.augment_images(imgRGB)
        imgBGR_aug = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)

        # save json format
        if instance is not None:

            img_info_tmp = edict()
            img_info_tmp.bbox = instance.bbox
            img_info_tmp.defect_name = instance.defect_name
            img_info_tmp.name = instance.name
            return imgBGR_aug, img_info_tmp
        else:
            return imgBGR_aug, None

    def aug_img(self, imgBGR, instance=None, img_info = None):
        bbs = self._mk_bbs(instance, img_info)
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        imgRGB_aug, bbs_aug = self.aug_img_seq(image = imgRGB, bounding_boxes = bbs)
        bbs_aug = bbs_aug.clip_out_of_image()
        imgBGR_aug = cv2.cvtColor(imgRGB_aug, cv2.COLOR_RGB2BGR)

        # # for debug to show
        # imgRGB_aug_with_box = bbs_aug.draw_on_image(imgRGB_aug,size = 2)
        # imgRGB_aug_with_box = cv2.cvtColor(imgRGB_aug_with_box, cv2.COLOR_RGB2BGR)
        # imgRGB_aug_with_box = cv2.resize(imgRGB_aug_with_box,(1333,800))
        # imgRGB_with_box = bbs.draw_on_image(imgRGB, size=2)
        # imgRGB_with_box = cv2.resize(imgRGB_with_box,(1333,800))
        # imgRGB_with_box = cv2.cvtColor(imgRGB_with_box, cv2.COLOR_RGB2BGR)
        # cv2.imshow('aug',imgRGB_aug_with_box)
        # cv2.imshow('raw',imgRGB_with_box)
        # cv2.waitKey(0)

        # save json format
        if len(bbs_aug.bounding_boxes) != 0:
            instance_aug = instance
            for i in range(len(bbs_aug.bounding_boxes)):
                box = []
                box.append(bbs_aug.bounding_boxes[i].x1)
                box.append(bbs_aug.bounding_boxes[i].y1)
                box.append(bbs_aug.bounding_boxes[i].x2)
                box.append(bbs_aug.bounding_boxes[i].y2)
                instance_aug[i].bbox = box
                instance_aug[i].w = box[2] - box[0]
                instance_aug[i].h = box[3] - box[1]
                return imgBGR_aug, instance_aug
        else:
            return imgBGR_aug, None

    def _mk_bbs(self, instance, img_info):
        BBox = [] #[ Bounding_box, Bounding_box,]
        w = img_info.img_w
        h = img_info.img_h
        for ins in instance:
            box = ins.bbox
            BBox.append(BoundingBox(x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3]))

        return BoundingBoxesOnImage(BBox,shape = (h,w))

def compute_wh(box):
    x1, y1, x2, y2 = box
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = max(x2, 0)
    y2 = max(y2, 0)
    w = x2 - x1
    h = y2 - y1
    return w, h
