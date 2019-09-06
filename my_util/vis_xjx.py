from vis_gt import DataAnalyze

class Config:
    def __init__(self):
        # self.json_paths = [
        #         # "/home/xjx/data/Kaggle/Json/Aug_json.json",
        #         '/home/xjx/data/Kaggle/guangdong1_round1_train1_20190818/Annotations/anno_train.json',
        #         '/home/xjx/data/Kaggle/guangdong1_round1_train2_20190828/Annotations/anno_train.json',
        #         '/home/xjx/ding/Aug_json.json'
        #                    ] # 训练json
        # ------------------------train json, path -----------------------------
        self.json_paths = [
                # "/home/xjx/data/Kaggle/Json/Aug_json.json",
                '/home/xjx/data/Kaggle/rawAddAug_train.json',
                '/home/xjx/data/Kaggle/rawAddAug_val.json',
                           ] # 训练json

        self.allimg_path = '/home/xjx/data/Kaggle/Images' # 训练 图片地址

        # ----------------------val json, path, result---------------------------
        # val result gt vis
        self.val_json_paths = ['/home/xjx/data/Kaggle/Json/instances_balance_val_coco.json'] # val json coco格式
        self.val_img_path = '/home/xjx/data/Kaggle/Images'    # val 图片地址

        self.result_json = ['/home/xjx/data/Kaggle/result/result_19_val.json'] # 结果 json 地址

        # ---------------------divide json --------------------------------------
        self.divide_json = self.json_paths

        # ---------------------aug path, json -----------------------------------
        self.aug_save_path = '/home/xjx/data/Kaggle/aug_data'
        self.json_file_path = '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/Aug_json.json'
        self.add_num = 150



if __name__ == "__main__":

    cfg = Config()
    dataer = DataAnalyze(cfg)
    flag_tool = 0

    if flag_tool == 0:
        # 可视化gt
        dataer.vis_gt(flag_show_raw_img=False) # 使用json_paths 和  allimg_path 路径

    elif flag_tool == 1:
        # 可视化 gt 和结果分析
        dataer.val_analyze() # 使用val_json_paths 和 val_img_path 作为val的路径，   result_json  作为结果的json

    elif flag_tool == 2:
        # 划分训练验证    # 使用 divide_json
        dataer.divide_trainval(ratio=0.2, train_json='/home/xjx/data/Kaggle/rawAddAug_train.json', val_json='/home/xjx/data/Kaggle/rawAddAug_val.json')

    # dataer.add_aug_data(add_num = cfg.add_num, aug_save_path=cfg.aug_save_path,
    #                     json_file_path= cfg.json_file_path)
    # dataer.draw_cls()


    z = 1

