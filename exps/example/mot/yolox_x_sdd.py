#python tools/train.py -f exps/example/mot/yolox_x_sdd.py -d 1 -b 8 --fp16 -o -c pretrained/bytetrack_sdd_converted.pth


import os
import torch
import torch.nn as nn
import torch.distributed as dist
from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir
from tqdm import tqdm  # 用于显示进度条
import psutil  # 用于内存监控
import random


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 6  # 自定义类别数
        self.depth = 0.33
        self.width = 0.375
        self.scale = (0.5, 1.5)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_dir = "d:/bytetrack/datasets/stanford_drone"
        self.train_ann = "d:/bytetrack/datasets/stanford_drone/annotations/train_split.json"
        self.val_ann = "d:/bytetrack/datasets/stanford_drone/annotations/val_split.json"

        self.input_size = (800, 1440) #修改了大小
        self.test_size = (800, 1440)
        self.random_size = (18, 19)
        self.max_epoch = 10 #修改了大小
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.001 #0.001
        self.nmsthre = 0.4 #0.7
        self.no_aug_epochs = 5

        self.basic_lr_per_img = 0.0005
        self.scheduler = "yoloxwarmcos"  # 使用余弦退火学习率
        self.min_lr_ratio = 0.05  # 最小学习率保持为初始值的 5%

        self.data_num_workers = 32
        self.batch_size = 32

        self.enable_mixup = False #防止混合导致丢失小框
        self.enable_mosaic = True
        self.mscale = (0.95, 1.05)  # 随机缩放范围

        self.iou_weight = 1.0  # 增加 IOU 损失权重
        self.cls_weight = 1.5  # 增加分类损失权重




    def log_memory_usage(self):
        """
        记录当前内存使用。
        """
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"[Memory Usage] RSS: {mem_info.rss / (1024 ** 2):.2f} MB, VMS: {mem_info.vms / (1024 ** 2):.2f} MB")

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        print("[INFO] 初始化训练数据集...")
        dataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "stanford_drone"),
            json_file=self.train_ann,
            name='',
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        print("[INFO] 应用 Mosaic 转换...")
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        print("[INFO] 加载训练数据...")
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        print("[INFO] 数据加载完成。")
        self.log_memory_usage()
        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        print("[INFO] 初始化评估数据集...")
        valdataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "stanford_drone"),
            json_file=self.val_ann,
            img_size=self.test_size,
            name='',
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        print("[INFO] 评估数据集加载完成。")
        self.log_memory_usage()
        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
