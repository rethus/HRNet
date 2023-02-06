import os
import copy

import transforms
import torch
import numpy as np
import cv2
import torch.utils.data as data
from pycocotools.coco import COCO
import json


class CocoKeypoint(data.Dataset):
    def __init__(self,
                 root,
                 dataset="train",
                 years="2017",
                 transforms=None,
                 det_json_path=None,
                 fixed_size=(256, 192)):
        super().__init__()
        assert dataset in ["train", "val"], 'dataset must be in ["train", "val"]'
        anno_file = f"person_keypoints_{dataset}{years}.json"
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        self.img_root = os.path.join(root, f"{dataset}{years}")
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        self.anno_path = os.path.join(root, "annotations", anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.fixed_size = fixed_size
        self.mode = dataset
        self.transforms = transforms
        self.coco = COCO(self.anno_path)
        img_ids = list(sorted(self.coco.imgs.keys()))

        if det_json_path is not None:
            det = self.coco.loadRes(det_json_path)
        else:
            det = self.coco

        self.valid_person_list = []
        obj_idx = 0
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = det.getAnnIds(imgIds=img_id)
            anns = det.loadAnns(ann_ids)
            for ann in anns:
                # only save person class
                if ann["category_id"] != 1:
                    print(f'warning: find not support id: {ann["category_id"]}, only support id: 1 (person)')
                    continue

                # COCO_val2017_detections_AP_H_56_person.json文件中只有det信息，没有keypoint信息，跳过检查
                if det_json_path is None:
                    # skip objs without keypoints annotation
                    if "keypoints" not in ann:
                        continue
                    if max(ann["keypoints"]) == 0:
                        continue

                xmin, ymin, w, h = ann['bbox']
                # Use only valid bounding boxes
                if w > 0 and h > 0:
                    info = {
                        "box": [xmin, ymin, w, h],
                        "image_path": os.path.join(self.img_root, img_info["file_name"]),
                        "image_id": img_id,
                        "image_width": img_info['width'],
                        "image_height": img_info['height'],
                        "obj_origin_hw": [h, w],
                        "obj_index": obj_idx,
                        "score": ann["score"] if "score" in ann else 1.
                    }

                    # COCO_val2017_detections_AP_H_56_person.json文件中只有det信息，没有keypoint信息，跳过
                    if det_json_path is None:
                        keypoints = np.array(ann["keypoints"]).reshape([-1, 3])
                        visible = keypoints[:, 2]
                        keypoints = keypoints[:, :2]
                        info["keypoints"] = keypoints
                        info["visible"] = visible

                    self.valid_person_list.append(info)
                    obj_idx += 1
        print("done ------------------------------------")

    def __getitem__(self, idx):
        target = copy.deepcopy(self.valid_person_list[idx])

        image = cv2.imread(target["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image, person_info = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.valid_person_list)

    @staticmethod
    def collate_fn(batch):
        # batch = torch.Tensor(batch)
        # print(type(batch))
        imgs_tuple, targets_tuple = tuple(zip(*batch))
        # print(type(imgs_tuple[0]))
        imgs_tensor = torch.stack(imgs_tuple)
        # print(type(imgs_tensor))
        # print(len(imgs_tensor), ", ", len(targets_tuple))
        return imgs_tensor, targets_tuple


if __name__ == '__main__':
    with open('./person_keypoints.json', "r") as f:
        person_kps_info = json.load(f)
    # print(person_kps_info)
    fixed_size = (256, 192)
    heatmap_hw = (fixed_size[0] // 4, fixed_size[1] // 4)
    kps_weights = np.array(person_kps_info["kps_weights"], dtype=np.float32).reshape((17,))

    data_transform = {
        "train": transforms.Compose([
            # 随机裁剪半身
            transforms.HalfBody(0.3, person_kps_info["upper_body_ids"], person_kps_info["lower_body_ids"]),
            # 2D仿射变换
            transforms.AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),
            # 随机水平翻转
            transforms.RandomHorizontalFlip(0.5, person_kps_info["flip_pairs"]),
            # 转化热力图
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            # 检测框放大1.25倍
            transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # train = CocoKeypoint("dataset/COCO2017/", dataset="train")
    train_dataset = CocoKeypoint("dataset/sCOCO2017/", "train", transforms=data_transform["train"],
                                 fixed_size=[256, 192])
    print(len(train_dataset))
    # t = train_dataset[0]
    # print(t)

    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=32,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=4,
                                        collate_fn=train_dataset.collate_fn)

    print(1)
    # # print(train_data_loader)
    # for data in train_data_loader:
    # #     print(data.shape)
    #     imgs, targets = data
    #     print(imgs.shape, " ***** ", len(targets))
