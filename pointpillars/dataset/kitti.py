import random
import numpy as np
import re
import os
import sys
from torch.utils.data import Dataset
from pointpillars.utils.quantize import (
    dequantize_octree,
    dequantize_precision,
    dequantize_resolution,
)


from pointpillars.utils import read_pickle, read_points, bbox_camera2lidar
from pointpillars.dataset import point_range_filter, data_augment

# ???? what is this thing doing here????
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))


class BaseSampler:
    def __init__(self, sampled_list, shuffle=True):
        self.total_num = len(sampled_list)
        self.sampled_list = np.array(sampled_list)
        self.indices = np.arange(self.total_num)
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle
        self.idx = 0

    def sample(self, num):
        if self.idx + num < self.total_num:
            ret = self.sampled_list[self.indices[self.idx : self.idx + num]]
            self.idx += num
        else:
            ret = self.sampled_list[self.indices[self.idx :]]
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        return ret


class Kitti(Dataset):
    CLASSES = {"Pedestrian": 0, "Cyclist": 1, "Car": 2}

    def __init__(
        self,
        data_root,
        split,
        pts_prefix="velodyne_reduced",
        file_name_changer=r"\1.\2",  # default do nothing, keep the original file name
        intensity_filling_strategy=None,
        intensity_forced_value=0,
        reference_pcd_prefix=None,
        dequantizer=None,
    ):
        assert split in ["train", "val", "trainval", "test"]
        self.data_root = data_root
        self.split = split
        self.pts_prefix = pts_prefix
        self.file_name_changer = file_name_changer
        self.reference_pcd_prefix = reference_pcd_prefix
        self.intensity_filling_strategy = intensity_filling_strategy
        self.intensity_forced_value = intensity_forced_value
        self.dequantizer = dequantizer
        self.data_infos = read_pickle(
            os.path.join(data_root, f"kitti_infos_{split}.pkl")
        )
        self.sorted_ids = list(self.data_infos.keys())
        db_infos = read_pickle(os.path.join(data_root, "kitti_dbinfos_train.pkl"))
        db_infos = self.filter_db(db_infos)

        db_sampler = {}
        for cat_name in self.CLASSES:
            db_sampler[cat_name] = BaseSampler(db_infos[cat_name], shuffle=True)
        self.data_aug_config = dict(
            db_sampler=dict(
                db_sampler=db_sampler,
                sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
            ),
            object_noise=dict(
                num_try=100,
                translation_std=[0.25, 0.25, 0.25],
                rot_range=[-0.15707963267, 0.15707963267],
            ),
            random_flip_ratio=0.5,
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0],
            ),
            point_range_filter=[0, -39.68, -3, 69.12, 39.68, 1],
            object_range_filter=[0, -39.68, -3, 69.12, 39.68, 1],
        )
    def create_sample_dataset(self, num_samples=100):
        """
        Returns a new Kitti object with a random sample of num_samples from data_infos.
        """
        new_kitti = Kitti(
            data_root=self.data_root,
            split=self.split,
            pts_prefix=self.pts_prefix,
            file_name_changer=self.file_name_changer,
            intensity_filling_strategy=self.intensity_filling_strategy,
            reference_pcd_prefix=self.reference_pcd_prefix,
            dequantizer=self.dequantizer,
        )
        sampled_keys = random.sample(list(self.data_infos.keys()), num_samples)
        new_kitti.data_infos = {k: self.data_infos[k] for k in sampled_keys}
        new_kitti.sorted_ids = list(new_kitti.data_infos.keys())
        return new_kitti

    def remove_dont_care(self, annos_info):
        keep_ids = [
            i for i, name in enumerate(annos_info["name"]) if name != "DontCare"
        ]
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info

    def filter_db(self, db_infos):
        # 1. filter_by_difficulty
        for k, v in db_infos.items():
            db_infos[k] = [item for item in v if item["difficulty"] != -1]

        # 2. filter_by_min_points, dict(Car=5, Pedestrian=10, Cyclist=10)
        filter_thrs = dict(Car=5, Pedestrian=10, Cyclist=10)
        for cat in self.CLASSES:
            filter_thr = filter_thrs[cat]
            db_infos[cat] = [
                item for item in db_infos[cat] if item["num_points_in_gt"] >= filter_thr
            ]

        return db_infos

    # remove the data infos with the given idxes
    def remove_skipped_idxes(self, skipped_idxes):
        self.data_infos = {
            k: v
            for i, (k, v) in enumerate(self.data_infos.items())
            if i not in skipped_idxes
        }
        self.sorted_ids = list(self.data_infos.keys())

    def __getitem__(self, index):
        data_info = self.data_infos[self.sorted_ids[index]]
        image_info, calib_info, annos_info = (
            data_info["image"],
            data_info["calib"],
            data_info["annos"],
        )

        # point cloud input
        velodyne_path = data_info["velodyne_path"].replace("velodyne", self.pts_prefix)
        pts_path = os.path.join(self.data_root, velodyne_path)
        pts_path = re.sub(r"(\d+)\.(bin)", self.file_name_changer, pts_path)

        # preload the reference pcd if specified
        if self.reference_pcd_prefix is not None:
            reference_bin_relative_path = data_info["velodyne_path"].replace(
                "velodyne", self.reference_pcd_prefix
            )
            reference_bin_path = os.path.join(
                self.data_root, reference_bin_relative_path
            )
            reference_pcd = read_points(reference_bin_path)

        suffix = os.path.splitext(pts_path)[1]
        if suffix == ".bin":
            pts = read_points(pts_path, dim=4)
        else:
            """
            the ply file need to be processed and if the ply file does not have intensity, we need to fill the intensity
            according to intensity_filling_strategy
            linear regression strategy need to refer to the x,y,z coordinates, therefore we need to dequantize the x,y,z
            so the dequantizer is passed to the read_points function
            moreover, the dequantized points may shift its centroid a bit, to avoid issue in later steps, we need to
            shift the centroid back to the original position, therefore the reference_pcd is also passed to the read_points function    
            """
            if self.intensity_filling_strategy in [
                "random_one",
                "random",
                "zeros",
                "ones",
                "force_fixed_value",
            ]:
                pts = read_points(
                    pts_path,
                    dim=4,
                    intensity_filling_strategy=self.intensity_filling_strategy,
                    dequantizer=self.dequantizer,
                    reference_pcd=reference_pcd if self.reference_pcd_prefix else None,
                )
            elif self.intensity_filling_strategy == "linear_regression":
                if self.reference_pcd_prefix is None:
                    raise ValueError(
                        "reference_pcd_prefix should be provided when intensity_filling_strategy is linear_regression"
                    )

                pts = read_points(
                    pts_path,
                    dim=4,
                    intensity_filling_strategy=self.intensity_filling_strategy,
                    dequantizer=self.dequantizer,
                    reference_pcd=reference_pcd if self.reference_pcd_prefix else None,
                )

            elif self.intensity_filling_strategy == "conform_to_target_intensity":
                if self.reference_pcd_prefix is None:
                    raise ValueError(
                        "reference_pcd_prefix should be provided when intensity_filling_strategy is conform_to_target_intensity"
                    )
                # use the intensity distribution of the training set as the target intensity
                target_intensity = reference_pcd[:, 3]
                pts = read_points(
                    pts_path,
                    dim=4,
                    intensity_filling_strategy=self.intensity_filling_strategy,
                    target_intensity=target_intensity,
                    dequantizer=self.dequantizer,
                    reference_pcd=reference_pcd if self.reference_pcd_prefix else None,
                )

            else:
                raise ValueError(
                    f"Not supported intensity_fillinng_strategy {self.intensity_filling_strategy}"
                )
        if self.intensity_filling_strategy == "fixed_value":
            fixed_value = (
                self.intensity_forced_value
            )  # you can change this value as needed
            pts_xyz = pts[:, :3]
            intensity = np.ones((pts_xyz.shape[0], 1), dtype=np.float32) * fixed_value
            pts = np.hstack([pts_xyz, intensity])
        # calib input: for bbox coordinates transformation between Camera and Lidar.
        # because
        tr_velo_to_cam = calib_info["Tr_velo_to_cam"].astype(np.float32)
        r0_rect = calib_info["R0_rect"].astype(np.float32)

        # annotations input
        annos_info = self.remove_dont_care(annos_info)
        annos_name = annos_info["name"]
        annos_location = annos_info["location"]
        annos_dimension = annos_info["dimensions"]
        rotation_y = annos_info["rotation_y"]
        gt_bboxes = np.concatenate(
            [annos_location, annos_dimension, rotation_y[:, None]], axis=1
        ).astype(np.float32)
        gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, tr_velo_to_cam, r0_rect)
        gt_labels = [self.CLASSES.get(name, -1) for name in annos_name]
        data_dict = {
            "pts": pts,
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels": np.array(gt_labels),
            "gt_names": annos_name,
            "difficulty": annos_info["difficulty"],
            "image_info": image_info,
            "calib_info": calib_info,
        }
        if self.split in ["train", "trainval"]:
            data_dict = data_augment(
                self.CLASSES, self.data_root, data_dict, self.data_aug_config
            )
        else:
            data_dict = point_range_filter(
                data_dict, point_range=self.data_aug_config["point_range_filter"]
            )
        if data_dict["pts"].shape[0] == 0:
            raise ValueError(f"No points left after data augmentation: {pts_path}")
        return data_dict

    def __len__(self):
        return len(self.data_infos)


if __name__ == "__main__":
    kitti_data = Kitti(data_root="/mnt/ssd1/lifa_rdata/det/kitti", split="train")
    kitti_data.__getitem__(9)
