import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import open3d as o3d
from pointpillars.utils import read_points


def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    """
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    """
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = (
        flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    )
    pts = pts[keep_mask]
    return pts


file_path = (
    f"/home/wuyuren/repos/PointPillars/pointpillars/dataset/demo_data/val/000134.bin"
)
points = read_points(file_path)
points_filter_range = point_range_filter(points)
image = o3d.geometry.Image(points_filter_range)
o3d.visualization.draw_geometries([image])
