import numpy as np
import os
import pickle
import open3d as o3d
import logging
from sklearn.linear_model import LinearRegression


def linear_regression_xyz_intensity(points: np.ndarray):
    """
    Given a Nx4 numpy array [x, y, z, intensity],
    perform linear regression with intensity as dependent variable and x, y, z as independent variables.
    Returns: model (sklearn LinearRegression), coefficients, intercept
    """
    X = points[:, :3]
    y = points[:, 3]
    model = LinearRegression()
    model.fit(X, y)
    return model


logging.basicConfig(level=logging.INFO)


def read_pickle(file_path, suffix=".pkl"):
    assert os.path.splitext(file_path)[1] == suffix
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(results, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(results, f)


def read_points(file_path, dim=4, **kwargs):
    suffix = os.path.splitext(file_path)[1]
    assert suffix in [".bin", ".ply"]
    if suffix == ".bin":
        pcd = np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
        if kwargs.get("dequantizer", None) is not None:
            dequantizer = kwargs.get("dequantizer", None)
            reference_pcd = kwargs.get("reference_pcd", None)
            assert reference_pcd is not None
            assert (
                isinstance(reference_pcd, np.ndarray)
                and reference_pcd.ndim == 2
                and reference_pcd.shape[1] >= 3
            )
            pcd_xyz = dequantizer.dequantize(pcd[:, :3])
            # the dequantized points may shift its centroid a bit, to avoid issue in later steps, we need to
            # shift the centroid back to the original position
            pcd_xyz = dequantizer.align_mid_point(pcd_xyz, ref_pts=reference_pcd[:, :3])
            pcd = np.hstack([pcd_xyz, pcd[:, 3:]])
        return pcd
    else:
        return ply_to_points(file_path, **kwargs)


def ply_to_points(file_path, **kwargs):
    """
    dequantizer and reference pcd are needed to correct the xyz coordinates if the points are quantized
    """
    dequantizer = kwargs.get("dequantizer", None)
    reference_pcd = kwargs.get("reference_pcd", None)
    target_intensity = kwargs.get("target_intensity", None)
    # read ply file using open3d
    # if color existed in the ply file, and no intensity_filling_strategy provided, use
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    points = dequantizer.dequantize(points) if dequantizer is not None else points
    points = (
        dequantizer.align_mid_point(points, ref_pts=reference_pcd[:, :3])
        if dequantizer is not None and reference_pcd is not None
        else points
    )
    size = points.shape[0]
    intensity_filling_strategy = kwargs.get("intensity_filling_strategy", None)
    if pcd.has_colors() and intensity_filling_strategy is None:
        # no intensity_filling_strategy provided, and color existed in the ply file, use color mean as intensity
        colors = np.asarray(pcd.colors, dtype=np.float32)
        intensity = colors.mean(axis=1, keepdims=True)
        # logging.info(f"colors = {colors}, intensity = {intensity}")
        points_with_intensity = np.hstack([points, intensity])
    else:
        # no color exists in the ply file, fill intensity according to intensity_filling_strategy
        if intensity_filling_strategy is None:
            raise ValueError(
                "intensity_filling_strategy should be provided when no color exists in the ply file"
            )
        if intensity_filling_strategy == "conform_to_target_intensity":
            if target_intensity is None:
                raise ValueError(
                    "target_intensity should be provided when intensity_filling_strategy is conform_to_target_intensity"
                )
            intensity = np.random.choice(target_intensity, size=(size, 1), replace=True)
        elif intensity_filling_strategy == "linear_regression":
            if reference_pcd is None:
                raise ValueError(
                    "reference_pcd (N x [x,y,z,intensity]) should be provided when intensity_filling_strategy is linear_regression"
                )
            intensity = (
                linear_regression_xyz_intensity(reference_pcd)
                .predict(points[:, :3])
                .reshape(-1, 1)
            )
        elif intensity_filling_strategy == "random":
            intensity = np.random.rand(size, 1).astype(points.dtype)
        elif intensity_filling_strategy == "random_one":
            intensity = np.ones((size, 1), dtype=points.dtype) * np.random.rand(1)
        elif intensity_filling_strategy == "zeros":
            intensity = np.zeros((size, 1), dtype=points.dtype)
        elif intensity_filling_strategy == "ones":
            intensity = np.ones((size, 1), dtype=points.dtype)
        else:
            raise ValueError(
                f"Not supported intensity_filling_strategy {intensity_filling_strategy}"
            )
        # logging.info(f"intensity_filling_strategy = {intensity_fillinng_strategy}, intensity = {intensity}")
        points_with_intensity = np.hstack([points, intensity])
    return points_with_intensity


def write_points(lidar_points, file_path):
    suffix = os.path.splitext(file_path)[1]
    assert suffix in [".bin", ".ply"]
    if suffix == ".bin":
        with open(file_path, "w") as f:
            lidar_points.tofile(f)
    else:
        raise NotImplementedError


def read_calib(file_path, extend_matrix=True):
    with open(file_path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    P0 = np.array([item for item in lines[0].split(" ")[1:]], dtype=np.float32).reshape(
        3, 4
    )
    P1 = np.array([item for item in lines[1].split(" ")[1:]], dtype=np.float32).reshape(
        3, 4
    )
    P2 = np.array([item for item in lines[2].split(" ")[1:]], dtype=np.float32).reshape(
        3, 4
    )
    P3 = np.array([item for item in lines[3].split(" ")[1:]], dtype=np.float32).reshape(
        3, 4
    )

    R0_rect = np.array(
        [item for item in lines[4].split(" ")[1:]], dtype=np.float32
    ).reshape(3, 3)
    Tr_velo_to_cam = np.array(
        [item for item in lines[5].split(" ")[1:]], dtype=np.float32
    ).reshape(3, 4)
    Tr_imu_to_velo = np.array(
        [item for item in lines[6].split(" ")[1:]], dtype=np.float32
    ).reshape(3, 4)

    if extend_matrix:
        P0 = np.concatenate([P0, np.array([[0, 0, 0, 1]])], axis=0)
        P1 = np.concatenate([P1, np.array([[0, 0, 0, 1]])], axis=0)
        P2 = np.concatenate([P2, np.array([[0, 0, 0, 1]])], axis=0)
        P3 = np.concatenate([P3, np.array([[0, 0, 0, 1]])], axis=0)

        R0_rect_extend = np.eye(4, dtype=R0_rect.dtype)
        R0_rect_extend[:3, :3] = R0_rect
        R0_rect = R0_rect_extend

        Tr_velo_to_cam = np.concatenate(
            [Tr_velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0
        )
        Tr_imu_to_velo = np.concatenate(
            [Tr_imu_to_velo, np.array([[0, 0, 0, 1]])], axis=0
        )

    calib_dict = dict(
        P0=P0,
        P1=P1,
        P2=P2,
        P3=P3,
        R0_rect=R0_rect,
        Tr_velo_to_cam=Tr_velo_to_cam,
        Tr_imu_to_velo=Tr_imu_to_velo,
    )
    return calib_dict


def read_label(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split(" ") for line in lines]
    annotation = {}
    annotation["name"] = np.array([line[0] for line in lines])
    annotation["truncated"] = np.array([line[1] for line in lines], dtype=np.float32)
    annotation["occluded"] = np.array([line[2] for line in lines], dtype=np.int32)
    annotation["alpha"] = np.array([line[3] for line in lines], dtype=np.float32)
    annotation["bbox"] = np.array([line[4:8] for line in lines], dtype=np.float32)
    annotation["dimensions"] = np.array(
        [line[8:11] for line in lines], dtype=np.float32
    )[:, [2, 0, 1]]  # hwl -> camera coordinates (lhw)
    annotation["location"] = np.array([line[11:14] for line in lines], dtype=np.float32)
    annotation["rotation_y"] = np.array([line[14] for line in lines], dtype=np.float32)

    return annotation


def write_label(result, file_path, suffix=".txt"):
    """
    result: dict,
    file_path: str
    """
    assert os.path.splitext(file_path)[1] == suffix
    name, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, score = (
        result["name"],
        result["truncated"],
        result["occluded"],
        result["alpha"],
        result["bbox"],
        result["dimensions"],
        result["location"],
        result["rotation_y"],
        result["score"],
    )

    with open(file_path, "w") as f:
        for i in range(len(name)):
            bbox_str = " ".join(map(str, bbox[i]))
            hwl = " ".join(map(str, dimensions[i]))
            xyz = " ".join(map(str, location[i]))
            line = f"{name[i]} {truncated[i]} {occluded[i]} {alpha[i]} {bbox_str} {hwl} {xyz} {rotation_y[i]} {score[i]}\n"
            f.writelines(line)
