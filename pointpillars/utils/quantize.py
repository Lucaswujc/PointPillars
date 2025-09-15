import numpy as np


class Dequantizer:
    def __init__(self, config_file):
        import yaml

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)
        self.method = self.config.get("method", None)
        assert self.method in [
            "precision",
            "resolution",
            "octree",
        ], "Unsupported dequantization method: {}".format(self.method)
        if self.method == "precision":
            self.precision = self.config.get("precision", 0.001)
            self.quant_mode = self.config.get("quant_mode", "round")
        elif self.method == "resolution":
            self.resolution = self.config.get("resolution", 65535)
            self.quant_mode = self.config.get("quant_mode", "round")
        elif self.method == "octree":
            self.qlevel = self.config.get("qlevel", 12)
            self.quant_mode = self.config.get("quant_mode", "round")

    def dequantize(self, points: np.ndarray, **kwargs) -> np.ndarray:
        assert (
            isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 3
        )
        if self.method == "precision":
            quant_error = kwargs.get("quant_error", 0)
            points_out = dequantize_precision(
                points,
                quant_error=quant_error,
                precision=self.precision,
            )
        elif self.method == "resolution":
            # max and min bounds are required, but it seems that we need to get this from reference file
            # kitti implemnetation is even more complicated . for now ignore
            # research shows that we may not need to dequantize the resolution quantized points
            max_bound = kwargs.get("max_bound", None)
            min_bound = kwargs.get("min_bound", None)
            quant_error = kwargs.get("quant_error", 0)
            assert max_bound is not None and min_bound is not None
            assert isinstance(max_bound, np.ndarray) and max_bound.shape == (3,)
            assert isinstance(min_bound, np.ndarray) and min_bound.shape == (3,)

            points_out = dequantize_resolution(
                points,
                max_bound=max_bound,
                min_bound=min_bound,
                quant_error=quant_error,
                resolution=self.resolution,
            )
        elif self.method == "octree":
            # min, max bounds and centroid are required, but it seems that we need to get this from reference file
            # research shows that we may not need to dequantize the octree quantized points
            min_bound = kwargs.get("min_bound", None)
            max_bound = kwargs.get("max_bound", None)
            centroid = kwargs.get("centroid", None)
            quant_error = kwargs.get("quant_error", 0)
            assert (
                min_bound is not None and max_bound is not None and centroid is not None
            )
            assert isinstance(min_bound, np.ndarray) and min_bound.shape == (3,)
            assert isinstance(max_bound, np.ndarray) and max_bound.shape == (3,)
            assert isinstance(centroid, np.ndarray) and centroid.shape == (3,)
            points_out = dequantize_octree(
                points,
                min_bound=min_bound,
                max_bound=max_bound,
                centroid=centroid,
                quant_error=quant_error,
                qlevel=self.qlevel,
            )
        return points_out

    def align_mid_point(self, pts: np.ndarray, ref_pts: np.ndarray) -> np.ndarray:
        """
        Align the centroid of pts to ref_pts
        using mean to calculate centroid is not a good idea, depends on
        the points density in the space, calculate the mean is not a good
        way since the density varies a lot
        A better approach is to align on spacial ranges
        To align two coordinates, using x value as an example,
        reference pcd has min(x) and max(x), the pcd has min(x) and max(x),
        the center point of both images's x should be the same value
        """
        pts_xyz = pts[:, :3]
        ref_xyz = ref_pts[:, :3]
        if pts_xyz.size == 0 or ref_xyz.size == 0:
            # Handle the empty case, e.g., return None or raise a custom error
            return pts
        # Compute midpoints for each axis
        pts_mid = (pts_xyz.min(axis=0) + pts_xyz.max(axis=0)) / 2
        ref_mid = (ref_xyz.min(axis=0) + ref_xyz.max(axis=0)) / 2

        # Compute shift
        shift = ref_mid - pts_mid

        # Shift pts
        pts_aligned = pts.copy()
        pts_aligned[:, :3] += shift

        return pts_aligned


######################## basic operation ########################
def normalize(points: np.ndarray, offset="min"):
    if offset == "mean":
        ref_point = points.mean(axis=0)
    elif offset == "min":
        ref_point = points.min(axis=0)
    else:
        ref_point = np.array([0, 0, 0])
    points = points - ref_point

    return points, ref_point


def quantize_precision(
    points: np.ndarray, precision=0.001, quant_mode="round", return_offset=False
):
    assert quant_mode in ["round", "floor"]
    points = points.astype("float")
    points_out = points / float(precision)
    points_quant = points_out
    if quant_mode == "round":
        points_quant = np.round(points_out)
    if quant_mode == "floor":
        points_quant = np.floor(points_out)
    points_quant = points_quant.astype("int")
    if not return_offset:
        return points_quant
    else:
        quant_error = points_out - points_quant
        return points_quant, quant_error


def dequantize_precision(points: np.ndarray, quant_error=0, precision=0.001):
    # points = points.astype('float')
    points_out = points + quant_error
    points_out = points_out * precision

    return points_out


def quantize_resolution(
    points: np.ndarray, resolution=65535, quant_mode="round", return_offset=False
):
    assert quant_mode in ["round", "floor"]
    min_bound = points.min(axis=0)
    points_out = points - min_bound
    max_bound = points_out.max()
    points_out = points_out / max_bound
    points_out = points_out * resolution
    points_quant = points_out
    if quant_mode == "round":
        points_quant = np.round(points_out)
    if quant_mode == "floor":
        points_quant = np.floor(points_out)
    points_quant = points_quant.astype("int")
    if not return_offset:
        return points_quant, max_bound, min_bound
    else:
        quant_error = points_out - points_quant
        return points_quant, max_bound, min_bound, quant_error


def dequantize_resolution(
    points: np.ndarray,
    max_bound: np.ndarray,
    min_bound: np.ndarray,
    quant_error=0,
    resolution=65535,
):
    points = points.astype("float")
    points_out = points + quant_error
    points_out = points_out / resolution
    points_out = points_out * max_bound
    points_out = points_out + min_bound
    return points_out


def quantize_octree(
    points: np.ndarray, qlevel=12, quant_mode="round", return_offset=False
):
    """Quantization method of OctAttention & VoxelContextNet"""
    assert quant_mode in ["round", "floor"]
    points_out = points.copy()
    # normalize
    centroid = points_out.mean(axis=0)
    points_out = points_out - centroid
    max_bound = np.abs(points_out).max()
    points_out = points_out / max_bound
    # print('DBG!!', points_raw.max(), points_raw.min())
    # attention
    min_bound = points_out.min(axis=0)
    points_out = points_out - min_bound
    resolution = (2**qlevel - 1) / 2
    points_out = points_out * resolution
    points_quant = points_out
    if quant_mode == "round":
        points_quant = np.round(points_out)
    if quant_mode == "floor":
        points_quant = np.floor(points_out)
    points_quant = points_quant.astype("int")
    if not return_offset:
        return points_quant, min_bound, max_bound, centroid
    else:
        quant_error = points_out - points_quant
        return points_quant, min_bound, max_bound, centroid, quant_error


def dequantize_octree(
    points: np.ndarray,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
    centroid: np.ndarray,
    quant_error=0,
    qlevel=12,
):
    points = points.astype("float")
    points_out = points + quant_error
    resolution = (2**qlevel - 1) / 2
    points_out = points_out / resolution
    points_out = points_out + min_bound
    points_out = points_out * max_bound
    points_out = points_out + centroid
    return points_out


def random_quantize(points: np.ndarray, factor=None, min_factor=0.5, max_factor=1):
    if factor is None:
        factor = np.random.uniform(min_factor, max_factor)
    pointsQ = quantize_precision(
        points, precision=1 / factor, quant_mode="round", return_offset=False
    )
    pointsQ = np.unique(pointsQ, axis=0).astype("int")

    return pointsQ


# def merge_points(points: np.ndarray, offset: np.ndarray):
#     """TODO"""
#     # quantize
#     points = points.astype("int64")
#     min_value = points.min(axis=0)
#     points_in = points - min_value
#     # collect duplicated points
#     step = points_in.max() + 1
#     points_1d = sum([points_in[:, i] * (step**i) for i in range(3)])
#     offset_dict = {}
#     for i, pt in enumerate(points_1d):
#         if not pt in offset_dict:
#             offset_dict[pt] = [offset[i]]
#         else:
#             offset_dict[pt].append(offset[i])
#     # average duplicated points
#     points_out = []
#     offset_out = []
#     for k, v in offset_dict.items():
#         points_out.append(k)
#         if len(v) == 1:
#             offset_out.append(v[0])
#         else:
#             offset_out.append(np.vstack(v).mean(axis=0))
#     # collection
#     points_out = np.array(points_out)
#     points_out = np.vstack(
#         [(points_out // (step**i)) % step for i in range(3)]
#     ).transpose(1, 0)
#     points_out = points_out + min_value
#     offset_out = np.vstack(offset_out)
#     assert (np.unique(points, axis=0) == np.unique(points_out, axis=0)).all()

#     return points_out, offset_out
