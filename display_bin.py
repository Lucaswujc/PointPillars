# Display a bin file and create a camera view , note : it requires view_point.json file under the 
# same directory as the program. You can find the file in pointpillars/utils folder. easiest thing to do 
# is make a copy to the program folder where this python file stores

import open3d as o3d
import numpy as np
import os
import argparse

view_point = os.path.join(os.path.dirname(__file__), "viewpoint.json")


def display_layers(
    image_layers: list = None, min_z: float = -10.0, max_z: float = 10.0
):
    """
    Display a list of Open3D geometries in a visualizer window.
    Args:
        image_layers (list): A list of Open3D geometries to be displayed.
        min_z (float): Minimum z value for the camera view.
        max_z (float): Maximum z value for the camera view.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    # Read the camera parameters from the JSON file
    param = o3d.io.read_pinhole_camera_parameters(view_point)

    # Add the point cloud to the visualizer
    for layer in image_layers:
        vis.add_geometry(layer)

    # following lines are used to setup the camera view, have not figured out the math yet
    # these controls are setup using try anad error,
    # so far this is created a camera view and looks about right and can be used for compare
    # directly with the png image files in the dataset
    ctr.convert_from_pinhole_camera_parameters(param)
    zoom = 1.0 / (max_z - min_z + 1e-6) * 0.4
    zoom = np.clip(zoom, 0.1, 2.0)
    # Create bird eye view camera parameters
    center = image_layers[0].get_center()
    distance = 50
    eye = center + np.array([0, 0, distance])
    up = np.array([1, 0, 0])  # Up direction in the Y direction
    ctr.set_lookat(center)
    ctr.set_up(up)
    ctr.set_front([-3, 0, 1])  # Looking down the Z axis
    ctr.set_zoom(zoom)
    # Run the visualizer
    vis.run()
    # Destroy the visualizer window
    vis.destroy_window()


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


def npy2ply(npy: np.ndarray) -> o3d.geometry.PointCloud:
    """
    convert a numpy array to an Open3D point cloud object.

    Args:
        npy : numpy.ndarray, a numpy array with shape (N, 4) where N is the number of points.
        [x,y,z,density] are the columns of the array.

    Returns:
        o3d.geometry.PointCloud: A point cloud object with points and colors.
    """
    ply = o3d.geometry.PointCloud()
    # select only three columns for each point in the array to construct the point cloud
    ply.points = o3d.utility.Vector3dVector(npy[:, :3])
    density = npy[:, 3]
    # use the same value in desntiy for RGB color, effectively making the point cloud grayscale
    colors = [[item, item, item] for item in density]
    ply.colors = o3d.utility.Vector3dVector(colors)
    return ply


def load_bin(file_path):
    image_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    if image_data.size == 0:
        print(f"File {file_path} is empty or not a valid binary file.")
        return
    image_data = point_range_filter(
        image_data
    )  # Filter points based on the specified range
    print(f"total points in the image : {image_data.size}")
    print(f"max_z = {np.max(image_data[:, 2])}, min_z = {np.min(image_data[:, 2])}")

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0, 0, 0]
    )
    display_layers(
        [npy2ply(image_data), mesh_frame],
        max_z=np.max(image_data[:, 2]),
        min_z=np.min(image_data[:, 2]),
    )


def main(args):
    file_path = args.file_path
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return
    load_bin(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument(
        "--file_path",
        type=str,
        default="pointpillars/dataset/demo_data/val/000134.bin",
        help="Path to the point cloud file in binary format.",
    )
    args = parser.parse_args()
    main(args)
 