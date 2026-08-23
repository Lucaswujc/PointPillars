# Custom Changes in This Workspace

This README only tracks custom work done in this workspace.

Original upstream README: [readme_original](readme_original)

## Project Summary (July 2025)

This project started with onboarding into LiDAR compression-for-detection research and understanding the PointPillars detection pipeline as the baseline system.

### Goals

- Build a practical understanding of PointPillars and its deployment constraints in our environment.
- Test whether aggressive point cloud compression can preserve detection quality.
- Evaluate an existing LiDAR compression framework (SparsePCGC) before proposing new model ideas.
- Create visualization tools to verify reconstruction quality and detection behavior.

### Process

- Familiarized myself with PointPillars internals and attempted end-to-end setup on NVIDIA Jetson hardware.
- Encountered version incompatibilities in legacy PyTorch dependencies; adapted execution strategy to available infrastructure.
- Ran a staged deletion experiment as a fast proxy for compression, increasing random point dropping from 20 percent to 90 percent.
- Evaluated SparsePCGC across its dense/sparse and lossy/lossless settings, with attention to MP-POV and SOPA design choices.
- Used AWS GPU hosts when deprecated dependencies blocked reliable local execution.
- Developed a visualizer to inspect reconstructed outputs with 2D and 3D bounding boxes for qualitative validation.

### Findings

- PointPillars setup friction was non-trivial due to legacy dependency stacks, especially on Jetson-class environments.
- In deletion-based simulation, detection performance stayed relatively stable until point removal exceeded roughly 80 percent, suggesting significant redundancy in LiDAR point clouds.
- SparsePCGC's MP-POV and SOPA strategy offered a useful reference for prioritizing informative spatial regions during compression.
- A recurring hypothesis from experiments was that a meaningful share of compression gains comes from stacked arithmetic encoding stages.

### Main Learnings

- Validate hypotheses with simple, controlled experiments first; they often reveal signal faster than complex redesigns.
- Environment and dependency constraints are core research risks, not side tasks.
- Combining quantitative metrics with visualization is essential for trustworthy iteration in reconstruction and detection workflows.

## 1) Custom evaluation pipeline work

- Extended and maintained local evaluation workflows in [evaluate.py](evaluate.py) and [evaluate2.py](evaluate2.py).
- Continued KITTI metric work around 2D, BEV, and 3D IoU paths, class/difficulty filtering, score thresholding, and AP reporting.
- Produced additional result artifacts in [results](results) and [results2](results2).

## 2) New large submission output set

- Added a new KITTI-format submission/result set in [results2/submit](results2/submit).
- Current structure in [results2](results2):
    - [results2/eval_results.txt](results2/eval_results.txt)
    - [results2/results.pkl](results2/results.pkl)
    - [results2/submit](results2/submit) containing a large number of per-frame prediction files.

## 3) Deletion robustness experiments

- Kept deletion-based evaluation flow for modified KITTI subsets.
- Batch script used: [delete_evaluation.sh](delete_evaluation.sh).
- Main experiment roots:
    - [kitti_deleted/0.6](kitti_deleted/0.6)
    - [kitti_deleted/0.8](kitti_deleted/0.8)
    - [kitti_deleted/0.9](kitti_deleted/0.9)
    - [results/del_results](results/del_results)

## 4) SparsePCGC reconstruction to detection evaluation

- Maintained evaluation flow from reconstructed point clouds to detection metrics.
- Main orchestration scripts:
    - [sparsepcgc_eval.sh](sparsepcgc_eval.sh)
    - [eval_first10_alltypes.sh](eval_first10_alltypes.sh)
- Dequantization configurations used:
    - [dequantizer_precision.yaml](dequantizer_precision.yaml)
    - [dequantizer_precision_0.1.yaml](dequantizer_precision_0.1.yaml)
    - [dequantizer_precision_0.001.yaml](dequantizer_precision_0.001.yaml)
    - [dequantizer_octree.yaml](dequantizer_octree.yaml)
    - [dequantizer_resolution.yaml](dequantizer_resolution.yaml)

Primary data roots:

- [sparsepcgc_data](sparsepcgc_data)
- [sparsepcgc_eval](sparsepcgc_eval)
- [structured_sparsepcgc_data](structured_sparsepcgc_data)
- [sparsepcgc_results](sparsepcgc_results)

## 5) Custom notebooks and utility workflows

Notebooks used for experiment iteration and analysis:

- [build_ply.ipynb](build_ply.ipynb)
- [deletion.ipynb](deletion.ipynb)
- [display_boxes.ipynb](display_boxes.ipynb)
- [evaluate_visual.ipynb](evaluate_visual.ipynb)
- [work.ipynb](work.ipynb)

Utility scripts used in custom runs:

- [display_bin.py](display_bin.py)
- [misc/vis_data_gt.py](misc/vis_data_gt.py)

## 6) Personal implementation notes

Detailed development notes are tracked in [misc/log.md](misc/log.md), including augmentation details, training/debug notes, and evaluation follow-ups.

## 7) Notebook-based evaluation summary

Source notebook: [evaluate_visual.ipynb](evaluate_visual.ipynb)

Work completed in this notebook:

- Loaded reconstructed-evaluation predictions, ground truth annotations, and baseline PointPillars predictions from pkl files.
- Built dataframe-based loaders for per-image analysis and class filtering (Pedestrian, Car, Cyclist).
- Added utility code for IoU and overlap checks to support comparison logic.
- Produced class-wise comparison plots between Ground Truth vs Reconstructed and Ground Truth vs PointPillars.
- Computed detection counts and derived Recall, Precision, and F1 for each class.

Detection results captured from notebook outputs:

| Class | Model | TP | FN | FP | Recall | Precision | F1 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pedestrian | Reconstructed | 687 | 137 | 718 | 0.834 | 0.489 | 0.616 |
| Pedestrian | PointPillars | 768 | 56 | 1127 | 0.932 | 0.405 | 0.565 |
| Car | Reconstructed | 3423 | 0 | 345 | 1.000 | 0.908 | 0.952 |
| Car | PointPillars | 3410 | 13 | 185 | 0.996 | 0.949 | 0.972 |
| Cyclist | Reconstructed | 286 | 341 | 266 | 0.456 | 0.518 | 0.485 |
| Cyclist | PointPillars | 496 | 131 | 564 | 0.791 | 0.468 | 0.588 |

Key findings from these results:

- Pedestrian: reconstructed output has lower recall but noticeably better precision than PointPillars, yielding a slightly higher F1.
- Car: reconstructed output reaches perfect recall, but PointPillars has much better precision and higher F1 overall.
- Cyclist: reconstructed output reduces false positives (higher precision) but misses many more ground-truth instances (much lower recall), so F1 is lower than PointPillars.
