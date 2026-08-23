import argparse
import os
import torch
from tqdm import tqdm
import pdb

from pointpillars.utils import setup_seed
from pointpillars.dataset import Kitti, get_dataloader
from pointpillars.model import PointPillars
from pointpillars.loss import Loss
from torch.utils.tensorboard import SummaryWriter


def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f"{tag}/{k}", v, global_step)
    if lr is not None:
        writer.add_scalar("lr", lr, global_step)
    if momentum is not None:
        writer.add_scalar("momentum", momentum, global_step)


def main(args):
    setup_seed()
    train_dataset = Kitti(
        data_root=args.data_root,
        split="train",
        intensity_filling_strategy=args.intensity_filling_strategy,
        intensity_forced_value=args.intensity_forced_value,
    )
    val_dataset = Kitti(
        data_root=args.data_root,
        split="val",
        intensity_filling_strategy=args.intensity_filling_strategy,
        intensity_forced_value=args.intensity_forced_value,
    )
    train_dataloader = get_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_dataloader = get_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    voxel_x = args.voxel_x
    voxel_y = args.voxel_y
    voxel_z = args.voxel_z
    voxel_size_base = [0.16, 0.16, 4]
    num_of_max_points_per_voxel_base = 32
    max_num_voxels_base = (16000, 40000)  # for train and test
    training_voxel_size = [voxel_x, voxel_y, voxel_z]
    training_max_num_points_per_voxel = int(
        num_of_max_points_per_voxel_base
        * (
            voxel_x
            * voxel_y
            * voxel_z
            / (voxel_size_base[0] * voxel_size_base[1] * voxel_size_base[2])
        )
    )
    training_max_num_voxels = (
        int(
            max_num_voxels_base[0]
            * (voxel_x * voxel_y / (voxel_size_base[0] * voxel_size_base[1]))
        ),
        int(
            max_num_voxels_base[1]
            * (voxel_x * voxel_y / (voxel_size_base[0] * voxel_size_base[1]))
        ),
    )
    print(
        f"voxel_size= {training_voxel_size}, training_max_num_points_per_voxel= {training_max_num_points_per_voxel}, training_max_num_voxels= {training_max_num_voxels}"
    )
    if not args.no_cuda:
        print("using cuda")

        pointpillars = PointPillars(
            nclasses=args.nclasses,
            voxel_size=training_voxel_size,
            max_num_points=training_max_num_points_per_voxel,
            max_voxels=max_num_voxels_base,
        ).cuda()
    else:
        print("using cpu")
        pointpillars = PointPillars(
            nclasses=args.nclasses,
            voxel_size=training_voxel_size,
            max_num_points=training_max_num_points_per_voxel,
            max_voxels=training_max_num_voxels,
        )
    loss_func = Loss()

    max_iters = len(train_dataloader) * (args.max_epoch + 40)
    init_lr = args.init_lr
    optimizer = torch.optim.AdamW(
        params=pointpillars.parameters(),
        lr=init_lr,
        betas=(0.95, 0.99),
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=init_lr * 5,
        total_steps=max_iters,
        pct_start=0.4,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.95 * 0.895,
        max_momentum=0.95,
        div_factor=10,
    )
    saved_logs_path = os.path.join(args.saved_path, "summary")
    os.makedirs(saved_logs_path, exist_ok=True)
    writer = SummaryWriter(saved_logs_path)
    saved_ckpt_path = os.path.join(args.saved_path, "checkpoints")
    os.makedirs(saved_ckpt_path, exist_ok=True)

    # Resume logic
    start_epoch = 0
    if args.resume_ckpt is not None:
        print(f"Resuming from checkpoint: {args.resume_ckpt}")
        checkpoint = torch.load(
            args.resume_ckpt, map_location="cuda" if not args.no_cuda else "cpu"
        )
        # Support both dict and raw state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            pointpillars.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
        else:
            pointpillars.load_state_dict(checkpoint)
            start_epoch = int(args.resume_ckpt.split("_")[-1].split(".")[0])
        print(
            f"Resumed weights from {args.resume_ckpt}, starting at epoch {start_epoch}"
        )

    for epoch in range(start_epoch, args.max_epoch):
        print("=" * 20, epoch, "=" * 20)
        train_step, val_step = 0, 0
        for i, data_dict in enumerate(tqdm(train_dataloader)):
            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()

            optimizer.zero_grad()

            batched_pts = data_dict["batched_pts"]
            batched_gt_bboxes = data_dict["batched_gt_bboxes"]
            batched_labels = data_dict["batched_labels"]
            batched_difficulty = data_dict["batched_difficulty"]
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = (
                pointpillars(
                    batched_pts=batched_pts,
                    mode="train",
                    batched_gt_bboxes=batched_gt_bboxes,
                    batched_gt_labels=batched_labels,
                )
            )

            bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
            bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

            batched_bbox_labels = anchor_target_dict["batched_labels"].reshape(-1)
            batched_label_weights = anchor_target_dict["batched_label_weights"].reshape(
                -1
            )
            batched_bbox_reg = anchor_target_dict["batched_bbox_reg"].reshape(-1, 7)
            # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
            batched_dir_labels = anchor_target_dict["batched_dir_labels"].reshape(-1)
            # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)

            pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
            bbox_pred = bbox_pred[pos_idx]
            batched_bbox_reg = batched_bbox_reg[pos_idx]
            # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
            bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(
                batched_bbox_reg[:, -1].clone()
            )
            batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(
                batched_bbox_reg[:, -1].clone()
            )
            bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
            batched_dir_labels = batched_dir_labels[pos_idx]

            num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
            bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
            batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
            batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

            loss_dict = loss_func(
                bbox_cls_pred=bbox_cls_pred,
                bbox_pred=bbox_pred,
                bbox_dir_cls_pred=bbox_dir_cls_pred,
                batched_labels=batched_bbox_labels,
                num_cls_pos=num_cls_pos,
                batched_bbox_reg=batched_bbox_reg,
                batched_dir_labels=batched_dir_labels,
            )

            loss = loss_dict["total_loss"]
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(pointpillars.parameters(), max_norm=35)
            optimizer.step()
            scheduler.step()

            global_step = epoch * len(train_dataloader) + train_step + 1

            if global_step % args.log_freq == 0:
                save_summary(
                    writer,
                    loss_dict,
                    global_step,
                    "train",
                    lr=optimizer.param_groups[0]["lr"],
                    momentum=optimizer.param_groups[0]["betas"][0],
                )
            train_step += 1
        if (epoch + 1) % args.ckpt_freq_epoch == 0:
            torch.save(
                {
                    "model_state_dict": pointpillars.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                },
                os.path.join(saved_ckpt_path, f"epoch_{epoch + 1}.pth"),
            )

        if epoch % 2 == 0:
            continue
        pointpillars.eval()
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(val_dataloader)):
                if not args.no_cuda:
                    # move the tensors to the cuda
                    for key in data_dict:
                        for j, item in enumerate(data_dict[key]):
                            if torch.is_tensor(item):
                                data_dict[key][j] = data_dict[key][j].cuda()

                batched_pts = data_dict["batched_pts"]
                batched_gt_bboxes = data_dict["batched_gt_bboxes"]
                batched_labels = data_dict["batched_labels"]
                batched_difficulty = data_dict["batched_difficulty"]
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = (
                    pointpillars(
                        batched_pts=batched_pts,
                        mode="train",
                        batched_gt_bboxes=batched_gt_bboxes,
                        batched_gt_labels=batched_labels,
                    )
                )

                bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(
                    -1, args.nclasses
                )
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                batched_bbox_labels = anchor_target_dict["batched_labels"].reshape(-1)
                batched_label_weights = anchor_target_dict[
                    "batched_label_weights"
                ].reshape(-1)
                batched_bbox_reg = anchor_target_dict["batched_bbox_reg"].reshape(-1, 7)
                # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
                batched_dir_labels = anchor_target_dict["batched_dir_labels"].reshape(
                    -1
                )
                # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)

                pos_idx = (batched_bbox_labels >= 0) & (
                    batched_bbox_labels < args.nclasses
                )
                bbox_pred = bbox_pred[pos_idx]
                batched_bbox_reg = batched_bbox_reg[pos_idx]
                # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
                bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1]) * torch.cos(
                    batched_bbox_reg[:, -1]
                )
                batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1]) * torch.sin(
                    batched_bbox_reg[:, -1]
                )
                bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                batched_dir_labels = batched_dir_labels[pos_idx]

                num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
                bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
                batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

                loss_dict = loss_func(
                    bbox_cls_pred=bbox_cls_pred,
                    bbox_pred=bbox_pred,
                    bbox_dir_cls_pred=bbox_dir_cls_pred,
                    batched_labels=batched_bbox_labels,
                    num_cls_pos=num_cls_pos,
                    batched_bbox_reg=batched_bbox_reg,
                    batched_dir_labels=batched_dir_labels,
                )

                global_step = epoch * len(val_dataloader) + val_step + 1
                if global_step % args.log_freq == 0:
                    save_summary(writer, loss_dict, global_step, "val")
                val_step += 1
        pointpillars.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration Parameters")
    parser.add_argument(
        "--data_root",
        default="/mnt/ssd1/lifa_rdata/det/kitti",
        help="your data root for kitti",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument("--saved_path", default="pillar_logs")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--nclasses", type=int, default=3)
    parser.add_argument("--init_lr", type=float, default=0.00025)
    parser.add_argument("--max_epoch", type=int, default=160)
    parser.add_argument("--log_freq", type=int, default=8)
    parser.add_argument("--ckpt_freq_epoch", type=int, default=20)
    parser.add_argument("--no_cuda", action="store_true", help="whether to use cuda")
    parser.add_argument(
        "--intensity_filling_strategy",
        type=str,
        default="random_one",
        help="the strategy to fill the intensity of the reconstructed point cloud, options: random_one, linear_regression, conform_to_target_intensity, fixed_value:<float>,random,zeros, ones",
    )
    parser.add_argument("--voxel_x", type=float, default=0.16, help="voxel x size")
    parser.add_argument("--voxel_y", type=float, default=0.16, help="voxel y size")
    parser.add_argument("--voxel_z", type=float, default=4.0, help="voxel z size")
    args = parser.parse_args()
    if args.intensity_filling_strategy.startswith("fixed_value"):
        try:
            fixed_value = float(args.intensity_filling_strategy.split(":")[-1])
            args.intensity_filling_strategy = "fixed_value"
            args.intensity_forced_value = fixed_value
        except:
            raise ValueError(
                f"the format of fixed_value is not correct, got {args.intensity_filling_strategy}"
            )
    else:
        args.intensity_forced_value = None

    args_dict = vars(args)
    width = 20  # Adjust as needed

    for k, v in args_dict.items():
        print(f"{k:<{width}}: {v}")
    main(args)
