#!/usr/bin/env python3
import sys
import logging
from collections import defaultdict
from logging import Logger
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import lightning as L
import torch
import torchvision.transforms as T
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from data import CUBDataset
from modeling.backbone import DINOv2Backbone, DINOv2BackboneExpanded, DINOBackboneExpanded
from modeling.pnp import PNP, PNPCriterion, SamMasks
from modeling.utils import print_parameters

import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumentationsTransform:
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.Sharpen(alpha=(0.2, 0.5), p=0.3),
                A.ToGray(p=0.3),
                A.CLAHE(p=0.3),
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        return self.transform(image=img)["image"]

transforms = AlbumentationsTransform()


def train(model: nn.Module, criterion: nn.Module | None, dataloader: DataLoader, epoch: int,
          optimizer: optim.Optimizer | None, logger: Logger, device: torch.device, sam_helper: SamMasks):
    model.train()
    running_losses = defaultdict(float)
    mca_train = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(item.to(device) for item in batch)
        images, labels = batch[:2]
        
        # Convert normalized images back to [0,1] range for SAM
        # Denormalize: (x - mean) / std -> x
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        images_for_sam = images * std + mean
        images_for_sam = torch.clamp(images_for_sam, 0, 1)
        
        sam_masks = sam_helper.generate_masks(images_for_sam)  # Use denormalized images for SAM
        # for i in range(sam_masks.shape[0]):
        #     plt.imshow(images_for_sam[i].permute(1, 2, 0).cpu().numpy())
        #     plt.imshow(sam_masks[i].cpu().numpy(), alpha=0.5, cmap="Greens")
        #     plt.axis("off")
        #     plt.title("SAM Mask")
        #     plt.show()
        outputs = model(images, labels=labels, sam_masks=sam_masks)

        if criterion is not None and optimizer is not None:
            loss_dict = criterion(outputs, batch)  # type: dict[str, torch.Tensor]
            loss = sum(val for key, val in loss_dict.items() if not key.startswith("_"))

            if not isinstance(loss, torch.Tensor):
                raise ValueError

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for k, v in loss_dict.items():
                running_losses[k] += v.item() * dataloader.batch_size

        mca_train(outputs["class_logits"], labels)

    for k, v in running_losses.items():
        loss_avg = v / len(dataloader.dataset)
        logger.info(f"EPOCH {epoch} train {k}: {loss_avg:.4f}")

    epoch_acc_train = mca_train.compute().item()
    logger.info(f"EPOCH {epoch} train acc: {epoch_acc_train:.4f}")


@torch.inference_mode()
def test(model: nn.Module, dataloader: DataLoader, epoch: int,
         logger: Logger, device: torch.device):
    model.eval()
    mca_test = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(item.to(device) for item in batch)
        images, labels = batch[:2]

        outputs = model(images)

        mca_test(outputs["class_logits"], labels)

    epoch_acc_test = mca_test.compute().item()
    logger.info(f"EPOCH {epoch} test acc: {epoch_acc_test:.4f}")

    return epoch_acc_test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data-root", type=str, default="./datasets")
    parser.add_argument("--dataset", type=str, default="CUB", choices=["CUB"])

    parser.add_argument("--backbone", type=str, default="dinov2_vitb14", choices=["dinov2_vitb14", "dinov2_vits14"])
    parser.add_argument("--num-splits", type=int, default=1)

    # Model related hyperparameters
    parser.add_argument("--num-prototypes", type=int, default=5, help="Number of prototypes per class")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sa-initial-value", type=float, default=0.5)

    # Optimization hyperparameters
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--classifier-lr", type=float, default=1.0e-6)
    parser.add_argument("--fine-tuning-start-epoch", type=int, default=0)

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler((log_dir / "train.log").as_posix()),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logger = logging.getLogger(__name__)

    L.seed_everything(args.seed)

    # normalize = T.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))
    # transforms = T.Compose([
    #     T.Resize((224, 224,)),
    #     T.ToTensor(),
    #     normalize
    # ])

    if args.dataset == "CUB":
        logger.info("Train on CUB-200-2011")
        n_classes = 69
        dataset_dir = Path(args.data_root)
        dataset_train = CUBDataset((dataset_dir / "train").as_posix(),
                                   transforms=transforms)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=128, num_workers=8, shuffle=True)

        dataset_test = CUBDataset((dataset_dir / "test").as_posix(),
                                   transforms=transforms)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, num_workers=8, shuffle=True)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented")

    if "dinov2" in args.backbone:
        if args.num_splits and args.num_splits > 0:
            backbone = DINOv2BackboneExpanded(
                name=args.backbone,
                n_splits=args.num_splits,
                mode="block_expansion",
                freeze_norm_layer=True
            )
        else:
            backbone = DINOv2Backbone(name=args.backbone)
        dim = backbone.dim
    elif "dino" in args.backbone:
        backbone = DINOBackboneExpanded(
            name=args.backbone,
            n_splits=args.num_splits,
            mode="block_expansion",
            freeze_norm_layer=True
        )
        dim = backbone
    else:
        raise NotImplementedError(f"Backbone {args.backbone} not implemented.")

    # Can be substituted with other off-the-shelf methods
    sam_helper = SamMasks(checkpoint_path="ckpt/sam_vit_b_01ec64.pth", device=device)

    net = PNP(
        backbone=backbone,
        dim=dim,
        n_prototypes=args.num_prototypes,
        n_classes=n_classes,
        gamma=args.gamma,
        temperature=args.temperature,
        sa_init=args.sa_initial_value,
        norm_prototypes=False
    )
    criterion = PNPCriterion(l_ppd_coef=0.8, n_prototypes=args.num_prototypes, num_classes=n_classes)

    net.to(device)

    best_epoch, best_test_epoch = 0, 0.0

    for epoch in range(args.epochs):
        is_fine_tuning = epoch >= args.fine_tuning_start_epoch

        # Stage 2 training
        if is_fine_tuning:
            logger.info("Start fine-tuning backbone...")
            for name, param in net.named_parameters():
                param.requires_grad = ("backbone" not in name)

            net.backbone.set_requires_grad()

            param_groups = [{'params': net.backbone.learnable_parameters(),
                             'lr': args.backbone_lr}]
            param_groups += [{'params': net.classifier.parameters(), 'lr': args.classifier_lr}]

            optimizer = optim.Adam(param_groups)

            net.optimizing_prototypes = False
        # Stage 1 training
        else:
            for params in net.parameters():
                params.requires_grad = False
            optimizer = None
            net.optimizing_prototypes = True

        if epoch > 0:
            net.initializing = False

        print_parameters(net=net, logger=logger)
        logger.info(f"net.initializing: {net.initializing}")
        logger.info(f"net.optimizing_prototypes: {net.optimizing_prototypes}")

        train(
            model=net,
            criterion=criterion if is_fine_tuning else None,
            dataloader=dataloader_train,
            epoch=epoch,
            optimizer=optimizer if is_fine_tuning else None,
            logger=logger,
            device=device,
            sam_helper=sam_helper
        )

        epoch_acc_test = test(model=net, dataloader=dataloader_test, epoch=epoch, logger=logger, device=device)
        
        torch.save(
            dict(
                state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
                hparams=vars(args),
            ),
            log_dir / "ckpt.pth"
        )
        logger.info("Model saved as ckpt.pth")

        if epoch_acc_test > best_test_epoch:
            best_val_acc = epoch_acc_test
            best_epoch = epoch

    logger.info(f"DONE! Best epoch is epoch {best_epoch} with accuracy {best_val_acc}.")


if __name__ == '__main__':
    main()
