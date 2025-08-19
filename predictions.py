#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

from modeling.backbone import DINOv2Backbone, DINOv2BackboneExpanded
from modeling.pnp import PNP, SamMasks


def load_model(checkpoint_path: str, device: torch.device) -> PNP:
    """Load the trained PNP model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args_dict = checkpoint.get("hparams", {})
    
    # Initialize backbone
    backbone_name = args_dict.get("backbone", "dinov2_vitb14")
    num_splits = args_dict.get("num_splits", 1)
    
    if num_splits and num_splits > 0:
        backbone = DINOv2BackboneExpanded(
            name=backbone_name,
            n_splits=num_splits,
            mode="block_expansion",
            freeze_norm_layer=True
        )
    else:
        backbone = DINOv2Backbone(name=backbone_name)
    
    # Initialize PNP model
    model = PNP(
        backbone=backbone,
        dim=backbone.dim,
        n_prototypes=args_dict.get("num_prototypes", 3),
        n_classes=args_dict.get("n_classes", 69),
        gamma=args_dict.get("gamma", 0.99),
        temperature=args_dict.get("temperature", 0.2),
        sa_init=args_dict.get("sa_initial_value", 0.5),
        # use_sinkhorn=True,
        norm_prototypes=False
    )
    
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: str, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    """Preprocess a single image for inference and return original dimensions."""
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        normalize
    ])
    
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    return transforms(image).unsqueeze(0).to(device), original_size


def preprocess_image_for_sam(image_path: str, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    """Preprocess a single image for SAM (without normalization) and return original dimensions."""
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    return transforms(image).unsqueeze(0).to(device), original_size


def load_original_image(image_path: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Load the original image at full resolution."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    image_np = np.array(image)
    return image_np, original_size


def visualize_foreground_background(image_path: str, sam_masks: torch.Tensor, original_size: tuple[int, int]):
    """Visualize foreground and background patches on the input image at original resolution."""
    # Load original image
    image_np, _ = load_original_image(image_path)
    
    # Get SAM mask and resize to original image dimensions
    mask_np = sam_masks.squeeze(0).cpu().numpy().astype(bool)
    
    # Resize mask to original image dimensions
    mask_resized = cv2.resize(mask_np.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
    mask_resized = mask_resized.astype(bool)
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    
    # Original image
    ax1.imshow(image_np)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Foreground mask
    ax2.imshow(image_np)
    foreground_overlay = np.zeros_like(image_np)
    foreground_overlay[mask_resized] = [0, 255, 0]  # Green
    ax2.imshow(foreground_overlay, alpha=0.3)
    ax2.set_title('Foreground (Green)')
    ax2.axis('off')
    
    # Background mask
    ax3.imshow(image_np)
    background_overlay = np.zeros_like(image_np)
    background_overlay[~mask_resized] = [255, 0, 0]  # Red
    ax3.imshow(background_overlay, alpha=0.3)
    ax3.set_title('Background (Red)')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_prototypes(image_path: str, patch_prototype_logits: torch.Tensor, predicted_class: int, original_size: tuple[int, int]):
    """Visualize prototype activations on top of the input image at original resolution."""
    # Load original image
    image_np, _ = load_original_image(image_path)
    
    # Get prototype activations for predicted class
    n_prototypes = patch_prototype_logits.shape[-1]
    
    # Create visualization
    fig, axes = plt.subplots(1, n_prototypes + 1, figsize=(6*(n_prototypes + 1), 6))
    if n_prototypes == 1:
        axes = [axes[0], axes[1]]
    
    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prototype activations
    for proto_idx in range(n_prototypes):
        # Get activation map for this prototype and predicted class
        activation_map = patch_prototype_logits[0, :, predicted_class, proto_idx]
        
        # Reshape to 2D (assuming square patches)
        n_patches = activation_map.shape[0]
        H = W = int(n_patches ** 0.5)
        activation_map = activation_map.reshape(H, W)
        
        # Upsample to original image size
        activation_map_upsampled = F.interpolate(
            activation_map.unsqueeze(0).unsqueeze(0), 
            size=original_size[::-1],  # PIL size is (width, height), but torch expects (height, width)
            mode='bilinear', 
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # Normalize activation map
        activation_map_upsampled = (activation_map_upsampled - activation_map_upsampled.min()) / (activation_map_upsampled.ptp() + 1e-8)
        
        # Plot
        axes[proto_idx + 1].imshow(image_np)
        axes[proto_idx + 1].imshow(activation_map_upsampled, cmap='jet', alpha=0.6)
        axes[proto_idx + 1].set_title(f'Prototype {proto_idx + 1}')
        axes[proto_idx + 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--visualize", action="store_true", help="Show foreground/background visualization")
    parser.add_argument("--show-prototypes", action="store_true", help="Show prototype activations", default=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Initialize SAM
    sam_helper = SamMasks(checkpoint_path="ckpt/sam_vit_b_01ec64.pth", device=device)
    
    # Preprocess image
    image_tensor, original_size = preprocess_image(args.image, device)
    image_tensor_for_sam, _ = preprocess_image_for_sam(args.image, device)
    
    print(f"Original image size: {original_size[0]}x{original_size[1]}")
    
    # Inference
    with torch.no_grad():
        sam_masks = sam_helper.generate_masks(image_tensor_for_sam)
        outputs = model(image_tensor, sam_masks=sam_masks)
        
        # Get prediction
        class_logits = outputs["class_logits"]
        probabilities = F.softmax(class_logits, dim=1)
        predicted_class = torch.argmax(class_logits, dim=1).item()
        confidence = torch.max(probabilities, dim=1)[0].item()
        
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        
        # Visualization
        if args.visualize:
            visualize_foreground_background(args.image, sam_masks, original_size)
        
        if args.show_prototypes:
            patch_prototype_logits = outputs.get("patch_prototype_logits")
            if patch_prototype_logits is not None:
                visualize_prototypes(args.image, patch_prototype_logits, predicted_class, original_size)
            else:
                print("No prototype logits available for visualization")


if __name__ == "__main__":
    main() 