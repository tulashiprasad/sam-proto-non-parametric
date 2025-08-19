from math import sqrt
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import nn
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np

from .utils import momentum_update, sinkhorn_knopp

class SamMasks:
    def __init__(self, model_type="vit_b", checkpoint_path="sam_vit_b_01ec64.pth", device="cuda"):
        self.device = device
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
        self.predictor = SamPredictor(self.model)

    @torch.no_grad()
    def generate_masks(self, images):
        """
        Args:
            images: Tensor of shape [B, C, H, W], values in [0, 1]

        Returns:
            Tensor of shape [B, H, W] with binary masks (bool or 0/1)
        """
        masks = []
        for img_tensor in images:
            # Convert to uint8 BGR image
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            # img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            self.predictor.set_image(img_np)
            # Prompt: use center point
            H, W = img_np.shape[:2]
            input_point = np.array([[W // 2, H // 2]])
            input_label = np.array([1])

            sam_masks, scores, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            best_idx = np.argmax(scores)

            best_mask = sam_masks[best_idx]  # [H, W]
            masks.append(torch.tensor(best_mask, dtype=torch.bool, device=images.device))

        return torch.stack(masks)  # [B, H, W]

class ScoreAggregation(nn.Module):
    def __init__(self, init_val: float = 0.2, n_classes: int = 200, n_prototypes: int = 5) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.full((n_classes, n_prototypes,), init_val, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        n_classes, n_prototypes = self.weights.shape
        sa_weights = F.softmax(self.weights, dim=-1) * n_prototypes
        x = x * sa_weights  # B C K
        x = x.sum(-1)  # B C
        return x

class PNP(nn.Module):
    def __init__(self, backbone: nn.Module,
                 *,
                 always_norm_patches: bool = True, gamma: float = 0.999, n_prototypes: int = 5,
                 n_classes: int = 200, norm_prototypes=False, temperature: float = 0.2,
                 sa_init: float = 0.5, dim: int = 768, use_gumbel: bool = False,
                 assignment_threshold: float = 0.8):
        super().__init__()
        self.gamma = gamma
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.C = n_classes + 1
        self.backbone = backbone
        self.use_gumbel = use_gumbel
        self.assignment_threshold = assignment_threshold

        self.dim = dim
        self.register_buffer("prototypes", torch.randn(self.C, self.n_prototypes, self.dim))
        self.temperature = temperature

        nn.init.trunc_normal_(self.prototypes, std=0.02)

        self.classifier = ScoreAggregation(init_val=sa_init, n_classes=n_classes, n_prototypes=n_prototypes)

        self.optimizing_prototypes = True
        self.initializing = True
        self.always_norm_patches = always_norm_patches
        self.norm_prototypes = norm_prototypes

    def soft_kmeans_assignment(self, logits: torch.Tensor, use_gumbel: bool = False):
        if use_gumbel:
            assignments = F.gumbel_softmax(logits, tau=0.5, hard=True, dim=-1)
            indices = torch.argmax(assignments, dim=-1)
        else:
            assignments = F.softmax(logits, dim=-1)
            max_vals, indices = torch.max(assignments, dim=-1)
            if self.assignment_threshold > 0:
                mask = max_vals < self.assignment_threshold
                assignments[mask] = 0.0
                assignments = F.normalize(assignments, p=1, dim=-1)
        return assignments, indices

    def online_clustering(self, prototypes: torch.Tensor,
                          patch_tokens: torch.Tensor,
                          patch_prototype_logits: torch.Tensor,
                          patch_labels: torch.Tensor,
                          *,
                          gamma: float = 0.999,
                          use_gumbel: bool = False):
        B, H, W = patch_labels.shape
        C, K, dim = prototypes.shape

        patch_labels_flat = patch_labels.flatten()
        patches_flat = rearrange(patch_tokens, "B n_patches dim -> (B n_patches) dim")
        L = rearrange(patch_prototype_logits, "B n_patches C K -> (B n_patches) C K")

        P_old = prototypes.clone()
        P_new = prototypes.clone()

        part_assignment_maps = torch.empty_like(patch_labels_flat)

        for c in patch_labels.unique().tolist():
            class_fg_mask = patch_labels_flat == c
            I_c = patches_flat[class_fg_mask]
            L_c = L[class_fg_mask, c, :]
            L_c_assignment, L_c_assignment_indices = self.soft_kmeans_assignment(L_c, use_gumbel=use_gumbel)
            P_c_new = torch.mm(L_c_assignment.t(), I_c)
            P_c_old = P_old[c, :, :]
            P_new[c, ...] = momentum_update(P_c_old, P_c_new, momentum=gamma)
            part_assignment_maps[class_fg_mask] = L_c_assignment_indices + c * K

        part_assignment_maps = rearrange(part_assignment_maps, "(B H W) -> B (H W)", B=B, H=H, W=W)

        return part_assignment_maps, P_new

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None, *,
                use_gumbel: bool = False, sam_masks: torch.Tensor | None = None):
        assert (not self.training) or (labels is not None)

        patch_tokens, raw_patch_tokens, cls_tokens = self.backbone(x)

        patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)
        prototype_norm = F.normalize(self.prototypes, p=2, dim=-1)

        patch_prototype_logits = einsum(patch_tokens, prototype_norm, "B N D, C K D -> B N C K")
        image_prototype_logits = patch_prototype_logits.max(1).values

        class_logits = self.classifier(image_prototype_logits[:, :-1, :]) / self.temperature

        outputs = {
            "patch_prototype_logits": patch_prototype_logits,
            "image_prototype_logits": image_prototype_logits,
            "class_logits": class_logits
        }

        if labels is not None:
            B, N, D = raw_patch_tokens.shape
            H = W = int(N ** 0.5)
            raw_patch_tokens = F.normalize(raw_patch_tokens, p=2, dim=-1)

            if sam_masks is None:
                raise ValueError("SAM masks must be provided for foreground extraction")

            if sam_masks.dtype != torch.float32:
                sam_masks = sam_masks.float()

            patch_masks = F.interpolate(sam_masks.unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)
            pseudo_patch_labels = torch.full((B, H, W), fill_value=self.n_classes, dtype=torch.long, device=x.device)
            for i in range(B):
                pseudo_patch_labels[i][patch_masks[i] == 1] = labels[i]

            part_assignment_maps, new_prototypes = self.online_clustering(
                prototypes=self.prototypes,
                patch_tokens=raw_patch_tokens.detach(),
                patch_prototype_logits=patch_prototype_logits.detach(),
                patch_labels=pseudo_patch_labels,
                gamma=self.gamma,
                use_gumbel=self.use_gumbel
            )

            if self.training and self.optimizing_prototypes:
                self.prototypes = F.normalize(new_prototypes, p=2, dim=-1) if self.norm_prototypes else new_prototypes

            outputs.update({
                "patches": raw_patch_tokens,
                "part_assignment_maps": part_assignment_maps,
                "pseudo_patch_labels": pseudo_patch_labels
            })

        return outputs

    def get_attn_maps(self, images: torch.Tensor, labels: torch.Tensor):
        outputs = self(images, labels)
        patch_prototype_logits = outputs["patch_prototype_logits"]

        batch_size, n_patches, C, K = patch_prototype_logits.shape
        H = W = int(sqrt(n_patches))

        patch_prototype_logits = rearrange(patch_prototype_logits, "B (H W) C K -> B C K H W", H=H, W=W)
        patch_prototype_logits = patch_prototype_logits[torch.arange(labels.numel()), labels, ...]  # B K H W

        pooled_logits = F.avg_pool2d(patch_prototype_logits, kernel_size=(2, 2,), stride=2)
        return patch_prototype_logits, pooled_logits

    def push_forward(self, x: torch.Tensor):
        patch_tokens, _, cls_tokens = self.backbone(x)  # shape: [B, n_patches, dim,]
        patch_tokens_norm = F.normalize(patch_tokens, p=2, dim=-1)
        prototype_norm = F.normalize(self.prototypes, p=2, dim=-1)
        if not self.initializing:
            patch_prototype_logits = einsum(patch_tokens_norm, prototype_norm,
                                            "B n_patches dim, C K dim -> B n_patches C K")
        else:
            patch_prototype_logits = einsum(patch_tokens_norm, prototype_norm,
                                            "B n_patches dim, C K dim -> B n_patches C K")
        batch_size, n_patches, C, K = patch_prototype_logits.shape
        H = W = int(sqrt(n_patches))
        prototype_logits = rearrange(patch_prototype_logits[:, :, :-1, :], "B (H W) C K -> B (C K) H W", H=H, W=W)
        return None, F.avg_pool2d(prototype_logits, kernel_size=(2, 2,), stride=2)


class PNPCriterion(nn.Module):
    def __init__(
            self,
            l_ppd_coef: float = 0,
            l_ppd_temp: float = 0.1,

            num_classes: int = 200,
            n_prototypes: int = 5,
            bg_class_weight: float = 0.1
    ) -> None:
        super().__init__()
        self.l_ppd_coef = l_ppd_coef
        self.l_ppd_temp = l_ppd_temp

        self.xe = nn.CrossEntropyLoss()

        self.C = num_classes
        self.K = n_prototypes
        self.class_weights = torch.tensor([1] * self.C * self.K + [bg_class_weight] * self.K)

    def forward(self, outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...]):
        logits = outputs["class_logits"]
        patch_prototype_logits = outputs["patch_prototype_logits"]
        part_assignment_maps = outputs["part_assignment_maps"]

        labels = batch[1]

        loss_dict = dict()
        loss_dict["l_y"] = self.xe(logits, labels)

        if self.l_ppd_coef != 0:
            l_ppd = self.ppd_criterion(
                patch_prototype_logits,
                part_assignment_maps,
                class_weight=self.class_weights.to(dtype=torch.float32, device=logits.device),
                temperature=self.l_ppd_temp
            )
            loss_dict["l_ppd"] = self.l_ppd_coef * l_ppd
            loss_dict["_l_ppd_unadjusted"] = l_ppd

        return loss_dict
    
    @staticmethod
    def ppd_criterion(patch_prototype_logits: torch.Tensor,
                      patch_prototype_assignments: torch.Tensor,
                      class_weight: torch.Tensor,
                      temperature: float = 0.1):
        patch_prototype_logits = rearrange(patch_prototype_logits, "B N C K -> B (C K) N") / temperature
        loss = F.cross_entropy(patch_prototype_logits, target=patch_prototype_assignments, weight=class_weight)
        return loss
