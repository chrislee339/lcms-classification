
"""
GradCAM visualization for LC-MS CNN classification models.

Generates:
  - Per-image 3-panel visualizations (Original | GradCAM Overlay | Heatmap)
  - Aggregate heatmaps (mean/std) for correctly classified Class A and Class B
  - Side-by-side comparison of mean Class A vs Class B attention

Usage:
  python run_gradcam_analysis.py \
      --model_dir models/resnet_v2/arch_resnet50_256_seed42_fold0/ \
      --data_dir data/kfold/v2_256_by_patient_rmS11/fold_0/test/ \
      --output_dir gradcam_outputs/fold_0/ \
      --variant resnet50
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm





RESNET_CONFIGS = {
    "resnet18":  (models.resnet18,  models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34":  (models.resnet34,  models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50":  (models.resnet50,  models.ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
    "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
}


class TransferResNet(nn.Module):
    def __init__(self, variant="resnet50"):
        super().__init__()
        model_fn, weights = RESNET_CONFIGS[variant]
        self.backbone = model_fn(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(-1)





class GradCAM:
    """Compute GradCAM heatmap for a given target layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None


        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class=None):
        """
        Compute GradCAM heatmap.

        Args:
            input_tensor: preprocessed image tensor [1, C, H, W]
            target_class: 0 for Class A, 1 for Class B, None to use predicted class

        Returns:
            heatmap: numpy array [H, W] in [0, 1]
            prediction: raw logit (float)
        """
        self.model.eval()
        input_tensor.requires_grad_(True)


        output = self.model(input_tensor)



        self.model.zero_grad()
        if target_class is not None:
            grad = torch.ones_like(output) if target_class == 1 else -torch.ones_like(output)
            output.backward(gradient=grad)
        else:
            output.backward()


        weights = self.gradients.mean(dim=(2, 3), keepdim=True)


        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)


        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()


        cam_resized = np.array(
            Image.fromarray(cam).resize(
                (input_tensor.shape[3], input_tensor.shape[2]),
                Image.BILINEAR,
            )
        )

        return cam_resized, output.item()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()





def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def denormalize(tensor):
    """Convert normalized tensor back to displayable image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    return img.detach().permute(1, 2, 0).numpy()


def overlay_heatmap(image_np, heatmap, alpha=0.4):
    """Overlay heatmap on image. image_np in [0,1], heatmap in [0,1]."""
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap)[:, :, :3]
    overlay = (1 - alpha) * image_np + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)


def save_three_panel(image_np, heatmap, overlay, filepath, title=""):
    """Save a 3-panel figure: Original | GradCAM Overlay | Heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("GradCAM Overlay")
    axes[1].axis("off")

    axes[2].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Heatmap")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_aggregate_figure(mean_a, std_a, mean_b, std_b, filepath):
    """Save 2x2 aggregate figure: mean/std for each class."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    im0 = axes[0, 0].imshow(mean_a, cmap="jet", vmin=0, vmax=1)
    axes[0, 0].set_title("Class A (Negative) - Mean")
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(std_a, cmap="viridis", vmin=0)
    axes[0, 1].set_title("Class A (Negative) - Std")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(mean_b, cmap="jet", vmin=0, vmax=1)
    axes[1, 0].set_title("Class B (Positive) - Mean")
    axes[1, 0].axis("off")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(std_b, cmap="viridis", vmin=0)
    axes[1, 1].set_title("Class B (Positive) - Std")
    axes[1, 1].axis("off")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    plt.suptitle("Aggregate GradCAM Heatmaps (Correctly Classified)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(mean_a, mean_b, filepath):
    """Save side-by-side comparison of mean Class A vs Class B attention."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(mean_a, cmap="jet", vmin=0, vmax=1)
    axes[0].set_title("Class A (Negative)\nMean Attention", fontsize=12)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(mean_b, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Class B (Positive)\nMean Attention", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)


    diff = mean_b - mean_a
    vabs = max(abs(diff.min()), abs(diff.max())) or 1.0
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    axes[2].set_title("Difference\n(Class B - Class A)", fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.suptitle("GradCAM Attention Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)





def load_model(model_dir, variant, device):
    """Load the best model from a directory."""
    model_path = os.path.join(model_dir, "best_model_1.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = TransferResNet(variant=variant)
    state = torch.load(model_path, map_location=device, weights_only=True)

    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v
    model.load_state_dict(new_state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def get_label_from_filename(filename):
    """Determine class label from filename: 'A' in name = 0, 'B' = 1."""
    base = os.path.splitext(filename)[0]
    if "_aug" in base:
        base = base.split("_aug")[0]
    if base.endswith("A"):
        return 0
    elif base.endswith("B"):
        return 1
    else:
        raise ValueError(f"Cannot determine label from filename: {filename}")


def run_gradcam_single_fold(model_dir, data_dir, output_dir, variant="resnet50"):
    """Run GradCAM analysis on a single fold."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {model_dir}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")


    model = load_model(model_dir, variant, device)
    target_layer = model.backbone.layer4
    gradcam = GradCAM(model, target_layer)

    transform = get_transform()


    individual_dir = os.path.join(output_dir, "individual")
    os.makedirs(individual_dir, exist_ok=True)


    image_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".png") and "_aug" not in f
    ])

    if not image_files:
        print(f"No test images found in {data_dir}")
        return None

    print(f"Processing {len(image_files)} test images...")


    heatmaps_a_correct = []
    heatmaps_b_correct = []
    results = []

    for fname in image_files:
        img_path = os.path.join(data_dir, fname)
        img = Image.open(img_path).convert("RGB")


        true_label = get_label_from_filename(fname)


        input_tensor = transform(img).unsqueeze(0).to(device)
        heatmap, logit = gradcam(input_tensor, target_class=true_label)

        pred_prob = torch.sigmoid(torch.tensor(logit)).item()
        pred_label = 1 if pred_prob > 0.5 else 0
        correct = pred_label == true_label

        true_class = "A" if true_label == 0 else "B"
        pred_class = "A" if pred_label == 0 else "B"
        status = "CORRECT" if correct else "WRONG"
        print(f"  {fname}: true={true_class} pred={pred_class} prob={pred_prob:.3f} [{status}]")

        results.append({
            "filename": fname,
            "true_label": true_label,
            "pred_label": pred_label,
            "pred_prob": pred_prob,
            "correct": correct,
        })


        if correct:
            if true_label == 0:
                heatmaps_a_correct.append(heatmap)
            else:
                heatmaps_b_correct.append(heatmap)


        image_np = denormalize(input_tensor.squeeze(0))
        overlay_img = overlay_heatmap(image_np, heatmap)
        panel_path = os.path.join(individual_dir, f"gradcam_{os.path.splitext(fname)[0]}.png")
        save_three_panel(
            image_np, heatmap, overlay_img, panel_path,
            title=f"{fname} | True: {true_class} | Pred: {pred_class} (p={pred_prob:.3f}) | {status}"
        )


    np.savez(
        os.path.join(output_dir, "heatmaps_raw.npz"),
        heatmaps_a=np.array(heatmaps_a_correct) if heatmaps_a_correct else np.array([]),
        heatmaps_b=np.array(heatmaps_b_correct) if heatmaps_b_correct else np.array([]),
    )


    aggregate_data = {}
    if heatmaps_a_correct:
        stack_a = np.stack(heatmaps_a_correct)
        mean_a = stack_a.mean(axis=0)
        std_a = stack_a.std(axis=0)
        aggregate_data["mean_a"] = mean_a
        aggregate_data["std_a"] = std_a
        print(f"  Class A correct: {len(heatmaps_a_correct)} samples")
    else:
        print("  WARNING: No correctly classified Class A samples")
        mean_a = np.zeros((224, 224))
        std_a = np.zeros((224, 224))
        aggregate_data["mean_a"] = mean_a
        aggregate_data["std_a"] = std_a

    if heatmaps_b_correct:
        stack_b = np.stack(heatmaps_b_correct)
        mean_b = stack_b.mean(axis=0)
        std_b = stack_b.std(axis=0)
        aggregate_data["mean_b"] = mean_b
        aggregate_data["std_b"] = std_b
        print(f"  Class B correct: {len(heatmaps_b_correct)} samples")
    else:
        print("  WARNING: No correctly classified Class B samples")
        mean_b = np.zeros((224, 224))
        std_b = np.zeros((224, 224))
        aggregate_data["mean_b"] = mean_b
        aggregate_data["std_b"] = std_b


    np.savez(os.path.join(output_dir, "aggregate_heatmaps.npz"), **aggregate_data)

    save_aggregate_figure(
        mean_a, std_a, mean_b, std_b,
        os.path.join(output_dir, "aggregate_heatmaps.png")
    )


    save_comparison_figure(
        mean_a, mean_b,
        os.path.join(output_dir, "class_comparison.png")
    )


    gradcam.remove_hooks()

    print(f"\nDone. Results saved to {output_dir}")
    return results


def aggregate_across_folds(fold_dirs, output_dir):
    """
    Aggregate GradCAM heatmaps across multiple folds.
    Reads heatmaps_raw.npz from each fold directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_a = []
    all_b = []

    for fd in fold_dirs:
        npz_path = os.path.join(fd, "heatmaps_raw.npz")
        if not os.path.exists(npz_path):
            print(f"  Skipping {fd} (no heatmaps_raw.npz)")
            continue

        data = np.load(npz_path, allow_pickle=True)
        ha = data["heatmaps_a"]
        hb = data["heatmaps_b"]

        if ha.ndim == 3 and ha.shape[0] > 0:
            all_a.append(ha)
        if hb.ndim == 3 and hb.shape[0] > 0:
            all_b.append(hb)

    if not all_a and not all_b:
        print("No heatmap data found across folds.")
        return

    if all_a:
        stack_a = np.concatenate(all_a, axis=0)
        mean_a = stack_a.mean(axis=0)
        std_a = stack_a.std(axis=0)
        print(f"  Class A total correct across folds: {stack_a.shape[0]}")
    else:
        mean_a = np.zeros((224, 224))
        std_a = np.zeros((224, 224))

    if all_b:
        stack_b = np.concatenate(all_b, axis=0)
        mean_b = stack_b.mean(axis=0)
        std_b = stack_b.std(axis=0)
        print(f"  Class B total correct across folds: {stack_b.shape[0]}")
    else:
        mean_b = np.zeros((224, 224))
        std_b = np.zeros((224, 224))

    np.savez(os.path.join(output_dir, "aggregate_heatmaps_all_folds.npz"),
             mean_a=mean_a, std_a=std_a, mean_b=mean_b, std_b=std_b)

    save_aggregate_figure(
        mean_a, std_a, mean_b, std_b,
        os.path.join(output_dir, "aggregate_heatmaps_all_folds.png")
    )

    save_comparison_figure(
        mean_a, mean_b,
        os.path.join(output_dir, "class_comparison_all_folds.png")
    )

    print(f"  Cross-fold aggregation saved to {output_dir}")





def main():
    parser = argparse.ArgumentParser(
        description="GradCAM visualization for LC-MS classification models"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")


    single_parser = subparsers.add_parser("single", help="Run GradCAM on a single fold")
    single_parser.add_argument("--model_dir", type=str, required=True,
                               help="Path to model directory containing best_model_1.pt")
    single_parser.add_argument("--data_dir", type=str, required=True,
                               help="Path to test data directory")
    single_parser.add_argument("--output_dir", type=str, required=True,
                               help="Where to save GradCAM outputs")
    single_parser.add_argument("--variant", type=str, default="resnet50",
                               choices=list(RESNET_CONFIGS.keys()),
                               help="ResNet variant (default: resnet50)")


    agg_parser = subparsers.add_parser("aggregate", help="Aggregate GradCAM across folds")
    agg_parser.add_argument("--fold_dirs", type=str, nargs="+", required=True,
                            help="Paths to per-fold GradCAM output directories")
    agg_parser.add_argument("--output_dir", type=str, required=True,
                            help="Where to save aggregated outputs")

    args = parser.parse_args()

    if args.command is None:

        parser2 = argparse.ArgumentParser(
            description="GradCAM visualization for LC-MS classification models"
        )
        parser2.add_argument("--model_dir", type=str, required=True,
                             help="Path to model directory containing best_model_1.pt")
        parser2.add_argument("--data_dir", type=str, required=True,
                             help="Path to test data directory")
        parser2.add_argument("--output_dir", type=str, required=True,
                             help="Where to save GradCAM outputs")
        parser2.add_argument("--variant", type=str, default="resnet50",
                             choices=list(RESNET_CONFIGS.keys()),
                             help="ResNet variant (default: resnet50)")
        args = parser2.parse_args()
        run_gradcam_single_fold(args.model_dir, args.data_dir, args.output_dir, args.variant)

    elif args.command == "single":
        run_gradcam_single_fold(args.model_dir, args.data_dir, args.output_dir, args.variant)

    elif args.command == "aggregate":
        aggregate_across_folds(args.fold_dirs, args.output_dir)


if __name__ == "__main__":
    main()
