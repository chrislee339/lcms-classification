
"""
Unified experiment suite for LC-MS CNN classification paper.

Experiments:
  1. Architecture ablation: ResNet-18/34/50/101/152 on full-spectrum 256×256
  2. Resolution ablation: Full-spectrum 256/512/1024 with ResNet-34
  3. Subregion ablation: Bottom-left quadrant 256/512/1024 with ResNet-50

All experiments use:
  - Patient-stratified 6-fold cross-validation (excluding patient S11)
  - 3 random seeds for statistical robustness
  - Same augmentation strategy (ColorJitter, GaussianBlur, Noise)
  - Cosine annealing LR schedule with warmup

Key features:
  - 3 seeds (42, 69, 137) for mean ± std reporting
  - Cosine annealing with warmup (better for fine-tuning than ReduceLROnPlateau)
  - Initial LR 5e-4 (slightly lower, gentler fine-tuning)
  - Backbone frozen for first 5 epochs then unfrozen (prevents early catastrophic forgetting)
  - Gradient clipping (max_norm=1.0) for training stability
"""

import os
import sys
import json
import csv
import random
import shutil
import time
import argparse
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd




from pathlib import Path as _Path_for_base
BASE_DIR = os.environ.get("LCMS_BASE", str(_Path_for_base(__file__).resolve().parent.parent))
RAW_BASE = os.path.join(BASE_DIR, "data/raw")
KFOLD_BASE = os.path.join(BASE_DIR, "data/kfold")
MODEL_BASE = os.path.join(BASE_DIR, "models/resnet_v2")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REFERENCE_KFOLD = os.path.join(KFOLD_BASE, "256_by_patient_rmS11")

os.makedirs(MODEL_BASE, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [42, 69, 137]
N_FOLDS = 6
NUM_EPOCHS = 100
WARMUP_EPOCHS = 5
INITIAL_LR = 5e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_EVAL = 8
N_AUGMENTS = 20
MAX_GRAD_NORM = 1.0

EXCLUDE_PATIENTS = {"S11"}
EXCLUDE_FILES = {"Fetuin_final.png", "S2B-5x-0302.png"}

RESNET_CONFIGS = {
    "resnet18":  (models.resnet18,  models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34":  (models.resnet34,  models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50":  (models.resnet50,  models.ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
    "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
}


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)





def get_patient_splits():
    """Extract exact patient splits from the 256 reference kfold or JSON fallback."""

    json_path = os.path.join(KFOLD_BASE, "patient_splits.json")
    if os.path.exists(json_path):
        import json
        with open(json_path) as f:
            raw = json.load(f)
        splits = {}
        for fold_str, fold_data in raw.items():
            splits[int(fold_str)] = {k: set(v) for k, v in fold_data.items()}
        log(f"  Loaded patient splits from {json_path}")
        return splits


    splits = {}
    for fold in range(N_FOLDS):
        splits[fold] = {}
        for split_name in ["train", "val", "test"]:
            split_dir = os.path.join(REFERENCE_KFOLD, f"fold_{fold}", split_name)
            files = os.listdir(split_dir)
            patients = set()
            for f in files:
                name = f.replace(".png", "")
                if "_aug" in name:
                    name = name.split("_aug")[0]
                patient_id = name[:-1]
                patients.add(patient_id)
            splits[fold][split_name] = patients
    return splits





def create_kfold_dirs(image_source_dir, kfold_output_dir, patient_splits):
    """Create kfold directory structure with augmented training images."""

    if os.path.exists(kfold_output_dir):

        test_path = os.path.join(kfold_output_dir, "fold_5", "test")
        if os.path.exists(test_path) and len(os.listdir(test_path)) > 0:
            log(f"  Kfold dir exists and looks complete: {kfold_output_dir}")
            return

    log(f"  Creating kfold dirs: {kfold_output_dir}")


    color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)
    gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
    noise_fn = transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x))

    class RandomSubsetAugmentation:
        def __init__(self, transforms_list, min_ops=1, max_ops=3):
            self.transforms_list = transforms_list
            self.min_ops = min_ops
            self.max_ops = max_ops
        def __call__(self, img):
            n_ops = random.randint(self.min_ops, self.max_ops)
            selected = random.sample(self.transforms_list, n_ops)
            return transforms.Compose(selected)(img)

    aug_pipeline = transforms.Compose([
        transforms.ToTensor(),
        RandomSubsetAugmentation([color_jitter, gaussian_blur, noise_fn]),
        transforms.ToPILImage()
    ])


    available = {}
    for f in os.listdir(image_source_dir):
        if not f.endswith('.png') or f in EXCLUDE_FILES:
            continue
        name = f.replace(".png", "")
        patient_id = name[:-1]
        if patient_id in EXCLUDE_PATIENTS:
            continue
        available[name] = os.path.join(image_source_dir, f)

    for fold in range(N_FOLDS):
        for split_name in ["train", "val", "test"]:
            split_dir = os.path.join(kfold_output_dir, f"fold_{fold}", split_name)
            os.makedirs(split_dir, exist_ok=True)

            patients = patient_splits[fold][split_name]

            for sample_name, src_path in available.items():
                patient_id = sample_name[:-1]
                if patient_id not in patients:
                    continue

                dst = os.path.join(split_dir, f"{sample_name}.png")
                shutil.copy2(src_path, dst)

                if split_name == "train":
                    img = Image.open(src_path).convert("RGB")
                    for aug_i in range(1, N_AUGMENTS + 1):
                        set_seed(69 + aug_i + hash(sample_name) % 10000)
                        aug_img = aug_pipeline(img)
                        aug_dst = os.path.join(split_dir, f"{sample_name}_aug{aug_i}.png")
                        aug_img.save(aug_dst)

    log(f"  Done creating kfold dirs")





class LCMSDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.labels = []
        for f in self.image_files:
            base = os.path.splitext(f)[0]
            if "_aug" in base:
                base = base.split("_aug")[0]
            self.labels.append(0 if base.endswith('A') else 1)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.image_files[idx])
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], self.image_files[idx]


class TransferResNet(nn.Module):
    def __init__(self, variant="resnet34"):
        super().__init__()
        model_fn, weights = RESNET_CONFIGS[variant]
        self.backbone = model_fn(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(-1)

    def freeze_backbone(self):
        """Freeze all layers except the final FC."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True





def get_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """Linear warmup for warmup_epochs, then cosine annealing."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)





def run_single_experiment(variant, kfold_dir, seed, experiment_name):
    """Train and evaluate one configuration with one seed. Returns per-fold metrics."""

    set_seed(seed)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold_results = []

    for fold in range(N_FOLDS):
        set_seed(seed)
        log(f"    Fold {fold}/{N_FOLDS-1}")

        train_dir = os.path.join(kfold_dir, f"fold_{fold}", "train")
        val_dir = os.path.join(kfold_dir, f"fold_{fold}", "val")
        test_dir = os.path.join(kfold_dir, f"fold_{fold}", "test")

        train_dataset = LCMSDataset(train_dir, transform=transform)
        val_dataset = LCMSDataset(val_dir, transform=transform)
        test_dataset = LCMSDataset(test_dir, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False,
                                num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False,
                                 num_workers=0, pin_memory=True)

        model = TransferResNet(variant=variant)
        model.freeze_backbone()
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)

        steps_per_epoch = len(train_loader)
        scheduler = get_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, steps_per_epoch)


        save_dir = os.path.join(MODEL_BASE, f"{experiment_name}_seed{seed}_fold{fold}")
        os.makedirs(save_dir, exist_ok=True)

        top_models = []
        best_val_loss = float('inf')
        global_step = 0

        for epoch in range(1, NUM_EPOCHS + 1):

            if epoch == WARMUP_EPOCHS + 1:
                model.unfreeze_backbone()

                optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
                scheduler = get_scheduler(optimizer, 0, NUM_EPOCHS - WARMUP_EPOCHS, steps_per_epoch)


            model.train()
            train_loss = 0
            for images, labels, _ in train_loader:
                images, labels = images.to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                global_step += 1


            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images, labels = images.to(device), labels.float().to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total if total > 0 else 0

            if epoch % 25 == 0 or epoch == NUM_EPOCHS:
                lr = optimizer.param_groups[0]['lr']
                log(f"      Epoch {epoch}: TrLoss={train_loss:.4f} ValLoss={val_loss:.4f} ValAcc={val_acc:.2f} LR={lr:.2e}")


            model_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
            torch.save(model.state_dict(), model_path)
            top_models.append((val_loss, epoch, model_path))

            if len(top_models) > 3:
                worst = max(top_models, key=lambda x: x[0])
                top_models.remove(worst)
                if os.path.exists(worst[2]):
                    os.remove(worst[2])


        top_models.sort()
        final_model_paths = []
        for i, (vl, ep, path) in enumerate(top_models, 1):
            final_name = os.path.join(save_dir, f"best_model_{i}.pt")
            if os.path.exists(path):
                os.rename(path, final_name)
            final_model_paths.append(final_name)


        model = TransferResNet(variant=variant)
        state = torch.load(final_model_paths[0], map_location=device)

        new_state = {}
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
        model.load_state_dict(new_state, strict=False)
        model = model.to(device)
        model.eval()

        all_preds, all_labels, all_probs, all_files = [], [], [], []
        with torch.no_grad():
            for images, labels, filenames in test_loader:
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_files.extend(filenames)

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)


        ensemble_probs = np.zeros_like(y_prob)
        for mp in final_model_paths:
            m = TransferResNet(variant=variant)
            st = torch.load(mp, map_location=device)
            new_st = {k.replace("module.", ""): v for k, v in st.items()}
            m.load_state_dict(new_st, strict=False)
            m = m.to(device)
            m.eval()
            fold_probs = []
            with torch.no_grad():
                for images, labels, _ in test_loader:
                    images = images.to(device)
                    outputs = m(images)
                    fold_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            ensemble_probs += np.array(fold_probs)
        ensemble_probs /= len(final_model_paths)
        ens_preds = (ensemble_probs > 0.5).astype(float)

        ens_acc = accuracy_score(y_true, ens_preds)
        ens_auc = roc_auc_score(y_true, ensemble_probs) if len(np.unique(y_true)) > 1 else 0.0
        ens_prec = precision_score(y_true, ens_preds, zero_division=0)
        ens_rec = recall_score(y_true, ens_preds, zero_division=0)
        ens_f1 = f1_score(y_true, ens_preds, zero_division=0)

        log(f"      -> Single: Acc={acc:.3f} AUC={auc:.3f} F1={f1:.3f} | Ensemble: Acc={ens_acc:.3f} AUC={ens_auc:.3f} F1={ens_f1:.3f}")


        details = {"TP": [], "TN": [], "FP": [], "FN": []}
        for i in range(len(all_files)):
            t, p = int(y_true[i]), int(y_pred[i])
            if t == 1 and p == 1: details["TP"].append(all_files[i])
            elif t == 0 and p == 0: details["TN"].append(all_files[i])
            elif t == 0 and p == 1: details["FP"].append(all_files[i])
            else: details["FN"].append(all_files[i])

        fold_results.append({
            "fold": fold,
            "accuracy": acc, "roc_auc": auc, "precision": prec, "recall": rec, "f1_score": f1,
            "ens_accuracy": ens_acc, "ens_roc_auc": ens_auc, "ens_precision": ens_prec,
            "ens_recall": ens_rec, "ens_f1_score": ens_f1,
            "details": details
        })

    return fold_results





def aggregate_results(all_seed_results, experiment_name):
    """Aggregate results across seeds, compute mean ± std."""


    records = []
    for seed, fold_results in all_seed_results.items():
        for fr in fold_results:
            records.append({
                "seed": seed,
                "fold": fr["fold"],
                "accuracy": fr["accuracy"],
                "roc_auc": fr["roc_auc"],
                "precision": fr["precision"],
                "recall": fr["recall"],
                "f1_score": fr["f1_score"],
                "ens_accuracy": fr["ens_accuracy"],
                "ens_roc_auc": fr["ens_roc_auc"],
                "ens_precision": fr["ens_precision"],
                "ens_recall": fr["ens_recall"],
                "ens_f1_score": fr["ens_f1_score"],
            })

    df = pd.DataFrame(records)


    detail_csv = os.path.join(RESULTS_DIR, f"{experiment_name}_detailed.csv")
    df.to_csv(detail_csv, index=False)


    seed_means = df.groupby("seed").mean(numeric_only=True).drop(columns=["fold"])

    summary = {}
    for col in seed_means.columns:
        summary[f"{col}_mean"] = seed_means[col].mean()
        summary[f"{col}_std"] = seed_means[col].std()

    summary_csv = os.path.join(RESULTS_DIR, f"{experiment_name}_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    return summary, df





def run_experiment_group(experiments, active_seeds=None):
    """Run a list of experiments, each defined as a dict."""

    if active_seeds is None:
        active_seeds = SEEDS
    is_subset = set(active_seeds) != set(SEEDS)

    all_summaries = {}

    for exp in experiments:
        name = exp["name"]
        variant = exp["variant"]
        image_dir = exp["image_dir"]
        kfold_dir = exp["kfold_dir"]



        detail_csv = os.path.join(RESULTS_DIR, f"{name}_detailed.csv")
        summary_csv = os.path.join(RESULTS_DIR, f"{name}_summary.csv")
        if os.path.exists(detail_csv) and os.path.exists(summary_csv):
            try:
                df = pd.read_csv(detail_csv)
                expected_rows = len(SEEDS) * N_FOLDS
                if len(df) == expected_rows:
                    log(f"\n{'='*60}")
                    log(f"SKIPPING {name} — already complete ({detail_csv})")
                    log(f"{'='*60}")
                    continue
            except Exception:
                pass


        if is_subset:
            seed_tag = "_".join(str(s) for s in sorted(active_seeds))
            frag_csv = os.path.join(RESULTS_DIR, f"{name}_seeds{seed_tag}_detailed.csv")
            if os.path.exists(frag_csv):
                try:
                    df = pd.read_csv(frag_csv)
                    expected_rows = len(active_seeds) * N_FOLDS
                    if len(df) == expected_rows:
                        log(f"\n{'='*60}")
                        log(f"SKIPPING {name} — seed fragment already complete ({frag_csv})")
                        log(f"{'='*60}")
                        continue
                except Exception:
                    pass

        log(f"\n{'='*60}")
        log(f"Experiment: {name}")
        log(f"  Variant: {variant}, Images: {image_dir}")
        log(f"{'='*60}")


        patient_splits = get_patient_splits()
        create_kfold_dirs(image_dir, kfold_dir, patient_splits)


        all_seed_results = {}
        for seed in active_seeds:
            log(f"\n  Seed {seed}:")
            results = run_single_experiment(variant, kfold_dir, seed, f"{name}_seed{seed}")
            all_seed_results[seed] = results

        if is_subset:

            seed_tag = "_".join(str(s) for s in sorted(active_seeds))
            frag_name = f"{name}_seeds{seed_tag}"
            summary, detail_df = aggregate_results(all_seed_results, frag_name)
            all_summaries[name] = summary
            log(f"\n  Saved fragment: {frag_name}_detailed.csv")
        else:

            summary, detail_df = aggregate_results(all_seed_results, name)
            all_summaries[name] = summary


            metrics = ["accuracy", "roc_auc", "precision", "recall", "f1_score"]
            log(f"\n  === {name} Summary (Single Model) ===")
            for m in metrics:
                log(f"    {m}: {summary[f'{m}_mean']:.4f} ± {summary[f'{m}_std']:.4f}")
            log(f"\n  === {name} Summary (Ensemble) ===")
            for m in metrics:
                log(f"    ens_{m}: {summary[f'ens_{m}_mean']:.4f} ± {summary[f'ens_{m}_std']:.4f}")

    return all_summaries





def merge_seed_results(experiment_names):
    """Merge per-seed CSV fragments into final aggregated CSVs."""
    for name in experiment_names:

        seed_csvs = sorted([
            f for f in os.listdir(RESULTS_DIR)
            if f.startswith(f"{name}_seeds") and f.endswith("_detailed.csv")
        ])
        if not seed_csvs:
            log(f"  No seed fragments found for {name}, skipping merge")
            continue


        dfs = []
        for csv_file in seed_csvs:
            dfs.append(pd.read_csv(os.path.join(RESULTS_DIR, csv_file)))
        combined = pd.concat(dfs, ignore_index=True)


        combined = combined.drop_duplicates(subset=["seed", "fold"])


        detail_csv = os.path.join(RESULTS_DIR, f"{name}_detailed.csv")
        combined.to_csv(detail_csv, index=False)
        log(f"  Merged {len(seed_csvs)} fragments -> {detail_csv} ({len(combined)} rows)")


        seed_means = combined.groupby("seed").mean(numeric_only=True).drop(columns=["fold"])
        summary = {}
        for col in seed_means.columns:
            summary[f"{col}_mean"] = seed_means[col].mean()
            summary[f"{col}_std"] = seed_means[col].std()
        summary_csv = os.path.join(RESULTS_DIR, f"{name}_summary.csv")
        pd.DataFrame([summary]).to_csv(summary_csv, index=False)
        log(f"  Summary -> {summary_csv}")


        for csv_file in seed_csvs:
            os.remove(os.path.join(RESULTS_DIR, csv_file))
        log(f"  Cleaned up {len(seed_csvs)} fragment files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "architecture", "resolution", "subregion"],
                        help="Which experiment group to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Subset of seeds to run (e.g. --seeds 42 69). Default: all seeds.")
    parser.add_argument("--variant", type=str, default="resnet34",
                        choices=list(RESNET_CONFIGS.keys()),
                        help="ResNet variant for resolution/subregion experiments (default: resnet34)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge per-seed CSV fragments and exit (run after parallel jobs finish)")
    args = parser.parse_args()


    if args.merge:
        log("Merging seed-specific CSV fragments...")

        all_names = set()
        for f in os.listdir(RESULTS_DIR):
            if "_seeds" in f and f.endswith("_detailed.csv"):

                name = f.split("_seeds")[0]
                all_names.add(name)
        if all_names:
            merge_seed_results(sorted(all_names))
            log("Merge complete!")
        else:
            log("No seed fragments found to merge.")
        return


    active_seeds = args.seeds if args.seeds else SEEDS

    log("=" * 60)
    log("LC-MS CNN Classification - Unified Experiment Suite")
    log("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            log(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    log(f"Seeds: {active_seeds}")
    log(f"Epochs: {NUM_EPOCHS}, Warmup: {WARMUP_EPOCHS}, LR: {INITIAL_LR}")


    if args.experiment in ["all", "architecture"]:
        arch_experiments = []
        for variant in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
            arch_experiments.append({
                "name": f"arch_{variant}_256",
                "variant": variant,
                "image_dir": os.path.join(RAW_BASE, "spectra_filter_256"),
                "kfold_dir": os.path.join(KFOLD_BASE, "v2_256_by_patient_rmS11"),
            })
        log("\n\n" + "=" * 60)
        log("EXPERIMENT GROUP 1: Architecture Ablation")
        log("=" * 60)
        arch_summaries = run_experiment_group(arch_experiments, active_seeds)


    if args.experiment in ["all", "resolution"]:
        variant = args.variant
        res_experiments = []
        for res in [256, 512, 1024]:
            res_experiments.append({
                "name": f"fullspec_{variant}_{res}",
                "variant": variant,
                "image_dir": os.path.join(RAW_BASE, f"spectra_filter_{res}"),
                "kfold_dir": os.path.join(KFOLD_BASE, f"v2_{res}_by_patient_rmS11"),
            })
        log("\n\n" + "=" * 60)
        log(f"EXPERIMENT GROUP 2: Resolution Ablation (Full Spectrum, {variant})")
        log("=" * 60)
        res_summaries = run_experiment_group(res_experiments, active_seeds)


    if args.experiment in ["all", "subregion"]:
        variant = args.variant
        sub_experiments = []
        for res in [256, 512, 1024]:
            sub_experiments.append({
                "name": f"subregion_{variant}_{res}",
                "variant": variant,
                "image_dir": os.path.join(RAW_BASE, f"focused_zoom_{res}"),
                "kfold_dir": os.path.join(KFOLD_BASE, f"v2_subregion_{res}_by_patient_rmS11"),
            })
        log("\n\n" + "=" * 60)
        log("EXPERIMENT GROUP 3: Subregion Ablation")
        log("=" * 60)
        sub_summaries = run_experiment_group(sub_experiments, active_seeds)


    log("\n\n" + "=" * 60)
    log("ALL EXPERIMENTS COMPLETE")
    log("=" * 60)
    log(f"\nResults saved to: {RESULTS_DIR}/")
    log("Files: *_detailed.csv (per-fold-per-seed) and *_summary.csv (aggregated)")


if __name__ == "__main__":
    main()
