"""
Stage 4b - Deep-learning benchmark: transfer learning with MobileNetV2.

The ImageNet-pretrained MobileNetV2 backbone is fine-tuned end-to-end on the
same stratified split used by the classical models.

Training recipe
  * input 224x224, ImageNet normalisation
  * augmentation: random resized crop (0.7-1.0), horizontal + vertical flip,
    rotation (+-20 deg), colour jitter (brightness/contrast/saturation 0.2)
  * AdamW (lr 1e-3 head / 3e-4 backbone, weight decay 1e-4), cosine schedule
    with 1 warm-up epoch, label smoothing 0.1, batch 32
  * the checkpoint with the best validation macro-F1 is kept
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import v2

from config import CNN_SIZE, MODEL_DIR, PRETRAINED_MNV2, SEED
from mobilenetv2 import build_mobilenetv2

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

TRAIN_TF = T.Compose([
    T.RandomResizedCrop(CNN_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(), T.Normalize(MEAN, STD)])
EVAL_TF = T.Compose([T.Resize((CNN_SIZE, CNN_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


class LeafDataset(Dataset):
    def __init__(self, paths, labels, tf, degrade=None):
        self.paths, self.labels, self.tf, self.degrade = list(paths), list(labels), tf, degrade

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB").resize((256, 256))
        if self.degrade is not None:
            img = Image.fromarray(self.degrade(np.asarray(img)))
        return self.tf(img), self.labels[i]


# Fast training input path: images are decoded once into a uint8 memory-mapped
# array (N x 3 x 256 x 256) and augmented as tensors with torchvision.transforms.v2.
TRAIN_TF_T = v2.Compose([
    v2.RandomResizedCrop(CNN_SIZE, scale=(0.7, 1.0), antialias=True),
    v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), v2.RandomRotation(20),
    v2.ColorJitter(0.2, 0.2, 0.2),
    v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])
EVAL_TF_T = v2.Compose([v2.Resize((CNN_SIZE, CNN_SIZE), antialias=True),
                        v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])


def image_cache(paths, name):
    """Decode images once into data/cache/<name>.u8 (memory-mapped uint8, CHW)."""
    from config import CACHE_DIR
    f = os.path.join(CACHE_DIR, f"{name}_{len(paths)}.u8")
    shape = (len(paths), 3, 256, 256)
    if not os.path.exists(f):
        mm = np.lib.format.open_memmap(f + ".tmp.npy", mode="w+", dtype=np.uint8, shape=shape)
        for i, p in enumerate(paths):
            mm[i] = np.asarray(Image.open(p).convert("RGB").resize((256, 256))).transpose(2, 0, 1)
        mm.flush()
        del mm
        os.replace(f + ".tmp.npy", f)
    return np.load(f, mmap_mode="r") if f.endswith(".npy") else np.lib.format.open_memmap(f, mode="r")


class CachedDataset(Dataset):
    def __init__(self, arr, labels, tf):
        self.arr, self.labels, self.tf = arr, list(labels), tf

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.tf(torch.from_numpy(np.array(self.arr[i]))), self.labels[i]


def _loader(paths, labels, tf, shuffle, degrade=None, bs=32, workers=1):
    return DataLoader(LeafDataset(paths, labels, tf, degrade), batch_size=bs, shuffle=shuffle,
                      num_workers=workers, persistent_workers=workers > 0)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x), 1).numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def train(df, num_classes, epochs=10, out_name="mobilenetv2_best.pt", log=print):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(os.cpu_count())
    tr, va = df[df.split == "train"], df[df.split == "val"]
    tl = DataLoader(CachedDataset(image_cache(list(tr.path), "train"), tr.label, TRAIN_TF_T),
                    batch_size=32, shuffle=True, num_workers=0)
    vl = DataLoader(CachedDataset(image_cache(list(va.path), "val"), va.label, EVAL_TF_T),
                    batch_size=64, shuffle=False, num_workers=0)

    model = build_mobilenetv2(num_classes, PRETRAINED_MNV2)
    head = list(model.classifier.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": backbone, "lr": 3e-4}, {"params": head, "lr": 1e-3}], weight_decay=1e-4)
    steps = epochs * len(tl)
    warm = len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    history, best, start = [], -1.0, 1
    path = os.path.join(MODEL_DIR, out_name)
    resume = os.path.join(MODEL_DIR, "cnn_resume_state.pt")
    if os.path.exists(resume):  # continue an interrupted run
        st = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        history, best, start = st["history"], st["best"], st["epoch"] + 1
        torch.set_rng_state(st["rng"])
        log(f"resumed after epoch {st['epoch']}")
    for ep in range(start, epochs + 1):
        model.train()
        t, tot, correct, n = time.time(), 0.0, 0, 0
        for x, y in tl:
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
            if (n // len(y)) % 50 == 0:
                log(f"  epoch {ep} batch {n // len(y)}/{len(tl)} loss {tot / n:.4f} acc {correct / n:.4f}")
        yv, pv = predict(model, vl)
        val_loss = float(nn.functional.nll_loss(torch.log(torch.tensor(pv) + 1e-9), torch.tensor(yv)))
        val_acc = float((pv.argmax(1) == yv).mean())
        val_f1 = float(f1_score(yv, pv.argmax(1), average="macro"))
        rec = {"epoch": ep, "train_loss": tot / n, "train_acc": correct / n, "val_loss": val_loss,
               "val_acc": val_acc, "val_f1": val_f1, "seconds": time.time() - t}
        history.append(rec)
        log(json.dumps(rec))
        if val_f1 > best:
            best = val_f1
            torch.save(model.state_dict(), path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "history": history, "best": best, "epoch": ep, "rng": torch.get_rng_state()}, resume + ".tmp")
        os.replace(resume + ".tmp", resume)
    if os.path.exists(resume):
        os.remove(resume)
    return history, path


def load_trained(num_classes, path):
    model = build_mobilenetv2(num_classes, None)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()
