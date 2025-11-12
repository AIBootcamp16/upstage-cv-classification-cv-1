# ============================================
# train_stage1.py
# 1차 coarse classifier (G0/G1/G2) 학습
# ============================================

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import random
from ultralytics import YOLO


# ----------------------------------------------
# ✅ coarse group mapping
# ----------------------------------------------
GROUP_MAP = {
    0: "G0", 2: "G1", 5: "G2",
    8: "G2", 9: "G3", 16: "G4",
    1:"G5", 3:"G5", 4:"G5", 6:"G5", 7:"G5",
    10:"G5", 11:"G5", 12:"G5", 13:"G5", 14:"G5", 15:"G5",
}


# ============================================
# ✅ Augmentation Modules (same as before)
# ============================================
def rotate(img, angle):
    if angle == 0:
        return img
    h, w = img.size[1], img.size[0]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rad = np.radians(angle)
    nw = int(abs(np.sin(rad)) * h + abs(np.cos(rad)) * w)
    nh = int(abs(np.cos(rad)) * h + abs(np.sin(rad)) * w)
    m[0, 2] += (nw - w) / 2
    m[1, 2] += (nh - h) / 2
    return Image.fromarray(cv2.warpAffine(np.array(img), m, (nw, nh)))


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)


def cutout(image, mask_size=60, p=0.5):
    if random.random() > p:
        return image
    _, h, w = image.shape
    y = np.random.randint(h)
    x = np.random.randint(w)
    y1 = np.clip(y - mask_size // 2, 0, h)
    y2 = np.clip(y + mask_size // 2, 0, h)
    x1 = np.clip(x - mask_size // 2, 0, w)
    x2 = np.clip(x + mask_size // 2, 0, w)
    image[:, y1:y2, x1:x2] = 0
    return image


def mixup_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(images, labels, alpha=1.0):
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size, C, H, W = images.size()
    rand_index = torch.randperm(batch_size).to(images.device)

    shuffled_images = images[rand_index]
    shuffled_labels = labels[rand_index]

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    w = int(W * np.sqrt(1 - lam))
    h = int(H * np.sqrt(1 - lam))

    x1 = np.clip(cx - w // 2, 0, W)
    x2 = np.clip(cx + w // 2, 0, W)
    y1 = np.clip(cy - h // 2, 0, H)
    y2 = np.clip(cy + h // 2, 0, H)

    images[:, :, y1:y2, x1:x2] = shuffled_images[:, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return images, labels, shuffled_labels, lam


def copy_paste(imgs, labels, p=0.5):
    if random.random() > p:
        return imgs, labels
    batch_size = imgs.size(0)
    idx = torch.randperm(batch_size)
    new_imgs = imgs.clone()
    _, _, h, w = imgs.size()

    for i in range(batch_size):
        donor = imgs[idx[i]]
        ph, pw = random.randint(30, 100), random.randint(30, 100)
        y = random.randint(0, h - ph)
        x = random.randint(0, w - pw)
        new_imgs[i, :, y:y+ph, x:x+pw] = donor[:, y:y+ph, x:x+pw]
    return new_imgs, labels


# ============================================
# ✅ Transform
# ============================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomChoice([
        transforms.RandomRotation(30),
        transforms.RandomRotation(45),
        transforms.RandomRotation(60),
        transforms.RandomRotation(90),
        transforms.RandomRotation(120),
        transforms.RandomRotation(150),
        transforms.RandomRotation(180),
        transforms.RandomRotation(210),
        transforms.RandomRotation(240),
        transforms.RandomRotation(270),
        transforms.RandomRotation(300)
    ]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.3),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.05),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================
# ✅ Dataset
# ============================================

class CoarseDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row['ID']}"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row['coarse_label'], dtype=torch.long)
        return image, label



# ============================================
# ✅ Load Train CSV & Create coarse labels
# ============================================

df = pd.read_csv("/root/CV_/datasets/data/train.csv")
df["coarse"] = df["target"].map(GROUP_MAP)

le_coarse = LabelEncoder()
df["coarse_label"] = le_coarse.fit_transform(df["coarse"])


# ✅ ✅ 3-WAY SPLIT: train / val / test
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["coarse_label"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["coarse_label"], random_state=42
)

print(
    f"✅ Split 완료: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
)


# ============================================
# ✅ Create Datasets & DataLoaders
# ============================================

train_dataset = CoarseDataset(train_df, "/root/CV_/datasets/data/train", train_transform)
val_dataset   = CoarseDataset(val_df,   "/root/CV_/datasets/data/train", test_transform)
test_dataset  = CoarseDataset(test_df,  "/root/CV_/datasets/data/train", test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16)
test_loader  = DataLoader(test_dataset,  batch_size=16)


# ============================================
# ✅ Load Model (ViT Large)
# ============================================

processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")

num_classes = 6
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

AUG_LIST = ["mixup", "cutmix", "copypaste", "none"]


# ============================================
# ✅ Train Stage1
# ============================================

for epoch in range(50):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        aug = random.choice(AUG_LIST)
        if aug == "mixup":
            images, ta, tb, lam = mixup_data(images, labels)
            loss_fn = lambda out: mixup_criterion(criterion, out, ta, tb, lam)

        elif aug == "cutmix":
            images, ta, tb, lam = cutmix_data(images, labels)
            loss_fn = lambda out: mixup_criterion(criterion, out, ta, tb, lam)

        elif aug == "copypaste":
            images, labels = copy_paste(images, labels)
            loss_fn = lambda out: criterion(out, labels)

        else:
            loss_fn = lambda out: criterion(out, labels)

        # CutOut
        for i in range(images.size(0)):
            images[i] = cutout(images[i], mask_size=60)

        optimizer.zero_grad()
        outputs = model(images).logits
        loss = loss_fn(outputs)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Stage1] Epoch {epoch+1}/50 Loss: {total_loss/len(train_loader):.4f}")


# ============================================
# ✅ 평가: val / test
# ============================================

def eval_model(loader, name):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"✅ {name} ACC: {correct/total:.4f}")


eval_model(val_loader, "VAL")
eval_model(test_loader, "TEST")

# ============================================
# ✅ Save Model + Encoder
# ============================================

torch.save(model.state_dict(), "coarse_model.pth")
import joblib
joblib.dump(le_coarse, "coarse_label_encoder.pkl")

print("✅ Stage1 coarse model saved ✔")