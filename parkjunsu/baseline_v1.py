import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from transformers import AutoImageProcessor, SwinForImageClassification
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2
from ultralytics import YOLO
import random

# ----------------------------------------------
# ✅ YOLO 모델 로드
# ----------------------------------------------
object_det_model = YOLO("yolov8x.pt", verbose=False)

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

def detect_best_class(model, img_path):
    img = Image.open(img_path).convert("RGB")
    angles = [0, 45, 90, 180]
    best = {"cls": None, "conf": 0}

    for a in angles:
        rimg = rotate(img, a)
        res = model(rimg)[0]
        if res.boxes:
            for b in res.boxes:
                conf = float(b.conf)
                cls = int(b.cls)
                if conf > best["conf"]:
                    best.update({"cls": cls, "conf": conf})

    if best["cls"] is None:
        return "None"
    return model.names[best["cls"]]


# ----------------------------------------------
# ✅ Gaussian Noise Transform
# ----------------------------------------------
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)


# ----------------------------------------------
# ✅ Mixup
# ----------------------------------------------
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


# ----------------------------------------------
# ✅ CutOut
# ----------------------------------------------
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


# ----------------------------------------------
# ✅ CutMix
# ----------------------------------------------
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


# ----------------------------------------------
# ✅ Copy-Paste
# ----------------------------------------------
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


# ----------------------------------------------
# ✅ Transform 정의
# ----------------------------------------------
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


# ----------------------------------------------
# ✅ Dataset
# ----------------------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, label_encoder=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_encoder = label_encoder

        if 'target' in self.data.columns and self.label_encoder is not None:
            self.data['target'] = self.label_encoder.transform(self.data['target'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.data.iloc[idx, 0]}"
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if 'target' in self.data.columns:
            label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.long)
            return image, label

        return image, -1


# ----------------------------------------------
# ✅ Dataset & Loader
# ----------------------------------------------
train_df = pd.read_csv("/root/CV_/datasets/data/train.csv")
le = LabelEncoder()
train_df['target'] = le.fit_transform(train_df['target'])

train_dataset = CustomImageDataset(
    "/root/CV_/datasets/data/train.csv",
    "/root/CV_/datasets/data/train",
    transform=train_transform,
    label_encoder=le
)

test_dataset = CustomImageDataset(
    "/root/CV_/datasets/data/sample_submission.csv",
    "/root/CV_/datasets/data/test",
    transform=test_transform,
    label_encoder=le
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


# ----------------------------------------------
# ✅ 모델 로드 (ViT Large)
# ----------------------------------------------
model_name = "microsoft/swin-large-patch4-window7-224"

processor = AutoImageProcessor.from_pretrained(model_name)
model = SwinForImageClassification.from_pretrained(model_name)

num_classes = len(le.classes_)
if model.config.num_labels != num_classes:
    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'score'):
        in_features = model.score.in_features
        model.score = nn.Linear(in_features, num_classes)
    else:
        print("⚠ 마지막 레이어명을 찾지 못했습니다.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)


# ----------------------------------------------
# ✅ 🔥 학습 루프 — Mixup, CutMix, Copy-Paste, CutOut 포함
# ----------------------------------------------
AUG_LIST = ["mixup", "cutmix", "copypaste", "none"]

num_epochs = 250

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        aug_type = random.choice(AUG_LIST)

        if aug_type == "mixup":
            images, ta, tb, lam = mixup_data(images, labels, alpha=1.0)
            compute_loss = lambda out: mixup_criterion(criterion, out, ta, tb, lam)

        elif aug_type == "cutmix":
            images, ta, tb, lam = cutmix_data(images, labels, alpha=1.0)
            compute_loss = lambda out: mixup_criterion(criterion, out, ta, tb, lam)

        elif aug_type == "copypaste":
            images, labels = copy_paste(images, labels, p=1.0)
            compute_loss = lambda out: criterion(out, labels)

        else:
            compute_loss = lambda out: criterion(out, labels)

        # ✅ CutOut 적용
        for i in range(images.size(0)):
            images[i] = cutout(images[i], mask_size=60, p=0.5)

        optimizer.zero_grad()
        outputs = model(images).logits

        loss = compute_loss(outputs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_loader):.4f}")


# ----------------------------------------------
# ✅ 테스트 + YOLO Car detection + 분류
# ----------------------------------------------
model.eval()
all_preds = []

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)

        for img_tensor in images:
            img_np = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            yolo_res = object_det_model(pil_img)[0]
            detected_car = False

            if yolo_res.boxes:
                for box in yolo_res.boxes:
                    cls = int(box.cls[0])
                    if object_det_model.names[cls] == "car":
                        detected_car = True
                        break

            if detected_car:
                all_preds.append(16)  # car idx
            else:
                outputs = model(img_tensor.unsqueeze(0)).logits
                pred = outputs.argmax(dim=1).cpu().item()
                all_preds.append(pred)


# ----------------------------------------------
# ✅ 결과 저장
# ----------------------------------------------
pred_labels = le.inverse_transform(all_preds)
result = pd.read_csv("/root/CV_/datasets/data/sample_submission.csv")
result['target'] = pred_labels

result.to_csv("aug_mixup_cutmix_cout_copypaste_yolo_vit_output.csv", index=False)
print("✅ 저장 완료: aug_mixup_cutmix_cout_copypaste_yolo_vit_output.csv")
result