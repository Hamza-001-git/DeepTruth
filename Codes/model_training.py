import sys 
import psutil
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import multiprocessing

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_memory_usage():
    memory = psutil.virtual_memory()
    return memory.percent

# Function to check memory usage
def check_ram_usage(threshold=80):
    usage = psutil.virtual_memory().percent
    if usage > threshold:
        raise MemoryError(f"RAM usage exceeded threshold: {usage}%")

# Check memory usage periodically
def validate_video(vid_path, train_transforms):
    check_ram_usage()  # Check RAM usage before processing
    transform = train_transforms
    count = 20
    video_path = vid_path
    frames = []
    a = int(100 / count)
    first_frame = np.random.randint(0, a)
    temp_video = video_path.split('/')[-1]
    for i, frame in enumerate(frame_extract(video_path)):
        frames.append(transform(frame))
        if len(frames) == count:
            break
    frames = torch.stack(frames)
    frames = frames[:count]
    return frames

# Extract a frame from video
def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image

# Transformations
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load video files
video_files = glob.glob(r"C:\Users\Hamid\Documents\DeepTruth\FF_Face_only_data\*.mp4")  # Modify this path as needed
print("Total no of videos:", len(video_files))
print(video_files)

count = 0
for i in video_files:
    try:
        count += 1
        validate_video(i, train_transforms)
    except Exception as e:
        print(f"Number of video processed: {count}, Remaining: {len(video_files) - count}")
        print(f"Corrupted video is: {i}, Error: {e}")
        continue
print(f"Remaining: {len(video_files) - count}")

# Define Dataset
class VideoDataset(Dataset):
    def __init__(self, video_names, labels, sequence_length=60, transform=None):
        self.video_names = video_names
        self.labels = labels
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):
        check_ram_usage()  # Check RAM usage before processing
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        temp_video = video_path.split('/')[-1]
        
        # Check if temp_video exists in self.labels["file"]
        if temp_video in self.labels["file"].values:
            label = self.labels.iloc[self.labels[self.labels["file"] == temp_video].index.values[0], 1]
            label = 0 if label == 'FAKE' else 1
        else:
            # Handle case where temp_video does not exist in labels (e.g., assign a default label)
            label = 0  # Or handle as needed
        
        for i, frame in enumerate(self.frame_extract(video_path)):
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames, label


    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

# Function to plot images
def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    b, g, r = cv2.split(image)
    image = cv2.merge((r, g, b))
    image = image * [0.22803, 0.22145, 0.216989] + [0.43216, 0.394666, 0.37645]
    image = image * 255.0
    plt.imshow(image.astype(int))
    plt.show()

# Load labels
header_list = ["file", "label"]
labels = pd.read_csv(r"C:\Users\Hamid\Documents\DeepTruth\FF_Face_only_data\metadata.csv", names=header_list)  # Modify this path as needed
train_videos, valid_videos = train_test_split(video_files, test_size=0.2, random_state=42)
print("Train:", len(train_videos))
print("Valid:", len(valid_videos))

# Set up DataLoaders
train_data = VideoDataset(train_videos, labels, sequence_length=10, transform=train_transforms)
val_data = VideoDataset(valid_videos, labels, sequence_length=10, transform=train_transforms)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)  # Set num_workers to 0 for Windows
valid_loader = DataLoader(val_data, batch_size=4, shuffle=True, num_workers=0)  # Set num_workers to 0 for Windows

# Model with feature visualization
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

# Initialize model
model = Model(2).to(device)

# Example forward pass
a, b = model(torch.from_numpy(np.empty((1, 20, 3, 112, 112))).type(torch.FloatTensor).to(device))

# Training and Testing the Model
def train_epoch(epoch, num_epochs, data_loader, model, criterion, optimizer):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    t = []
    for i, (inputs, targets) in enumerate(data_loader):
        check_ram_usage()  # Check RAM usage during training
        inputs, targets = inputs.to(device), targets.to(device).long()
        _, outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d / %d] [Loss: %f, Acc: %.2f%%]"
            % (
                epoch,
                num_epochs,
                i,
                len(data_loader),
                losses.avg,
                accuracies.avg))
    torch.save(model.state_dict(), 'checkpoint.pt')
    return losses.avg, accuracies.avg

def test(epoch, model, data_loader, criterion):
    print('Testing')
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pred = []
    true = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            check_ram_usage()  # Check RAM usage during testing
            inputs, targets = inputs.to(device), targets.to(device).long()
            _, outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)
            _, p = torch.max(outputs, 1)
            true += targets.cpu().numpy().tolist()
            pred += p.cpu().numpy().tolist()
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            sys.stdout.write(
                "\r[Batch %d / %d]  [Loss: %f, Acc: %.2f%%]"
                % (
                    i,
                    len(data_loader),
                    losses.avg,
                    accuracies.avg
                )
            )
        print('\nAccuracy {}'.format(accuracies.avg))
    return true, pred, losses.avg, accuracies.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return (correct / total) * 100

# Set up training parameters
num_epochs = 10
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(epoch + 1, num_epochs, train_loader, model, criterion, optimizer)
    print(f"\n[TRAIN] Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

    # Validation loop
    with torch.no_grad():
        true_labels, pred_labels, val_loss, val_acc = test(epoch + 1, model, valid_loader, criterion)
        print(f"\n[VALID] Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(true_labels, pred_labels))
print(confusion_matrix(true_labels, pred_labels))

# Saving the model
torch.save(model.state_dict(), 'video_model.pth')
