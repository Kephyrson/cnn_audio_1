import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa

class AudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, sr=22050, n_mels=128, duration=15):
        """
        Args:
            csv_path (str): CSV文件路径
            audio_dir (str): 音频文件夹路径
            sr (int): 采样率（默认22050）
            n_mels (int): 梅尔频谱维度（默认128）
            duration (int): 统一音频长度（秒）
        """
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.max_samples = sr * duration

    def __len__(self):
        return len(self.df)

    def _load_audio(self, file_path):
        # 加载音频并进行时间轴标准化
        audio, _ = librosa.load(file_path, sr=self.sr)
        
        # 统一音频长度
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        else:
            padding = self.max_samples - len(audio)
            audio = np.pad(audio, (0, padding))
        
        return audio

    def _extract_melspectrogram(self, audio):
        # 提取梅尔频谱
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512
        )
        # 转换为对数刻度
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        # 归一化到[-1,1]
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min()) * 2 -1
        return log_mel

    def __getitem__(self, idx):
        # 从CSV获取元数据
        audio_name = self.df.iloc[idx, 0]  # 假设第一列是音频文件名
        label = self.df.iloc[idx, 3]      # 假设第二列是数字标签
        # 构建完整路径
        ofile_path = os.path.join(self.audio_dir,f"{audio_name}.wav")
        file_path = ofile_path.replace(".txt", "", 1)  # 输出 ID=1.wav
        # 加载并处理音频
        raw_audio = self._load_audio(file_path)
        mel = self._extract_melspectrogram(raw_audio)
        
        # 转换为PyTorch张量
        tensor_mel = torch.FloatTensor(mel).unsqueeze(0)  # 添加通道维度
        tensor_label = torch.LongTensor([label]).squeeze()
        
        return tensor_mel, tensor_label

# 使用示例

if __name__ == '__main__':

    dataset = AudioDataset(
        csv_path="sorted_labels.csv",
        audio_dir="new/dataset1/train_audio",
        duration=15
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    # 验证数据形状
    sample, label = next(iter(dataloader))
    print(f"梅尔频谱形状: {sample.shape}")  # [32,1,128,129]
    print(f"标签形状: {label.shape}")        # 






import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        # 输入形状：(batch, 1, n_mels, time_steps)
        
        # 卷积模块
        self.conv_layers = nn.Sequential(
            # 第一卷积块
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 第二卷积块
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 第三卷积块
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 全连接模块
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * (n_mels//8) * (time_steps//8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 卷积部分
        x = self.conv_layers(x)
        
        # 展平特征
        x = x.view(x.size(0), -1)
        
        # 全连接分类
        return self.fc_layers(x)

# 初始化模型（假设num_classes=10）
n_mels = 128  # 需与Dataset中的n_mels一致
time_steps = (22050*15)//512 +1  # 根据音频长度计算
model = AudioCNN(num_classes=10)
num_epoch = 3



import torch.optim as optim
from sklearn.model_selection import train_test_split

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

# 划分训练集和验证集（假设全量数据集为dataset）
train_idx, val_idx = train_test_split(
    range(len(dataset)), 
    test_size=0.2,
    stratify=dataset.df.iloc[:,1]  # 按标签分层抽样
)

train_set = torch.utils.data.Subset(dataset, train_idx)
val_set = torch.utils.data.Subset(dataset, val_idx)

# 创建DataLoader
train_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_size=32,
    shuffle=False,
    num_workers=2
)


def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计指标
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 每50个batch打印一次
        if batch_idx % 50 == 49:  
            print(f'Epoch: {epoch} | Batch: {batch_idx+1} | '
                  f'Loss: {running_loss/50:.3f} | Acc: {100.*correct/total:.2f}%')
            running_loss = 0.0

    return running_loss / len(train_loader)

def validate():
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    acc = 100. * correct / total
    print(f'Validation Loss: {val_loss/len(val_loader):.3f} | Acc: {acc:.2f}%')
    return acc

# 训练主循环
best_acc = 0
for epoch in range(num_epoch):  # 训练50个epoch
    train_loss = train(epoch)
    val_acc = validate()
    
    # 学习率调整
    scheduler.step(val_acc)
    
    # 保存最佳模型
    if val_acc > best_acc:
        print(f'Saving improved model (Acc: {val_acc:.2f}%)...')
        torch.save(model.state_dict(), f'best_model_epoch{epoch}.pth')
        best_acc = val_acc