import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torchaudio
from torchaudio.transforms import TimeMasking, FrequencyMasking

class AudioAugmentor:
    def __init__(self, sr=22050):
        self.sr = sr
        self.config = {
            'noise_factor': 0.03,
            'time_shift_max': 0.3,  
            'speed_range': (0.9, 1.2),  
            'pitch_steps': (-3, 3),  
            'mixup_alpha': 0.4,  
            'time_mask_param': 20, 
            'freq_mask_param': 5    
        }
    
    def __call__(self, audio):

        aug_methods = np.random.choice([
            self.add_noise,
            self.time_shift,
            self.speed_tuning,
            self.pitch_shift,
            self.spec_augment,
            self.mixup
        ], 3, replace=False)
        
        for method in aug_methods:
            audio = method(audio)
        return audio
    
    def add_noise(self, audio):
        noise = np.random.randn(len(audio))
        return audio + self.config['noise_factor'] * noise
    
    def time_shift(self, audio):
        shift = np.random.randint(-int(self.sr * self.config['time_shift_max']), 
                                 int(self.sr * self.config['time_shift_max']))
        return np.roll(audio, shift)
    
    def speed_tuning(self, audio):
        speed_factor = np.random.uniform(*self.config['speed_range'])
        return librosa.effects.time_stretch(audio, rate=speed_factor)
    
    def pitch_shift(self, audio):
        n_steps = np.random.randint(*self.config['pitch_steps'])
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
    
    def spec_augment(self, audio):

        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr)
        mel_spec = torch.tensor(mel_spec).unsqueeze(0)
        
        if np.random.rand() > 0.5:
            mel_spec = TimeMasking(self.config['time_mask_param'])(mel_spec)
        if np.random.rand() > 0.5:
            mel_spec = FrequencyMasking(self.config['freq_mask_param'])(mel_spec)
            
        return librosa.feature.inverse.mel_to_audio(mel_spec.squeeze().numpy(), sr=self.sr)
    
    def mixup(self, audio):

        shifted = np.roll(audio, int(self.sr * 0.2))
        lam = np.random.beta(self.config['mixup_alpha'], self.config['mixup_alpha'])
        return lam * audio + (1 - lam) * shifted

if __name__ == "__main__":
    augmentor = AudioAugmentor(sr=16000)
    
class AudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, sr=22050, n_mels=128, duration=15):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.max_samples = sr * duration

    def __len__(self):
        return len(self.df)

    def _load_audio(self, file_path):
        
        audio, _ = librosa.load(file_path, sr=self.sr)
        return audio

    def _extract_melspectrogram(self, audio):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min()) * 2 -1
        return log_mel

    
    def __getitem__(self, idx):
        
        audio_name = self.df.iloc[idx, 0]  
        label = self.df.iloc[idx, 3]      
        ofile_path = os.path.join(self.audio_dir,f"{audio_name}.wav")
        file_path = ofile_path.replace(".txt", "", 1)  

        raw_audio = self._load_audio(file_path)
        audio = augmentor(raw_audio)
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        else:
            padding = self.max_samples - len(audio)
            audio = np.pad(audio, (0, padding))

        
        mel = self._extract_melspectrogram(audio)
        
        tensor_mel = torch.FloatTensor(mel).unsqueeze(0)  
        tensor_label = torch.LongTensor([label]).squeeze()
        
        return tensor_mel, tensor_label


if __name__ == '__main__':
 
    train_dataset = AudioDataset(
        csv_path="../sorted_labels.csv",
        audio_dir="../new/Dataset1/train_audio",
        duration=15
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    test_dataset = AudioDataset(
        csv_path="../sorted_labels0.csv",
        audio_dir="../new/Dataset1/val_audio",
        duration=15
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

n_mels = 128  
time_steps = (22050*15)//512 +1  
num_epoch = 50

import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
        
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * (n_mels//8) * (time_steps//8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        
        x = x.view(x.size(0), -1)
        
        return self.fc_layers(x)

model = AudioCNN(num_classes=3)

import torch.optim as optim
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 50 == 49:  
            print(f'Epoch: {epoch} | Batch: {batch_idx+1} | '
                  f'Loss: {running_loss/50:.3f} | Acc: {100.*correct/total:.2f}%')
            running_loss = 0.0

    return running_loss / len(train_dataloader)

def validate():
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    acc = 100. * correct / total
    print(f'Validation Loss: {val_loss/len(test_dataloader):.3f} | Acc: {acc:.2f}%')
    return acc

best_acc = 0
for epoch in range(num_epoch):  
    train_loss = train(epoch)
    val_acc = validate()
    
    scheduler.step(val_acc)
    
    if val_acc > best_acc:
        print(f'Saving improved model (Acc: {val_acc:.2f}%)...')
        torch.save(model.state_dict(), f'best_model_epoch{epoch}.pth')
        best_acc = val_acc