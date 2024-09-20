import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import random
from datetime import datetime


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# HJZDataset class definition to load images and embeddings from filenames


class HJZDataset(Dataset):
    def __init__(self, root_dir, shuffle=True):
        """
        Args:
            root_dir (string): save_dir with all the images.
        """
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        if shuffle:
            random.shuffle(self.image_files)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((201, 201)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        image = self.transform(image)

        # Extract probing_port and decap from file name
        probing_port, decap, pb_x, pb_y, dp_x, dp_y = self.extract_embeddings(img_name)
        
        # Combine embeddings into a single tensor for the model

        # hj_modified (flag)
        embeddings = torch.tensor([pb_x, pb_y, dp_x, dp_y], dtype=torch.float) #, dtype=torch.long)
        return image, embeddings

    def extract_embeddings(self, img_name):
        basename = os.path.basename(img_name)
        parts = basename.split('_')  # Split by '_'
        probing_port = int(parts[1])  # Extract probing port value
        pb_y = ((probing_port // 10) + 1)/10
        pb_x = ((probing_port % 10) + 1)/10

        decap = int(parts[2].split('.')[0])  # Extract decap value, removing the '.png' part

        dp_x = ((decap %10) +1)/10
        dp_y = ((decap //10) +1)/10

        return probing_port, decap, pb_x, pb_y, dp_x, dp_y

def test_HJZDataset(root_dir):
    
    dataset = HJZDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"데이터셋 크기: {len(dataset)}")
    
    images, embeddings = next(iter(dataloader))

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4):
        img = images[i].squeeze().numpy()
        ax = axs[i // 2, i % 2]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Probing Port: {embeddings[i][0]}, Decap: {embeddings[i][1]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 임베딩 분포 확인
    all_embeddings = []
    for _, emb in dataset:
        all_embeddings.append(emb)
        print("_", _)
        print("len(_)", len(_))
        print("emb", emb)
        break
    all_embeddings = torch.stack(all_embeddings)
    
    print("임베딩 통계:")
    print(f"Probing Port - Min: {all_embeddings[:, 0].min()}, Max: {all_embeddings[:, 0].max()}")
    print(f"Decap - Min: {all_embeddings[:, 1].min()}, Max: {all_embeddings[:, 1].max()}")



# not png file, only txt file treatment
class HJZDataset_txt(Dataset):
    def __init__(self, root_dir, filename="impedance_data.txt", shuffle=True):
        """
        Args:
            root_dir (string): 데이터 파일이 있는 디렉토리 경로
            filename (string): 데이터 파일의 이름
            shuffle (bool): 데이터를 섞을지 여부
        """
        self.root_dir = root_dir
        self.data_path = os.path.join(root_dir, filename)
        
        with open(self.data_path, 'r') as f:
            self.data_lines = f.readlines()
        
        if shuffle:
            np.random.shuffle(self.data_lines)

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        # 한 줄의 데이터 가져오기
        line = self.data_lines[idx]
        
        # 데이터 파싱 (쉼표로 구분된 값들 처리)
        values = line.strip().split(',')
        probing_port = int(values[0])
        decap = int(values[1])
        impedance_data = np.array([float(v) for v in values[2:]], dtype=np.float32)

        # 나머지 코드는 그대로 유지
        # probing_port와 decap 정규화
        pb_x = ((probing_port % 10) + 1) / 10
        pb_y = ((probing_port // 10) + 1) / 10
        dp_x = ((decap % 10) + 1) / 10
        dp_y = ((decap // 10) + 1) / 10

        # 임베딩 생성
        embeddings = torch.tensor([pb_x, pb_y, dp_x, dp_y], dtype=torch.float)
        
        # impedance_data를 2D 텐서로 변환 (1 x 201)
        impedance_tensor = torch.tensor(impedance_data, dtype=torch.float).unsqueeze(0)

        return impedance_tensor, embeddings

    def get_original_values(self, idx):
        """원본 probing_port와 decap 값을 반환하는 메서드"""
        line = self.data_lines[idx]
        values = line.strip().split(',')
        return int(values[0]), int(values[1])


def test_HJZDataset_txt(root_dir):
    dataset = HJZDataset_txt(root_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"데이터셋 크기: {len(dataset)}")
    
    impedance_data, embeddings = next(iter(dataloader))

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    for i in range(4):
        data = impedance_data[i].squeeze().numpy()
        ax = axs[i // 2, i % 2]
        ax.plot(data)
        ax.set_title(f"Probing Port: {dataset.get_original_values(i)[0]}, Decap: {dataset.get_original_values(i)[1]}")
        ax.set_xlabel("Frequency Index")
        ax.set_ylabel("Impedance")
    
    plt.tight_layout()
    plt.show()
    
def test_HJZDataset_txt(root_dir):
    dataset = HJZDataset_txt(root_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"데이터셋 크기: {len(dataset)}")
    
    impedance_data, embeddings = next(iter(dataloader))

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    for i in range(4):
        data = impedance_data[i].squeeze().numpy()
        ax = axs[i // 2, i % 2]
        ax.plot(data)
        ax.set_title(f"Probing Port: {dataset.get_original_values(i)[0]}, Decap: {dataset.get_original_values(i)[1]}")
        ax.set_xlabel("Frequency Index")
        ax.set_ylabel("Impedance")
    
    plt.tight_layout()
    plt.show()
    
    # 임베딩 분포 확인
    all_embeddings = []
    for imp, emb in dataset:
        all_embeddings.append(emb)
        # print("Impedance shape:", imp.shape)
    all_embeddings = torch.stack(all_embeddings)
    
    print("\n임베딩 통계:")
    for i, name in enumerate(["Probing Port X", "Probing Port Y", "Decap X", "Decap Y"]):
        print(f"{name} - Min: {all_embeddings[:, i].min():.4f}, Max: {all_embeddings[:, i].max():.4f}, "
              f"Mean: {all_embeddings[:, i].mean():.4f}, Std: {all_embeddings[:, i].std():.4f}")

if __name__ == "__main__":
    root_dir = "./hj_z_data_3_vectors_240912"  # 데이터 폴더 경로를 적절히 수정하세요
    test_HJZDataset_txt(root_dir)