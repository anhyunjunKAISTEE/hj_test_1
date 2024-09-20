import torch
import habana_frameworks.torch.core as htcore  #gaudi
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from DNN import *
from utils_dataset import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# GAUDI 설정
device = torch.device("hpu")  #gaudi

class ImpedanceDataset(Dataset):
    def __init__(self, embeddings, impedance_data):
        self.embeddings = embeddings
        self.impedance_data = impedance_data

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.impedance_data[idx]

def setup_visualization():
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    return fig, ax

def update_plot(ax, train_losses, val_losses, batch_num):
    ax.clear()
    ax.plot(range(len(train_losses)), train_losses, label='Training Loss', alpha=0.7)
    
    val_steps = np.linspace(0, len(train_losses), len(val_losses), endpoint=False).astype(int)
    ax.scatter(val_steps, val_losses, color='red', label='Validation Loss', s=50)
    
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Batch number')
    ax.set_ylabel('Loss')
    ax.legend()
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

def load_data(root_dir):
    dataset = HJZDataset_txt(root_dir)
    embeddings = []
    impedance_data = []

    for imp, emb in dataset:
        embeddings.append(emb)
        impedance_data.append(imp)

    embeddings = torch.stack(embeddings)
    impedance_data = torch.stack(impedance_data)

    X_train, X_val, y_train, y_val = train_test_split(embeddings, impedance_data, test_size=0.2, random_state=42)

    train_dataset = ImpedanceDataset(X_train, y_train)
    val_dataset = ImpedanceDataset(X_val, y_val)

    return train_dataset, val_dataset

def train(model_name, train_dataset, val_dataset, batch_size=32, num_epochs=10, learning_rate=0.001):
    # 모델 초기화
    if model_name == "DNNModel":
        model = DNNModel().to(device)  #gaudi
    elif model_name == "DNNModel2":
        model = DNNModel2().to(device)  #gaudi
    else:
        raise ValueError(f"Unknown model: {model_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    fig, ax = setup_visualization()

    train_losses = []
    val_losses = []
    batch_num = 0

    for epoch in range(num_epochs):
        model.train()
        for embeddings, impedance in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            embeddings = embeddings.to(device)  #gaudi
            impedance = impedance.to(device)  #gaudi
            optimizer.zero_grad()
            outputs = model(embeddings.float())
            loss = criterion(outputs.squeeze(), impedance.float())
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            batch_num += 1
            
            if batch_num % 10 == 0:
                update_plot(ax, train_losses, val_losses, batch_num)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for embeddings, impedance in val_loader:
                embeddings = embeddings.to(device)  #gaudi
                impedance = impedance.to(device)  #gaudi
                outputs = model(embeddings.float())
                loss = criterion(outputs.squeeze(), impedance.float())
                val_loss += loss.item() * embeddings.size(0)

        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

        update_plot(ax, train_losses, val_losses, batch_num)

    print("Training complete!")
    plt.ioff()
    plt.show()

    return model

if __name__ == "__main__":
    root_dir = "./hj_z_data_3_vectors_240912"
    train_dataset, val_dataset = load_data(root_dir)
    
    # model_name = "DNNModel"  # 또는 "AdvancedDNNModel"
    model_name = "DNNModel2"
    trained_model = train(model_name, train_dataset, val_dataset)

    # torch.save(trained_model.state_dict(), f"{model_name}_trained.pth")