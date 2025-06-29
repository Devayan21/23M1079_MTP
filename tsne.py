import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

n_epochs = 10
batch_size = 64
random_seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(random_seed)
torch.manual_seed(random_seed)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                          worker_init_fn=lambda _: np.random.seed(random_seed))


class CNN10Layer(nn.Module):
    def __init__(self):
        super(CNN10Layer, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc1 = nn.Linear(256, 128)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv_layers(x)
        feat1 = x.view(x.size(0), -1)
        x = self.relu_fc(self.fc1(feat1))
        feat2 = x
        x = self.fc2(x)
        return feat1, feat2, x


def train_model(model, dataloader, n_epochs):
    features_layer1 = []
    features_layer2 = []
    labels = []

    for epoch in range(n_epochs):
        model.train()
        epoch_features1 = []
        epoch_features2 = []
        epoch_labels = []

        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            x1, x2, output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                epoch_features1.append(x1.detach().cpu().numpy())
                epoch_features2.append(x2.detach().cpu().numpy())
                epoch_labels.append(target.detach().cpu().numpy())

        features_layer1.append(np.concatenate(epoch_features1))
        features_layer2.append(np.concatenate(epoch_features2))
        labels.append(np.concatenate(epoch_labels))

        print(f"Epoch {epoch+1}/{n_epochs} completed.")

    return features_layer1, features_layer2, labels

# --------------------
# t-SNE: Epoch-wise
# --------------------
def plot_tsne_epochwise(features_list, labels_list, max_points=3000, layer_name='Penultimate Layer', output_dir='tsne_epochwise', random_seed=42):
    os.makedirs(output_dir, exist_ok=True)

    for epoch, (features, labels) in enumerate(zip(features_list, labels_list)):
        np.random.seed(random_seed)
        if len(features) > max_points:
            idx = np.random.choice(len(features), max_points, replace=False)
            features = features[idx]
            labels = labels[idx]

        features_pca = PCA(n_components=50).fit_transform(features)
        tsne = TSNE(n_components=2, random_state=random_seed)
        reduced = tsne.fit_transform(features_pca)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=5, alpha=0.7)
        plt.colorbar(scatter, ticks=range(10), label='Class')
        plt.title(f't-SNE of {layer_name} at Epoch {epoch+1}')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.grid(True)

        filename = os.path.join(output_dir, f"tsne_{layer_name.replace(' ', '_').lower()}_epoch_{epoch+1}.png")
        plt.savefig(filename)
        plt.close()
        print(f"Saved: {filename}")

# --------------------
# t-SNE: Layer-wise
# --------------------
def plot_tsne_layerwise(features_dict, labels, max_points=3000, output_dir='tsne_layerwise', random_seed=42):
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(random_seed)
    if len(labels) > max_points:
        idx = np.random.choice(len(labels), max_points, replace=False)
        labels = labels[idx]

    for layer_name, features in features_dict.items():
        if len(features) > max_points:
            features = features[idx]

        features_pca = PCA(n_components=50).fit_transform(features)
        tsne = TSNE(n_components=2, random_state=random_seed)
        reduced = tsne.fit_transform(features_pca)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=5, alpha=0.7)
        plt.colorbar(scatter, ticks=range(10), label='Class')
        plt.title(f't-SNE of Layer: {layer_name}')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.grid(True)

        filename = os.path.join(output_dir, f"tsne_layer_{layer_name.replace(' ', '_').lower()}.png")
        plt.savefig(filename)
        plt.close()
        print(f"Saved: {filename}")


model = CNN10Layer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train and collect features
features_layer1, features_layer2, labels = train_model(model, train_loader, n_epochs)

# Plot Epoch-wise t-SNE of Penultimate Layer
plot_tsne_epochwise(features_layer2, labels, layer_name='Penultimate Layer', output_dir='tsne_epochwise')

# Plot Layer-wise t-SNE at Last Epoch
layerwise_features = {
    'Conv Output': features_layer1[-1],
    'Penultimate Layer': features_layer2[-1]
}
plot_tsne_layerwise(layerwise_features, labels[-1], output_dir='tsne_layerwise')