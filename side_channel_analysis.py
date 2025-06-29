import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 (adjust for your dataset if needed)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=2)
num_classes = 10

# Define ShuffleNetV2 with layer access
class ShuffleNetV2Custom(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def get_layer_features(self, x):
        features = {}
        x = self.model.conv1(x)
        features['conv1'] = x
        x = self.model.maxpool(x)
        features['maxpool'] = x
        x = self.model.stage2(x)
        features['stage2'] = x
        x = self.model.stage3(x)
        features['stage3'] = x
        x = self.model.stage4(x)
        features['stage4'] = x
        x = self.model.conv5(x)
        features['conv5'] = x
        x = x.view(x.size(0), -1)
        features['penultimate'] = x
        return features

# Define a simple classifier for side-channel
class SideChannelClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Load pretrained NC model
model_nc = ShuffleNetV2Custom(num_classes)
model_nc.load_state_dict(torch.load('model_nc.pth', map_location=device))
model_nc.to(device).eval()

# Train side-channel classifier on features from one layer
def train_side_channel(model, layer_name, data_loader, epochs=5, lr=0.001):
    features_all, labels_all = [], []
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(data_loader, desc=f"Extracting {layer_name}"):
            x, y = x.to(device), y.to(device)
            feats = model.get_layer_features(x)[layer_name]
            feats = feats.view(feats.size(0), -1).cpu()
            features_all.append(feats)
            labels_all.append(y.cpu())

    features_all = torch.cat(features_all)
    labels_all = torch.cat(labels_all)
    dataset = TensorDataset(features_all, labels_all)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    classifier = SideChannelClassifier(features_all.shape[1], num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    classifier.train()
    for epoch in range(epochs):
        correct, total, loss_sum = 0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = classifier(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item()
        print(f"[{layer_name}] Epoch {epoch+1}/{epochs} - Loss: {loss_sum/len(loader):.4f}, Acc: {100*correct/total:.2f}%")
    return classifier

# Evaluate on test set
def evaluate(classifier, model, layer_name, loader):
    classifier.eval()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feats = model.get_layer_features(x)[layer_name]
            feats = feats.view(feats.size(0), -1)
            out = classifier(feats)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = 100 * correct / total
    print(f"[{layer_name}] Test Accuracy: {acc:.2f}%")
    return acc

# Run side-channel for selected layers
layer_list = ['stage2', 'stage3', 'stage4', 'conv5', 'penultimate']
results = {}

for layer in layer_list:
    print(f"\n--- Side-channel analysis for {layer} ---")
    clf = train_side_channel(model_nc, layer, train_loader)
    acc = evaluate(clf, model_nc, layer, test_loader)
    results[layer] = acc

# Summary
print("\n=== Side-Channel Accuracy Summary ===")
for layer, acc in results.items():
    print(f"{layer:>10}: {acc:.2f}%")
