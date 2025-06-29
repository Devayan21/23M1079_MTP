import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import logging
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
from transformers import AutoModelForImageClassification
from timm.data.transforms_factory import create_transform
from timm.models.layers import trunc_normal_

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'mambavision_t_1k_nc_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()
logger.info("Starting Neural Collapse analysis with MambaVision-T-1K on CIFAR-10, EuroSAT, Food-101, and DTD datasets.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
logger.info(f"Device properties: {torch.cuda.get_device_properties(device) if torch.cuda.is_available() else 'CPU'}")

# Define transforms for data preprocessing (using timm's create_transform for compatibility)
transform = create_transform(
    input_size=224,
    is_training=False,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

logger.info("Loading datasets...")
cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_loader = DataLoader(cifar10_train, batch_size=32, shuffle=True, num_workers=4)

eurosat_dataset = datasets.EuroSAT(root='./data', download=True, transform=transform)
eurosat_loader = DataLoader(eurosat_dataset, batch_size=32, shuffle=True, num_workers=4)

food101_dataset = datasets.Food101(root='./data', split='train', download=True, transform=transform)
food101_loader = DataLoader(food101_dataset, batch_size=32, shuffle=True, num_workers=4)

dtd_dataset = datasets.DTD(root='./data', split='train', download=True, transform=transform)
dtd_loader = DataLoader(dtd_dataset, batch_size=32, shuffle=True, num_workers=4)

# Define datasets for experiments
datasets = {
    'CIFAR-10': (cifar10_loader, 10),
    'EuroSAT': (eurosat_loader, 10),
    'Food-101': (food101_loader, 101),
    'DTD': (dtd_loader, 47)
}

# Define MambaVision-T-1K Model with custom feature extraction
class MambaVisionTCustom(nn.Module):
    def __init__(self, num_classes):
        super(MambaVisionTCustom, self).__init__()
        # Load MambaVision-T-1K with pretrained weights
        self.model = AutoModelForImageClassification.from_pretrained(
            "nvidia/MambaVision-T-1K", trust_remote_code=True
        )
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

        trunc_normal_(self.model.classifier.weight, std=0.02)
        nn.init.constant_(self.model.classifier.bias, 0)
        self.model.to(device)

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits

    def get_layer_features(self, x):
        features = {}
        x = self.model.backbone.stem(x)
        features['backbone.stem'] = x
        for i in range(4):
            x = self.model.backbone.stages[i](x)
            features[f'backbone.stages[{i}]'] = x
        x = self.model.backbone.pool(x)
        features['backbone.pool'] = x
        return features

    def get_penultimate_features(self, x):
        x = self.model.backbone.stem(x)
        for i in range(4):
            x = self.model.backbone.stages[i](x)
        x = self.model.backbone.pool(x)
        return x

# Define Neural Collapse (NC) Loss
def compute_nc_loss(features, labels, n_classes, lambda_wc=1.0, lambda_etf=0.5, lambda_norm=0.5):
    features = features.view(features.size(0), -1)
    features = features / (features.norm(dim=1, keepdim=True) + 1e-6)

    class_features_list = [[] for _ in range(n_classes)]
    for idx, label in enumerate(labels):
        class_features_list[label.item()].append(features[idx])

    within_class_loss = 0
    valid_classes = 0
    class_means = torch.zeros(n_classes, features.size(1), device=features.device)
    for c in range(n_classes):
        if len(class_features_list[c]) == 0:
            continue
        class_features = torch.stack(class_features_list[c])
        class_mean = class_features.mean(dim=0)
        class_means[c] = class_mean
        within_class_loss += ((class_features - class_mean) ** 2).mean()
        valid_classes += 1
    within_class_loss = within_class_loss / valid_classes if valid_classes > 0 else 0

    global_mean = class_means.mean(dim=0)
    centered_means = class_means - global_mean
    etf_loss = 0
    valid_pairs = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            if class_means[i].norm() > 0 and class_means[j].norm() > 0:
                cos_sim = torch.nn.functional.cosine_similarity(centered_means[i], centered_means[j], dim=0)
                etf_loss += (cos_sim + 1 / (n_classes - 1)) ** 2
                valid_pairs += 1
    etf_loss = etf_loss / valid_pairs if valid_pairs > 0 else 0

    norm_loss = 0
    valid_norms = 0
    mean_norm_squared = (centered_means.norm(dim=1) ** 2).mean()
    for i in range(n_classes):
        if class_means[i].norm() > 0:
            norm_loss += (centered_means[i].norm() ** 2 - mean_norm_squared) ** 2
            valid_norms += 1
    norm_loss = norm_loss / valid_norms if valid_norms > 0 else 0

    nc_loss = lambda_wc * within_class_loss + lambda_etf * etf_loss + lambda_norm * norm_loss
    return nc_loss

# Training function
def train(model, train_loader, optimizer, criterion, num_classes, use_nc_loss=False, nc_layers=None, nc_lambda=0.3, layerwise_lambdas=None, track_penultimate=False, experiment_name=""):
    logger.info(f"Starting training for {experiment_name}...")
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    logger.info(f"Trainable parameters: {trainable_params}")
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    nc_loss_total = 0.0
    nc_losses_penultimate = [] if track_penultimate else None
    accuracies = [] if track_penultimate else None
    layer_nc_losses_final_epoch = {} if track_penultimate else None

    for epoch in range(num_epochs):
        correct_epoch = 0
        total_epoch = 0
        running_loss_epoch = 0.0
        all_features_penultimate = [] if track_penultimate else None
        all_labels = [] if track_penultimate else None

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            ce_loss = criterion(outputs, labels)

            total_loss = ce_loss
            if use_nc_loss:
                features = model.get_layer_features(inputs)
                nc_loss = 0
                if nc_layers and layerwise_lambdas:
                    for layer, lambda_i in zip(nc_layers, layerwise_lambdas):
                        layer_features = features[layer]
                        nc_loss += lambda_i * compute_nc_loss(layer_features, labels, num_classes)
                elif nc_layers:
                    for layer in nc_layers:
                        layer_features = features[layer]
                        nc_loss += nc_lambda * compute_nc_loss(layer_features, labels, num_classes)
                else:
                    features = model.get_penultimate_features(inputs)
                    nc_loss = nc_lambda * compute_nc_loss(features, labels, num_classes)
                total_loss += nc_loss
                nc_loss_total += nc_loss.item()

            total_loss.backward()
            optimizer.step()

            running_loss_epoch += ce_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_epoch += labels.size(0)
            correct_epoch += (predicted == labels).sum().item()

            if track_penultimate:
                features = model.get_layer_features(inputs)
                all_features_penultimate.append(features['backbone.pool'].detach())
                all_labels.append(labels)

        if track_penultimate:
            all_features_penultimate = torch.cat(all_features_penultimate, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            nc_loss = compute_nc_loss(all_features_penultimate, all_labels, num_classes)
            nc_losses_penultimate.append(nc_loss.item())
            accuracy = 100 * correct_epoch / total_epoch
            accuracies.append(accuracy)

            if epoch == num_epochs - 1:
                for layer_name, layer_features in features.items():
                    layer_features = layer_features.view(layer_features.size(0), -1).detach()
                    nc_loss_layer = compute_nc_loss(layer_features, labels, num_classes)
                    layer_nc_losses_final_epoch[layer_name] = nc_loss_layer.item()

        running_loss += running_loss_epoch
        correct += correct_epoch
        total += total_epoch

        logger.info(f"{experiment_name} - Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss_epoch / len(train_loader):.4f}, Accuracy: {100 * correct_epoch / total_epoch:.2f}%")

    accuracy = 100 * correct / total
    avg_nc_loss = nc_loss_total / len(train_loader) if use_nc_loss else 0
    logger.info(f"{experiment_name} - Final Accuracy: {accuracy:.2f}%, Average NC Loss: {avg_nc_loss:.4f}")
    if track_penultimate:
        return accuracy, avg_nc_loss, nc_losses_penultimate, accuracies, layer_nc_losses_final_epoch
    return accuracy, avg_nc_loss

# Common parameters
num_epochs = 10
nc_lambda = 0.3
layerwise_lambdas = [0.1, 0.5, 0.5]

# Combined Part 1 and Baseline for CIFAR-10
logger.info("\n=== Combined Part 1 and Baseline: NC Loss Analysis and Baseline on CIFAR-10 ===")
model_cifar10 = MambaVisionTCustom(num_classes=10).to(device)
for name, param in model_cifar10.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_cifar10.parameters()), lr=0.001)

cifar10_accuracy, _, nc_losses_penultimate, accuracies, layer_nc_losses_final_epoch = train(
    model_cifar10, cifar10_loader, optimizer, criterion, num_classes=10, track_penultimate=True, experiment_name="Part 1 + Baseline (CIFAR-10)"
)

results_baseline = {'CIFAR-10': cifar10_accuracy}

# Baseline for Other Datasets
logger.info("\n=== Baseline: No NC Loss for Other Datasets ===")
for dataset_name, (loader, num_classes) in datasets.items():
    if dataset_name == 'CIFAR-10':
        continue
    logger.info(f"\nTraining on {dataset_name} (Baseline)...")
    model_baseline = MambaVisionTCustom(num_classes).to(device)
    for name, param in model_baseline.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_baseline.parameters()), lr=0.001)

    baseline_accuracy, _ = train(
        model_baseline, loader, optimizer, criterion, num_classes, use_nc_loss=False, experiment_name=f"Baseline ({dataset_name})"
    )
    results_baseline[dataset_name] = baseline_accuracy

# Part 2: NC Loss on Penultimate Layer
logger.info("\n=== Part 2: NC Loss on Penultimate Layer ===")
results_penultimate = {}
for dataset_name, (loader, num_classes) in datasets.items():
    logger.info(f"\nTraining on {dataset_name} (Penultimate NC)...")
    model_nc = MambaVisionTCustom(num_classes).to(device)
    for name, param in model_nc.named_parameters():
        if not any(n in name for n in ["backbone.stages.3", "backbone.pool", "classifier"]):
            param.requires_grad = False
    optimizer_nc = optim.Adam(filter(lambda p: p.requires_grad, model_nc.parameters()), lr=0.001)

    nc_accuracy, nc_loss_value = train(
        model_nc, loader, optimizer_nc, criterion, num_classes, use_nc_loss=True, nc_layers=['backbone.pool'], nc_lambda=nc_lambda, experiment_name=f"Penultimate NC ({dataset_name})"
    )
    results_penultimate[dataset_name] = {
        'Accuracy': nc_accuracy,
        'NC Loss': nc_loss_value
    }

# Part 3: NC Loss on Last Three Layers
logger.info("\n=== Part 3: NC Loss on Last Three Layers ===")
nc_layers = ['backbone.stages[3]', 'backbone.pool', 'backbone.pool']
results_multi_layer = {}
for dataset_name, (loader, num_classes) in datasets.items():
    logger.info(f"\nTraining on {dataset_name} (Multi-Layer NC)...")
    model_nc = MambaVisionTCustom(num_classes).to(device)
    for name, param in model_nc.named_parameters():
        if not any(n in name for n in ["backbone.stages.3", "backbone.pool", "classifier"]):
            param.requires_grad = False
    optimizer_nc = optim.Adam(filter(lambda p: p.requires_grad, model_nc.parameters()), lr=0.001)

    nc_accuracy, nc_loss_value = train(
        model_nc, loader, optimizer_nc, criterion, num_classes, use_nc_loss=True, nc_layers=nc_layers, layerwise_lambdas=layerwise_lambdas, experiment_name=f"Multi-Layer NC ({dataset_name})"
    )
    results_multi_layer[dataset_name] = {
        'Accuracy': nc_accuracy,
        'NC Loss': nc_loss_value
    }

# Analysis: Compare Results to Determine Where NC Helps
logger.info("\n=== Analysis: Where Does NC Enforcement Help? ===")
print("\n=== Analysis: Where Does NC Enforcement Help? ===")
for dataset_name in datasets.keys():
    baseline_acc = results_baseline[dataset_name]
    pen_acc = results_penultimate[dataset_name]['Accuracy']
    pen_nc_loss = results_penultimate[dataset_name]['NC Loss']
    multi_acc = results_multi_layer[dataset_name]['Accuracy']
    multi_nc_loss = results_multi_layer[dataset_name]['NC Loss']

    pen_change = pen_acc - baseline_acc
    pen_helps = pen_change > 0
    pen_analysis = (
        f"Penultimate NC {'improves' if pen_helps else 'hurts'} performance "
        f"(Baseline: {baseline_acc:.2f}%, Penultimate NC: {pen_acc:.2f}%, Change: {pen_change:.2f}%, NC Loss: {pen_nc_loss:.4f})"
    )

    multi_change = multi_acc - baseline_acc
    multi_helps = multi_change > 0
    multi_analysis = (
        f"Multi-Layer NC {'improves' if multi_helps else 'hurts'} performance "
        f"(Baseline: {baseline_acc:.2f}%, Multi-Layer NC: {multi_acc:.2f}%, Change: {multi_change:.2f}%, NC Loss: {multi_nc_loss:.4f})"
    )

    logger.info(f"\n{dataset_name}:")
    logger.info(pen_analysis)
    logger.info(multi_analysis)
    print(f"\n{dataset_name}:")
    print(pen_analysis)
    print(multi_analysis)

# Log end of experiment
logger.info("Experiment completed.")