import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import timm
import logging
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
from torch.amp import GradScaler, autocast

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'swinbase_nc_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()
logger.info("Starting Neural Collapse analysis with Swin-Tiny on CIFAR-10, EuroSAT, Food-101, and DTD datasets.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
num_epochs = 10

# Define transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets with batch_size=32
logger.info("Loading datasets...")
cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_loader = DataLoader(cifar10_train, batch_size=32, shuffle=True)

eurosat_dataset = datasets.EuroSAT(root='./data', download=True, transform=transform)
eurosat_loader = DataLoader(eurosat_dataset, batch_size=32, shuffle=True)

food101_dataset = datasets.Food101(root='./data', split='train', download=True, transform=transform)
food101_loader = DataLoader(food101_dataset, batch_size=32, shuffle=True)

dtd_dataset = datasets.DTD(root='./data', split='train', download=True, transform=transform)
dtd_loader = DataLoader(dtd_dataset, batch_size=32, shuffle=True)

# Define Swin-Base Model
class SwinBaseCustom(nn.Module):
    def __init__(self, num_classes):
        super(SwinBaseCustom, self).__init__()
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
        self.to(device)

    def forward(self, x):
        return self.model(x)

    def get_layer_features(self, x, nc_layers=None):
        features = {}
        x = self.model.patch_embed(x)
        if not nc_layers or 'patch_embed' in nc_layers:
            features['patch_embed'] = x
        x = self.model.layers[0](x)
        if not nc_layers or 'layers[0]' in nc_layers:
            features['layers[0]'] = x
        x = self.model.layers[1](x)
        if not nc_layers or 'layers[1]' in nc_layers:
            features['layers[1]'] = x
        x = self.model.layers[2](x)
        if not nc_layers or 'layers[2]' in nc_layers:
            features['layers[2]'] = x
        x = self.model.layers[3](x)
        if not nc_layers or 'layers[3]' in nc_layers:
            features['layers[3]'] = x
        x = self.model.norm(x)
        if not nc_layers or 'norm' in nc_layers:
            features['norm'] = x
        x = self.model.head.global_pool(x)
        if not nc_layers or 'head.global_pool' in nc_layers:
            features['head.global_pool'] = x
        return {k: v.to(device) for k, v in features.items()}

    def get_penultimate_features(self, x):
        x = self.model.patch_embed(x)
        x = self.model.layers[0](x)
        x = self.model.layers[1](x)
        x = self.model.layers[2](x)
        x = self.model.layers[3](x)
        x = self.model.norm(x)
        x = self.model.head.global_pool(x)
        return x.to(device)

# Define Neural Collapse (NC) Loss
def compute_nc_loss(features, labels, n_classes, lambda_wc=1.0, lambda_etf=0.5, lambda_norm=0.01):
    features = features.reshape(features.size(0), -1)
    within_class_loss = 0
    for c in range(n_classes):
        class_features = features[labels == c]
        if len(class_features) == 0:
            continue
        class_mean = class_features.mean(dim=0)
        within_class_loss += ((class_features - class_mean) ** 2).mean()
    within_class_loss = within_class_loss / n_classes if within_class_loss != 0 else 0

    class_means = torch.zeros(n_classes, features.size(1), device=features.device)
    for c in range(n_classes):
        class_features = features[labels == c]
        if len(class_features) > 0:
            class_means[c] = class_features.mean(dim=0)
    global_mean = class_means.mean(dim=0)
    centered_means = class_means - global_mean
    etf_loss = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            cos_sim = torch.nn.functional.cosine_similarity(centered_means[i], centered_means[j], dim=0)
            etf_loss += (cos_sim + 1 / (n_classes - 1)) ** 2
    etf_loss = etf_loss / (n_classes * (n_classes - 1) / 2) if etf_loss != 0 else 0

    norm_loss = 0
    mean_norm_squared = (centered_means.norm(dim=1) ** 2).mean()
    for i in range(n_classes):
        norm_loss += (centered_means[i].norm() ** 2 - mean_norm_squared) ** 2
    norm_loss = norm_loss / n_classes if norm_loss != 0 else 0

    nc_loss = lambda_wc * within_class_loss + lambda_etf * etf_loss + lambda_norm * norm_loss
    return nc_loss

# Training function
def train(model, train_loader, optimizer, criterion, num_classes, use_nc_loss=False, nc_layers=None, nc_lambda=0.3, nc_layer_lambdas=None, track_penultimate=False, experiment_name="", accumulation_steps=1, num_epochs=num_epochs, lambda_wc=1.0, lambda_etf=0.5, lambda_norm=0.01):
    logger.info(f"Starting training for {experiment_name}...")
    model.train()
    scaler = GradScaler('cuda')
    correct = 0
    total = 0
    running_loss = 0.0
    nc_loss_total = 0.0
    nc_losses_penultimate = [] if track_penultimate else None
    accuracies = [] if track_penultimate else None
    layer_nc_losses_epoch = {} if track_penultimate and nc_layers else {}
    layer_nc_losses_final_epoch = {} if track_penultimate else {}

    for epoch in range(num_epochs):
        correct_epoch = 0
        total_epoch = 0
        running_loss_epoch = 0.0
        nc_loss_epoch = 0.0
        layer_nc_losses_epoch.clear() if track_penultimate and nc_layers else None
        last_inputs = None
        last_labels = None

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if inputs is None or labels is None:
                logger.warning(f"Batch {batch_idx} has None inputs or labels, skipping.")
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            last_inputs, last_labels = inputs, labels
            optimizer.zero_grad()

            with autocast('cuda'):
                outputs = model(inputs)
                ce_loss = criterion(outputs, labels)
                total_loss = ce_loss

                if use_nc_loss:
                    if nc_layers:
                        features = model.get_layer_features(inputs, nc_layers)
                        nc_loss = 0
                        for layer in nc_layers:
                            if layer not in features:
                                logger.warning(f"Layer {layer} not found in features, skipping.")
                                continue
                            layer_features = features[layer]
                            layer_lambda = nc_layer_lambdas.get(layer, nc_lambda)
                            nc_loss += layer_lambda * compute_nc_loss(layer_features, labels, num_classes, lambda_wc, lambda_etf, lambda_norm)
                        total_loss += nc_loss
                        nc_loss_epoch += nc_loss.item()
                    else:
                        features = model.get_penultimate_features(inputs)
                        nc_loss = compute_nc_loss(features, labels, num_classes, lambda_wc, lambda_etf, lambda_norm)
                        total_loss += nc_lambda * nc_loss
                        nc_loss_epoch += nc_loss.item()

            scaler.scale(total_loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss_epoch += ce_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_epoch += labels.size(0)
            correct_epoch += (predicted == labels).sum().item()

            if track_penultimate:
                with torch.no_grad():
                    features = model.get_layer_features(inputs, nc_layers) if nc_layers else model.get_layer_features(inputs)
                    nc_loss = compute_nc_loss(features.get('head.global_pool', features['head.global_pool']), labels, num_classes, lambda_wc, lambda_etf, lambda_norm)
                    nc_losses_penultimate.append(nc_loss.item())
                    if nc_layers:
                        for layer in nc_layers:
                            if layer not in layer_nc_losses_epoch:
                                layer_nc_losses_epoch[layer] = []
                            layer_nc_losses_epoch[layer].append(compute_nc_loss(features[layer], labels, num_classes, lambda_wc, lambda_etf, lambda_norm).item())
                    else:
                        accuracy = 100 * correct_epoch / total_epoch
                        accuracies.append(accuracy)

        avg_nc_loss_epoch = nc_loss_epoch / len(train_loader) if use_nc_loss else 0
        nc_loss_total += avg_nc_loss_epoch

        running_loss += running_loss_epoch
        correct += correct_epoch
        total += total_epoch

        logger.info(f"{experiment_name} - Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss_epoch / len(train_loader):.4f}, Accuracy: {100 * correct_epoch / total_epoch:.2f}%, Avg NC Loss: {avg_nc_loss_epoch:.4f}")

        if track_penultimate and epoch == num_epochs - 1:
            if nc_layers:
                for layer in layer_nc_losses_epoch:
                    layer_nc_losses_final_epoch[layer] = np.mean(layer_nc_losses_epoch[layer])
            else:
                if last_inputs is not None and last_labels is not None:
                    features = model.get_layer_features(last_inputs, nc_layers)
                    for layer_name, layer_features in features.items():
                        nc_loss_layer = compute_nc_loss(layer_features, last_labels, num_classes, lambda_wc, lambda_etf, lambda_norm)
                        layer_nc_losses_final_epoch[layer_name] = nc_loss_layer.item()

    accuracy = 100 * correct / total
    avg_nc_loss = nc_loss_total / num_epochs if use_nc_loss else 0
    logger.info(f"{experiment_name} - Final Accuracy: {accuracy:.2f}%, Average NC Loss: {avg_nc_loss:.4f}")
    if track_penultimate:
        return accuracy, avg_nc_loss, nc_losses_penultimate, accuracies, layer_nc_losses_final_epoch
    return accuracy, avg_nc_loss

# Common parameters
nc_lambda = 0.3
nc_layer_lambdas = {
    'patch_embed': 0.0,
    'layers[0]': 0.0,
    'layers[1]': 0.0,
    'layers[2]': 0.1,
    'layers[3]': 0.5,
    'norm': 0.5,
    'head.global_pool': 0.5
}

# All datasets for experiments
datasets = {
    'CIFAR-10': (cifar10_loader, 10),
    'EuroSAT': (eurosat_loader, 10),
    'Food-101': (food101_loader, 101),
    'DTD': (dtd_loader, 47)
}

# Combined Part 1 and Baseline for CIFAR-10
logger.info("\n=== Combined Part 1 and Baseline: NC Loss Analysis and Baseline on CIFAR-10 ===")
model_cifar10 = SwinBaseCustom(num_classes=10).to(device)
for name, param in model_cifar10.named_parameters():
    if "head" not in name:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_cifar10.parameters()), lr=0.001)

cifar10_accuracy, _, nc_losses_penultimate, accuracies, layer_nc_losses_final_epoch = train(
    model_cifar10, cifar10_loader, optimizer, criterion, num_classes=10, track_penultimate=True, experiment_name="Part 1 + Baseline (CIFAR-10)"
)

results_baseline = {'CIFAR-10': cifar10_accuracy}

# Baseline for Other Datasets (EuroSAT, Food-101, DTD)
logger.info("\n=== Baseline: No NC Loss for Other Datasets ===")
for dataset_name, (loader, num_classes) in datasets.items():
    if dataset_name == 'CIFAR-10':
        continue
    logger.info(f"\nTraining on {dataset_name} (Baseline)...")
    model_baseline = SwinBaseCustom(num_classes).to(device)
    for name, param in model_baseline.named_parameters():
        if "head" not in name:
            param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_baseline.parameters()), lr=0.001)

    baseline_accuracy, _ = train(
        model_baseline, loader, optimizer, criterion, num_classes, track_penultimate=False, experiment_name=f"Baseline ({dataset_name})"
    )
    results_baseline[dataset_name] = baseline_accuracy

# Part 2: NC Loss on Penultimate Layer
logger.info("\n=== Part 2: NC Loss on Penultimate Layer ===")
results_penultimate = {}
criterion = nn.CrossEntropyLoss()
for dataset_name, (loader, num_classes) in datasets.items():
    logger.info(f"\nTraining on {dataset_name} (Penultimate NC)...")
    model_nc = SwinBaseCustom(num_classes).to(device)
    for name, param in model_nc.named_parameters():
        if any(k in name for k in ['patch_embed', 'layers.0', 'layers.1', 'layers.2']):
            param.requires_grad = False
        else:
            param.requires_grad = True
    optimizer_nc = optim.Adam(filter(lambda p: p.requires_grad, model_nc.parameters()), lr=0.001)

    nc_accuracy, nc_loss_value, _, _, _ = train(
        model_nc, loader, optimizer_nc, criterion, num_classes, use_nc_loss=True, track_penultimate=True, experiment_name=f"Penultimate NC ({dataset_name})"
    )
    results_penultimate[dataset_name] = {
        'Accuracy': nc_accuracy,
        'NC Loss': nc_loss_value
    }

# Part 3: NC Loss on Random Combination of Middle and Last Layers
logger.info("\n=== Part 3: NC Loss on Middle and Last Layers ===")
nc_layers = ['patch_embed', 'layers[0]', 'layers[1]', 'layers[2]', 'layers[3]', 'norm', 'head.global_pool']
results_multi_layer = {}
for dataset_name, (loader, num_classes) in datasets.items():
    logger.info(f"\nTraining on {dataset_name} (Multi-Layer NC with ? per Layer)...")
    model_nc = SwinBaseCustom(num_classes).to(device)
    for name, param in model_nc.named_parameters():
        if any(k in name for k in ['patch_embed', 'layers.0', 'layers.1', 'layers.2']):
            param.requires_grad = False
        else:
            param.requires_grad = True
    optimizer_nc = optim.Adam(filter(lambda p: p.requires_grad, model_nc.parameters()), lr=0.001)

    nc_accuracy, nc_loss_value, _, _, _ = train(
        model_nc, loader, optimizer_nc, criterion, num_classes,
        use_nc_loss=True,
        nc_layers=nc_layers,
        nc_layer_lambdas=nc_layer_lambdas,
        track_penultimate=True,
        experiment_name=f"Multi-Layer NC ({dataset_name})"
    )
    results_multi_layer[dataset_name] = {
        'Accuracy': nc_accuracy,
        'NC Loss': nc_loss_value
    }

# Analysis: Compare Results to Determine Where NC Helps
logger.info("\n=== Analysis: Where Does NC Enforcement Help? ===")
print("\n=== Analysis: Where Does NC Enforcement Help? ===")
for dataset_name in datasets.keys():
    baseline_acc = results_baseline.get(dataset_name, 0.0)
    pen_acc = results_penultimate.get(dataset_name, {}).get('Accuracy', 0.0)
    pen_nc_loss = results_penultimate.get(dataset_name, {}).get('NC Loss', 0.0)
    multi_acc = results_multi_layer.get(dataset_name, {}).get('Accuracy', 0.0)
    multi_nc_loss = results_multi_layer.get(dataset_name, {}).get('NC Loss', 0.0)

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

    if "Part 1" in dataset_name:
        logger.info(f"\n{dataset_name}:")
        logger.info(pen_analysis.replace(", NC Loss: {pen_nc_loss:.4f}", ""))
        logger.info(multi_analysis.replace(", NC Loss: {multi_nc_loss:.4f}", ""))
        print(f"\n{dataset_name}:")
        print(pen_analysis.replace(", NC Loss: {pen_nc_loss:.4f}", ""))
        print(multi_analysis.replace(", NC Loss: {multi_nc_loss:.4f}", ""))
    else:
        logger.info(f"\n{dataset_name}:")
        logger.info(pen_analysis)
        logger.info(multi_analysis)
        print(f"\n{dataset_name}:")
        print(pen_analysis)
        print(multi_analysis)

# Log end of experiment
logger.info("Experiment completed.")