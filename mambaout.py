import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import logging
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'mambaout_kobe_nc_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()
logger.info("Starting Neural Collapse analysis with MambaOut Kobe on CIFAR-10, EuroSAT, Food-101, and DTD datasets.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
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

# MambaOut implementation
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'mambaout_kobe': _cfg(url='https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_kobe.pth'),
}

class StemLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=96, act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1)
        self.norm1 = norm_layer(out_channels // 2)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        return x

class DownsampleLayer(nn.Module):
    def __init__(self, in_channels=96, out_channels=198, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x

class MlpHead(nn.Module):
    def __init__(self, dim, num_classes=1000, act_layer=nn.GELU, mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x

class GatedCNNBlock(nn.Module):
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop_path=0., **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2)
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1)
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        return x + shortcut

DOWNSAMPLE_LAYERS_FOUR_STAGES = [StemLayer] + [DownsampleLayer]*3

class MambaOut(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 15, 3], dims=[48, 96, 192, 288],
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU, conv_ratio=1.0, kernel_size=7, drop_path_rate=0.,
                 output_norm=partial(nn.LayerNorm, eps=1e-6), head_fn=MlpHead, head_dropout=0.0, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        if not isinstance(depths, (list, tuple)):
            depths = [depths]
        if not isinstance(dims, (list, tuple)):
            dims = [dims]
        num_stage = len(depths)
        self.num_stage = num_stage
        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i+1]) for i in range(num_stage)]
        )
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[GatedCNNBlock(dim=dims[i], norm_layer=norm_layer, act_layer=act_layer, kernel_size=kernel_size,
                                conv_ratio=conv_ratio, drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.norm = output_norm(dims[-1])
        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, x):
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([1, 2]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def mambaout_kobe(pretrained=False, **kwargs):
    model = MambaOut(depths=[3, 3, 15, 3], dims=[48, 96, 192, 288], **kwargs)
    model.default_cfg = default_cfgs['mambaout_kobe']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
        model.load_state_dict(filtered_state_dict, strict=False)
        logger.info("Loaded pretrained weights, skipping head.fc2 due to class number mismatch.")
    return model

# Custom wrapper for MambaOut Kobe with feature extraction
class MambaOutKobeCustom(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = mambaout_kobe(pretrained=True, num_classes=num_classes)
        self.to(device)

    def forward(self, x):
        return self.model(x)

    def get_layer_features(self, x, nc_layers=None):
        features = {}
        x = self.model.downsample_layers[0](x)
        if not nc_layers or 'downsample_layers[0]' in nc_layers:
            features['downsample_layers[0]'] = x
        x = self.model.stages[0](x)
        if not nc_layers or 'stages[0]' in nc_layers:
            features['stages[0]'] = x
        x = self.model.downsample_layers[1](x)
        if not nc_layers or 'downsample_layers[1]' in nc_layers:
            features['downsample_layers[1]'] = x
        x = self.model.stages[1](x)
        if not nc_layers or 'stages[1]' in nc_layers:
            features['stages[1]'] = x
        x = self.model.downsample_layers[2](x)
        if not nc_layers or 'downsample_layers[2]' in nc_layers:
            features['downsample_layers[2]'] = x
        x = self.model.stages[2](x)
        if not nc_layers or 'stages[2]' in nc_layers:
            features['stages[2]'] = x
        x = self.model.downsample_layers[3](x)
        if not nc_layers or 'downsample_layers[3]' in nc_layers:
            features['downsample_layers[3]'] = x
        x = self.model.stages[3](x)
        if not nc_layers or 'stages[3]' in nc_layers:
            features['stages[3]'] = x
        x = self.model.norm(x)
        if not nc_layers or 'norm' in nc_layers:
            features['norm'] = x
        x = x.mean([1, 2])
        if not nc_layers or 'global_pool' in nc_layers:
            features['global_pool'] = x
        return {k: v.to(device) for k, v in features.items()}

    def get_penultimate_features(self, x):
        x = self.model.downsample_layers[0](x)
        x = self.model.stages[0](x)
        x = self.model.downsample_layers[1](x)
        x = self.model.stages[1](x)
        x = self.model.downsample_layers[2](x)
        x = self.model.stages[2](x)
        x = self.model.downsample_layers[3](x)
        x = self.model.stages[3](x)
        x = self.model.norm(x)
        x = x.mean([1, 2])
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
def train(model, train_loader, optimizer, criterion, num_classes, use_nc_loss=False, nc_layers=None, nc_lambda=0.3, nc_layer_lambdas=None, track_penultimate=False, experiment_name="", num_epochs=10, lambda_wc=1.0, lambda_etf=0.5, lambda_norm=0.01):
    logger.info(f"Starting training for {experiment_name}...")
    model.train()
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

            total_loss.backward()
            optimizer.step()

            running_loss_epoch += ce_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_epoch += labels.size(0)
            correct_epoch += (predicted == labels).sum().item()

            if track_penultimate:
                with torch.no_grad():
                    features = model.get_layer_features(inputs, nc_layers) if nc_layers else model.get_layer_features(inputs)
                    nc_loss = compute_nc_loss(features.get('global_pool', features['global_pool']), labels, num_classes, lambda_wc, lambda_etf, lambda_norm)
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

num_epochs = 10
nc_lambda = 0.3
nc_layer_lambdas = {
    'downsample_layers[0]': 0.0,
    'stages[0]': 0.0,
    'downsample_layers[1]': 0.0,
    'stages[1]': 0.0,
    'downsample_layers[2]': 0.0,
    'stages[2]': 0.0,
    'downsample_layers[3]': 0.1,
    'stages[3]': 0.5,
    'norm': 0.5,
    'global_pool': 0.5
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
model_cifar10 = MambaOutKobeCustom(num_classes=10).to(device)
for name, param in model_cifar10.named_parameters():
    if "head" not in name:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_cifar10.parameters()), lr=0.001)

cifar10_accuracy, _, nc_losses_penultimate, accuracies, layer_nc_losses_final_epoch = train(
    model_cifar10, cifar10_loader, optimizer, criterion, num_classes=10, track_penultimate=True, experiment_name="Part 1 + Baseline (CIFAR-10)"
)

results_baseline = {'CIFAR-10': cifar10_accuracy}

logger.info("Plotting NC Loss vs. Layer at Final Epoch for CIFAR-10...")
layers = list(layer_nc_losses_final_epoch.keys())
nc_values = list(layer_nc_losses_final_epoch.values())

# Baseline for Other Datasets (EuroSAT, Food-101, DTD)
logger.info("\n=== Baseline: No NC Loss for Other Datasets ===")
for dataset_name, (loader, num_classes) in datasets.items():
    if dataset_name == 'CIFAR-10':
        continue
    logger.info(f"\nTraining on {dataset_name} (Baseline)...")
    model_baseline = MambaOutKobeCustom(num_classes).to(device)
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
    model_nc = MambaOutKobeCustom(num_classes).to(device)
    for name, param in model_nc.named_parameters():
        if any(k in name for k in ['downsample_layers.0', 'stages.0', 'downsample_layers.1', 'stages.1']):
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
nc_layers = ['downsample_layers[0]', 'stages[0]', 'downsample_layers[1]', 'stages[1]', 'downsample_layers[2]', 'stages[2]', 'downsample_layers[3]', 'stages[3]', 'norm', 'global_pool']
results_multi_layer = {}
for dataset_name, (loader, num_classes) in datasets.items():
    logger.info(f"\nTraining on {dataset_name} (Multi-Layer NC with ? per Layer)...")
    model_nc = MambaOutKobeCustom(num_classes).to(device)
    for name, param in model_nc.named_parameters():
        if any(k in name for k in ['downsample_layers.0', 'stages.0', 'downsample_layers.1', 'stages.1']):
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