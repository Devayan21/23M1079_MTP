import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
from torch.autograd import Function
import pywt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Helper functions for wavelet transforms
def sfb1d(lo, hi, g0, g1, mode='zero', dim=-1):
    C = lo.shape[1]
    d = dim % 4
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()), dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()), dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1,1,1,1]
    shape[d] = L
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)
    s = (2, 1) if d == 2 else (1,2)
    g0 = torch.cat([g0]*C, dim=0)
    g1 = torch.cat([g1]*C, dim=0)
    if mode == 'per' or mode == 'periodization':
        y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + F.conv_transpose2d(hi, g1, stride=s, groups=C)
        if d == 2:
            y[:,:,:L-2] = y[:,:,:L-2] + y[:,:,N:N+L-2]
            y = y[:,:,:N]
        else:
            y[:,:,:,:L-2] = y[:,:,:,:L-2] + y[:,:,:,N:N+L-2]
            y = y[:,:,:,:N]
        y = roll(y, 1-L//2, dim=dim)
    else:
        if mode == 'zero' or mode == 'symmetric' or mode == 'reflect' or mode == 'periodic':
            pad = (L-2, 0) if d == 2 else (0, L-2)
            y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
        else:
            raise ValueError("Unknown pad type: {}".format(mode))
    return y

def roll(x, shift, dim):
    return torch.roll(x, shift, dim)

def reflect(x, minx, maxx):
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)

def mode_to_int(mode):
    if mode == 'zero': return 0
    elif mode == 'symmetric': return 1
    elif mode == 'per' or mode == 'periodization': return 2
    elif mode == 'constant': return 3
    elif mode == 'reflect': return 4
    elif mode == 'replicate': return 5
    elif mode == 'periodic': return 6
    else: raise ValueError("Unknown pad type: {}".format(mode))

def int_to_mode(mode):
    if mode == 0: return 'zero'
    elif mode == 1: return 'symmetric'
    elif mode == 2: return 'periodization'
    elif mode == 3: return 'constant'
    elif mode == 4: return 'reflect'
    elif mode == 5: return 'replicate'
    elif mode == 6: return 'periodic'
    else: raise ValueError("Unknown pad type: {}".format(mode))

def afb1d(x, h0, h1, mode='zero', dim=-1):
    C = x.shape[1]
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d]
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]), dtype=torch.float, device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]), dtype=torch.float, device=x.device)
    L = h0.numel()
    L2 = L // 2
    shape = [1,1,1,1]
    shape[d] = L
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)
    if mode == 'per' or mode == 'periodization':
        if x.shape[dim] % 2 == 1:
            if d == 2:
                x = torch.cat((x, x[:,:,-1:]), dim=2)
            else:
                x = torch.cat((x, x[:,:,:,-1:]), dim=3)
            N += 1
        x = roll(x, -L2, dim=d)
        pad = (L-1, 0) if d == 2 else (0, L-1)
        lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        N2 = N//2
        if d == 2:
            lohi[:,:,:L2] = lohi[:,:,:L2] + lohi[:,:,N2:N2+L2]
            lohi = lohi[:,:,:N2]
        else:
            lohi[:,:,:,:L2] = lohi[:,:,:,:L2] + lohi[:,:,:,N2:N2+L2]
            lohi = lohi[:,:,:,:N2]
    else:
        outsize = pywt.dwt_coeff_len(N, L, mode=mode)
        p = 2 * (outsize - 1) - N + L
        if mode == 'zero':
            if p % 2 == 1:
                pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
                x = F.pad(x, pad)
            pad = (p//2, 0) if d == 2 else (0, p//2)
            lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        elif mode == 'symmetric' or mode == 'reflect' or mode == 'periodic':
            pad = (0, 0, p//2, (p+1)//2) if d == 2 else (p//2, (p+1)//2, 0, 0)
            x = F.pad(x, pad, mode=mode)
            lohi = F.conv2d(x, h, stride=s, groups=C)
        else:
            raise ValueError("Unknown pad type: {}".format(mode))
    return lohi

class AFB2D(Function):
    @staticmethod
    def forward(ctx, x, h0_row, h1_row, h0_col, h1_col, mode):
        ctx.save_for_backward(h0_row, h1_row, h0_col, h1_col)
        ctx.shape = x.shape[-2:]
        mode = int_to_mode(mode)
        ctx.mode = mode
        lohi = afb1d(x, h0_row, h1_row, mode=mode, dim=3)
        y = afb1d(lohi, h0_col, h1_col, mode=mode, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        low = y[:,:,0].contiguous()
        highs = y[:,:,1:].contiguous()
        return low, highs

    @staticmethod
    def backward(ctx, low, highs):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0_row, h1_row, h0_col, h1_col = ctx.saved_tensors
            lh, hl, hh = torch.unbind(highs, dim=2)
            lo = sfb1d(low, lh, h0_col, h1_col, mode=mode, dim=2)
            hi = sfb1d(hl, hh, h0_col, h1_col, mode=mode, dim=2)
            dx = sfb1d(lo, hi, h0_row, h1_row, mode=mode, dim=3)
            if dx.shape[-2] > ctx.shape[-2] and dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:,:,:ctx.shape[-2], :ctx.shape[-1]]
            elif dx.shape[-2] > ctx.shape[-2]:
                dx = dx[:,:,:ctx.shape[-2]]
            elif dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:,:,:,:ctx.shape[-1]]
        return dx, None, None, None, None, None

def prep_filt_afb2d(h0_col, h1_col, h0_row=None, h1_row=None, device=device):
    h0_col, h1_col = prep_filt_afb1d(h0_col, h1_col, device)
    if h0_row is None:
        h0_row, h1_col = h0_col, h1_col
    else:
        h0_row, h1_row = prep_filt_afb1d(h0_row, h1_row, device)
    h0_col = h0_col.reshape((1, 1, -1, 1))
    h1_col = h1_col.reshape((1, 1, -1, 1))
    h0_row = h0_row.reshape((1, 1, 1, -1))
    h1_row = h1_row.reshape((1, 1, 1, -1))
    return h0_col, h1_col, h0_row, h1_row

def prep_filt_afb1d(h0, h1, device=device):
    h0 = np.array(h0[::-1]).ravel()
    h1 = np.array(h1[::-1]).ravel()
    t = torch.get_default_dtype()
    h0 = torch.tensor(h0, device=device, dtype=t).reshape((1, 1, -1))
    h1 = torch.tensor(h1, device=device, dtype=t).reshape((1, 1, -1))
    return h0, h1

class DWTForward(nn.Module):
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]
        filts = prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        self.J = J
        self.mode = mode

    def forward(self, x):
        yh = []
        ll = x
        mode = mode_to_int(self.mode)
        for j in range(self.J):
            ll, high = AFB2D.apply(ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
            yh.append(high)
        return ll, yh

# Initialize DWTForward instances
xf1 = DWTForward(J=1, mode='zero', wave='db1').to(device)
xf2 = DWTForward(J=2, mode='zero', wave='db1').to(device)
xf3 = DWTForward(J=3, mode='zero', wave='db1').to(device)
xf4 = DWTForward(J=4, mode='zero', wave='db1').to(device)

# Define Level*Waveblock classes
class Level1Waveblock(nn.Module):
    def __init__(self, *, mult=2, ff_channel=16, final_dim=16, dropout=0.5):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Conv2d(final_dim, final_dim*mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(final_dim*mult, ff_channel, 1),
            nn.ConvTranspose2d(ff_channel, final_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(final_dim)
        )
        self.reduction = nn.Conv2d(final_dim, int(final_dim/4), 1)
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.reduction(x)
        Y1, Yh = xf1(x)
        x = torch.reshape(Yh[0], (b, int(c*3/4), int(h/2), int(w/2)))
        x = torch.cat((Y1, x), dim=1)
        x = self.feedforward(x)
        return x

class Level2Waveblock(nn.Module):
    def __init__(self, *, mult=2, ff_channel=16, final_dim=16, dropout=0.5):
        super().__init__()
        self.feedforward1 = nn.Sequential(
            nn.Conv2d(final_dim + int(final_dim/2), final_dim*mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(final_dim*mult, ff_channel, 1),
            nn.ConvTranspose2d(ff_channel, final_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(final_dim)
        )
        self.feedforward2 = nn.Sequential(
            nn.Conv2d(final_dim, final_dim*mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(final_dim*mult, ff_channel, 1),
            nn.ConvTranspose2d(ff_channel, int(final_dim/2), 4, stride=2, padding=1),
            nn.BatchNorm2d(int(final_dim/2))
        )
        self.reduction = nn.Conv2d(final_dim, int(final_dim/4), 1)
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.reduction(x)
        Y1, Yh = xf1(x)
        Y2, Yh = xf2(x)
        x1 = torch.reshape(Yh[0], (b, int(c*3/4), int(h/2), int(w/2)))
        x2 = torch.reshape(Yh[1], (b, int(c*3/4), int(h/4), int(w/4)))
        x1 = torch.cat((Y1, x1), dim=1)
        x2 = torch.cat((Y2, x2), dim=1)
        x2 = self.feedforward2(x2)
        x1 = torch.cat((x1, x2), dim=1)
        x = self.feedforward1(x1)
        return x

class Level3Waveblock(nn.Module):
    def __init__(self, *, mult=2, ff_channel=16, final_dim=16, dropout=0.5):
        super().__init__()
        self.feedforward1 = nn.Sequential(
            nn.Conv2d(final_dim + int(final_dim/2), final_dim*mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(final_dim*mult, ff_channel, 1),
            nn.ConvTranspose2d(ff_channel, final_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(final_dim)
        )
        self.feedforward2 = nn.Sequential(
            nn.Conv2d(final_dim + int(final_dim/2), final_dim*mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(final_dim*mult, ff_channel, 1),
            nn.ConvTranspose2d(ff_channel, int(final_dim/2), 4, stride=2, padding=1),
            nn.BatchNorm2d(int(final_dim/2))
        )
        self.feedforward3 = nn.Sequential(
            nn.Conv2d(final_dim, final_dim*mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(final_dim*mult, ff_channel, 1),
            nn.ConvTranspose2d(ff_channel, int(final_dim/2), 4, stride=2, padding=1),
            nn.BatchNorm2d(int(final_dim/2))
        )
        self.reduction = nn.Conv2d(final_dim, int(final_dim/4), 1)
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.reduction(x)
        Y1, Yh = xf1(x)
        Y2, Yh = xf2(x)
        Y3, Yh = xf3(x)
        x1 = torch.reshape(Yh[0], (b, int(c*3/4), int(h/2), int(w/2)))
        x2 = torch.reshape(Yh[1], (b, int(c*3/4), int(h/4), int(w/4)))
        x3 = torch.reshape(Yh[2], (b, int(c*3/4), int(h/8), int(w/8)))
        x1 = torch.cat((Y1, x1), dim=1)
        x2 = torch.cat((Y2, x2), dim=1)
        x3 = torch.cat((Y3, x3), dim=1)
        x3 = self.feedforward3(x3)
        x2 = torch.cat((x2, x3), dim=1)
        x2 = self.feedforward2(x2)
        x1 = torch.cat((x1, x2), dim=1)
        x = self.feedforward1(x1)
        return x

class Level4Waveblock(nn.Module):
    def __init__(self, *, mult=2, ff_channel=16, final_dim=16, dropout=0.5):
        super().__init__()
        self.feedforward1 = nn.Sequential(
            nn.Conv2d(final_dim + int(final_dim/2), final_dim*mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(final_dim*mult, ff_channel, 1),
            nn.ConvTranspose2d(ff_channel, final_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(final_dim)
        )
        self.feedforward2 = nn.Sequential(
            nn.Conv2d(final_dim + int(final_dim/2), final_dim*mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(final_dim*mult, ff_channel, 1),
            nn.ConvTranspose2d(ff_channel, int(final_dim/2), 4, stride=2, padding=1),
            nn.BatchNorm2d(int(final_dim/2))
        )
        self.feedforward3 = nn.Sequential(
            nn.Conv2d(final_dim + int(final_dim/2), final_dim*mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(final_dim*mult, ff_channel, 1),
            nn.ConvTranspose2d(ff_channel, int(final_dim/2), 4, stride=2, padding=1),
            nn.BatchNorm2d(int(final_dim/2))
        )
        self.feedforward4 = nn.Sequential(
            nn.Conv2d(final_dim, final_dim*mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(final_dim*mult, ff_channel, 1),
            nn.ConvTranspose2d(ff_channel, int(final_dim/2), 4, stride=2, padding=1),
            nn.BatchNorm2d(int(final_dim/2))
        )
        self.reduction = nn.Conv2d(final_dim, int(final_dim/4), 1)
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.reduction(x)
        Y1, Yh = xf1(x)
        Y2, Yh = xf2(x)
        Y3, Yh = xf3(x)
        Y4, Yh = xf4(x)
        x1 = torch.reshape(Yh[0], (b, int(c*3/4), int(h/2), int(w/2)))
        x2 = torch.reshape(Yh[1], (b, int(c*3/4), int(h/4), int(w/4)))
        x3 = torch.reshape(Yh[2], (b, int(c*3/4), int(h/8), int(w/8)))
        x4 = torch.reshape(Yh[3], (b, int(c*3/4), int(h/16), int(w/16)))
        x1 = torch.cat((Y1, x1), dim=1)
        x2 = torch.cat((Y2, x2), dim=1)
        x3 = torch.cat((Y3, x3), dim=1)
        x4 = torch.cat((Y4, x4), dim=1)
        x4 = self.feedforward4(x4)
        x3 = torch.cat((x3, x4), dim=1)
        x3 = self.feedforward3(x3)
        x2 = torch.cat((x2, x3), dim=1)
        x2 = self.feedforward2(x2)
        x1 = torch.cat((x1, x2), dim=1)
        x = self.feedforward1(x1)
        return x

# Define the WaveMix class
class WaveMix(nn.Module):
    def __init__(self, *, num_classes=1000, depth=16, mult=2, ff_channel=192, final_dim=192, dropout=0.5, level=3, initial_conv='pachify', patch_size=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if level == 4:
                self.layers.append(Level4Waveblock(mult=mult, ff_channel=ff_channel, final_dim=final_dim, dropout=dropout))
            elif level == 3:
                self.layers.append(Level3Waveblock(mult=mult, ff_channel=ff_channel, final_dim=final_dim, dropout=dropout))
            elif level == 2:
                self.layers.append(Level2Waveblock(mult=mult, ff_channel=ff_channel, final_dim=final_dim, dropout=dropout))
            else:
                self.layers.append(Level1Waveblock(mult=mult, ff_channel=ff_channel, final_dim=final_dim, dropout=dropout))
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(final_dim, num_classes)
        )
        if initial_conv == 'strided':
            self.conv = nn.Sequential(
                nn.Conv2d(3, int(final_dim/2), 3, stride=1, padding=1),
                nn.Conv2d(int(final_dim/2), final_dim, 3, stride=1, padding=1)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(3, int(final_dim/4), 3, 1, 1),
                nn.Conv2d(int(final_dim/4), int(final_dim/2), 3, 1, 1),
                nn.Conv2d(int(final_dim/2), final_dim, patch_size, patch_size),
                nn.GELU(),
                nn.BatchNorm2d(final_dim)
            )
        # Initialize classifier
        trunc_normal_(self.pool[2].weight, std=0.02)
        nn.init.constant_(self.pool[2].bias, 0)

    def forward(self, img):
        x = self.conv(img)
        for attn in self.layers:
            x = attn(x) + x
        out = self.pool(x)
        return out

# Define WaveMix Model with custom feature extraction
class WaveMixCustom(nn.Module):
    def __init__(self, num_classes):
        super(WaveMixCustom, self).__init__()
        self.model = WaveMix(
            num_classes=num_classes,
            depth=16,
            mult=2,
            ff_channel=192,
            final_dim=192,
            dropout=0.5,
            level=3,
            initial_conv='pachify',
            patch_size=4
        )
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

    def get_layer_features(self, x):
        features = {}
        x = self.model.conv(x)
        features['conv'] = x
        for i, block in enumerate(self.model.layers):
            x = block(x) + x
            if i in [3, 7, 11, 13, 14, 15]:
                features[f'blocks[{i}]'] = x
        for i, layer in enumerate(self.model.pool):
            if i == 2:
                break
            x = layer(x)
            if i == 1:
                features['pool'] = x
        return features

    def get_penultimate_features(self, x):
        x = self.model.conv(x)
        for block in self.model.layers:
            x = block(x) + x
        for i, layer in enumerate(self.model.pool):
            if i == 2:
                break
            x = layer(x)
        return x

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'wavemix_nc_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()
logger.info("Starting Neural Collapse analysis with WaveMix on CIFAR-10, EuroSAT, Food-101, and DTD datasets.")

# Define transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
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

# Define Neural Collapse (NC) Loss
def compute_nc_loss(features, labels, n_classes, lambda_wc=1.0, lambda_etf=0.5, lambda_norm=0.01):
    features = features.reshape(features.size(0), -1)
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
                all_features_penultimate.append(features['pool'].detach())
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
model_cifar10 = WaveMixCustom(num_classes=10).to(device)
for name, param in model_cifar10.named_parameters():
    if "pool.2" not in name:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_cifar10.parameters()), lr=0.005)

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
    model_baseline = WaveMixCustom(num_classes).to(device)
    for name, param in model_baseline.named_parameters():
        if "pool.2" not in name:
            param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_baseline.parameters()), lr=0.005)

    baseline_accuracy, _ = train(
        model_baseline, loader, optimizer, criterion, num_classes, use_nc_loss=False, experiment_name=f"Baseline ({dataset_name})"
    )
    results_baseline[dataset_name] = baseline_accuracy

# Part 2: NC Loss on Penultimate Layer
logger.info("\n=== Part 2: NC Loss on Penultimate Layer ===")
results_penultimate = {}
for dataset_name, (loader, num_classes) in datasets.items():
    logger.info(f"\nTraining on {dataset_name} (Penultimate NC)...")
    model_nc = WaveMixCustom(num_classes).to(device)
    for name, param in model_nc.named_parameters():
        if not any(n in name for n in ["layers.13", "layers.14", "layers.15", "pool.2"]):
            param.requires_grad = False
    optimizer_nc = optim.Adam(filter(lambda p: p.requires_grad, model_nc.parameters()), lr=0.001)

    nc_accuracy, nc_loss_value = train(
        model_nc, loader, optimizer_nc, criterion, num_classes, use_nc_loss=True, nc_layers=['pool'], nc_lambda=nc_lambda, experiment_name=f"Penultimate NC ({dataset_name})"
    )
    results_penultimate[dataset_name] = {
        'Accuracy': nc_accuracy,
        'NC Loss': nc_loss_value
    }

# Part 3: NC Loss on Last Three Layers
logger.info("\n=== Part 3: NC Loss on Last Three Layers ===")
nc_layers = ['blocks[13]', 'blocks[15]', 'pool']
results_multi_layer = {}
for dataset_name, (loader, num_classes) in datasets.items():
    logger.info(f"\nTraining on {dataset_name} (Multi-Layer NC)...")
    model_nc = WaveMixCustom(num_classes).to(device)
    for name, param in model_nc.named_parameters():
        if not any(n in name for n in ["layers.13", "layers.14", "layers.15", "pool.2"]):
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