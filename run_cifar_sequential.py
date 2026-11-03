"""SETA-FL development scaffold; see docs/MATH.md for conditional assumptions.

FedAvg remains the aggregation baseline. Fitted historical centers are not
conditional validity certificates. Current updates never auto-enter clean history.
"""
import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from seta_statistics import BoundedMeanEProcess, ClientHistory


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*8*8, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dirichlet_partition(targets, n_clients, alpha, seed):
    if n_clients < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('clients and Dirichlet alpha must be positive')
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    if targets.ndim != 1:
        raise ValueError('targets must be one-dimensional')
    out = [[] for _ in range(n_clients)]
    for label in np.unique(targets):
        ids = np.where(targets == label)[0]
        rng.shuffle(ids)
        proportions = rng.dirichlet(np.full(n_clients, alpha))
        cuts = (np.cumsum(proportions)*len(ids)).astype(int)[:-1]
        for client, chunk in enumerate(np.split(ids, cuts)):
            out[client].extend(chunk.tolist())
    for ids in out:
        rng.shuffle(ids)
    # Empty partitions are valid; the round scheduler skips them.
    return out


class LabelFlippedDataset(Dataset):
    """Seeded label permutation with no fixed points; no base dataset mutation."""
    def __init__(self, dataset, seed, num_classes=10):
        if num_classes < 2:
            raise ValueError('label flipping needs at least two classes')
        self.dataset = dataset
        cycle = np.random.default_rng(seed).permutation(num_classes)
        self.mapping = np.empty(num_classes, dtype=np.int64)
        self.mapping[cycle] = np.roll(cycle, -1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        label = int(label)
        if not 0 <= label < len(self.mapping):
            raise ValueError('label outside class range')
        return image, int(self.mapping[label])


def flat_update(new, old):
    return torch.cat([(new[key].detach().cpu()-old[key].detach().cpu()).flatten()
                      for key in old if old[key].is_floating_point()])


def update_features(update, dimension=16, scale=.01):
    """Fixed signed block-mean sketch, norm <= 1; not adversarially sufficient.

    Scale/dimension must be fixed before the run. Opposite updates give opposite
    features, unlike the old norm diagnostic. Orthogonal attacks can evade it.
    """
    if dimension < 1 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('invalid feature dimension or scale')
    flat = torch.cat([value.detach().cpu().float().flatten()
                      for value in update.values() if value.is_floating_point()])
    if not torch.isfinite(flat).all():
        raise ValueError('nonfinite client update')
    chunks = torch.tensor_split(flat, min(dimension, flat.numel()))
    return np.asarray([math.tanh(float(chunk.mean())/scale)
                       for chunk in chunks])/math.sqrt(len(chunks))


def load_state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def apply_aggregate(global_model, updates, weights):
    """Normalize over actual participants; empty rounds are no-ops.

    Validate before changing server state. Integer buffers keep server values;
    this is an explicit policy, not a general BatchNorm aggregation rule.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) != len(updates):
        raise ValueError('one weight is required per update')
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError('weights must be finite and nonnegative')
    if not updates:
        return weights
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError('participating weights need a positive finite sum')
    weights = weights/total
    base = load_state(global_model)
    for update in updates:
        if set(update) != set(base):
            raise ValueError('update keys differ from server')
        for key, value in update.items():
            if value.shape != base[key].shape or not torch.isfinite(value).all():
                raise ValueError('invalid shape or nonfinite update')
    new = {key: value.clone() for key, value in base.items()}
    for key, value in base.items():
        if value.is_floating_point():
            delta = torch.zeros_like(value)
            for weight, update in zip(weights, updates):
                delta.add_(update[key].to(device=value.device, dtype=value.dtype),
                           alpha=float(weight))
            new[key] = value+delta
    global_model.load_state_dict(new)
    return weights


def local_train(global_model, loader, device, epochs=1, lr=.03, prox=0.):
    if epochs < 1 or not math.isfinite(lr) or lr <= 0 or not math.isfinite(prox) or prox < 0:
        raise ValueError('invalid training hyperparameters')
    model = copy.deepcopy(global_model).to(device)
    reference = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.)
    model.train()
    for _ in range(epochs):
        for image, label in loader:
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(image), label)
            if prox:
                loss = loss+prox/2*sum((value-reference[name]).pow(2).sum()
                                      for name, value in model.named_parameters())
            loss.backward()
            optimizer.step()
    state = model.state_dict()
    return {key: state[key].detach().cpu()-value.detach().cpu()
            if value.is_floating_point() else torch.zeros_like(value, device='cpu')
            for key, value in global_model.state_dict().items()}


def evaluate(model, loader, device):
    model.eval()
    loss, correct, total = 0., 0, 0
    with torch.no_grad():
        for image, label in loader:
            image, label = image.to(device), label.to(device)
            logits = model(image)
            loss += nn.functional.cross_entropy(logits, label, reduction='sum').item()
            correct += (logits.argmax(1) == label).sum().item()
            total += len(label)
    if not total:
        raise ValueError('evaluation dataset is empty')
    return loss/total, correct/total


def select_participants(parts, fraction, rng):
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('participation must lie in (0,1]')
    eligible = [index for index, ids in enumerate(parts) if len(ids)]
    if not eligible:
        return []
    count = max(1, math.ceil(len(eligible)*fraction))
    return sorted(int(index) for index in rng.choice(eligible, count, replace=False))


def attack_active(round_id, start, mode='persistent', period=4, duration=1):
    if start < 1 or period < 1 or not 1 <= duration <= period:
        raise ValueError('invalid attack schedule')
    if mode not in ('persistent', 'intermittent'):
        raise ValueError('unknown attack mode')
    return round_id >= start and (mode == 'persistent' or (round_id-start) % period < duration)


def train_round(model, dataset, parts, device, *, seed, round_id, participation=1.,
                attack='none', attack_start=16, attack_mode='persistent',
                attack_period=4, attack_duration=1, malicious_fraction=.2):
    if attack not in ('none', 'signflip', 'labelflip') or not 0 <= malicious_fraction <= 1:
        raise ValueError('invalid attack configuration')
    selected = select_participants(parts, participation, np.random.default_rng([seed, round_id]))
    updates, sizes = [], []
    malicious_count = math.ceil(len(parts)*malicious_fraction)
    active = attack_active(round_id, attack_start, attack_mode, attack_period, attack_duration)
    for client in selected:
        subset = Subset(dataset, parts[client])
        malicious = client < malicious_count and active
        if attack == 'labelflip' and malicious:
            subset = LabelFlippedDataset(subset, seed+client)
        generator = torch.Generator().manual_seed(seed+round_id*len(parts)+client)
        loader = DataLoader(subset, batch_size=64, shuffle=True, generator=generator)
        update = local_train(model, loader, device)
        if attack == 'signflip' and malicious:
            update = {key: -value for key, value in update.items()}
        updates.append(update)
        sizes.append(len(parts[client]))
    weights = apply_aggregate(model, updates, sizes)
    return selected, updates, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./data')
    parser.add_argument('--out', default='./results')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=.5, help='Dirichlet concentration')
    parser.add_argument('--clients', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--participation', type=float, default=1.)
    parser.add_argument('--attack', choices=['none', 'signflip', 'labelflip'], default='signflip')
    parser.add_argument('--attack-start', type=int, default=16)
    parser.add_argument('--attack-mode', choices=['persistent', 'intermittent'], default='persistent')
    parser.add_argument('--attack-period', type=int, default=4)
    parser.add_argument('--attack-duration', type=int, default=1)
    parser.add_argument('--malicious-fraction', type=float, default=.2)
    parser.add_argument('--drift-round', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--trusted-history', help='externally validated JSON feature history')
    parser.add_argument('--mean-error-bound', type=float, default=.2,
                        help='ASSUMED conditional mean error bound, not a certificate')
    parser.add_argument('--family-alpha', type=float, default=.05)
    parser.add_argument('--feature-scale', type=float, default=.01)
    args = parser.parse_args()
    if args.clients < 1 or args.rounds < 1 or not math.isfinite(args.alpha) or args.alpha <= 0:
        parser.error('clients, rounds and Dirichlet alpha must be positive')
    if not 0 < args.participation <= 1 or not 0 < args.family_alpha < 1:
        parser.error('invalid participation or family alpha')
    if not 0 <= args.mean_error_bound <= 2 or not math.isfinite(args.feature_scale) or args.feature_scale <= 0:
        parser.error('invalid error bound or feature scale')
    if not 0 <= args.malicious_fraction <= 1 or args.drift_round < 1 or args.seed < 0:
        parser.error('invalid malicious fraction, drift round or seed')
    attack_active(1, args.attack_start, args.attack_mode, args.attack_period, args.attack_duration)
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    histories = [ClientHistory() for _ in range(args.clients)]
    if args.trusted_history:
        records = json.loads(Path(args.trusted_history).read_text())
        for client, entries in records.items():
            client = int(client)
            if not 0 <= client < args.clients:
                parser.error('trusted-history client out of range')
            for entry in entries:
                if len(entry['features']) != 16:
                    parser.error('trusted history needs 16-dimensional features')
                histories[client].add_trusted(entry['features'], provenance=entry['provenance'])
    monitors = [BoundedMeanEProcess(alpha=args.family_alpha/args.clients) for _ in histories]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (.4914, .4822, .4465), (.247, .243, .261))])
    train = datasets.CIFAR10(args.root, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(args.root, train=False, download=True, transform=transform)
    parts = dirichlet_partition(train.targets, args.clients, args.alpha, args.seed)
    drift_parts = dirichlet_partition(train.targets, args.clients, args.alpha, args.seed+10000)
    test_loader = DataLoader(test, batch_size=256, shuffle=False)
    model = SmallCNN().to(device)
    rows, client_rows = [], []
    print('Conditional validity assumed, not certified. FedAvg baseline; no automatic trusted-history admission.')
    for round_id in range(1, args.rounds+1):
        # Freeze ALL predictions before any current-round training or updates.
        predictions = [history.predict(args.mean_error_bound) for history in histories]
        current_parts = drift_parts if round_id >= args.drift_round else parts
        selected, updates, weights = train_round(
            model, train, current_parts, device, seed=args.seed, round_id=round_id,
            participation=args.participation, attack=args.attack, attack_start=args.attack_start,
            attack_mode=args.attack_mode, attack_period=args.attack_period,
            attack_duration=args.attack_duration, malicious_fraction=args.malicious_fraction)
        for client, update, weight in zip(selected, updates, weights):
            features = update_features(update, scale=args.feature_scale)
            prediction = predictions[client]
            score, log_e, alarm = None, None, False
            if prediction is not None:
                score = prediction.score(features)
                log_e, alarm = monitors[client].update(score, prediction.mean, prediction.mean_error_bound)
            histories[client].observe(features, round_id=round_id, alarm=alarm)
            client_rows.append(dict(round=round_id, client=client, score=score, log_e=log_e,
                                    alarm=alarm, detector_enabled=prediction is not None,
                                    predicted_mean=prediction.mean if prediction else None,
                                    mean_error_bound=args.mean_error_bound,
                                    trusted_count=len(histories[client].trusted), weight=float(weight)))
        loss, accuracy = evaluate(model, test_loader, device)
        rows.append(dict(round=round_id, test_loss=loss, test_accuracy=accuracy,
                         participants=len(selected), alarm_clients=sum(m.alarmed for m in monitors)))
        pd.DataFrame(rows).to_csv(output/f'rounds_seed{args.seed}.csv', index=False)
        pd.DataFrame(client_rows).to_csv(output/f'clients_seed{args.seed}.csv', index=False)
        print(f'round={round_id:03d} loss={loss:.4f} acc={accuracy:.4f} participants={len(selected)}')
    (output/f'config_seed{args.seed}.json').write_text(json.dumps(vars(args), indent=2))


if __name__ == '__main__':
    main()
