import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]  # один логит -> BCEWithLogitsLoss
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_nn(X_train, y_train, X_val, y_val, cfg,
             max_epochs=100, patience=10, seed=0, device="cpu"):
    """Внутренняя валидация для early stopping — из валидационного окна,
    переданного отдельно (временная задача: конец обучающего окна)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(X_train.shape[1], cfg["hidden"], cfg.get("dropout", 0.0)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg.get("wd", 0.0))
    loss_fn = nn.BCEWithLogitsLoss()

    tr_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.float32))
    tr_loader = DataLoader(tr_ds, batch_size=cfg["batch"], shuffle=True)

    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)
    # для train-loss на фиксированной части (без dropout) — честная кривая
    Xt_eval = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt_eval = torch.tensor(y_train, dtype=torch.float32, device=device)

    best = {"AP": -np.inf, "state": None, "epoch": -1}
    history = []
    bad = 0
    n_params = count_params(model)
    for ep in range(max_epochs):
        model.train()
        # средний train-loss по батчам за эпоху (для кривой)
        ep_loss_sum, ep_loss_n = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            ep_loss_sum += float(loss.item()) * len(xb)
            ep_loss_n += len(xb)
        train_loss_epoch = ep_loss_sum / max(ep_loss_n, 1)

        model.eval()
        with torch.no_grad():
            # eval-loss на train (без dropout) — сопоставим с val-loss
            train_eval_loss = float(loss_fn(model(Xt_eval), yt_eval).item())
            val_loss = float(loss_fn(model(Xv), yv).item())
            logits_val = model(Xv)
            scores_val = torch.sigmoid(logits_val).cpu().numpy()
        ap = average_precision_score(y_val, scores_val)
        history.append({
            "epoch": ep,
            "train_loss_batch": train_loss_epoch,
            "train_loss_eval": train_eval_loss,
            "val_loss": val_loss,
            "val_AP": ap,
            "train_params": n_params,
        })
        if ap > best["AP"] + 1e-6:
            best = {"AP": ap, "state": {k: v.detach().cpu().clone()
                                        for k, v in model.state_dict().items()},
                    "epoch": ep}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best["state"])
    return model, history, best

def predict_scores_nn(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32, device=device))
        return torch.sigmoid(out).cpu().numpy()

def nn_prepare(pre, X):
    """Препроцессор (ColumnTransformer) обучается на train — вызывается снаружи."""
    return pre.transform(X)