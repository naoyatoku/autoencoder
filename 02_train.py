"""
オートエンコーダーの学習 (PyTorch)

戦略：正常データだけで学習 → 異常データは再構成誤差が大きくなる
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # GUIなし環境でも動作

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

EPOCHS     = 100
BATCH_SIZE = 64
LR         = 1e-3
VAL_RATIO  = 0.1
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def reconstruction_error(model, x):
    """サンプルごとの平均二乗誤差 (MSE)"""
    with torch.no_grad():
        x_hat = model(x)
    return ((x - x_hat) ** 2).mean(dim=1)


def main():
    print(f"デバイス: {DEVICE}")

    print("\n=== 1. データ読み込み ===")
    X = np.load(DATA_DIR / "X_scaled.npy")
    y = np.load(DATA_DIR / "y.npy")

    # 正常データだけで学習
    X_normal = X[y == -1]
    X_anom   = X[y ==  1]
    print(f"  正常: {len(X_normal)}件  異常: {len(X_anom)}件  センサー: {X.shape[1]}")

    X_tensor = torch.tensor(X_normal, dtype=torch.float32)
    dataset  = TensorDataset(X_tensor)
    n_val    = int(len(dataset) * VAL_RATIO)
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    print("\n=== 2. モデル構築 ===")
    input_dim = X.shape[1]
    model     = Autoencoder(input_dim, latent_dim=16).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    print(f"  入力次元: {input_dim}  潜在次元: 16")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  パラメータ数: {total_params:,}")

    print("\n=== 3. 学習 ===")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        for (batch,) in val_loader:
            batch = batch.to(DEVICE)
            with torch.no_grad():
                val_loss += criterion(model(batch), batch).item() * len(batch)
        val_loss /= n_val

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / "autoencoder_best.pth")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.6f}  val={val_loss:.6f}")

    model.load_state_dict(torch.load(MODEL_DIR / "autoencoder_best.pth", weights_only=True))
    print(f"\n  最良 val_loss = {best_val_loss:.6f}")

    print("\n=== 4. 異常スコア計算・閾値決定 ===")
    model.eval()
    X_all = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    errors = reconstruction_error(model, X_all).cpu().numpy()
    errors_normal = errors[y == -1]
    errors_anom   = errors[y ==  1]

    print(f"  正常の再構成誤差: mean={errors_normal.mean():.4f}  std={errors_normal.std():.4f}")
    print(f"  異常の再構成誤差: mean={errors_anom.mean():.4f}  std={errors_anom.std():.4f}")

    # 閾値の候補を複数表示してユーザーが選べるようにする
    print("\n  【閾値候補と精度のトレードオフ】")
    print(f"  {'閾値設定':>20}  {'閾値':>7}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}")
    print("  " + "-" * 62)
    candidates = {}
    for pct in [90, 95, 97, 99]:
        t = np.percentile(errors_normal, pct)
        yp = (errors > t).astype(int)
        yt = (y == 1).astype(int)
        tp_ = ((yp==1)&(yt==1)).sum(); fp_ = ((yp==1)&(yt==0)).sum()
        fn_ = ((yp==0)&(yt==1)).sum()
        pr = tp_/(tp_+fp_+1e-9); rc = tp_/(tp_+fn_+1e-9)
        f1_ = 2*pr*rc/(pr+rc+1e-9)
        print(f"  正常データの{pct:2d}パーセンタイル  {t:>7.4f}  {pr:>10.3f}  {rc:>8.3f}  {f1_:>6.3f}")
        candidates[pct] = t

    # デフォルト閾値: 95パーセンタイル（感度と特異度のバランス）
    threshold = candidates[95]
    print(f"\n  → デフォルト採用: 95パーセンタイル = {threshold:.4f}")

    y_pred = (errors > threshold).astype(int)
    y_true = (y == 1).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    print(f"\n  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    # 閾値を保存
    np.save(MODEL_DIR / "threshold.npy", np.array([threshold], dtype=np.float32))

    print("\n=== 5. ONNX エクスポート ===")
    model.cpu().eval()
    dummy = torch.zeros(1, input_dim)
    torch.onnx.export(
        model, dummy,
        MODEL_DIR / "autoencoder.onnx",
        input_names=["sensor_input"],
        output_names=["reconstructed"],
        dynamic_axes={"sensor_input": {0: "batch_size"}, "reconstructed": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"  保存: models/autoencoder.onnx")

    print("\n=== 6. グラフ保存 ===")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="train")
    axes[0].plot(val_losses,   label="val")
    axes[0].set_title("Learning Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True)

    bins = np.linspace(0, np.percentile(errors, 99), 60)
    axes[1].hist(errors_normal, bins=bins, alpha=0.6, label="正常", color="steelblue")
    axes[1].hist(errors_anom,   bins=bins, alpha=0.6, label="異常", color="tomato")
    axes[1].axvline(threshold, color="black", linestyle="--", label=f"閾値={threshold:.3f}")
    axes[1].set_title("Reconstruction Error Distribution")
    axes[1].set_xlabel("MSE")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(MODEL_DIR / "result.png", dpi=150)
    print(f"  保存: models/result.png")
    print("\n完了。次のステップ: python 03_inference.py")


if __name__ == "__main__":
    main()
