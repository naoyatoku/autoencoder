"""
LSTM-Autoencoder の学習 (PyTorch)

アーキテクチャ:
  入力:  (batch, window=50, sensors=20)  ← 5秒分 × 20センサー
  Encoder: LSTM → 最後の隠れ状態 (latent=32)
  Decoder: latent を window 回繰り返し → LSTM → 元の窓を再構成
  損失: MSE（再構成誤差）

推論時:
  窓をスライドさせながら各時刻の異常スコアを計算
  スコアが閾値を超えたら異常と判定
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Meiryo'

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

WINDOW       = 200      # 20秒分 (200点 × 100ms) — グループ因子1〜2周期が収まる
STRIDE       = 20       # 学習時のスライド幅
LATENT       = 32       # 潜在空間の次元
HIDDEN       = 64       # LSTMの隠れ層サイズ
EPOCHS       = 100
BATCH        = 128
LR           = 1e-3
CORR_LAMBDA  = 0.2      # グループ内相関損失の重み
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_sensors: int, hidden: int, latent: int):
        super().__init__()
        self.n_sensors = n_sensors
        self.hidden    = hidden
        self.latent    = latent

        self.encoder_lstm = nn.LSTM(n_sensors, hidden, batch_first=True)
        self.enc_fc       = nn.Linear(hidden, latent)

        self.dec_fc       = nn.Linear(latent, hidden)
        self.decoder_lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.out_fc       = nn.Linear(hidden, n_sensors)

    def forward(self, x):
        # x: (batch, window, n_sensors)
        _, (h, _) = self.encoder_lstm(x)
        z = self.enc_fc(h[-1])                    # (batch, latent)

        dec_in = self.dec_fc(z).unsqueeze(1)      # (batch, 1, hidden)
        dec_in = dec_in.repeat(1, x.size(1), 1)   # (batch, window, hidden)
        out, _ = self.decoder_lstm(dec_in)
        return self.out_fc(out)                    # (batch, window, n_sensors)


def group_corr_loss(x: torch.Tensor, recon: torch.Tensor, groups: list) -> torch.Tensor:
    """グループ内センサー間相関をモデルが保持するよう促す補助損失。
    正常データでの学習中に各グループのセンサー間相関をエンコーダに刷り込む。"""
    loss = torch.zeros(1, device=x.device, dtype=x.dtype)
    n = 0
    for g in groups:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                xi = x[:, :, g[i]] - x[:, :, g[i]].mean(dim=1, keepdim=True)
                xj = x[:, :, g[j]] - x[:, :, g[j]].mean(dim=1, keepdim=True)
                ri = recon[:, :, g[i]] - recon[:, :, g[i]].mean(dim=1, keepdim=True)
                rj = recon[:, :, g[j]] - recon[:, :, g[j]].mean(dim=1, keepdim=True)
                cx = (xi * xj).sum(dim=1) / (xi.norm(dim=1) * xj.norm(dim=1) + 1e-8)
                cr = (ri * rj).sum(dim=1) / (ri.norm(dim=1) * rj.norm(dim=1) + 1e-8)
                loss = loss + ((cx - cr) ** 2).mean()
                n += 1
    return loss / max(n, 1)


def make_windows(X: np.ndarray, window: int, stride: int) -> np.ndarray:
    """(T, sensors) → (N_windows, window, sensors)"""
    idxs = range(0, len(X) - window + 1, stride)
    return np.stack([X[i:i + window] for i in idxs])


def compute_sensor_err_stats(model, X: np.ndarray, window: int, batch: int = 256):
    """正常データからセンサーごとの再構成誤差の平均・標準偏差を計算して返す"""
    model.eval()
    wins   = make_windows(X, window, stride=1)
    tensor = torch.tensor(wins, dtype=torch.float32)
    all_err = []
    for i in range(0, len(tensor), batch):
        chunk = tensor[i:i + batch].to(DEVICE)
        with torch.no_grad():
            recon = model(chunk)
        err = ((chunk - recon) ** 2).cpu().numpy()  # (B, window, sensors)
        all_err.append(err.reshape(-1, X.shape[1]))
    all_err = np.concatenate(all_err, axis=0)        # (N, sensors)
    return all_err.mean(axis=0), all_err.std(axis=0).clip(min=1e-6)


def window_scores(model, X: np.ndarray, window: int,
                  sensor_mean=None, sensor_std=None, groups=None, batch: int = 256) -> np.ndarray:
    """
    各時刻の異常スコアを計算。
    グループ構造が与えられた場合、グループ内z-スコア平均→グループ間最大値 を使う。
    2センサー分の異常信号が18センサーで希釈されなくなる。
    """
    T, S   = X.shape
    scores = np.zeros(T, dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)
    model.eval()

    windows = make_windows(X, window, stride=1)
    tensor  = torch.tensor(windows, dtype=torch.float32)

    s_mean = torch.tensor(sensor_mean, dtype=torch.float32).to(DEVICE) if sensor_mean is not None else None
    s_std  = torch.tensor(sensor_std,  dtype=torch.float32).to(DEVICE) if sensor_std  is not None else None

    for i in range(0, len(tensor), batch):
        chunk = tensor[i:i + batch].to(DEVICE)
        with torch.no_grad():
            recon = model(chunk)
        sq_err = (chunk - recon) ** 2                        # (B, window, sensors)
        if s_mean is not None:
            z = ((sq_err - s_mean) / s_std).clamp(min=0)    # 正常以下は0に切り捨て
            if groups is not None:
                # グループ内（全timestep × 全sensor）を平均 → グループ間最大 → 窓スコア (B,)
                g_means = torch.stack([z[:, :, g].mean(dim=[1, 2]) for g in groups], dim=-1)  # (B, G)
                win_sc  = g_means.max(dim=-1).values.cpu().numpy()  # (B,)
                for j, sc in enumerate(win_sc):
                    t0 = i + j
                    scores[t0:t0 + window] += sc
                    counts[t0:t0 + window] += 1
                continue  # 下の共通ループをスキップ
            else:
                err = z.mean(dim=2).cpu().numpy()            # (B, window)
        else:
            err = sq_err.mean(dim=2).cpu().numpy()
        for j, e in enumerate(err):
            t0 = i + j
            scores[t0:t0 + window] += e
            counts[t0:t0 + window] += 1

    return scores / np.maximum(counts, 1)


def main():
    print(f"デバイス: {DEVICE}")

    print("\n=== 1. データ読み込み ===")
    X_normal = np.load(DATA_DIR / "ts_X_normal.npy")
    X_test   = np.load(DATA_DIR / "ts_X_test.npy")
    y_test   = np.load(DATA_DIR / "ts_y_test.npy")
    N_SENSORS = X_normal.shape[1]
    print(f"  正常: {X_normal.shape}  テスト: {X_test.shape}  センサー数: {N_SENSORS}")

    groups_np   = np.load(DATA_DIR / "ts_groups.npy")   # (5, 4)
    groups_list = [list(g) for g in groups_np]
    print(f"  グループ構造: {groups_list}")

    print("\n=== 2. 窓データ作成 ===")
    wins = make_windows(X_normal, WINDOW, STRIDE)
    print(f"  窓数: {len(wins)}  shape: {wins.shape}")

    # 学習:検証 = 9:1
    n_val   = int(len(wins) * 0.1)
    n_train = len(wins) - n_val
    perm    = np.random.default_rng(42).permutation(len(wins))
    t_idx, v_idx = perm[:n_train], perm[n_train:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(wins[t_idx])), batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(
        TensorDataset(torch.tensor(wins[v_idx])), batch_size=BATCH)

    print("\n=== 3. モデル構築 ===")
    model     = LSTMAutoencoder(N_SENSORS, HIDDEN, LATENT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
    criterion = nn.MSELoss()
    params    = sum(p.numel() for p in model.parameters())
    print(f"  センサー={N_SENSORS}  window={WINDOW}  latent={LATENT}  params={params:,}")

    print("\n=== 4. 学習 ===")
    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch) + CORR_LAMBDA * group_corr_loss(batch, recon, groups_list)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(batch)
        tr_loss /= n_train

        model.eval()
        vl_loss = 0.0
        for (batch,) in val_loader:
            batch = batch.to(DEVICE)
            with torch.no_grad():
                recon    = model(batch)
                vl_loss += (criterion(recon, batch)
                            + CORR_LAMBDA * group_corr_loss(batch, recon, groups_list)).item() * len(batch)
        vl_loss /= n_val

        scheduler.step(vl_loss)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), MODEL_DIR / "lstm_ae_best.pth")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  train={tr_loss:.6f}  val={vl_loss:.6f}")

    model.load_state_dict(torch.load(MODEL_DIR / "lstm_ae_best.pth", weights_only=True))
    print(f"\n  最良 val_loss = {best_val:.6f}")

    print("\n=== 5. センサー誤差統計の計算 ===")
    print("  (正常データで各センサーの再構成誤差の平均・標準偏差を計算中...)")
    s_mean, s_std = compute_sensor_err_stats(model, X_normal, WINDOW)
    np.save(MODEL_DIR / "sensor_err_stats.npy", np.stack([s_mean, s_std]))
    print(f"  保存: models/sensor_err_stats.npy")

    print("\n=== 6. テストデータで異常スコア計算 ===")
    print("  (グループ最大z-スコアを計算中... しばらくかかります)")
    scores = window_scores(model, X_test, WINDOW, sensor_mean=s_mean, sensor_std=s_std,
                           groups=groups_list)

    # 正常区間スコアから閾値を決定（99パーセンタイルで誤報率を抑制）
    normal_scores = scores[y_test == 0]
    threshold = np.percentile(normal_scores, 99)
    y_pred    = (scores > threshold).astype(int)

    tp = int(((y_pred==1) & (y_test==1)).sum())
    fp = int(((y_pred==1) & (y_test==0)).sum())
    fn = int(((y_pred==0) & (y_test==1)).sum())
    tn = int(((y_pred==0) & (y_test==0)).sum())
    pr = tp / (tp + fp + 1e-9)
    rc = tp / (tp + fn + 1e-9)
    f1 = 2 * pr * rc / (pr + rc + 1e-9)
    print(f"  閾値={threshold:.4f}  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision={pr:.3f}  Recall={rc:.3f}  F1={f1:.3f}")

    np.save(MODEL_DIR / "ts_threshold.npy", np.array([threshold], dtype=np.float32))

    print("\n=== 7. ONNX エクスポート ===")
    model.cpu().eval()
    dummy = torch.zeros(1, WINDOW, N_SENSORS)
    torch.onnx.export(
        model, dummy,
        MODEL_DIR / "lstm_ae.onnx",
        input_names=["windows"],
        output_names=["reconstructed"],
        dynamic_axes={"windows": {0: "batch"}, "reconstructed": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"  保存: models/lstm_ae.onnx")

    print("\n=== 8. グラフ保存 ===")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(train_losses, label="train")
    axes[0].plot(val_losses,   label="val")
    axes[0].set_title("Learning Curve")
    axes[0].set_ylabel("MSE Loss"); axes[0].legend(); axes[0].grid(True)

    t_axis = np.arange(len(scores)) / (10 * 60)   # 分単位
    axes[1].plot(t_axis, scores, linewidth=0.5, label="anomaly score")
    axes[1].axhline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.3f}")
    axes[1].fill_between(t_axis, 0, scores.max(),
                         where=(y_test==1), alpha=0.2, color="red", label="true anomaly")
    axes[1].set_title("Anomaly Score (Test Data)")
    axes[1].set_xlabel("Time [min]"); axes[1].set_ylabel("Score"); axes[1].legend(); axes[1].grid(True)

    # 先頭3センサーの波形
    t_show = np.arange(3000) / (10 * 60)
    for s in range(3):
        axes[2].plot(t_show, X_test[:3000, s], linewidth=0.6, alpha=0.7, label=f"Sensor {s}")
    axes[2].fill_between(t_show, X_test[:3000].min(), X_test[:3000].max(),
                         where=(y_test[:3000]==1), alpha=0.25, color="red", label="anomaly")
    axes[2].set_title("Sensor Waveform (first 5 min, sensors 0-2)")
    axes[2].set_xlabel("Time [min]"); axes[2].legend(); axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(MODEL_DIR / "ts_result.png", dpi=150)
    print(f"  保存: models/ts_result.png")
    print("\n完了。次のステップ: python 06_inference_lstm.py")


if __name__ == "__main__":
    main()
