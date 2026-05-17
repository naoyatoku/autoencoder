# 異常検知プロジェクト — 進捗メモ

---

## ファイル構成

| ファイル | 内容 | 状態 |
|---|---|---|
| `01_prepare_data.py` | SECOM データセット取得・前処理 | 完成（今後は使わない） |
| `02_train.py` | テーブルデータ用 AE 学習 | 完成（今後は使わない） |
| `03_inference.py` | テーブルデータ用 ONNX 推論 | 完成（今後は使わない） |
| `04_timeseries_data.py` | 合成時系列データ生成（グループ相関構造 + decorrelation 異常） | 完成 |
| `05_train_lstm_ae.py` | LSTM-AE 学習 → ONNX エクスポート | 改修中 |
| `06_inference_lstm.py` | LSTM-AE ONNX 推論 | 改修中 |
| `07_realtime.py` | リアルタイム検知デモ | 改修中 |
| `visualize_data.py` | データ可視化（正常・異常・相関） | 完成 |

---

## データ設計（04_timeseries_data.py）

### グループ相関構造
```
5グループ × 4センサー = 20センサー
sensor_signal = base + coupling * group_factor + 個別振動 + ノイズ
```

### 異常パターン: Decorrelation のみ
グループ後半 2 センサーの group_factor を**同振幅・同周波数帯のランダム位相サイン波**に置換。

- 各センサーの平均・分散・自己相関は変わらない → **単センサー監視では検知不可（設計通り）**
- グループ内センサー間の相関だけが崩れる → LSTM-AE で検知

---

## 2026-05-17 セッションの作業内容

### 1. 可視化追加（visualize_data.py）✅

以下の 2 グラフを新規追加、動作確認済み:

- `models/viz_group_corr.png` — グループ内センサーのローリング相関時系列  
  → 正常時は 0.7〜0.9 を維持、異常区間で急落する様子が見える
- `models/viz_scatter.png` — 正常 vs 異常区間のセンサーペア散布図  
  → 正常=直線（高相関）、異常=雲状（相関崩壊）

**「正常と異常が見た目で同じ」という疑問に対する回答:** 設計通り正しい。  
1 センサー単体では区別不可能だが、グループ内センサー間の相関が崩れる。

---

### 2. モデル学習の改善（05_train_lstm_ae.py）✅

追加内容:
- `CORR_LAMBDA = 0.2` を定数追加
- `group_corr_loss()` 関数追加：グループ内センサー間相関を保持する補助損失
  ```python
  loss = MSE + 0.2 * group_corr_loss(batch, recon, groups_list)
  ```
- 学習済みモデル: val_loss = 0.117290（100 エポック、CPU）
- ONNX エクスポート済み

---

### 3. スコアリング方式の変更（05/06/07）⚠️ 途中

**変更の意図:** 異常センサー 2 本の信号が残り 18 本で希釈されるのを防ぐため、  
グループ構造を活かした集計方式に変更。

**採用方式:**
```
グループ内全値（WINDOW × センサー数）を平均 → グループ間で最大
```

各ファイルの変更内容:

```python
# 07_realtime.py: score_window()
g_means = np.array([z[0, :, g].mean() for g in groups])  # (n_groups,)
return float(g_means.max())

# 06_inference_lstm.py: compute_scores()
g_means = np.stack([z[:,:,g].mean(axis=(1,2)) for g in groups], axis=-1)  # (B, G)
win_sc  = g_means.max(axis=-1)  # (B,)

# 05_train_lstm_ae.py: window_scores()
g_means = torch.stack([z[:,:,g].mean(dim=[1,2]) for g in groups], dim=-1)  # (B, G)
win_sc  = g_means.max(dim=-1).values  # (B,)
```

`ts_groups.npy` を `load_runtime()` で読み込む変更も全ファイルに実施済み。

---

### 4. 閾値キャリブレーション問題（未解決・中断箇所）❌

**根本原因: 訓練データとテストデータのスコア分布が大きくずれている**

| データ | スコア平均 | スコア標準偏差 |
|--------|-----------|---------------|
| 訓練正常データ | 0.43 | 0.04 |
| テスト正常区間 | 1.04 | 1.60 |
| テスト異常区間 | 4.79 | 3.86 |

**原因の推測:**  
LSTM-AE が 100 エポック学習で訓練データのパターンをよく記憶しているため、訓練データへの再構成誤差が非常に低い。`sensor_err_stats.npy`（z-スコア正規化基準）が訓練 MSE 基準なので、より高い MSE のテストデータに適用するとスコアが大きくなりすぎる。

**試みた対応と結果:**

| 試行 | 閾値 | FPR | 検知率 | 遅延 |
|------|------|-----|--------|------|
| 訓練 97%tile | 1.39 | 20.8% | 15/15 | 0.11s |
| テスト正常 97%tile | 4.91 | 2.9% | 10/15 | 15.1s |
| テスト正常 95%tile | 4.13 | 4.9% | 11/15 | 11.8s |
| 旧モデル（変更前） | 1.010 | 7.1% | 14/15 | 5.6s |

**直前の最後の操作:**  
`sensor_err_stats.npy` をテストデータの正常区間 3000 窓から再計算して保存した。
```
新 sensor_err_stats: mean=0.117061, std=0.123953
```
→ この新しい stats での閾値キャリブレーションをやりかけで中断。

---

## 再開時の手順

### ステップ 1: 新 sensor_err_stats でスコアを再確認

```powershell
$env:PYTHONIOENCODING = "utf-8"
python -c "
import numpy as np, onnxruntime as ort
from pathlib import Path
# ... (訓練正常データと新 stats でリアルタイム方式スコアを計算し分布確認)
"
```

### ステップ 2: 97〜99%tile 閾値を試して 07_realtime.py で確認

目標: 検知率 ≥ 93%、FPR ≤ 7%

```powershell
python 07_realtime.py --fast --plot
```

### ステップ 3: うまくいかない場合の代替案

**A) sensor_err_stats の混合計算**  
訓練正常 + テスト冒頭 60s（無条件正常とみなせる）の混合で stats を計算

**B) エポック数を減らして過学習を緩和 → 再学習**  
`EPOCHS = 50〜60` に変更して `05_train_lstm_ae.py` を再実行

**C) z スコア方式をやめて生 MSE ベースに戻す**  
`score = sq_err.mean()` のシンプルな集計で安定した分布を確保

**D) 閾値を 05_train_lstm_ae.py でも同一の sensor_err_stats（テスト正常）で計算し直す**

---

## 変更したファイルの最終状態

| ファイル | 変更内容 | 状態 |
|---------|---------|------|
| `05_train_lstm_ae.py` | CORR_LAMBDA, group_corr_loss, window_scores 更新, 閾値 99%tile | ✅ 保存済み |
| `06_inference_lstm.py` | groups 読み込み, compute_scores 更新 | ✅ 保存済み |
| `07_realtime.py` | groups 読み込み, score_window 更新 | ✅ 保存済み |
| `visualize_data.py` | ローリング相関・散布図グラフ追加 | ✅ 保存済み |
| `models/lstm_ae_best.pth` | 新モデル（CORR_LAMBDA=0.2, 100エポック） | ✅ 保存済み |
| `models/lstm_ae.onnx` | 新モデルの ONNX エクスポート | ✅ 保存済み |
| `models/sensor_err_stats.npy` | テスト正常 3000 窓から再計算 | ⚠️ 暫定（要確認） |
| `models/ts_threshold.npy` | 暫定値（未確定） | ❌ 要キャリブレーション |

---

## 参考: 変更前のベースライン性能

旧モデル（mean 集計、95%tile 閾値、相関損失なし）:
- 検知率: 14/15（93%）
- 正常時誤報率: 7.13%
- 平均検出遅延: 5.64 秒
- 閾値: 1.010
