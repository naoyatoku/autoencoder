# 異常検知プロジェクト — 進捗メモ

## 概要

半導体製造装置センサーを想定した時系列データに対し、
LSTM-Autoencoder (ONNX Runtime) でリアルタイム異常検知を行うパイプライン。

---

## ファイル構成

| ファイル | 内容 | 状態 |
|---|---|---|
| `01_prepare_data.py` | SECOM データセット取得・前処理 | 完成（今後は使わない） |
| `02_train.py` | テーブルデータ用 AE 学習 | 完成（今後は使わない） |
| `03_inference.py` | テーブルデータ用 ONNX 推論 | 完成（今後は使わない） |
| `04_timeseries_data.py` | 合成時系列データ生成（20センサー×10Hz） | 完成・使用中 |
| `05_train_lstm_ae.py` | LSTM-AE 学習 → ONNX エクスポート | 完成・使用中 |
| `06_inference_lstm.py` | LSTM-AE ONNX 推論（バッチ） | 完成 |
| `07_realtime.py` | **リアルタイム検知デモ（本命）** | 完成・結果あり |

> SECOM 関連（01〜03）は今後は触らない方針。04〜07 が主軸。

---

## 実行手順（別 PC でのセットアップ）

### 1. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

> **Windows の場合**: PyTorch が動かない場合は Visual C++ 再頒布可能パッケージが必要。  
> https://aka.ms/vs/17/release/vc_redist.x64.exe をインストールすること。

### 2. パイプライン実行順序

```bash
python 04_timeseries_data.py   # データ生成（data/ 以下に保存）
python 05_train_lstm_ae.py     # LSTM-AE 学習（models/ 以下に保存）
python 07_realtime.py --fast --plot  # リアルタイム検知デモ
```

> `data/` と `models/` は .gitignore に含まれるため、別 PC では上記を再実行する必要がある。

---

## 直近の実験結果（2026-05-08）

### モデル設定

| パラメータ | 値 |
|---|---|
| 窓幅 (WINDOW) | 50 点 (= 5 秒) |
| stride (学習) | 10 |
| Latent 次元 | 32 |
| LSTM Hidden | 64 |
| Epochs | 60 |
| Optimizer | Adam (lr=1e-3) |
| LR Scheduler | ReduceLROnPlateau (patience=8) |

### 学習結果

| 指標 | 値 |
|---|---|
| Best val_loss | ~0.166 |
| Precision (閾値=95%ile) | 0.674 |
| Recall | 0.284 |
| F1 | 0.400 |

### リアルタイム検知結果（07_realtime.py --fast）

| 指標 | 値 |
|---|---|
| 真の異常区間 | 22 件 |
| 検出できた | **14 件 (64%)** |
| 見逃し | 8 件 |
| 正常時誤報率 | **1.70%** (224/13,146 点) |
| 平均検出遅延 | **8.86 秒** |
| 最短遅延 | 0.80 秒 |
| 最長遅延 | 23.40 秒 |

> グラフ: `models/rt_result.png` (実行後に生成される)

---

## 現状の課題

1. **Recall が低い（36% 見逃し）**  
   - リアルタイムモードのスコアは「直前 50 点窓だけの MSE」を使用。  
     学習時評価（複数窓の平均）より高くなるため、閾値との余裕が薄い。  
   - 正常スコアが ~47〜55、閾値が 56.3 と非常に近い。

2. **見逃しやすい異常パターン**  
   - 短いスパイク（6〜9 秒）
   - 影響センサー数が少ない level_shift / volatility

---

## 次にやること（候補）

### A. 閾値を下げて Recall を上げる
```python
# 05_train_lstm_ae.py の閾値計算行を変更
threshold = np.percentile(normal_scores, 90)  # 95 → 90
```
→ 誤報率が増えるトレードオフあり。

### B. スコアに移動平均を適用（07_realtime.py）
短い揺らぎをなめらかにして安定した検出に。
```python
# 例: 直近 N 点のスコア平均で判定
SMOOTH = 5
score_ma = np.convolve(scores, np.ones(SMOOTH)/SMOOTH, mode='same')
```

### C. モデル強化
- Epochs を 100〜150 に増やす
- Latent 次元を 64 に拡大
- Attention 機構の追加

### D. 評価の改善
- 時系列的な「区間単位」ではなく「点単位」の AUC-ROC を計算
- `sklearn.metrics.roc_auc_score` で定量評価

---

## 備考・ハマりポイント

- **Windows で PyTorch が動かない**: Visual C++ 再頒布可能パッケージが必要（→ インストール済み）
- **matplotlib の日本語**: `font.family = 'Meiryo'` で対応（Linux では要変更）
- **ONNX export 警告**: LSTM の batch_size に関する警告は動作に影響なし
- **リアルタイムスコアの特性**: stride=1 の単一窓スコアは学習評価の多窓平均より高くなる傾向
