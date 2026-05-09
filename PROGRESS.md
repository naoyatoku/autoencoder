# 異常検知プロジェクト — 進捗メモ

## 概要

半導体製造装置センサーを想定した時系列データに対し、
LSTM-Autoencoder (ONNX Runtime) でリアルタイム異常検知を行うパイプライン。

**テーマ: 単センサー監視では検知不可能な異常を LSTM-AE で検知する**

---

## ファイル構成

| ファイル | 内容 | 状態 |
|---|---|---|
| `01_prepare_data.py` | SECOM データセット取得・前処理 | 完成（今後は使わない） |
| `02_train.py` | テーブルデータ用 AE 学習 | 完成（今後は使わない） |
| `03_inference.py` | テーブルデータ用 ONNX 推論 | 完成（今後は使わない） |
| `04_timeseries_data.py` | 合成時系列データ生成（グループ相関構造 + decorrelation 異常） | 完成・使用中 |
| `05_train_lstm_ae.py` | LSTM-AE 学習 → ONNX エクスポート（z スコア正規化スコア） | 完成・使用中 |
| `06_inference_lstm.py` | LSTM-AE ONNX 推論（単センサー監視との比較付き） | 完成 |
| `07_realtime.py` | **リアルタイム検知デモ（本命）** | 完成（z スコア対応済み） |
| `compare_anomaly_types.py` | 旧 4 異常タイプの検出能力比較（参考用） | 完成 |
| `visualize_anomalies.py` | 旧 4 異常タイプの波形サンプル描画 | 完成 |

> SECOM 関連（01〜03）は今後は触らない方針。04〜07 が主軸。

---

## データ生成の設計（04_timeseries_data.py）

### グループ相関構造

```
5グループ × 4センサー = 20センサー
sensor_signal = base + coupling * group_factor + 個別振動 + ノイズ
```

同グループのセンサーは共通の潜在因子 (group_factor) で連動している。
正常時はグループ内の相関係数が高い (avg ~0.7〜0.9)。

### 異常パターン: Decorrelation

グループ後半 2 センサーの group_factor を **同振幅・同周波数帯のランダム位相サイン波**に置換。

- 各センサーの平均・分散・自己相関は変わらない → **単センサー監視では検知不可**
- グループ内センサー間の相関だけが崩れる → **LSTM-AE で初めて検知可能**

白色ノイズでなくサイン波を使う理由：白色ノイズは高周波成分で単センサー監視でも
ローリング平均偏差が上がってしまうため、時系列特性を保つサイン波を採用。

---

## 異常スコアの設計（05_train / 06_inference / 07_realtime）

### z スコア正規化

再構成誤差を「正常データでの各センサーの再構成誤差統計」で正規化：

```
z = clip((sq_err - sensor_mean) / sensor_std, 0, None)
score = z.mean()
```

- `sensor_err_stats.npy` に正常データの (mean, std) を保存
- 正常以下（z < 0）はゼロに切り捨て → 正常区間スコアが安定する
- decorrelation 時は複数センサーの z が同時上昇 → 合算で検知感度が上がる
- 閾値は正常スコアの 80%ile（05_train が計算して ts_threshold.npy に保存）

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
python 04_timeseries_data.py        # データ生成（data/ 以下に保存）
python 05_train_lstm_ae.py          # LSTM-AE 学習（models/ 以下に保存）
python 07_realtime.py --fast --plot # リアルタイム検知デモ
python 06_inference_lstm.py         # バッチ推論 + 単センサー監視との比較
```

> `data/` と `models/` は .gitignore に含まれるため、別 PC では上記を再実行する必要がある。

---

## 直近の実験結果（2026-05-08、旧アプローチ）

> ⚠️ 以下は旧アプローチ（spike / level_shift / volatility / stuck）での結果。
> 現在は decorrelation アプローチに移行済みのため、再実行で結果が変わる。

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

| 指標 | 旧結果 |
|---|---|
| Best val_loss | ~0.166 |
| F1 (閾値=95%ile) | 0.400 |
| Precision | 0.674 |
| Recall | 0.284 |

> **旧アプローチの課題**: 正常スコアと閾値の余裕が薄く Recall が低かった。
> decorrelation + z スコア正規化 + 閾値 80%ile で改善を期待。

---

## 次にやること

1. **パイプラインを再実行して新アプローチの F1 を計測**
   ```
   python 04_timeseries_data.py && python 05_train_lstm_ae.py
   python 06_inference_lstm.py   # 単センサー比較も確認
   ```

2. **07_realtime.py で検知率を確認**
   ```
   python 07_realtime.py --fast --plot
   ```

3. （オプション）**モデル強化**
   - Epochs を 100〜150 に増やす
   - Latent 次元を 64 に拡大

---

## 備考・ハマりポイント

- **Windows で PyTorch が動かない**: Visual C++ 再頒布可能パッケージが必要（→ インストール済み）
- **matplotlib の日本語**: `font.family = 'Meiryo'` で対応（Linux では要変更）
- **ONNX export 警告**: LSTM の batch_size に関する警告は動作に影響なし
- **閾値の空間**: ts_threshold.npy は z スコア空間の値。07_realtime.py も z スコアで統一済み
