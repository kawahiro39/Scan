# Document Scanner API

書類画像を自動的にトリミングしてPDF出力するCloud Run対応のAPIです。OpenCVを使用して書類の輪郭を自動検出し、パースペクティブ変換を適用して正面から見た画像に補正します。

## 機能

- 📸 **高精度境界検出**: 複数スケールエッジ検出とモルフォロジー演算で書類の輪郭を高精度に自動検出
- 🔄 **パースペクティブ変換**: 斜めから撮影した画像を正面から見た状態に補正
- 🔃 **自動回転補正**: 斜めにスキャンされた書類を自動的に正しい向きに補正
- 🌑 **影除去**: モルフォロジー演算を用いて書類の影を自動除去
- 📰 **裏透け防止**: 適応的閾値処理で裏面の文字透けを軽減
- ✨ **画像強化**: ガンマ補正、CLAHE、コントラスト調整、シャープニングで画像品質を向上
- 📄 **PDF生成**: スキャンした画像をPDFファイルとして出力
- 🌐 **Cloud Run対応**: スケーラブルなサーバーレス環境で動作

## アーキテクチャ

```
├── main.py                 # FastAPI アプリケーション
├── document_scanner.py     # 書類スキャン処理
├── pdf_generator.py        # PDF生成処理
├── requirements.txt        # Python依存関係
├── Dockerfile             # コンテナイメージ定義
├── cloudbuild.yaml        # Cloud Build設定
└── .gcloudignore          # デプロイ時の除外ファイル
```

## API エンドポイント

### `POST /scan`

書類画像をアップロードしてスキャンし、PDFとして返します。

**パラメータ:**
- `file` (required): 画像ファイル (JPEG, PNG等)
- `enhance` (optional, default: true): 画像強化の有効化
- `page_size` (optional, default: "A4"): PDFのページサイズ ("A4" or "letter")

**レスポンス:** PDF ファイル

**cURLサンプル:**
```bash
curl -X POST "https://your-service-url.run.app/scan" \
  -F "file=@document.jpg" \
  -F "enhance=true" \
  -F "page_size=A4" \
  -o scanned.pdf
```

### `POST /scan/preview`

書類画像をスキャンして、処理後の画像をJPEGで返します（PDF化なし）。

**パラメータ:**
- `file` (required): 画像ファイル
- `enhance` (optional, default: true): 画像強化の有効化

**レスポンス:** JPEG 画像

**cURLサンプル:**
```bash
curl -X POST "https://your-service-url.run.app/scan/preview" \
  -F "file=@document.jpg" \
  -F "enhance=true" \
  -o preview.jpg
```

### `GET /health`

ヘルスチェックエンドポイント

**レスポンス:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00"
}
```

## ローカル開発

### 必要要件

- Python 3.11+
- pip

### セットアップ

1. リポジトリのクローン:
```bash
git clone <repository-url>
cd webapp
```

2. 仮想環境の作成と有効化:
```bash
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
```

3. 依存関係のインストール:
```bash
pip install -r requirements.txt
```

4. アプリケーションの起動:
```bash
python main.py
```

または

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

5. ブラウザで `http://localhost:8080` を開く

### Dockerでの実行

```bash
# イメージのビルド
docker build -t document-scanner .

# コンテナの実行
docker run -p 8080:8080 document-scanner

# ブラウザで http://localhost:8080 を開く
```

## Cloud Runへのデプロイ

### 前提条件

- Google Cloud プロジェクト
- gcloud CLI のインストールと認証
- Cloud Run API と Cloud Build API の有効化

### デプロイ手順

#### 方法1: gcloud コマンドを使用

```bash
# Google Cloudプロジェクトの設定
gcloud config set project YOUR_PROJECT_ID

# Cloud Run APIの有効化
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# デプロイ
gcloud run deploy document-scanner \
  --source . \
  --region asia-northeast1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --max-instances 10 \
  --timeout 300
```

#### 方法2: Cloud Build を使用

```bash
# Cloud Buildの実行
gcloud builds submit --config cloudbuild.yaml
```

### デプロイ後

デプロイが完了すると、サービスURLが表示されます:
```
Service [document-scanner] revision [document-scanner-xxxxx] has been deployed
and is serving 100 percent of traffic.
Service URL: https://document-scanner-xxxxx-an.a.run.app
```

## 使用方法

### Python での利用例

```python
import requests

# スキャンしてPDFを取得
url = "https://your-service-url.run.app/scan"
files = {"file": open("document.jpg", "rb")}
data = {"enhance": "true", "page_size": "A4"}

response = requests.post(url, files=files, data=data)

# PDFを保存
with open("scanned.pdf", "wb") as f:
    f.write(response.content)
```

### JavaScriptでの利用例

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('enhance', 'true');
formData.append('page_size', 'A4');

fetch('https://your-service-url.run.app/scan', {
  method: 'POST',
  body: formData
})
  .then(response => response.blob())
  .then(blob => {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'scanned.pdf';
    a.click();
  });
```

## 技術スタック

- **フレームワーク**: FastAPI
- **画像処理**: OpenCV, NumPy, Pillow
- **PDF生成**: ReportLab
- **サーバー**: Uvicorn
- **コンテナ**: Docker
- **デプロイ**: Google Cloud Run

## 画像処理の流れ

### 1. 前処理と輪郭検出
- **グレースケール変換**: カラー画像を白黒に変換
- **バイラテラルフィルタ**: エッジを保持しながらノイズ除去
- **モルフォロジー演算**: 輪郭の途切れを接続
- **適応的閾値処理**: 照明ムラに強い二値化
- **マルチスケールエッジ検出**: 複数の閾値でCannyエッジ検出を実行し統合

### 2. 書類輪郭の検出と検証
- **輪郭抽出**: 面積順に上位15個の輪郭を抽出
- **4点近似**: 複数のイプシロン値で輪郭を4点に近似
- **検証**: アスペクト比、充実度（Solidity）、面積比で書類らしさを判定

### 3. パースペクティブ変換
- **頂点の並び替え**: 左上、右上、右下、左下の順に整列
- **出力サイズ計算**: 各辺の長さから最適な出力サイズを算出
- **射影変換**: cv2.getPerspectiveTransformで正面視に補正

### 4. 回転補正
- **テキスト方向検出**: 二値化画像から最小外接矩形を計算
- **角度算出**: テキストラインの傾きを検出
- **自動回転**: 0.5度以上の傾きを自動補正

### 5. 画像強化 (enhance=true の場合)

#### 周波数分離処理 (DocShadow-SD7K技術)
- **低周波成分**: ガウシアンブラー（σ=20）で背景・照明情報を抽出
- **高周波成分**: 元画像 - 低周波で文字・エッジ情報を保持
- **周波数別処理**: 低周波に影除去、高周波はそのまま保持して再結合

#### 高度な影除去 (DocShadow-SD7K技術)
- **照明マップ推定**: 重いガウシアンブラー（σ=50）で背景照明を推定
- **影マスク検出**: LAB色空間のLチャンネルで照明比率を計算
  - 比率 < 0.92 の領域を影として検出
  - モルフォロジー演算（楕円カーネル15×15）で精緻化
  - ガウシアンブラー（31×31）でソフトマスク生成
- **適応的補正**: 影領域と非影領域の明度中央値から補正係数を算出
- **スムーズ遷移**: マスクグラデーションで自然な境界を実現

#### 色保持処理
- **LAB色空間処理**: L（明度）のみ補正、a/b（色相）は保持
- **周波数再結合**: 補正した低周波 + 元の高周波で文字鮮明度維持

#### 最終仕上げ
- **ガンマ補正**: 明るさ調整（γ=1.15、控えめ）
- **CLAHE**: clipLimit=2.0で局所的なコントラスト向上
- **コントラスト強化**: α=1.05, β=2（控えめ、色保持重視）
- **シャープニング**: 強化ラプラシアンカーネルで鮮明さ向上

## 参考資料・技術ソース

このプロジェクトは以下の最新研究技術を統合実装しています:

### 影除去技術
- **[DocShadow-SD7K (ICCV 2023)](https://github.com/CXH-Research/DocShadow-SD7K)**: 
  - 周波数分離処理（低周波/高周波分離）
  - 照明マップ推定と適応的影補正
  - LAB色空間での色保持処理

### 書類検出技術
- **[DocScanner (IJCV 2025)](https://github.com/fh2019ustc/DocScanner)**:
  - ロバストな書類位置特定
  - マルチスケールエッジ検出
  - 段階的補正アルゴリズム

### 基本実装
- **[Python-Document-Scanner-OpenCV](https://github.com/ArashNasrEsfahani/Python-Document-Scanner-OpenCV)**:
  - 基本的な輪郭検出とパースペクティブ変換

## トラブルシューティング

### 書類が検出されない場合

- 書類と背景のコントラストを高くする
- 照明を改善する
- 書類全体が画像内に収まるようにする
- より正面から撮影する

### メモリ不足エラー

Cloud Runのメモリ設定を増やす:
```bash
gcloud run deploy document-scanner --memory 1Gi
```

### タイムアウトエラー

Cloud Runのタイムアウト設定を増やす:
```bash
gcloud run deploy document-scanner --timeout 600
```

## ライセンス

MIT

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。
