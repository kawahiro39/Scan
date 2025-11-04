# Document Scanner API

書類画像を自動的にトリミングしてPDF出力するCloud Run対応のAPIです。OpenCVを使用して書類の輪郭を自動検出し、パースペクティブ変換を適用して正面から見た画像に補正します。

## 機能

- 📸 **自動境界検出**: OpenCVを使用して書類の輪郭を自動検出
- 🔄 **パースペクティブ変換**: 斜めから撮影した画像を正面から見た状態に補正
- ✨ **画像強化**: ガンマ補正、CLAHE、シャープニングで画像品質を向上
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

1. **前処理**: グレースケール変換 → ガウシアンブラー
2. **エッジ検出**: Cannyアルゴリズムでエッジを検出
3. **輪郭検出**: 最大の4点の輪郭を書類として認識
4. **パースペクティブ変換**: 4点を基に正面視に補正
5. **画像強化** (オプション):
   - ガンマ補正で明るさ調整
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) で局所的なコントラスト向上
   - シャープニングで鮮明さ向上

## 参考資料

このプロジェクトは以下のリポジトリを参考に作成されました:
- [Python-Document-Scanner-OpenCV](https://github.com/ArashNasrEsfahani/Python-Document-Scanner-OpenCV)

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
