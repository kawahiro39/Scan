# API使用方法ガイド

## エンドポイント: POST /scan

書類画像をアップロードして、自動トリミング＆PDF変換します。

### リクエスト形式

**Content-Type**: `multipart/form-data`

### パラメータ

| パラメータ | 型 | 必須 | デフォルト | 説明 |
|----------|-----|------|-----------|------|
| file | File | ✅ Yes | - | 画像ファイル（JPEG, PNG等） |
| enhance | boolean | ❌ No | true | 画像強化の有効化 |
| page_size | string | ❌ No | "A4" | PDFページサイズ（"A4" or "letter"） |

### レスポンス

- **Content-Type**: `application/pdf`
- **Status Code**: 200 (成功時)
- **Body**: PDFファイル

### エラーレスポンス

```json
{
  "detail": "エラーメッセージ"
}
```

## 使用例

### 1. cURLコマンド

#### 基本的な使用方法
```bash
curl -X POST "https://your-service-url.run.app/scan" \
  -F "file=@/path/to/document.jpg" \
  -o scanned.pdf
```

#### 全パラメータ指定
```bash
curl -X POST "https://your-service-url.run.app/scan" \
  -F "file=@/path/to/document.jpg" \
  -F "enhance=true" \
  -F "page_size=A4" \
  -o scanned.pdf
```

#### 画像強化なし
```bash
curl -X POST "https://your-service-url.run.app/scan" \
  -F "file=@/path/to/document.jpg" \
  -F "enhance=false" \
  -o scanned.pdf
```

#### Letterサイズ指定
```bash
curl -X POST "https://your-service-url.run.app/scan" \
  -F "file=@/path/to/document.jpg" \
  -F "page_size=letter" \
  -o scanned.pdf
```

### 2. Python

#### 基本的な使用方法
```python
import requests

url = "https://your-service-url.run.app/scan"

# ファイルを開く
with open("document.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

# PDFを保存
if response.status_code == 200:
    with open("scanned.pdf", "wb") as f:
        f.write(response.content)
    print("PDF saved successfully!")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

#### パラメータ付き
```python
import requests

url = "https://your-service-url.run.app/scan"

with open("document.jpg", "rb") as f:
    files = {"file": f}
    data = {
        "enhance": "true",
        "page_size": "A4"
    }
    response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    with open("scanned.pdf", "wb") as f:
        f.write(response.content)
    print("Success!")
```

#### エラーハンドリング付き
```python
import requests

def scan_document(image_path, enhance=True, page_size="A4"):
    """書類をスキャンしてPDFを生成"""
    url = "https://your-service-url.run.app/scan"
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {
                "enhance": str(enhance).lower(),
                "page_size": page_size
            }
            
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            
            # ステータスメッセージを取得
            status_msg = response.headers.get("X-Scan-Status", "")
            print(f"Scan Status: {status_msg}")
            
            return response.content
            
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

# 使用例
pdf_data = scan_document("document.jpg", enhance=True, page_size="A4")
if pdf_data:
    with open("output.pdf", "wb") as f:
        f.write(pdf_data)
```

### 3. JavaScript (Node.js)

```javascript
const fs = require('fs');
const FormData = require('form-data');
const axios = require('axios');

async function scanDocument(imagePath) {
  const url = 'https://your-service-url.run.app/scan';
  
  const form = new FormData();
  form.append('file', fs.createReadStream(imagePath));
  form.append('enhance', 'true');
  form.append('page_size', 'A4');
  
  try {
    const response = await axios.post(url, form, {
      headers: form.getHeaders(),
      responseType: 'arraybuffer'
    });
    
    fs.writeFileSync('scanned.pdf', response.data);
    console.log('PDF saved successfully!');
    
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
  }
}

// 使用
scanDocument('document.jpg');
```

### 4. JavaScript (Browser)

```javascript
async function scanDocument(fileInput) {
  const url = 'https://your-service-url.run.app/scan';
  
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('enhance', 'true');
  formData.append('page_size', 'A4');
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const blob = await response.blob();
    
    // ダウンロード
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = 'scanned.pdf';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(downloadUrl);
    
    console.log('PDF downloaded!');
    
  } catch (error) {
    console.error('Error:', error);
  }
}

// HTML例
// <input type="file" id="fileInput" accept="image/*">
// <button onclick="scanDocument(document.getElementById('fileInput'))">Scan</button>
```

### 5. PHP

```php
<?php

function scanDocument($imagePath, $enhance = true, $pageSize = 'A4') {
    $url = 'https://your-service-url.run.app/scan';
    
    $cfile = new CURLFile($imagePath, mime_content_type($imagePath), basename($imagePath));
    
    $data = [
        'file' => $cfile,
        'enhance' => $enhance ? 'true' : 'false',
        'page_size' => $pageSize
    ];
    
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_POST, 1);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $data);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);
    
    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    
    if ($httpCode === 200) {
        file_put_contents('scanned.pdf', $response);
        echo "PDF saved successfully!\n";
        return true;
    } else {
        echo "Error: HTTP $httpCode\n";
        echo $response . "\n";
        return false;
    }
}

// 使用例
scanDocument('document.jpg', true, 'A4');
?>
```

## エンドポイント: POST /scan/preview

書類をスキャンして、処理後の画像をJPEGで返します（PDF化なし、プレビュー用）。

### パラメータ

| パラメータ | 型 | 必須 | デフォルト | 説明 |
|----------|-----|------|-----------|------|
| file | File | ✅ Yes | - | 画像ファイル |
| enhance | boolean | ❌ No | true | 画像強化の有効化 |

### 使用例

```bash
curl -X POST "https://your-service-url.run.app/scan/preview" \
  -F "file=@document.jpg" \
  -F "enhance=true" \
  -o preview.jpg
```

## よくある質問

### Q: 対応している画像形式は？
A: JPEG, PNG, BMP, TIFF など、OpenCVが対応している一般的な画像形式に対応しています。

### Q: 最大ファイルサイズは？
A: Cloud Runのデフォルト設定では32MBまでです。必要に応じて設定を変更できます。

### Q: 複数ページのPDFを作成できますか？
A: 現在のバージョンは1画像＝1ページのPDFです。複数ページが必要な場合は、複数回呼び出して結合してください。

### Q: 書類が検出されない場合は？
A: 元の画像（強化処理のみ適用）がPDFとして返されます。レスポンスヘッダー `X-Scan-Status` で詳細を確認できます。

### Q: 処理時間はどのくらい？
A: 画像サイズにもよりますが、通常1〜3秒程度です。

## トラブルシューティング

### エラー: "Field required" (422)

**原因**: `file`パラメータが送信されていません。

**解決策**: 
- `multipart/form-data`形式でリクエストを送信しているか確認
- ファイルフィールド名が`file`になっているか確認

```bash
# 正しい例
curl -X POST "URL" -F "file=@image.jpg"

# 間違った例
curl -X POST "URL" -d "file=@image.jpg"  # -d ではなく -F を使用
```

### エラー: "Could not decode image" (400)

**原因**: アップロードされたファイルが画像として認識できません。

**解決策**:
- ファイルが破損していないか確認
- 対応形式（JPEG, PNG等）か確認

### エラー: "No document boundary detected"

**原因**: 書類の輪郭を自動検出できませんでした。

**解決策**:
- 書類と背景のコントラストを高くする
- 照明を改善する
- より正面から撮影する
- この場合でも、強化された元画像がPDFとして返されます

## テストスクリプト

プロジェクトに含まれる`test_api.sh`を使用してAPIをテストできます：

```bash
# 実行権限を付与
chmod +x test_api.sh

# テスト実行
./test_api.sh https://your-service-url.run.app
```
