from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import whisper
import uuid
import os
import aiofiles

app = FastAPI(title="Subtitle Generator")


UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


print("Загрузка модели Whisper...")
model = whisper.load_model("tiny")  # или "base", "small"
print("Модель загружена!")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Генератор субтитров</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 50px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #f8f9ff;
        }
        .upload-area:hover {
            background: #e8ecff;
            border-color: #764ba2;
        }
        #fileInfo {
            margin-top: 20px;
            padding: 15px;
            background: #e8ecff;
            border-radius: 10px;
            display: none;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 20px;
        }
        button:hover {
            transform: translateY(-2px);
        }
        .progress {
            display: none;
            margin-top: 20px;
        }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }
        .result {
            display: none;
            margin-top: 30px;
        }
        .subtitles-preview {
            background: #f5f5f5;
            border-radius: 10px;
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 14px;
            white-space: pre-wrap;
        }
        .download-links {
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }
        .download-btn {
            background: #4CAF50;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            display: inline-block;
        }
        .error {
            color: red;
            margin-top: 10px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Генератор субтитров</h1>
        <p class="subtitle">Загрузите видео или аудио файл</p>

        <div class="upload-area" id="uploadArea">
            <div style="font-size: 48px;">📁</div>
            <p>Нажмите или перетащите файл сюда</p>
            <input type="file" id="fileInput" accept="audio/*,video/*" style="display: none;">
        </div>

        <div id="fileInfo"></div>
        <button id="generateBtn" style="display: none;">🎯 Сгенерировать субтитры</button>

        <div class="progress" id="progress">
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill">0%</div>
            </div>
            <p id="statusText" style="margin-top: 10px; text-align: center;">Обработка...</p>
        </div>

        <div class="error" id="error"></div>

        <div class="result" id="result">
            <h3>📝 Результат транскрипции</h3>
            <div class="subtitles-preview" id="preview"></div>
            <div class="download-links" id="downloadLinks"></div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const generateBtn = document.getElementById('generateBtn');
        const progressDiv = document.getElementById('progress');
        const errorDiv = document.getElementById('error');
        const resultDiv = document.getElementById('result');
        const preview = document.getElementById('preview');
        const downloadLinks = document.getElementById('downloadLinks');
        const progressFill = document.getElementById('progressFill');
        const statusText = document.getElementById('statusText');

        let selectedFile = null;

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = '#e8ecff';
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.background = '#f8f9ff';
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = '#f8f9ff';
            handleFile(e.dataTransfer.files[0]);
        });

        fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

        function handleFile(file) {
            if (!file) return;
            selectedFile = file;
            fileInfo.innerHTML = `<strong>Выбран файл:</strong> ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
            fileInfo.style.display = 'block';
            generateBtn.style.display = 'block';
            resultDiv.style.display = 'none';
            errorDiv.style.display = 'none';
        }

        generateBtn.addEventListener('click', async () => {
            if (!selectedFile) return;

            const formData = new FormData();
            formData.append('file', selectedFile);

            generateBtn.disabled = true;
            progressDiv.style.display = 'block';
            errorDiv.style.display = 'none';
            resultDiv.style.display = 'none';
            progressFill.style.width = '0%';
            statusText.textContent = 'Обработка файла...';

            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Ошибка обработки');
                }

                const result = await response.json();
                displayResult(result);
            } catch (error) {
                showError(error.message);
            } finally {
                generateBtn.disabled = false;
                progressDiv.style.display = 'none';
            }
        });

        function displayResult(result) {
            resultDiv.style.display = 'block';
            preview.textContent = result.text;

            downloadLinks.innerHTML = `
                <a href="/api/download/${result.file_id}" class="download-btn">📄 Скачать TXT</a>
            `;
        }

        function showError(message) {
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }
    </script>
</body>
</html>
"""


@app.get("/")
async def root():
    return HTMLResponse(HTML_TEMPLATE)


@app.post("/api/generate")
async def gee_subtitlesnerat(file: UploadFile = File(...)):
    try:

        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"{file_id}{file_extension}"

        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)


        result = model.transcribe(str(file_path), language="ru")


        txt_path = OUTPUT_DIR / f"{file_id}.txt"
        async with aiofiles.open(txt_path, 'w', encoding='utf-8') as f:
            await f.write(result["text"])


        os.remove(file_path)

        return {
            "text": result["text"],
            "file_id": file_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{file_id}")
async def download_subtitle(file_id: str):
    """Скачивание субтитров"""
    file_path = OUTPUT_DIR / f"{file_id}.txt"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(file_path, filename=f"subtitle_{file_id}.txt")


# Правильный способ запуска приложения
if __name__ == "__main__":
    import uvicorn

    print("🚀 Запуск сервера: http://localhost:8000")
    print("📝 Нажмите Ctrl+C для остановки")
    uvicorn.run(
        "main:app",  # Импортируем как строку, а не объект
        host="127.0.0.1",
        port=8000,
        reload=True,  # Теперь reload будет работать
        log_level="info"
    )