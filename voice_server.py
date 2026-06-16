from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import whisper
import numpy as np
from datetime import datetime
import logging
import os
import yt_dlp
import uuid
import re
from collections import Counter
import math
import tempfile
import wave

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Player with Subtitles")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
AUDIO_DIR = Path("audio_downloads")
DEBUG_DIR = Path("debug_audio")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)

# Хранилище статусов
task_status = {}

# Загружаем модель
MODEL_SIZE = "small"
print("[loading] Загрузка модели Whisper (" + MODEL_SIZE + ")...")
model = whisper.load_model(MODEL_SIZE)
print("[ok] Модель загружена")

# Стоп-слова
STOP_WORDS = {
    'это', 'все', 'так', 'же', 'есть', 'даже', 'очень', 'такой',
    'более', 'менее', 'можно', 'нужно', 'будет', 'который', 'потом',
    'сейчас', 'тогда', 'здесь', 'там', 'тут', 'когда', 'если'
}


def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt(segments):
    srt = []
    for i, seg in enumerate(segments, 1):
        start = format_srt_time(seg["start"])
        end = format_srt_time(seg["end"])
        text = seg["text"].strip()
        srt.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(srt)


def generate_summary(text, num_sentences=3):
    if not text or len(text) < 100:
        return "Текст слишком короткий."
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    if len(sentences) <= num_sentences:
        return '. '.join(sentences) + '.'
    return '. '.join(sentences[:num_sentences]) + '.'


def extract_keywords(text, top_n=8):
    words = re.findall(r'\b[а-яА-ЯёЁ]{4,}\b', text.lower())
    words = [w for w in words if w not in STOP_WORDS]
    word_counts = Counter(words)
    return [word for word, _ in word_counts.most_common(top_n)]


# ============== ОСНОВНЫЕ ЭНДПОИНТЫ ==============

@app.post("/api/process")
async def process_video(url: str = Form(None), file: UploadFile = File(None)):
    task_id = str(uuid.uuid4())
    video_path = None
    audio_path = None

    try:
        task_status[task_id] = {"progress": 10, "status": "Скачивание..."}

        if url:
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': str(UPLOAD_DIR / f'{task_id}.%(ext)s'),
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                ext = info.get('ext', 'mp4')
                video_path = UPLOAD_DIR / f"{task_id}.{ext}"

            task_status[task_id] = {"progress": 30, "status": "Извлечение аудио..."}

            audio_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '16000'}],
                'outtmpl': str(AUDIO_DIR / f'{task_id}.%(ext)s'),
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(audio_opts) as ydl:
                ydl.extract_info(url, download=True)
                audio_path = AUDIO_DIR / f"{task_id}.wav"

        elif file:
            video_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
            content = await file.read()
            with open(video_path, 'wb') as f:
                f.write(content)
            audio_path = video_path

        task_status[task_id] = {"progress": 50, "status": "Распознавание речи..."}
        result = model.transcribe(str(audio_path), language="ru", task="transcribe")

        for seg in result["segments"]:
            seg["text"] = clean_text(seg["text"])

        task_status[task_id] = {"progress": 80, "status": "Генерация субтитров..."}

        srt_content = generate_srt(result["segments"])
        with open(OUTPUT_DIR / f"{task_id}.srt", 'w', encoding='utf-8') as f:
            f.write(srt_content)

        full_text = " ".join([seg["text"] for seg in result["segments"]])
        summary = generate_summary(full_text)
        keywords = extract_keywords(full_text)

        duration = result["segments"][-1]["end"] if result["segments"] else 0
        word_count = len(full_text.split())

        if audio_path != video_path and audio_path.exists():
            os.remove(audio_path)

        task_status[task_id] = {
            "progress": 100,
            "status": "completed",
            "result": {
                "video_url": f"/api/video/{video_path.name}",
                "srt_url": f"/api/download/{task_id}.srt",
                "segments": result["segments"],
                "summary": summary,
                "keywords": keywords,
                "stats": {
                    "duration": f"{int(duration // 60)}:{int(duration % 60):02d}",
                    "segments": len(result["segments"]),
                    "words": word_count
                }
            }
        }

        return task_status[task_id]["result"]

    except Exception as e:
        logger.error(f"Ошибка: {e}")
        task_status[task_id] = {"progress": 0, "status": "error", "error": str(e)}
        raise HTTPException(status_code=500, detail=str(e))


# ============== ЭНДПОИНТ ДЛЯ РАСШИРЕНИЯ ==============

@app.post("/api/recognize")
async def recognize_audio(audio: UploadFile = File(...)):
    """Получает аудио от расширения и распознаёт речь"""
    print("=" * 50)
    print("[recognize] Вызван эндпоинт распознавания")

    try:
        content = await audio.read()
        print(f"[recognize] Размер файла: {len(content)} байт")

        if len(content) < 2000:
            print("[recognize] Файл слишком маленький")
            return {"text": ""}

        # Сохраняем для отладки
        debug_file = DEBUG_DIR / f"audio_{datetime.now().strftime('%H%M%S')}.wav"
        with open(debug_file, 'wb') as f:
            f.write(content)
        print(f"[recognize] Сохранён отладочный файл: {debug_file}")

        # Сохраняем во временный файл для Whisper
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Распознаём
        result = model.transcribe(tmp_path, language="ru", task="transcribe", fp16=False)
        text = result["text"].strip()

        print(f"[recognize] Распознано: '{text}'")

        # Очищаем
        os.unlink(tmp_path)

        if text and len(text) > 1:
            return JSONResponse(
                content={"text": text},
                headers={"Content-Type": "application/json; charset=utf-8"}
            )
        else:
            return {"text": ""}

    except Exception as e:
        print(f"[recognize] Ошибка: {e}")
        return {"text": ""}


# ============== ВСПОМОГАТЕЛЬНЫЕ ЭНДПОИНТЫ ==============

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    return task_status.get(task_id, {"progress": 0, "status": "not_found"})


@app.get("/api/video/{filename}")
async def get_video(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4")


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)


# ============== ГЛАВНАЯ СТРАНИЦА ==============

@app.get("/")
async def root():
    html_content = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Видео-плеер с субтитрами</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #1a1a2e;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            color: #eee;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 600;
            letter-spacing: -0.5px;
        }

        .header p {
            font-size: 1.1em;
            opacity: 0.8;
        }

        .card {
            background: #16213e;
            border-radius: 16px;
            padding: 30px;
            border: 1px solid #2a3a5e;
        }

        .url-input {
            width: 100%;
            padding: 14px 16px;
            font-size: 16px;
            border: 1px solid #2a3a5e;
            border-radius: 10px;
            background: #0f0f2a;
            color: #eee;
            margin-bottom: 15px;
            transition: all 0.2s;
        }

        .url-input:focus {
            outline: none;
            border-color: #3b82f6;
            background: #1a1a3a;
        }

        .url-input::placeholder {
            color: #4a5568;
        }

        .file-input {
            width: 100%;
            padding: 30px;
            border: 2px dashed #2a3a5e;
            border-radius: 12px;
            background: #0f0f2a;
            cursor: pointer;
            margin-bottom: 15px;
            color: #718096;
            text-align: center;
        }

        .file-input:hover {
            border-color: #3b82f6;
            background: #1a1a3a;
        }

        button {
            background: #3b82f6;
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            transition: background 0.2s;
        }

        button:hover {
            background: #2563eb;
        }

        .progress {
            display: none;
            margin-top: 20px;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: #2a3a5e;
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            width: 0%;
            height: 100%;
            background: #3b82f6;
            transition: width 0.3s;
            border-radius: 4px;
        }

        .status-text {
            margin-top: 12px;
            text-align: center;
            color: #a0aec0;
            font-size: 14px;
        }

        .result {
            display: none;
            margin-top: 30px;
        }

        .subtitles-preview {
            background: #0f0f2a;
            border-radius: 12px;
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #2a3a5e;
        }

        .subtitle-item {
            padding: 12px;
            border-bottom: 1px solid #1e2a4a;
        }

        .subtitle-item:last-child {
            border-bottom: none;
        }

        .subtitle-time {
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 12px;
            color: #3b82f6;
            margin-bottom: 6px;
        }

        .subtitle-text {
            color: #e2e8f0;
            line-height: 1.5;
        }

        .download-links {
            margin-top: 20px;
            display: flex;
            gap: 12px;
            justify-content: center;
        }

        .download-btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #2d3748;
            color: #e2e8f0;
            padding: 10px 20px;
            border-radius: 8px;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s;
        }

        .download-btn:hover {
            background: #4a5568;
        }

        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 24px;
            border-bottom: 1px solid #2a3a5e;
        }

        .tab {
            padding: 10px 24px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 15px;
            color: #a0aec0;
            font-weight: 500;
        }

        .tab.active {
            color: #3b82f6;
            border-bottom: 2px solid #3b82f6;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .error {
            background: #7f1a1a;
            color: #fecaca;
            padding: 14px;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
            border: 1px solid #991b1b;
        }

        .stats {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .stat-card {
            background: #0f0f2a;
            padding: 12px 20px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #2a3a5e;
        }

        .stat-value {
            font-size: 24px;
            font-weight: 600;
            color: #3b82f6;
        }

        .stat-label {
            font-size: 12px;
            color: #718096;
            margin-top: 4px;
        }

        .keywords {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
            margin-bottom: 20px;
        }

        .keyword {
            background: #1e2a4a;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            color: #93c5fd;
        }

        .summary {
            background: #0f0f2a;
            padding: 16px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 3px solid #3b82f6;
            color: #cbd5e1;
            font-size: 14px;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Видео-плеер с субтитрами</h1>
            <p>Загрузите видео или вставьте ссылку с YouTube</p>
        </div>
        <div class="card">
            <div class="tabs">
                <button class="tab active" data-tab="url">По ссылке</button>
                <button class="tab" data-tab="file">Загрузить файл</button>
            </div>
            <div id="url-tab" class="tab-content active">
                <input type="text" id="urlInput" class="url-input" placeholder="https://www.youtube.com/watch?v=...">
                <button id="processUrlBtn">Создать субтитры</button>
            </div>
            <div id="file-tab" class="tab-content">
                <input type="file" id="fileInput" class="file-input" accept="video/*,audio/*">
                <button id="processFileBtn">Создать субтитры</button>
            </div>

            <div class="progress" id="progress">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="status-text" id="statusText">Обработка...</div>
            </div>

            <div class="error" id="error"></div>

            <div class="result" id="result">
                <div class="stats" id="stats"></div>
                <div class="keywords" id="keywords"></div>
                <div class="summary" id="summary"></div>
                <div class="subtitles-preview" id="preview"></div>
                <div class="download-links" id="downloadLinks"></div>
            </div>
        </div>
    </div>

    <script>
        const urlInput = document.getElementById('urlInput');
        const fileInput = document.getElementById('fileInput');
        const processUrlBtn = document.getElementById('processUrlBtn');
        const processFileBtn = document.getElementById('processFileBtn');
        const progressDiv = document.getElementById('progress');
        const errorDiv = document.getElementById('error');
        const resultDiv = document.getElementById('result');
        const previewDiv = document.getElementById('preview');
        const downloadLinks = document.getElementById('downloadLinks');
        const progressFill = document.getElementById('progressFill');
        const statusText = document.getElementById('statusText');
        const statsDiv = document.getElementById('stats');
        const keywordsDiv = document.getElementById('keywords');
        const summaryDiv = document.getElementById('summary');

        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
                document.getElementById(tab.dataset.tab + '-tab').classList.add('active');
            });
        });

        async function processRequest(formData) {
            progressDiv.style.display = 'block';
            errorDiv.style.display = 'none';
            resultDiv.style.display = 'none';
            progressFill.style.width = '0%';
            statusText.textContent = 'Обработка...';

            try {
                const response = await fetch('/api/process', { method: 'POST', body: formData });
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Ошибка обработки');
                }
                const data = await response.json();
                displayResult(data);
                progressDiv.style.display = 'none';
            } catch (error) {
                showError(error.message);
                progressDiv.style.display = 'none';
            }
        }

        function displayResult(data) {
            resultDiv.style.display = 'block';

            if (data.stats) {
                statsDiv.innerHTML = `
                    <div class="stat-card"><div class="stat-value">${data.stats.duration}</div><div class="stat-label">Длительность</div></div>
                    <div class="stat-card"><div class="stat-value">${data.stats.segments}</div><div class="stat-label">Сегментов</div></div>
                    <div class="stat-card"><div class="stat-value">${data.stats.words}</div><div class="stat-label">Слов</div></div>
                `;
            }

            if (data.keywords && data.keywords.length) {
                keywordsDiv.innerHTML = data.keywords.map(kw => `<span class="keyword">${escapeHtml(kw)}</span>`).join('');
            } else {
                keywordsDiv.innerHTML = '';
            }

            if (data.summary) {
                summaryDiv.innerHTML = escapeHtml(data.summary);
            }

            if (data.segments) {
                previewDiv.innerHTML = data.segments.map(seg => `
                    <div class="subtitle-item">
                        <div class="subtitle-time">${formatTime(seg.start)} → ${formatTime(seg.end)}</div>
                        <div class="subtitle-text">${escapeHtml(seg.text)}</div>
                    </div>
                `).join('');
            }

            if (data.srt_url) {
                downloadLinks.innerHTML = `<a href="${data.srt_url}" class="download-btn">Скачать субтитры (SRT)</a>`;
            }
        }

        function formatTime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return h > 0 ? `${h.toString().padStart(2,'0')}:${m.toString().padStart(2,'0')}:${s.toString().padStart(2,'0')}` : `${m.toString().padStart(2,'0')}:${s.toString().padStart(2,'0')}`;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function showError(msg) {
            errorDiv.textContent = msg;
            errorDiv.style.display = 'block';
            setTimeout(() => errorDiv.style.display = 'none', 5000);
        }

        processUrlBtn.addEventListener('click', () => {
            const url = urlInput.value.trim();
            if (!url) { showError('Введите ссылку на видео'); return; }
            const fd = new FormData();
            fd.append('url', url);
            processRequest(fd);
        });

        processFileBtn.addEventListener('click', () => {
            const file = fileInput.files[0];
            if (!file) { showError('Выберите видеофайл'); return; }
            const fd = new FormData();
            fd.append('file', file);
            processRequest(fd);
        });
    </script>
</body>
</html>
    '''
    return HTMLResponse(html_content)


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Видео-плеер с субтитрами")
    print("=" * 60)
    print("Адрес: http://localhost:8017")
    print("Расширение использует микрофон для распознавания")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8017)