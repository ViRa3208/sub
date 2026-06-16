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


task_status = {}

# Загружаем модель
MODEL_SIZE = "small"
print("[loading] Загрузка модели Whisper (" + MODEL_SIZE + ")...")
model = whisper.load_model(MODEL_SIZE)
print("[ok] Модель загружена")

# Стоп-слова для фильтрации
STOP_WORDS = {
    'это', 'все', 'так', 'же', 'есть', 'даже', 'очень', 'такой', 'такие',
    'более', 'менее', 'можно', 'нужно', 'будет', 'который', 'которая',
    'которые', 'потом', 'сейчас', 'тогда', 'здесь', 'там', 'тут', 'когда',
    'если', 'потому', 'поэтому', 'также', 'например', 'вообще', 'конечно',
    'наверное', 'ладно', 'хорошо', 'и', 'в', 'на', 'с', 'к', 'у', 'о', 'об',
    'от', 'до', 'по', 'за', 'под', 'над', 'без', 'для', 'через', 'между'
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


def compute_sentence_importance(sentences, text):
    """
    Вычисляет важность предложений на основе:
    1. TF-IDF (ключевые слова)
    2. Позиции в тексте (первые предложения важнее)
    3. Длины предложения (слишком короткие штрафуются)
    4. Наличия ключевых фраз-индикаторов
    """
    if len(sentences) < 2:
        return [(sentences[0] if sentences else "", 1.0)]

    # Ключевые фразы, которые указывают на важность
    importance_indicators = [
        'важно', 'ключевой', 'основной', 'главный', 'суть', 'суть в том',
        'вывод', 'итак', 'таким образом', 'следовательно', 'поэтому',
        'во-первых', 'во-вторых', 'наконец', 'резюмируя', 'подводя итог'
    ]

    # Токенизируем предложения
    tokenized_sentences = []
    for sentence in sentences:
        words = re.findall(r'\b[а-яА-ЯёЁ]{3,}\b', sentence.lower())
        words = [w for w in words if w not in STOP_WORDS]
        tokenized_sentences.append(words)

    # Вычисляем TF для каждого предложения
    tf_scores = []
    for words in tokenized_sentences:
        word_count = Counter(words)
        tf = {word: count / len(words) for word, count in word_count.items()} if words else {}
        tf_scores.append(tf)

    # Вычисляем IDF для каждого слова
    idf_scores = {}
    total_docs = len(sentences)

    for words in tokenized_sentences:
        unique_words = set(words)
        for word in unique_words:
            idf_scores[word] = idf_scores.get(word, 0) + 1

    for word in idf_scores:
        idf_scores[word] = math.log(total_docs / idf_scores[word]) + 1

    # Вычисляем итоговые оценки
    sentence_scores = []
    text_lower = text.lower()

    for idx, (sentence, tf) in enumerate(zip(sentences, tf_scores)):
        # TF-IDF оценка
        tfidf_score = sum(tf.get(word, 0) * idf_scores.get(word, 0) for word in tf)

        # Оценка по позиции (первые предложения важнее)
        position_score = 1.0 / (idx + 1) * 0.3

        # Оценка длины (предложения средней длины лучше)
        length = len(sentence.split())
        if length < 5:
            length_score = 0.1  # штраф за слишком короткие
        elif length > 40:
            length_score = 0.7  # немного штрафуем за очень длинные
        else:
            length_score = 1.0

        # Оценка по индикаторам важности
        indicator_score = 0
        sentence_lower = sentence.lower()
        for indicator in importance_indicators:
            if indicator in sentence_lower:
                indicator_score += 0.15

        # Итоговая оценка
        total_score = tfidf_score * 0.5 + position_score + length_score * 0.2 + indicator_score

        sentence_scores.append((sentence, total_score))

    return sorted(sentence_scores, key=lambda x: x[1], reverse=True)


def generate_intelligent_summary(text, top_k=6):
    """
    Генерирует информативное резюме из самых важных предложений
    """
    if not text or len(text) < 200:
        if text and len(text) > 50:
            return text
        return "Текст видео слишком короткий для создания подробного резюме."

    # Разбиваем на предложения
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 40]

    if len(sentences) < 3:
        return text

    # Вычисляем важность предложений
    scored_sentences = compute_sentence_importance(sentences, text)

    # Берём топ-k самых важных (но не больше половины от общего числа)
    top_k = min(top_k, max(3, len(sentences) // 2))
    important_sentences = [sent for sent, score in scored_sentences[:top_k]]

    # Сортируем в порядке появления в тексте
    important_sentences.sort(key=lambda s: text.find(s))

    # Добавляем связующие фразы для лучшего понимания контекста
    summary_parts = []
    for i, sentence in enumerate(important_sentences):
        # Добавляем вводные слова для лучшей связности
        if i == 0:
            summary_parts.append(sentence)
        else:
            summary_parts.append(sentence)

    summary = '. '.join(summary_parts) + '.'

    # Если резюме получилось слишком коротким, добавляем ещё предложений
    if len(summary.split()) < 50 and len(important_sentences) < len(sentences) // 2:
        remaining = [sent for sent, score in scored_sentences[top_k:top_k + 3]]
        if remaining:
            summary += ' ' + '. '.join(remaining) + '.'

    return summary


def extract_keywords(text, top_n=12):
    """Извлекает ключевые слова с учётом частоты и значимости"""
    words = re.findall(r'\b[а-яА-ЯёЁ]{3,}\b', text.lower())
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]

    # Убираем повторяющиеся подряд слова
    filtered_words = []
    prev = None
    for w in words:
        if w != prev:
            filtered_words.append(w)
        prev = w

    word_counts = Counter(filtered_words)

    # Добавляем нормализацию по частотности
    total = sum(word_counts.values())
    normalized = [(word, count / total) for word, count in word_counts.items()]
    sorted_words = sorted(normalized, key=lambda x: x[1], reverse=True)

    return [word for word, _ in sorted_words[:top_n]]


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

        task_status[task_id] = {"progress": 80, "status": "Генерация субтитров и резюме..."}

        srt_content = generate_srt(result["segments"])
        with open(OUTPUT_DIR / f"{task_id}.srt", 'w', encoding='utf-8') as f:
            f.write(srt_content)

        full_text = " ".join([seg["text"] for seg in result["segments"]])

        # Генерируем улучшенное резюме
        print(f"[summary] Длина текста: {len(full_text)} символов")
        summary = generate_intelligent_summary(full_text, top_k=6)
        print(f"[summary] Длина резюме: {len(summary)} символов")

        keywords = extract_keywords(full_text, top_n=12)

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

        debug_file = DEBUG_DIR / f"audio_{datetime.now().strftime('%H%M%S')}.wav"
        with open(debug_file, 'wb') as f:
            f.write(content)
        print(f"[recognize] Сохранён отладочный файл: {debug_file}")

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = model.transcribe(tmp_path, language="ru", task="transcribe", fp16=False)
        text = result["text"].strip()

        print(f"[recognize] Распознано: '{text}'")

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
            max-width: 1400px;
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

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 24px;
        }

        .card {
            background: #16213e;
            border-radius: 16px;
            padding: 24px;
            border: 1px solid #2a3a5e;
        }

        .video-card {
            background: #0f0f2a;
        }

        .video-container {
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 16px;
        }

        video {
            width: 100%;
            max-height: 500px;
            background: black;
        }

        .current-subtitle {
            background: rgba(0, 0, 0, 0.85);
            color: #3b82f6;
            font-size: 24px;
            font-weight: 500;
            padding: 16px 20px;
            text-align: center;
            border-radius: 12px;
            margin-top: 16px;
            backdrop-filter: blur(8px);
            font-family: monospace;
        }

        .controls {
            display: flex;
            gap: 12px;
            justify-content: center;
            margin-top: 16px;
        }

        button {
            background: #3b82f6;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s;
        }

        button:hover {
            background: #2563eb;
        }

        .url-input, .file-input {
            width: 100%;
            padding: 12px 16px;
            font-size: 14px;
            border: 1px solid #2a3a5e;
            border-radius: 10px;
            background: #0f0f2a;
            color: #eee;
            margin-bottom: 12px;
        }

        .url-input:focus, .file-input:focus {
            outline: none;
            border-color: #3b82f6;
        }

        .url-input::placeholder {
            color: #4a5568;
        }

        .file-input {
            padding: 30px;
            text-align: center;
            cursor: pointer;
        }

        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            border-bottom: 1px solid #2a3a5e;
        }

        .tab {
            background: none;
            padding: 10px 20px;
            font-size: 14px;
        }

        .tab.active {
            color: #3b82f6;
            border-bottom: 2px solid #3b82f6;
            background: none;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .progress {
            display: none;
            margin-top: 20px;
        }

        .progress-bar {
            width: 100%;
            height: 6px;
            background: #2a3a5e;
            border-radius: 3px;
            overflow: hidden;
        }

        .progress-fill {
            width: 0%;
            height: 100%;
            background: #3b82f6;
            transition: width 0.3s;
            border-radius: 3px;
        }

        .status-text {
            margin-top: 12px;
            text-align: center;
            color: #a0aec0;
            font-size: 13px;
        }

        .error {
            background: #7f1a1a;
            color: #fecaca;
            padding: 12px;
            border-radius: 10px;
            margin-top: 16px;
            display: none;
            font-size: 14px;
        }

        .summary-section {
            background: #0f0f2a;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
            border-left: 3px solid #3b82f6;
        }

        .summary-title {
            font-size: 14px;
            font-weight: 600;
            color: #3b82f6;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .summary-content {
            font-size: 14px;
            line-height: 1.6;
            color: #cbd5e1;
        }

        .stats {
            display: flex;
            gap: 16px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }

        .stat-card {
            background: #16213e;
            padding: 10px 16px;
            border-radius: 8px;
            text-align: center;
            flex: 1;
            min-width: 80px;
        }

        .stat-value {
            font-size: 22px;
            font-weight: 600;
            color: #3b82f6;
        }

        .stat-label {
            font-size: 11px;
            color: #718096;
            margin-top: 4px;
        }

        .keywords {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 16px;
        }

        .keyword {
            background: #1e2a4a;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            color: #93c5fd;
        }

        .subtitles-list {
            background: #0f0f2a;
            border-radius: 12px;
            padding: 16px;
            max-height: 400px;
            overflow-y: auto;
        }

        .subtitles-list h3 {
            font-size: 14px;
            font-weight: 600;
            color: #a0aec0;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .subtitle-item {
            padding: 10px;
            border-bottom: 1px solid #1e2a4a;
            cursor: pointer;
            transition: background 0.2s;
        }

        .subtitle-item:hover {
            background: #1e2a4a;
        }

        .subtitle-item.active {
            background: #1e3a5f;
            border-left: 3px solid #3b82f6;
        }

        .subtitle-time {
            font-family: monospace;
            font-size: 11px;
            color: #3b82f6;
            margin-bottom: 4px;
        }

        .subtitle-text {
            font-size: 13px;
            color: #cbd5e1;
            line-height: 1.4;
        }

        .right-panel {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        @media (max-width: 1000px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Видео-плеер с субтитрами</h1>
            <p>Загрузите видео или вставьте ссылку с YouTube</p>
        </div>

        <div class="card" style="margin-bottom: 24px;">
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
        </div>

        <div id="playerSection" style="display: none;">
            <div class="main-grid">
                <div class="video-card card">
                    <div class="video-container">
                        <video id="videoPlayer" controls>
                            <source id="videoSource" src="">
                        </video>
                    </div>
                    <div class="current-subtitle" id="currentSubtitle">—</div>
                    <div class="controls">
                        <button id="playPauseBtn">Пауза</button>
                        <button id="fullscreenBtn">Полный экран</button>
                    </div>
                </div>

                <div class="right-panel">
                    <div class="card">
                        <div class="summary-title">Краткое содержание</div>
                        <div class="summary-content" id="summaryContent"></div>

                        <div class="stats" id="stats"></div>

                        <div class="summary-title" style="margin-top: 12px;">Ключевые слова</div>
                        <div class="keywords" id="keywords"></div>
                    </div>

                    <div class="subtitles-list">
                        <h3>Список субтитров</h3>
                        <div id="subtitlesContent"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let subtitleSegments = [];

        const urlInput = document.getElementById('urlInput');
        const fileInput = document.getElementById('fileInput');
        const processUrlBtn = document.getElementById('processUrlBtn');
        const processFileBtn = document.getElementById('processFileBtn');
        const progressDiv = document.getElementById('progress');
        const errorDiv = document.getElementById('error');
        const playerSection = document.getElementById('playerSection');
        const videoPlayer = document.getElementById('videoPlayer');
        const currentSubtitle = document.getElementById('currentSubtitle');
        const subtitlesContent = document.getElementById('subtitlesContent');
        const progressFill = document.getElementById('progressFill');
        const statusText = document.getElementById('statusText');
        const summaryContent = document.getElementById('summaryContent');
        const keywordsDiv = document.getElementById('keywords');
        const statsDiv = document.getElementById('stats');

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
            playerSection.style.display = 'none';
            progressFill.style.width = '0%';
            statusText.textContent = 'Обработка...';

            try {
                const response = await fetch('/api/process', { method: 'POST', body: formData });
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Ошибка обработки');
                }
                const data = await response.json();
                loadPlayer(data);
                progressDiv.style.display = 'none';
                playerSection.style.display = 'block';
            } catch (error) {
                showError(error.message);
                progressDiv.style.display = 'none';
            }
        }

        function loadPlayer(data) {
            subtitleSegments = data.segments;
            videoPlayer.src = data.video_url;
            videoPlayer.load();
            displaySubtitlesList();

            if (data.summary) {
                summaryContent.innerHTML = escapeHtml(data.summary);
            }

            if (data.keywords && data.keywords.length) {
                keywordsDiv.innerHTML = data.keywords.map(k => `<span class="keyword">${escapeHtml(k)}</span>`).join('');
            }

            if (data.stats) {
                statsDiv.innerHTML = `
                    <div class="stat-card"><div class="stat-value">${data.stats.duration}</div><div class="stat-label">Длительность</div></div>
                    <div class="stat-card"><div class="stat-value">${data.stats.segments}</div><div class="stat-label">Фрагментов</div></div>
                    <div class="stat-card"><div class="stat-value">${data.stats.words}</div><div class="stat-label">Слов</div></div>
                `;
            }

            videoPlayer.addEventListener('timeupdate', updateSubtitle);

            document.getElementById('playPauseBtn').onclick = () => {
                if (videoPlayer.paused) {
                    videoPlayer.play();
                    document.getElementById('playPauseBtn').textContent = 'Пауза';
                } else {
                    videoPlayer.pause();
                    document.getElementById('playPauseBtn').textContent = 'Воспроизвести';
                }
            };

            videoPlayer.addEventListener('play', () => {
                document.getElementById('playPauseBtn').textContent = 'Пауза';
            });

            videoPlayer.addEventListener('pause', () => {
                document.getElementById('playPauseBtn').textContent = 'Воспроизвести';
            });

            document.getElementById('fullscreenBtn').onclick = () => {
                if (videoPlayer.requestFullscreen) {
                    videoPlayer.requestFullscreen();
                }
            };
        }

        function displaySubtitlesList() {
            subtitlesContent.innerHTML = subtitleSegments.map((seg, i) => `
                <div class="subtitle-item" data-start="${seg.start}">
                    <div class="subtitle-time">${formatTime(seg.start)} → ${formatTime(seg.end)}</div>
                    <div class="subtitle-text">${escapeHtml(seg.text)}</div>
                </div>
            `).join('');

            document.querySelectorAll('.subtitle-item').forEach(item => {
                item.addEventListener('click', () => {
                    videoPlayer.currentTime = parseFloat(item.dataset.start);
                    videoPlayer.play();
                });
            });
        }

        function updateSubtitle() {
            const t = videoPlayer.currentTime;
            const active = subtitleSegments.find(seg => t >= seg.start && t <= seg.end);

            if (active) {
                currentSubtitle.textContent = active.text;
                document.querySelectorAll('.subtitle-item').forEach(item => {
                    const start = parseFloat(item.dataset.start);
                    if (Math.abs(start - active.start) < 0.1) {
                        item.classList.add('active');
                    } else {
                        item.classList.remove('active');
                    }
                });
            } else {
                currentSubtitle.textContent = '—';
            }
        }

        function formatTime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return h > 0 
                ? `${h.toString().padStart(2,'0')}:${m.toString().padStart(2,'0')}:${s.toString().padStart(2,'0')}`
                : `${m.toString().padStart(2,'0')}:${s.toString().padStart(2,'0')}`;
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
    print("Видео-плеер с субтитрами и умным резюме")
    print("=" * 60)
    print("Адрес: http://localhost:8017")
    print("Улучшенная генерация резюме на основе TF-IDF, позиции и индикаторов")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8017)