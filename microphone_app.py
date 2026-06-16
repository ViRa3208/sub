from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pathlib import Path
import whisper
import numpy as np
import wave
import io
from datetime import datetime
import logging
import tempfile
import os
import subprocess
import asyncio
import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice to Text - Микрофон")


BASE_DIR = Path(__file__).parent
RECORDINGS_DIR = BASE_DIR / "voice_recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)


print("🔄 Загрузка модели Whisper...")
model = whisper.load_model("tiny")
print("✅ Модель загружена!")


# Хранилище для истории распознавания
class TranscriptionHistory:
    def __init__(self):
        self.entries = []
        self.current_text = ""

    def add_entry(self, text, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "text": text
        }
        self.entries.append(entry)
        self.current_text += text + " "
        return entry

    def get_all(self):
        return self.entries

    def clear(self):
        self.entries = []
        self.current_text = ""

    def save_to_file(self, filename=None):
        if filename is None:
            filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        filepath = RECORDINGS_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.current_text)
        return filepath


history = TranscriptionHistory()

# HTML шаблон с непрерывной записью
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice to Text - Непрерывное распознавание</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .main-card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }

        .record-section {
            text-align: center;
            padding: 20px;
        }

        .record-btn {
            width: 160px;
            height: 160px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin: 0 auto 20px;
        }

        .record-btn:hover {
            transform: scale(1.05);
        }

        .record-btn.recording {
            animation: pulse 1.5s infinite;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        .record-icon {
            font-size: 60px;
            color: white;
        }

        .timer {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
            margin: 15px 0;
            font-family: monospace;
        }

        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            margin-top: 10px;
        }

        .status.idle {
            background: #e0e0e0;
            color: #666;
        }

        .status.recording {
            background: #ff6b6b;
            color: white;
            animation: blink 1s infinite;
        }

        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }

        .status.processing {
            background: #ffa500;
            color: white;
        }

        .result-section {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9ff;
            border-radius: 15px;
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
            flex-wrap: wrap;
            gap: 10px;
        }

        .button-group {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .btn {
            padding: 8px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }

        .btn-primary {
            background: #4CAF50;
            color: white;
        }

        .btn-danger {
            background: #ff6b6b;
            color: white;
        }

        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }

        .btn-warning {
            background: #ff9800;
            color: white;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .transcription-container {
            margin-top: 15px;
        }

        .current-text {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            border: 2px solid #667eea;
            min-height: 100px;
            font-size: 18px;
            line-height: 1.5;
        }

        .current-label {
            color: #667eea;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 14px;
        }

        .history-text {
            background: #f5f5f5;
            border-radius: 10px;
            padding: 15px;
            max-height: 300px;
            overflow-y: auto;
            font-size: 14px;
            line-height: 1.5;
        }

        .history-entry {
            padding: 8px;
            border-bottom: 1px solid #e0e0e0;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .timestamp {
            color: #667eea;
            font-size: 11px;
            font-weight: bold;
            display: inline-block;
            margin-right: 10px;
        }

        .level-indicator {
            width: 100%;
            height: 4px;
            background: #e0e0e0;
            border-radius: 2px;
            margin-top: 10px;
            overflow: hidden;
        }

        .level-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.05s;
        }

        @media (max-width: 768px) {
            .record-btn {
                width: 120px;
                height: 120px;
            }

            .record-icon {
                font-size: 48px;
            }

            .current-text {
                font-size: 14px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎤 Voice to Text - Непрерывный режим</h1>
            <p>Нажмите один раз и говорите непрерывно</p>
        </div>

        <div class="main-card">
            <div class="record-section">
                <button class="record-btn" id="recordBtn">
                    <div class="record-icon">🎙️</div>
                </button>
                <div class="timer" id="timer">00:00</div>
                <div class="level-indicator">
                    <div class="level-fill" id="levelFill"></div>
                </div>
                <div class="status idle" id="status">Готов к записи</div>
            </div>

            <div class="result-section">
                <div class="result-header">
                    <h3>📝 Распознавание в реальном времени</h3>
                    <div class="button-group">
                        <button class="btn btn-primary" id="saveBtn">💾 Сохранить всё</button>
                        <button class="btn btn-danger" id="clearBtn">🗑️ Очистить</button>
                        <button class="btn btn-warning" id="copyBtn">📋 Копировать</button>
                    </div>
                </div>

                <div class="transcription-container">
                    <div class="current-label">🎯 Сейчас вы сказали:</div>
                    <div class="current-text" id="currentText">
                        <span style="color: #999;">Ожидание речи...</span>
                    </div>

                    <div class="current-label" style="margin-top: 20px;">📜 Вся история:</div>
                    <div class="history-text" id="historyText">
                        <div style="color: #999; text-align: center;">Здесь будет вся история распознанного текста...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let startTime = null;
        let timerInterval = null;
        let audioContext = null;
        let animationFrame = null;
        let ws = null;
        let totalText = "";

        const recordBtn = document.getElementById('recordBtn');
        const timerElement = document.getElementById('timer');
        const statusElement = document.getElementById('status');
        const levelFill = document.getElementById('levelFill');
        const currentText = document.getElementById('currentText');
        const historyText = document.getElementById('historyText');

        // Подключаем WebSocket для непрерывного распознавания
        function connectWebSocket() {
            ws = new WebSocket(`ws://localhost:8001/ws`);

            ws.onopen = function() {
                console.log('WebSocket connected');
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'transcription') {
                    updateCurrentText(data.text);
                    addToHistory(data.text, data.timestamp);
                }
            };

            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };

            ws.onclose = function() {
                console.log('WebSocket disconnected');
                if (isRecording) {
                    setTimeout(connectWebSocket, 1000);
                }
            };
        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        channelCount: 1,
                        sampleRate: 16000,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });

                // Подключаем WebSocket
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    connectWebSocket();
                    await new Promise(resolve => setTimeout(resolve, 500));
                }

                // Настраиваем визуализатор
                setupAudioVisualization(stream);

                // Создаем MediaRecorder для отправки аудио
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0 && ws && ws.readyState === WebSocket.OPEN) {
                        sendAudioToServer(event.data);
                    }
                };

                mediaRecorder.onstop = () => {
                    stream.getTracks().forEach(track => track.stop());
                    if (animationFrame) {
                        cancelAnimationFrame(animationFrame);
                    }
                    if (audioContext) {
                        audioContext.close();
                    }
                    stopTimer();
                };

                mediaRecorder.start(500); // Отправляем каждые 500 мс для быстрого отклика
                isRecording = true;
                startTimer();

                recordBtn.classList.add('recording');
                statusElement.className = 'status recording';
                statusElement.textContent = '🔴 Слушаю... Говорите непрерывно';

            } catch (error) {
                console.error('Error:', error);
                alert('Ошибка доступа к микрофону');
            }
        }

        function setupAudioVisualization(stream) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);

            const dataArray = new Uint8Array(analyser.frequencyBinCount);

            function updateLevel() {
                if (!isRecording) return;
                analyser.getByteFrequencyData(dataArray);
                let average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
                let percent = (average / 255) * 100;
                levelFill.style.width = percent + '%';
                animationFrame = requestAnimationFrame(updateLevel);
            }

            audioContext.resume();
            updateLevel();
        }

        function sendAudioToServer(audioChunk) {
            const reader = new FileReader();
            reader.onload = function() {
                const base64Data = reader.result.split(',')[1];
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'audio',
                        data: base64Data
                    }));
                }
            };
            reader.readAsDataURL(audioChunk);
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                recordBtn.classList.remove('recording');
                if (timerInterval) {
                    clearInterval(timerInterval);
                    timerInterval = null;
                }
                statusElement.className = 'status idle';
                statusElement.textContent = 'Готов к записи';
            }
        }

        function startTimer() {
            startTime = Date.now();
            if (timerInterval) clearInterval(timerInterval);
            timerInterval = setInterval(() => {
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                timerElement.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }, 1000);
        }

        function stopTimer() {
            if (timerInterval) {
                clearInterval(timerInterval);
                timerInterval = null;
            }
            timerElement.textContent = '00:00';
        }

        function updateCurrentText(text) {
            currentText.innerHTML = `<span style="color: #333;">${escapeHtml(text)}</span>`;
            // Через 2 секунды возвращаем подсказку
            setTimeout(() => {
                if (currentText.innerHTML !== '<span style="color: #999;">Ожидание речи...</span>') {
                    currentText.innerHTML = '<span style="color: #999;">Говорите...</span>';
                }
            }, 2000);
        }

        function addToHistory(text, timestamp) {
            totalText += text + " ";

            // Удаляем заглушку если есть
            if (historyText.children.length === 1 && 
                historyText.children[0].innerHTML.includes('Здесь будет вся история')) {
                historyText.innerHTML = '';
            }

            const entryDiv = document.createElement('div');
            entryDiv.className = 'history-entry';
            entryDiv.innerHTML = `
                <span class="timestamp">${timestamp}</span>
                <span>${escapeHtml(text)}</span>
            `;
            historyText.appendChild(entryDiv);
            historyText.scrollTop = historyText.scrollHeight;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function clearHistory() {
            totalText = "";
            historyText.innerHTML = '<div style="color: #999; text-align: center;">Здесь будет вся история распознанного текста...</div>';
            currentText.innerHTML = '<span style="color: #999;">Ожидание речи...</span>';
            await fetch('/api/clear-history', { method: 'POST' });
        }

        async function saveHistory() {
            const response = await fetch('/api/save-history');
            const data = await response.json();
            alert(data.success ? `✅ Сохранено: ${data.filename}` : '❌ Нет текста для сохранения');
        }

        function copyToClipboard() {
            if (totalText.trim()) {
                navigator.clipboard.writeText(totalText);
                alert('✅ Текст скопирован в буфер обмена!');
            } else {
                alert('❌ Нет текста для копирования');
            }
        }

        // Обработчики событий
        recordBtn.addEventListener('click', () => {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        });

        document.getElementById('clearBtn').addEventListener('click', clearHistory);
        document.getElementById('saveBtn').addEventListener('click', saveHistory);
        document.getElementById('copyBtn').addEventListener('click', copyToClipboard);

        // Подключаем WebSocket при загрузке
        connectWebSocket();

        window.addEventListener('beforeunload', () => {
            if (isRecording) stopRecording();
            if (ws) ws.close();
        });
    </script>
</body>
</html>
"""


# WebSocket для непрерывного распознавания
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get('type') == 'audio':
                    # Декодируем аудио из base64
                    audio_data = base64.b64decode(message['data'])

                    # Конвертируем в numpy array
                    try:
                        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                        if len(audio_array) > 8000:  # Минимум 0.5 секунды
                            # Распознаем
                            result = model.transcribe(audio_array, language="ru", fp16=False)
                            text = result["text"].strip()

                            if text and len(text) > 2:
                                entry = history.add_entry(text)
                                await websocket.send_json({
                                    'type': 'transcription',
                                    'text': text,
                                    'timestamp': entry['timestamp']
                                })
                    except Exception as e:
                        logger.warning(f"Ошибка распознавания чанка: {e}")

            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/", response_class=HTMLResponse)
async def get_microphone_page():
    """Главная страница с микрофоном"""
    return HTML_TEMPLATE


@app.post("/api/clear-history")
async def clear_history():
    """Очистить историю транскрипции"""
    history.clear()
    return {"success": True}


@app.get("/api/save-history")
async def save_history():
    """Сохранить историю в файл"""
    if not history.entries:
        return {"success": False, "message": "Нет текста для сохранения"}

    filepath = history.save_to_file()
    return {"success": True, "filename": filepath.name}


@app.get("/api/download-txt")
async def download_txt():
    """Скачать текст в формате TXT"""
    if not history.entries:
        raise HTTPException(status_code=404, detail="Нет текста для сохранения")

    txt_content = history.current_text
    txt_file = RECORDINGS_DIR / f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(txt_content)

    return FileResponse(txt_file, filename=txt_file.name)


def convert_to_srt(entries):
    """Конвертирует записи в формат SRT"""
    srt = []
    for i, entry in enumerate(entries, 1):
        time_str = entry['timestamp']
        parts = time_str.split(':')

        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
        else:
            hours, minutes, seconds = 0, 0, 0

        start_seconds = hours * 3600 + minutes * 60 + seconds
        end_seconds = start_seconds + 3

        start = format_srt_time(start_seconds)
        end = format_srt_time(end_seconds)

        srt.append(f"{i}\n{start} --> {end}\n{entry['text']}\n")

    return "\n".join(srt)


def format_srt_time(seconds):
    """Форматирует время для SRT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


if __name__ == "__main__":
    import uvicorn
    import base64

    print("=" * 50)
    print("🎤 Voice to Text - Непрерывное распознавание")
    print("=" * 50)
    print("🌐 Откройте в браузере: http://localhost:8001")
    print("🎙️ Нажмите на кнопку ОДИН раз и говорите непрерывно")
    print("💡 Текст будет появляться автоматически")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8001)