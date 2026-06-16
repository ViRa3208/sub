from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import whisper
import numpy as np
from datetime import datetime
import logging
import subprocess
import tempfile
import os
import io
import wave
import struct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice to Text")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RECORDINGS_DIR = Path("voice_recordings")
RECORDINGS_DIR.mkdir(exist_ok=True)



def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        logger.info("FFmpeg найден")
        return True
    except:
        logger.error("FFmpeg НЕ найден!")
        return False


HAS_FFMPEG = check_ffmpeg()

# Загружаем модель побольше (base вместо tiny)
print("🔄 Загрузка модели Whisper (base)...")
model = whisper.load_model("base")  # base более точная, чем tiny
print("✅ Модель загружена!")


def remove_noise(audio_array, sample_rate=16000):
    """Простое шумоподавление"""
    # Убираем DC смещение
    audio_array = audio_array - np.mean(audio_array)

    # Применяем фильтр низких частот (убираем высокочастотный шум)
    from scipy import signal
    b, a = signal.butter(4, 0.5, btype='low', analog=False)
    audio_array = signal.filtfilt(b, a, audio_array)

    return audio_array


def detect_speech(audio_array, threshold=0.02):
    """Определяет, есть ли речь в аудио"""
    energy = np.mean(np.abs(audio_array))
    return energy > threshold, energy


def convert_webm_to_wav(webm_data):
    """Конвертация WebM в WAV"""
    if not HAS_FFMPEG:
        return None

    temp_webm = None
    temp_wav = None

    try:
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(webm_data)
            temp_webm = f.name

        temp_wav = temp_webm.replace('.webm', '.wav')

        cmd = ['ffmpeg', '-i', temp_webm, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', temp_wav]
        subprocess.run(cmd, capture_output=True, check=True)

        with open(temp_wav, 'rb') as f:
            return f.read()
    except Exception as e:
        logger.error(f"FFmpeg error: {e}")
        return None
    finally:
        if temp_webm and os.path.exists(temp_webm):
            os.unlink(temp_webm)
        if temp_wav and os.path.exists(temp_wav):
            os.unlink(temp_wav)


# HTML код с улучшенным интерфейсом
HTML_CODE = '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice to Text - Улучшенный</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 30px;
            max-width: 900px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { text-align: center; color: #333; }
        .subtitle { text-align: center; color: #666; margin: 10px 0 30px; }
        .mic-container { text-align: center; margin: 30px 0; }
        .record-btn {
            width: 180px;
            height: 180px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            font-size: 70px;
        }
        .record-btn:hover { transform: scale(1.05); }
        .record-btn.recording {
            animation: pulse 1s infinite;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .status {
            text-align: center;
            padding: 12px;
            margin: 20px 0;
            border-radius: 10px;
            font-weight: bold;
        }
        .status.idle { background: #e0e0e0; color: #666; }
        .status.recording { background: #ff6b6b; color: white; }
        .status.processing { background: #ffa500; color: white; }
        .energy-bar {
            width: 100%;
            height: 6px;
            background: #e0e0e0;
            border-radius: 3px;
            margin: 10px 0;
            overflow: hidden;
        }
        .energy-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #ff9800, #f44336);
            width: 0%;
            transition: width 0.1s;
        }
        .result {
            background: #f8f9ff;
            border-radius: 15px;
            padding: 20px;
            min-height: 120px;
            margin-top: 20px;
        }
        .result-label { color: #667eea; font-weight: bold; margin-bottom: 10px; }
        .result-text { font-size: 20px; line-height: 1.5; color: #333; font-weight: 500; }
        .history {
            background: #f5f5f5;
            border-radius: 15px;
            padding: 15px;
            max-height: 250px;
            overflow-y: auto;
            margin-top: 20px;
        }
        .history-item { 
            padding: 10px; 
            border-bottom: 1px solid #ddd; 
            font-size: 14px;
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .timestamp { color: #667eea; font-size: 11px; margin-right: 10px; font-weight: bold; }
        .buttons {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 20px;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: 0.2s;
        }
        .btn-save { background: #4CAF50; color: white; }
        .btn-clear { background: #ff6b6b; color: white; }
        .btn-copy { background: #ff9800; color: white; }
        .btn:hover { transform: translateY(-2px); }
        .debug {
            margin-top: 15px;
            padding: 10px;
            background: #f0f0f0;
            border-radius: 10px;
            font-family: monospace;
            font-size: 11px;
            color: #666;
        }
        .confidence {
            font-size: 12px;
            color: #4CAF50;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Voice to Text</h1>
        <div class="subtitle">Нажмите и ГОВОРИТЕ ЧЕТКО, отпустите - появится текст</div>

        <div class="mic-container">
            <button class="record-btn" id="recordBtn">🎙️</button>
        </div>

        <div class="energy-bar">
            <div class="energy-fill" id="energyFill"></div>
        </div>

        <div class="status idle" id="status">Готов</div>

        <div class="result">
            <div class="result-label">📝 Распознано:</div>
            <div class="result-text" id="resultText">—</div>
            <div class="confidence" id="confidence"></div>
        </div>

        <div class="history">
            <div class="result-label">📜 История (последние фразы):</div>
            <div id="historyList"></div>
        </div>

        <div class="buttons">
            <button class="btn btn-save" id="saveBtn">💾 Сохранить всё</button>
            <button class="btn btn-copy" id="copyBtn">📋 Копировать всё</button>
            <button class="btn btn-clear" id="clearBtn">🗑️ Очистить</button>
        </div>

        <div class="debug" id="debug">💡 Нажмите и удерживайте кнопку, говорите ЧЕТКО, отпустите</div>
    </div>

    <script>
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let animationId = null;
        let audioContext = null;

        const recordBtn = document.getElementById('recordBtn');
        const statusDiv = document.getElementById('status');
        const resultText = document.getElementById('resultText');
        const historyList = document.getElementById('historyList');
        const debugDiv = document.getElementById('debug');
        const energyFill = document.getElementById('energyFill');
        const confidenceDiv = document.getElementById('confidence');

        let fullHistory = "";
        let lastText = "";

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }
                });

                // Визуализация уровня звука
                setupEnergyMeter(stream);

                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    stream.getTracks().forEach(t => t.stop());
                    if (animationId) cancelAnimationFrame(animationId);
                    if (audioContext) audioContext.close();

                    statusDiv.className = 'status processing';
                    statusDiv.textContent = '🔄 Распознавание...';
                    debugDiv.innerHTML = 'Отправка на сервер...';

                    if (audioChunks.length) {
                        const blob = new Blob(audioChunks, { type: 'audio/webm' });
                        await sendAudio(blob);
                    }

                    recordBtn.classList.remove('recording');
                    statusDiv.className = 'status idle';
                    statusDiv.textContent = 'Готов';
                    debugDiv.innerHTML = 'Готов. Нажмите и удерживайте для записи.';
                    energyFill.style.width = '0%';
                };

                mediaRecorder.start();
                isRecording = true;
                recordBtn.classList.add('recording');
                statusDiv.className = 'status recording';
                statusDiv.textContent = '🔴 Запись... Говорите!';
                debugDiv.innerHTML = '🔴 ГОВОРИТЕ четко, не торопитесь!';

            } catch(e) {
                debugDiv.innerHTML = '❌ Ошибка: ' + e.message;
                alert('Ошибка доступа к микрофону');
            }
        }

        function setupEnergyMeter(stream) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);

            const dataArray = new Uint8Array(analyser.frequencyBinCount);

            function updateEnergy() {
                if (!isRecording) return;
                analyser.getByteFrequencyData(dataArray);
                let avg = dataArray.reduce((a,b) => a + b, 0) / dataArray.length;
                let percent = (avg / 255) * 100;
                energyFill.style.width = percent + '%';
                animationId = requestAnimationFrame(updateEnergy);
            }

            audioContext.resume();
            updateEnergy();
        }

        async function sendAudio(blob) {
            const form = new FormData();
            form.append('audio', blob);

            try {
                const resp = await fetch('/api/transcribe', { method: 'POST', body: form });
                const data = await resp.json();
                debugDiv.innerHTML = 'Ответ: "' + (data.text || 'пусто') + '"';

                if (data.text && data.text.trim() && data.text !== lastText) {
                    lastText = data.text;
                    addToHistory(data.text);
                    if (data.confidence) {
                        confidenceDiv.innerHTML = `🎯 Уверенность: ${data.confidence}`;
                    }
                } else if (data.error) {
                    resultText.innerHTML = '<span style="color:#ff6b6b;">❌ ' + data.error + '</span>';
                    confidenceDiv.innerHTML = '';
                } else {
                    resultText.innerHTML = '<span style="color:#999;">😕 Не распознано. Попробуйте говорить громче и четче.</span>';
                    confidenceDiv.innerHTML = '';
                }
            } catch(e) {
                debugDiv.innerHTML = '❌ Ошибка: ' + e.message;
            }
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
            }
        }

        function addToHistory(text) {
            resultText.innerHTML = '<span style="color:#333; font-size:22px;">✨ ' + escapeHtml(text) + '</span>';
            const time = new Date().toLocaleTimeString();
            fullHistory += text + " ";

            const div = document.createElement('div');
            div.className = 'history-item';
            div.innerHTML = '<span class="timestamp">' + time + '</span> ' + escapeHtml(text);
            historyList.insertBefore(div, historyList.firstChild);

            // Ограничиваем историю 20 записями
            while (historyList.children.length > 20) {
                historyList.removeChild(historyList.lastChild);
            }

            setTimeout(() => {
                if (resultText.innerHTML.includes('✨')) {
                    resultText.innerHTML = '<span style="color:#999;">Говорите...</span>';
                    confidenceDiv.innerHTML = '';
                }
            }, 3000);
        }

        function escapeHtml(t) {
            const div = document.createElement('div');
            div.textContent = t;
            return div.innerHTML;
        }

        async function saveHistory() {
            if (!fullHistory.trim()) { alert('Нет текста'); return; }
            const resp = await fetch('/api/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: fullHistory })
            });
            const data = await resp.json();
            alert(data.success ? '✅ Сохранено: ' + data.filename : '❌ Ошибка');
        }

        function copyHistory() {
            if (!fullHistory.trim()) { alert('Нет текста'); return; }
            navigator.clipboard.writeText(fullHistory);
            alert('✅ Скопировано');
        }

        function clearHistory() {
            fullHistory = '';
            lastText = '';
            historyList.innerHTML = '';
            resultText.innerHTML = '<span style="color:#999;">—</span>';
            confidenceDiv.innerHTML = '';
            fetch('/api/clear', { method: 'POST' });
        }

        recordBtn.addEventListener('mousedown', startRecording);
        recordBtn.addEventListener('mouseup', stopRecording);
        recordBtn.addEventListener('mouseleave', stopRecording);
        recordBtn.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
        recordBtn.addEventListener('touchend', (e) => { e.preventDefault(); stopRecording(); });

        document.getElementById('saveBtn').onclick = saveHistory;
        document.getElementById('copyBtn').onclick = copyHistory;
        document.getElementById('clearBtn').onclick = clearHistory;
    </script>
</body>
</html>'''


@app.get("/")
async def root():
    return HTMLResponse(HTML_CODE)


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        data = await audio.read()

        if len(data) < 2000:
            return {"text": "", "error": f"Слишком коротко ({len(data)} байт)"}

        # Конвертируем в WAV
        wav_data = convert_webm_to_wav(data)

        if wav_data is None:
            # Пробуем прямой метод
            try:
                audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            except:
                return {"text": "", "error": "Ошибка конвертации (нужен FFmpeg)"}
        else:
            # Читаем WAV
            with io.BytesIO(wav_data) as buf:
                with wave.open(buf, 'rb') as wav:
                    frames = wav.readframes(wav.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_array) < 4000:
            return {"text": "", "error": "Аудио слишком короткое"}

        # Проверяем энергию (есть ли речь)
        energy = np.mean(np.abs(audio_array))
        logger.info(f"Энергия аудио: {energy:.4f}")

        if energy < 0.01:
            return {"text": "", "error": f"Слишком тихо (энергия={energy:.3f})"}

        # Нормализуем громкость
        if energy > 0:
            audio_array = audio_array / (energy + 0.1)
            audio_array = np.clip(audio_array, -1, 1)

        # Распознаем с разными параметрами
        result = model.transcribe(
            audio_array,
            language="ru",
            fp16=False,
            temperature=0.0,
            compression_ratio_threshold=2.0,
            logprob_threshold=-1.0,
            no_speech_threshold=0.3
        )

        text = result["text"].strip()
        logger.info(f"Распознано: '{text}'")

        # Фильтруем мусорные слова
        garbage = [
            "динамичная музыка", "динамичная", "музыка",
            "ты", "это", "и", "в", "на", "к", "с", "у", "а",
            "так", "вот", "ну", "ой", "эх", "мм", "хм"
        ]

        if text and len(text) >= 2:
            text_lower = text.lower()
            is_garbage = any(g in text_lower for g in garbage if len(g) > 1)
            is_garbage = is_garbage or text_lower in garbage

            if not is_garbage and len(text) > 2:
                # Вычисляем "уверенность" (просто для информации)
                confidence = min(1.0, energy * 10)
                return {"text": text, "confidence": f"{confidence:.0%}"}

        return {"text": "", "error": "Речь не распознана"}

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"text": "", "error": str(e)}


@app.post("/api/save")
async def save(data: dict):
    text = data.get('text', '')
    if not text:
        return {"success": False}
    filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(RECORDINGS_DIR / filename, 'w', encoding='utf-8') as f:
        f.write(text)
    return {"success": True, "filename": filename}


@app.post("/api/clear")
async def clear():
    return {"success": True}


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("🎤 Voice to Text - Улучшенная версия")
    print("=" * 60)
    print("🌐 http://localhost:8012")
    print("📌 Нажмите и УДЕРЖИВАЙТЕ кнопку")
    print("🎙️ ГОВОРИТЕ ЧЕТКО, не торопитесь")
    print("✅ Отпустите - текст появится")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8012)