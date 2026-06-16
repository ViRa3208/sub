from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import whisper
import numpy as np
from datetime import datetime
import logging
import io
import wave
import re

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


print("[loading] Загрузка модели Whisper...")
model = whisper.load_model("base")
print("[ok] Модель загружена")


BLACKLIST_PHRASES = [
    "редактор субтитров", "корректор", "динамичная музыка", "спокойная музыка",
    "музыка", "тишина", "шум", "фон", "распознай четко и точно",
    "спасибо за внимание", "все права защищены"
]


def is_valid_text(text):
    """Проверка, является ли текст валидной речью (не мусором)"""
    if not text or len(text) < 2:
        return False

    text_lower = text.lower().strip()

    # Проверяем черный список
    for bad in BLACKLIST_PHRASES:
        if bad in text_lower:
            return False

    # Проверяем, что это не просто одно короткое слово-паразит
    parasites = ['э', 'м', 'ээ', 'мм', 'хм', 'ну', 'вот', 'типа', 'короче']
    if text_lower in parasites:
        return False

    # Должны быть гласные (осмысленная речь)
    vowels = 'аеёиоуыэюя'
    vowel_count = sum(1 for c in text_lower if c in vowels)
    if vowel_count < 2 and len(text) < 5:
        return False

    return True


HTML_CODE = '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Голосовое распознавание</title>
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

        .mic-container {
            text-align: center;
            margin-bottom: 30px;
        }

        .mic-btn {
            width: 160px;
            height: 160px;
            border-radius: 50%;
            background: #3b82f6;
            border: none;
            cursor: pointer;
            font-size: 64px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s;
            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3);
        }

        .mic-btn:hover {
            transform: scale(1.05);
            background: #2563eb;
        }

        .mic-btn.recording {
            animation: pulse 1.5s infinite;
            background: #ef4444;
            box-shadow: 0 8px 20px rgba(239, 68, 68, 0.3);
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.08); }
            100% { transform: scale(1); }
        }

        .status {
            text-align: center;
            padding: 12px;
            border-radius: 10px;
            margin-top: 16px;
            font-weight: 500;
            font-size: 14px;
        }

        .status.idle {
            background: #0f0f2a;
            color: #718096;
            border: 1px solid #2a3a5e;
        }

        .status.recording {
            background: #7f1a1a;
            color: #fecaca;
            border: 1px solid #991b1b;
        }

        .status.processing {
            background: #92400e;
            color: #fef3c7;
            border: 1px solid #b45309;
        }

        .energy-bar {
            width: 100%;
            height: 6px;
            background: #2a3a5e;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 16px;
        }

        .energy-fill {
            width: 0%;
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #ef4444);
            border-radius: 3px;
            transition: width 0.05s;
        }

        .live-box {
            background: #0f0f2a;
            border-radius: 12px;
            padding: 20px;
            margin-top: 24px;
            border: 1px solid #2a3a5e;
        }

        .live-box strong {
            color: #3b82f6;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .live-text {
            font-size: 20px;
            padding: 20px;
            background: #16213e;
            border-radius: 10px;
            border: 1px solid #2a3a5e;
            min-height: 100px;
            margin-top: 12px;
            color: #cbd5e1;
            line-height: 1.5;
        }

        .history-box {
            background: #0f0f2a;
            border-radius: 12px;
            padding: 20px;
            margin-top: 24px;
            border: 1px solid #2a3a5e;
        }

        .history-box strong {
            color: #3b82f6;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .history-list {
            max-height: 300px;
            overflow-y: auto;
            margin-top: 12px;
        }

        .history-item {
            padding: 12px;
            border-bottom: 1px solid #1e2a4a;
            font-size: 14px;
            animation: fadeIn 0.3s;
            color: #cbd5e1;
        }

        .history-item:last-child {
            border-bottom: none;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }

        .timestamp {
            color: #3b82f6;
            font-size: 11px;
            margin-right: 12px;
            font-weight: 600;
        }

        .buttons {
            display: flex;
            gap: 12px;
            justify-content: center;
            margin-top: 24px;
            flex-wrap: wrap;
        }

        .btn {
            padding: 10px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
        }

        .btn:hover {
            transform: translateY(-2px);
        }

        .btn-save {
            background: #10b981;
            color: white;
        }

        .btn-save:hover {
            background: #059669;
        }

        .btn-clear {
            background: #6b7280;
            color: white;
        }

        .btn-clear:hover {
            background: #4b5563;
        }

        .btn-stop {
            background: #ef4444;
            color: white;
        }

        .btn-stop:hover {
            background: #dc2626;
        }

        .debug {
            text-align: center;
            font-size: 12px;
            color: #718096;
            margin-top: 20px;
            padding: 10px;
            background: #0f0f2a;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Голосовое распознавание</h1>
            <p>Нажмите на микрофон и говорите — текст появится на экране</p>
        </div>

        <div class="card">
            <div class="mic-container">
                <button class="mic-btn" id="micBtn">🎙️</button>
                <div class="status idle" id="status">Готов к записи</div>
                <div class="energy-bar">
                    <div class="energy-fill" id="energyFill"></div>
                </div>
            </div>

            <div class="live-box">
                <strong>Распознаётся</strong>
                <div class="live-text" id="liveText">—</div>
            </div>

            <div class="history-box">
                <strong>История распознанного</strong>
                <div class="history-list" id="historyList"></div>
            </div>

            <div class="buttons">
                <button class="btn btn-save" id="saveBtn">Сохранить всё</button>
                <button class="btn btn-clear" id="clearBtn">Очистить историю</button>
                <button class="btn btn-stop" id="stopBtn">Остановить запись</button>
            </div>

            <div class="debug" id="debug">Нажмите на микрофон и начинайте говорить</div>
        </div>
    </div>

    <script>
        let mediaStream = null;
        let isListening = false;
        let audioContext = null;
        let sourceNode = null;
        let processorNode = null;
        let animationId = null;
        let analyserNode = null;
        let lastTranscript = "";
        let fullText = "";
        let audioBuffer = [];

        const micBtn = document.getElementById('micBtn');
        const statusDiv = document.getElementById('status');
        const liveText = document.getElementById('liveText');
        const historyList = document.getElementById('historyList');
        const debugDiv = document.getElementById('debug');
        const energyFill = document.getElementById('energyFill');

        async function startListening() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }
                });

                mediaStream = stream;
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

                analyserNode = audioContext.createAnalyser();
                analyserNode.fftSize = 256;
                const dataArray = new Uint8Array(analyserNode.frequencyBinCount);

                sourceNode = audioContext.createMediaStreamSource(stream);
                sourceNode.connect(analyserNode);

                function updateMeter() {
                    if (!isListening) return;
                    analyserNode.getByteFrequencyData(dataArray);
                    let avg = dataArray.reduce((a,b) => a+b, 0) / dataArray.length;
                    let percent = Math.min(100, (avg / 255) * 100);
                    energyFill.style.width = percent + '%';
                    animationId = requestAnimationFrame(updateMeter);
                }

                audioContext.resume();
                updateMeter();

                processorNode = audioContext.createScriptProcessor(8192, 1, 1);
                sourceNode.connect(processorNode);
                processorNode.connect(audioContext.destination);

                processorNode.onaudioprocess = (event) => {
                    if (!isListening) return;
                    const inputData = event.inputBuffer.getChannelData(0);
                    const samples = new Float32Array(inputData.length);
                    samples.set(inputData);
                    audioBuffer.push(samples);

                    if (audioBuffer.length >= 2) {
                        const chunks = audioBuffer;
                        audioBuffer = [];
                        sendAudio(chunks);
                    }
                };

                isListening = true;
                micBtn.classList.add('recording');
                statusDiv.className = 'status recording';
                statusDiv.textContent = 'Запись активна';
                debugDiv.innerHTML = 'Говорите четко, не торопитесь';
                liveText.innerHTML = '<span style="color:#718096;">Слушаю...</span>';

            } catch(e) {
                debugDiv.innerHTML = 'Ошибка доступа к микрофону: ' + e.message;
                alert('Ошибка доступа к микрофону. Проверьте разрешения.');
            }
        }

        async function sendAudio(chunks) {
            let totalLength = 0;
            for (let chunk of chunks) {
                totalLength += chunk.length;
            }

            const combined = new Float32Array(totalLength);
            let offset = 0;
            for (let chunk of chunks) {
                combined.set(chunk, offset);
                offset += chunk.length;
            }

            // Нормализация
            let maxVal = 0.001;
            for (let i = 0; i < combined.length; i++) {
                maxVal = Math.max(maxVal, Math.abs(combined[i]));
            }
            for (let i = 0; i < combined.length; i++) {
                combined[i] = combined[i] / maxVal;
            }

            const int16Data = new Int16Array(combined.length);
            for (let i = 0; i < combined.length; i++) {
                let s = Math.max(-1, Math.min(1, combined[i]));
                int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }

            const wav = createWav(int16Data, 16000);

            const formData = new FormData();
            formData.append('audio', new Blob([wav], { type: 'audio/wav' }), 'recording.wav');

            try {
                const response = await fetch('/api/transcribe', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.text && data.text.trim() && data.text !== lastTranscript) {
                    lastTranscript = data.text;
                    showText(data.text);
                }
            } catch(e) {
                console.error('Error:', e);
            }
        }

        function createWav(samples, sampleRate) {
            const buffer = new ArrayBuffer(44 + samples.length * 2);
            const view = new DataView(buffer);

            writeString(view, 0, 'RIFF');
            view.setUint32(4, 36 + samples.length * 2, true);
            writeString(view, 8, 'WAVE');
            writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, 1, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true);
            view.setUint16(32, 2, true);
            view.setUint16(34, 16, true);
            writeString(view, 36, 'data');
            view.setUint32(40, samples.length * 2, true);

            let offset = 44;
            for (let i = 0; i < samples.length; i++) {
                view.setInt16(offset, samples[i], true);
                offset += 2;
            }

            return buffer;
        }

        function writeString(view, offset, str) {
            for (let i = 0; i < str.length; i++) {
                view.setUint8(offset + i, str.charCodeAt(i));
            }
        }

        function showText(text) {
            liveText.innerHTML = '✨ ' + escapeHtml(text);

            const time = new Date().toLocaleTimeString();
            fullText += text + " ";

            const div = document.createElement('div');
            div.className = 'history-item';
            div.innerHTML = '<span class="timestamp">' + time + '</span> ' + escapeHtml(text);
            historyList.insertBefore(div, historyList.firstChild);

            setTimeout(() => {
                if (liveText.innerHTML.includes('✨')) {
                    liveText.innerHTML = '<span style="color:#718096;">Слушаю...</span>';
                }
            }, 2500);
        }

        function stopListening() {
            if (isListening) {
                isListening = false;

                if (processorNode) {
                    processorNode.disconnect();
                    processorNode = null;
                }
                if (sourceNode) {
                    sourceNode.disconnect();
                    sourceNode = null;
                }
                if (analyserNode) {
                    analyserNode.disconnect();
                    analyserNode = null;
                }
                if (audioContext) {
                    audioContext.close();
                    audioContext = null;
                }
                if (mediaStream) {
                    mediaStream.getTracks().forEach(track => track.stop());
                    mediaStream = null;
                }
                if (animationId) {
                    cancelAnimationFrame(animationId);
                    animationId = null;
                }

                micBtn.classList.remove('recording');
                statusDiv.className = 'status idle';
                statusDiv.textContent = 'Готов к записи';
                debugDiv.innerHTML = 'Запись остановлена';
                energyFill.style.width = '0%';
            }
        }

        function escapeHtml(t) {
            const div = document.createElement('div');
            div.textContent = t;
            return div.innerHTML;
        }

        async function saveAll() {
            if (!fullText.trim()) { 
                debugDiv.innerHTML = 'Нет текста для сохранения';
                return; 
            }
            const resp = await fetch('/api/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: fullText })
            });
            const data = await resp.json();
            if (data.success) {
                debugDiv.innerHTML = 'Сохранено: ' + data.filename;
            } else {
                debugDiv.innerHTML = 'Ошибка сохранения';
            }
        }

        function clearAll() {
            fullText = '';
            lastTranscript = '';
            historyList.innerHTML = '';
            liveText.innerHTML = '—';
            debugDiv.innerHTML = 'История очищена';
            fetch('/api/clear', { method: 'POST' });
        }

        micBtn.addEventListener('click', () => {
            if (isListening) {
                stopListening();
            } else {
                startListening();
            }
        });

        document.getElementById('stopBtn').addEventListener('click', stopListening);
        document.getElementById('saveBtn').addEventListener('click', saveAll);
        document.getElementById('clearBtn').addEventListener('click', clearAll);

        window.addEventListener('beforeunload', () => {
            if (isListening) stopListening();
        });
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
            return {"text": ""}

        # Читаем WAV
        with io.BytesIO(data) as buf:
            with wave.open(buf, 'rb') as wav:
                n_frames = wav.getnframes()
                frames = wav.readframes(n_frames)
                audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_array) < 8000:
            return {"text": ""}

        # Проверяем громкость
        energy = np.mean(np.abs(audio_array))
        if energy < 0.008:
            return {"text": ""}

        # Распознаем
        result = model.transcribe(
            audio_array,
            language="ru",
            fp16=False,
            temperature=0.0,
            no_speech_threshold=0.5
        )

        text = result["text"].strip()

        # Очистка текста
        text = re.sub(r'\s+', ' ', text)

        logger.info(f"Raw: '{text}'")

        # Фильтруем
        if is_valid_text(text):
            # Дополнительно убираем типичные галлюцинации модели
            if "субтитров" in text.lower() or "корректор" in text.lower():
                # Проверяем, начинается ли фраза с реальной речи после этого
                parts = re.split(r'[.!?]', text)
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 5 and not any(b in part.lower() for b in ["субтитров", "корректор"]):
                        text = part
                        break
                else:
                    return {"text": ""}

            logger.info(f"Accepted: '{text}'")
            return {"text": text}

        return {"text": ""}

    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return {"text": ""}


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
    print("Голосовое распознавание - тёмная тема")
    print("=" * 60)
    print("Адрес: http://localhost:8018")
    print("Работает через микрофон, очищает текст от мусора")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8018)