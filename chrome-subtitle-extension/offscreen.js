// offscreen.js - ВЕРСИЯ ДЛЯ МИКРОФОНА (ТЕСТОВАЯ)
let mediaStream = null;
let audioContext = null;
let isCapturing = false;
let audioBuffer = [];

chrome.runtime.onMessage.addListener(async (msg) => {
  console.log('Offscreen received:', msg.type);

  if (msg.type === 'start-capture-offscreen') {
    await startMicrophoneCapture(msg.tabId);
  }

  if (msg.type === 'stop-capture-offscreen') {
    stopCapture();
  }
});

// ЗАХВАТ С МИКРОФОНА (ПРОСТОЙ ТЕСТ)
async function startMicrophoneCapture(tabId) {
  try {
    console.log('Starting MICROPHONE capture');

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 16000
      }
    });

    console.log('Microphone stream obtained');
    mediaStream = stream;
    isCapturing = true;

    audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(8192, 1, 1);

    source.connect(processor);
    processor.connect(audioContext.destination);

    processor.onaudioprocess = (event) => {
      if (!isCapturing) return;

      const input = event.inputBuffer.getChannelData(0);
      audioBuffer.push(new Float32Array(input));

      // Отправляем каждые 2 секунды
      if (audioBuffer.length >= 2) {
        const chunks = audioBuffer;
        audioBuffer = [];

        let totalLen = 0;
        for (let c of chunks) totalLen += c.length;
        const combined = new Float32Array(totalLen);
        let offset = 0;
        for (let c of chunks) {
          combined.set(c, offset);
          offset += c.length;
        }

        // Нормализация
        let maxVal = 0.001;
        for (let i = 0; i < combined.length; i++) {
          maxVal = Math.max(maxVal, Math.abs(combined[i]));
        }

        if (maxVal < 0.01) {
          return; // Слишком тихо
        }

        // Усиление
        const gain = 3.0;
        for (let i = 0; i < combined.length; i++) {
          combined[i] = Math.min(0.99, Math.max(-0.99, combined[i] * gain));
        }

        sendToServer(combined, tabId);
      }
    };

    await audioContext.resume();
    console.log('Microphone capture active');

    chrome.runtime.sendMessage({
      type: 'subtitle',
      tabId: tabId,
      text: '🎤 Говорите в микрофон...'
    });

  } catch (err) {
    console.error('Microphone error:', err);
    chrome.runtime.sendMessage({
      type: 'subtitle',
      tabId: tabId,
      text: '❌ Ошибка микрофона: ' + err.message
    });
  }
}

async function sendToServer(samples, tabId) {
  const int16 = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }

  const wavBlob = createWavBlob(int16, 16000);
  const formData = new FormData();
  formData.append('audio', wavBlob, 'recording.wav');

  try {
    const response = await fetch('http://localhost:8017/api/recognize', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    console.log('Server response:', data);

    if (data.text && data.text.trim()) {
      chrome.runtime.sendMessage({
        type: 'subtitle',
        tabId: tabId,
        text: data.text
      });
    }
  } catch (err) {
    console.error('Send error:', err);
  }
}

function createWavBlob(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    view.setInt16(offset, samples[i], true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function stopCapture() {
  isCapturing = false;
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  console.log('Capture stopped');
}