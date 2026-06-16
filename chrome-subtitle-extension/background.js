let isCapturing = false;
let offscreenCreated = false;

async function createOffscreen() {
  if (offscreenCreated) return;
  try {
    await chrome.offscreen.createDocument({
      url: 'offscreen.html',
      reasons: ['USER_MEDIA'],
      justification: 'Захват звука с вкладки'
    });
    offscreenCreated = true;
    console.log('Offscreen документ создан');
  } catch (e) {
    console.log('Offscreen уже существует');
  }
}

chrome.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
  if (message.type === 'start-capture') {
    await createOffscreen();

    chrome.tabCapture.getMediaStreamId({ targetTabId: message.tabId }, (streamId) => {
      if (chrome.runtime.lastError) {
        console.error('Error:', chrome.runtime.lastError);
        sendResponse({ error: chrome.runtime.lastError.message });
        return;
      }

      chrome.runtime.sendMessage({
        type: 'start-capture-offscreen',
        streamId: streamId,
        tabId: message.tabId
      });
      isCapturing = true;
      sendResponse({ success: true });
    });
    return true;
  }

  if (message.type === 'stop-capture') {
    chrome.runtime.sendMessage({ type: 'stop-capture-offscreen' });
    isCapturing = false;
    sendResponse({ success: true });
  }

  if (message.type === 'subtitle') {
    chrome.tabs.sendMessage(message.tabId, { type: 'subtitle', text: message.text });
  }
});