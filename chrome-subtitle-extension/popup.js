let currentTabId = null;

async function getCurrentTabId() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab.id;
}

document.getElementById('startBtn').addEventListener('click', async () => {
  const tabId = await getCurrentTabId();

  chrome.runtime.sendMessage({ type: 'start-capture', tabId: tabId }, (response) => {
    if (response && response.error) {
      document.getElementById('status').innerHTML = '❌ ' + response.error;
    } else {
      document.getElementById('status').innerHTML = '🔴 Активен';
      document.getElementById('status').style.color = '#4CAF50';
    }
  });
});

document.getElementById('stopBtn').addEventListener('click', () => {
  chrome.runtime.sendMessage({ type: 'stop-capture' });
  document.getElementById('status').innerHTML = '⚪ Не активен';
  document.getElementById('status').style.color = '#f44336';
});