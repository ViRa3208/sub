
console.log('Content ready');

let div = null;

function showMessage(text) {
  if (!div) {
    div = document.createElement('div');
    div.style.cssText = `
      position: fixed;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      background: black;
      color: lime;
      font-size: 24px;
      padding: 10px 20px;
      border-radius: 10px;
      z-index: 999999;
      font-family: monospace;
      white-space: nowrap;
    `;
    document.body.appendChild(div);
  }
  div.textContent = text;
  div.style.opacity = '1';
  setTimeout(() => { if (div) div.style.opacity = '0.3'; }, 3000);
}

chrome.runtime.onMessage.addListener((msg) => {
  console.log('Got:', msg.type, msg.text);
  if (msg.type === 'subtitle') {
    showMessage(msg.text);
  }
});