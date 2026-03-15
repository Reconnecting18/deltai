const { contextBridge, ipcRenderer } = require('electron')
const { ipcMain, BrowserWindow } = require('electron')

process.once('loaded', () => {
  window.addEventListener('message', (event) => {
    if (event.data === 'minimize') {
      ipcRenderer.send('minimize')
    }
    if (event.data === 'maximize') {
      ipcRenderer.send('maximize')
    }
    if (event.data === 'close') {
      ipcRenderer.send('close')
    }
  })
})
