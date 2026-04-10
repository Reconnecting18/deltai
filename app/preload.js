const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('deltai', {
  minimize: () => ipcRenderer.send('win-minimize'),
  maximize: () => ipcRenderer.send('win-maximize'),
  close: () => ipcRenderer.send('win-close'),
})
