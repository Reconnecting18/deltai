const { app, BrowserWindow, shell, ipcMain } = require('electron')
const { spawn } = require('child_process')
const path = require('path')
const http = require('http')

const E3N_PROJECT = 'C:\\e3n\\project'
const E3N_PYTHON  = path.join(E3N_PROJECT, 'venv', 'Scripts', 'python.exe')
const PORT        = 8000
const URL         = `http://127.0.0.1:${PORT}`

let backendProcess = null
let mainWindow     = null

function startBackend() {
  backendProcess = spawn(E3N_PYTHON, [
    '-m', 'uvicorn', 'main:app',
    '--host', '127.0.0.1',
    '--port', String(PORT),
    '--log-level', 'warning'
  ], { cwd: E3N_PROJECT, windowsHide: true })
  backendProcess.on('error', (err) => {
    console.error('Failed to start backend:', err.message)
  })
}

function waitForBackend(retries = 30) {
  return new Promise((resolve, reject) => {
    let attempts = 0
    const check = () => {
      attempts++
      const req = http.get(URL, () => resolve())
      req.on('error', (err) => {
        if (attempts >= retries) {
          console.error(`Backend failed to start after ${attempts} attempts:`, err.message)
          return reject(err)
        }
        setTimeout(check, 500)
      })
      req.setTimeout(400, () => { req.destroy(); setTimeout(check, 500) })
    }
    setTimeout(check, 1000)
  })
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    title: 'E3N',
    transparent: true,
    backgroundColor: '#00000000',
    frame: false,
    hasShadow: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      autoplayPolicy: 'no-user-gesture-required',
    },
    show: false,
  })

  mainWindow.webContents.session.clearCache()

  // Grant media permissions (microphone/camera/audio) automatically for localhost
  mainWindow.webContents.session.setPermissionRequestHandler((webContents, permission, callback) => {
    const allowed = ['media', 'microphone', 'audioCapture', 'speaker', 'audioOutput'].includes(permission)
    callback(allowed)
  })
  mainWindow.webContents.session.setPermissionCheckHandler((webContents, permission) => {
    return ['media', 'microphone', 'audioCapture', 'speaker', 'audioOutput'].includes(permission)
  })

  mainWindow.loadURL(URL)
  mainWindow.once('ready-to-show', () => mainWindow.show())
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })
}

ipcMain.on('win-minimize', () => { if (mainWindow) mainWindow.minimize() })
ipcMain.on('win-maximize', () => {
  if (!mainWindow) return
  mainWindow.isMaximized() ? mainWindow.unmaximize() : mainWindow.maximize()
})
ipcMain.on('win-close', () => { if (mainWindow) mainWindow.close() })

app.whenReady().then(async () => {
  startBackend()
  try {
    await waitForBackend()
    createWindow()
  } catch { app.quit() }
})

app.on('window-all-closed', () => {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill('SIGTERM')
    setTimeout(() => {
      if (backendProcess && !backendProcess.killed) backendProcess.kill('SIGKILL')
    }, 3000)
  }
  app.quit()
})
