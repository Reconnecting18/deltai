const { app, BrowserWindow, shell, ipcMain, dialog } = require('electron')
const { spawn } = require('child_process')
const path = require('path')
const http = require('http')
const fs = require('fs')
const os = require('os')

const E3N_PROJECT = 'C:\\e3n\\project'
const E3N_PYTHON  = path.join(E3N_PROJECT, 'venv', 'Scripts', 'python.exe')
const PORT        = 8000
const URL         = `http://127.0.0.1:${PORT}`
const REPO_ROOT   = path.resolve(__dirname, '..')
/** Always use %TEMP% so logs exist even if C:\\e3n is read-only or app lives elsewhere. */
const BACKEND_LOG = path.join(os.tmpdir(), 'e3n-backend-live.log')
const BOOT_LOG    = path.join(os.tmpdir(), 'e3n-boot.log')

let backendProcess = null
let mainWindow     = null
let backendLogBuf = ''

function logPathsHint () {
  return (
    `Backend stream:\n${BACKEND_LOG}\n\nBoot log:\n${BOOT_LOG}`
  )
}

function appendBoot (msg, data) {
  const line = JSON.stringify({ t: Date.now(), msg, ...data }) + '\n'
  try {
    fs.appendFileSync(BOOT_LOG, line, 'utf8')
  } catch (e) {
    try {
      process.stderr.write(`[E3N] boot log failed: ${e}\n`)
    } catch (_) {}
  }
}

function mirrorToRepo (baseName, chunk) {
  try {
    fs.appendFileSync(path.join(REPO_ROOT, baseName), chunk, 'utf8')
  } catch (_) {}
}

appendBoot('main.js loaded', { __dirname, REPO_ROOT, tmpdir: os.tmpdir() })

function appendBackendChunk (chunk) {
  const s = chunk.toString()
  backendLogBuf = (backendLogBuf + s).slice(-8000)
  try {
    fs.appendFileSync(BACKEND_LOG, s, 'utf8')
  } catch (e) {
    appendBoot('backend log write failed', { err: String(e) })
  }
  mirrorToRepo('e3n-backend-live.log', s)
}

function startBackend () {
  const pyOk = fs.existsSync(E3N_PYTHON)
  const projOk = fs.existsSync(E3N_PROJECT)
  if (!pyOk) {
    throw new Error(`Python not found at ${E3N_PYTHON} — create the venv in project (see README).`)
  }
  if (!projOk) {
    throw new Error(`E3N project folder not found: ${E3N_PROJECT}`)
  }
  backendProcess = spawn(E3N_PYTHON, [
    '-m', 'uvicorn', 'main:app',
    '--host', '127.0.0.1',
    '--port', String(PORT),
    '--log-level', 'warning',
  ], { cwd: E3N_PROJECT, windowsHide: true })

  backendProcess.on('error', (err) => {
    console.error('[E3N backend] spawn error:', err.message)
  })
  backendProcess.on('exit', (code, signal) => {
    console.error('[E3N backend] process exited', { code, signal })
  })
  const pipe = (buf, label) => {
    appendBackendChunk(`[${label}] ${buf}`)
    const s = buf.toString().trim()
    if (s) console.error(`[E3N backend ${label}]`, s)
  }
  if (backendProcess.stderr) backendProcess.stderr.on('data', (d) => pipe(d, 'stderr'))
  if (backendProcess.stdout) backendProcess.stdout.on('data', (d) => pipe(d, 'stdout'))
}

/** Waits for FastAPI to accept HTTP (Ollama can still be down). */
const PROBE_TIMEOUT_MS = 20000

function waitForBackend (retries = 120) {
  return new Promise((resolve, reject) => {
    let attempts = 0
    const check = () => {
      attempts++
      const req = http.get(URL, (res) => {
        req.setTimeout(0)
        // Drain body so the socket can close cleanly (large index.html / keep-alive).
        res.resume()
        resolve()
      })
      req.on('error', () => {
        if (attempts >= retries) {
          return reject(new Error(`Backend did not respond at ${URL} after ${attempts} attempts`))
        }
        setTimeout(check, 500)
      })
      req.setTimeout(PROBE_TIMEOUT_MS, () => {
        req.destroy()
        if (attempts >= retries) {
          return reject(new Error(`Backend probe timed out repeatedly for ${URL}`))
        }
        setTimeout(check, 500)
      })
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

process.on('uncaughtException', (err) => {
  appendBoot('uncaughtException', { err: String(err && err.stack || err) })
  try {
    dialog.showErrorBox('E3N — uncaught error', `${err && err.message || err}\n\n${logPathsHint()}`)
  } catch (_) {}
})

process.on('unhandledRejection', (reason) => {
  appendBoot('unhandledRejection', { reason: String(reason) })
})

ipcMain.on('win-minimize', () => { if (mainWindow) mainWindow.minimize() })
ipcMain.on('win-maximize', () => {
  if (!mainWindow) return
  mainWindow.isMaximized() ? mainWindow.unmaximize() : mainWindow.maximize()
})
ipcMain.on('win-close', () => { if (mainWindow) mainWindow.close() })

app.whenReady().then(async () => {
  try {
    startBackend()
  } catch (e) {
    dialog.showErrorBox('E3N — backend not started', `${e.message || e}\n\n${logPathsHint()}`)
    app.quit()
    return
  }
  try {
    await waitForBackend()
    createWindow()
  } catch (e) {
    const tail = backendLogBuf.trim().slice(-1200) || '(no backend output captured)'
    console.error('[E3N]', e.message || e)
    console.error('Hint: Ensure Python venv exists at', E3N_PYTHON, 'and project at', E3N_PROJECT)
    console.error('Backend log tail:\n', tail)
    dialog.showErrorBox(
      'E3N — API did not become ready',
      `${e.message || e}\n\nPython: ${E3N_PYTHON}\nProject: ${E3N_PROJECT}\n\nLast backend output:\n${tail}\n\n${logPathsHint()}`,
    )
    app.quit()
  }
})

app.on('window-all-closed', () => {
  if (backendProcess) backendProcess.kill()
  app.quit()
})
