'use strict'

/**
 * Launches Electron with the parent console attached on Windows so PowerShell
 * shows main-process stderr (ELECTRON_ATTACH_CONSOLE).
 */
const { spawn } = require('child_process')
const path = require('path')

process.env.ELECTRON_ATTACH_CONSOLE = '1'

const appDir = path.join(__dirname, '..')
const isWin = process.platform === 'win32'
const electronCmd = isWin ? 'electron.cmd' : 'electron'
const bin = path.join(appDir, 'node_modules', '.bin', electronCmd)

const child = spawn(bin, ['.'], {
  cwd: appDir,
  env: process.env,
  stdio: 'inherit',
  shell: isWin,
})

child.on('exit', (code, signal) => {
  process.exit(code == null ? (signal ? 1 : 0) : code)
})
