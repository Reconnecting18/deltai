'use strict'

const fs = require('fs')
const path = require('path')

const appRoot = path.join(__dirname, '..')
const electronPkgPath = path.join(appRoot, 'node_modules', 'electron', 'package.json')
const binName = process.platform === 'win32' ? 'electron.cmd' : 'electron'
const electronBinPath = path.join(appRoot, 'node_modules', '.bin', binName)

const hasElectronPackage = fs.existsSync(electronPkgPath)
const hasElectronBin = fs.existsSync(electronBinPath)

if (!hasElectronPackage || !hasElectronBin) {
  console.error(
    'E3N: Electron is not installed in this folder (missing node_modules).\n' +
      'From the app directory, run:\n' +
      '  npm install\n' +
      `\nChecked: ${electronPkgPath}`
  )
  process.exit(1)
}
