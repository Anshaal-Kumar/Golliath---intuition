// app/main.js - Fixed Electron Main Process
const { app, BrowserWindow, Menu, ipcMain, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

let mainWindow;
let pythonProcess;
const serverPort = 8501;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1200,
        minHeight: 800,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            enableRemoteModule: false,
            preload: path.join(__dirname, 'preload.js'),
            webSecurity: true
        },
        show: false,
        backgroundColor: '#0f172a',
        titleBarStyle: 'default',
        resizable: true
    });

    mainWindow.setTitle('Digital Twin Intelligence Platform');
    
    // Load the app
    const indexPath = path.join(__dirname, 'renderer', 'index.html');
    console.log('Loading HTML from:', indexPath);
    
    // Check if file exists
    if (!fs.existsSync(indexPath)) {
        console.error('HTML file not found:', indexPath);
        return;
    }
    
    mainWindow.loadFile(indexPath);
    
    mainWindow.once('ready-to-show', () => {
        console.log('Window ready to show');
        mainWindow.show();
        startPythonBackend();
    });

    mainWindow.on('closed', () => {
        console.log('Window closed');
        mainWindow = null;
        stopPythonBackend();
    });

    // Development tools
    if (process.argv.includes('--dev')) {
        mainWindow.webContents.openDevTools();
    }

    // Handle navigation
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        console.log('Blocked navigation to:', url);
        return { action: 'deny' };
    });
}

function startPythonBackend() {
    console.log('Starting Python backend...');
    
    // Check if Python is available
    const pythonCommands = ['python', 'py', 'python3'];
    const scriptPath = path.join(__dirname, '..', 'python', 'api_server.py');
    
    console.log('Python script path:', scriptPath);
    
    if (!fs.existsSync(scriptPath)) {
        console.error('Python script not found:', scriptPath);
        return;
    }
    
    let pythonFound = false;
    
    for (const pythonCmd of pythonCommands) {
        try {
            pythonProcess = spawn(pythonCmd, [scriptPath, '--port', serverPort.toString()], {
                cwd: path.join(__dirname, '..', 'python'),
                stdio: ['ignore', 'pipe', 'pipe']
            });

            pythonProcess.stdout.on('data', (data) => {
                console.log(`Python stdout: ${data}`);
            });

            pythonProcess.stderr.on('data', (data) => {
                console.log(`Python stderr: ${data}`);
            });

            pythonProcess.on('error', (error) => {
                console.error(`Python process error: ${error}`);
                if (!pythonFound) {
                    console.log(`Failed to start with ${pythonCmd}, trying next...`);
                }
            });

            pythonProcess.on('close', (code) => {
                console.log(`Python process exited with code ${code}`);
                pythonProcess = null;
            });

            pythonFound = true;
            console.log(`Python backend started with ${pythonCmd}`);
            break;
        } catch (error) {
            console.log(`Could not start Python with ${pythonCmd}:`, error.message);
            continue;
        }
    }
    
    if (!pythonFound) {
        console.error('Could not start Python backend. Make sure Python is installed.');
        dialog.showErrorBox('Python Required', 'Python is required to run this application. Please install Python and try again.');
    }
}

function stopPythonBackend() {
    if (pythonProcess && !pythonProcess.killed) {
        console.log('Stopping Python backend...');
        pythonProcess.kill('SIGTERM');
        
        // Force kill after 5 seconds if still running
        setTimeout(() => {
            if (pythonProcess && !pythonProcess.killed) {
                console.log('Force killing Python process...');
                pythonProcess.kill('SIGKILL');
            }
        }, 5000);
        
        pythonProcess = null;
    }
}

function createMenu() {
    const template = [
        {
            label: 'File',
            submenu: [
                {
                    label: 'Import Data...',
                    accelerator: 'CmdOrCtrl+O',
                    click: async () => {
                        try {
                            const result = await dialog.showOpenDialog(mainWindow, {
                                properties: ['openFile'],
                                filters: [{ name: 'CSV Files', extensions: ['csv'] }]
                            });
                            
                            if (!result.canceled && result.filePaths.length > 0) {
                                mainWindow.webContents.send('file-selected', result.filePaths[0]);
                            }
                        } catch (error) {
                            console.error('Error opening file dialog:', error);
                        }
                    }
                },
                { type: 'separator' },
                { 
                    label: 'Quit',
                    accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
                    click: () => {
                        app.quit();
                    }
                }
            ]
        },
        {
            label: 'View',
            submenu: [
                { role: 'reload' },
                { role: 'forceReload' },
                { role: 'toggleDevTools' },
                { type: 'separator' },
                { role: 'resetZoom' },
                { role: 'zoomIn' },
                { role: 'zoomOut' },
                { type: 'separator' },
                { role: 'togglefullscreen' }
            ]
        },
        {
            label: 'Help',
            submenu: [
                {
                    label: 'About',
                    click: () => {
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            title: 'About Digital Twin Intelligence',
                            message: 'Digital Twin Intelligence Platform',
                            detail: 'Version 1.0.0\n\nAI-powered data analysis and digital twin modeling platform.'
                        });
                    }
                }
            ]
        }
    ];

    // macOS specific menu adjustments
    if (process.platform === 'darwin') {
        template.unshift({
            label: app.getName(),
            submenu: [
                { role: 'about' },
                { type: 'separator' },
                { role: 'services' },
                { type: 'separator' },
                { role: 'hide' },
                { role: 'hideOthers' },
                { role: 'unhide' },
                { type: 'separator' },
                { role: 'quit' }
            ]
        });
    }

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
}

// IPC Handlers
ipcMain.handle('get-server-port', () => {
    console.log('Returning server port:', serverPort);
    return serverPort;
});

ipcMain.handle('show-open-dialog', async (event, options) => {
    try {
        return await dialog.showOpenDialog(mainWindow, options);
    } catch (error) {
        console.error('Error in show-open-dialog:', error);
        return { canceled: true };
    }
});

// App event handlers
app.whenReady().then(() => {
    console.log('App ready, creating window...');
    
    try {
        createWindow();
        createMenu();
    } catch (error) {
        console.error('Error during app initialization:', error);
    }

    app.on('activate', () => {
        console.log('App activated');
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    console.log('All windows closed');
    stopPythonBackend();
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('before-quit', (event) => {
    console.log('App about to quit');
    stopPythonBackend();
});

app.on('will-quit', () => {
    console.log('App will quit');
    stopPythonBackend();
});

// Error handling
process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

// Prevent new window creation
app.on('web-contents-created', (event, contents) => {
    contents.on('new-window', (event, navigationUrl) => {
        event.preventDefault();
        console.log('Prevented new window:', navigationUrl);
    });
});

console.log('Electron main process loaded');