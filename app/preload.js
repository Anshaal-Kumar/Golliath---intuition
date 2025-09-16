// app/preload.js - Security Bridge
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // Server communication
    getServerPort: () => ipcRenderer.invoke('get-server-port'),
    
    // File dialogs
    showOpenDialog: (options) => ipcRenderer.invoke('show-open-dialog', options),
    
    // Menu events
    onFileSelected: (callback) => ipcRenderer.on('file-selected', callback),
    
    // Cleanup
    removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel)
});

contextBridge.exposeInMainWorld('appInfo', {
    version: '1.0.0',
    name: 'Digital Twin Intelligence Platform'
});