// app/preload.js - Fixed Security Bridge
const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
    // Server communication
    getServerPort: () => {
        console.log('Preload: Getting server port...');
        return ipcRenderer.invoke('get-server-port');
    },
    
    // File dialogs
    showOpenDialog: (options) => {
        console.log('Preload: Opening file dialog...');
        return ipcRenderer.invoke('show-open-dialog', options);
    },
    
    // Menu events - using proper event handling
    onFileSelected: (callback) => {
        console.log('Preload: Setting up file selection listener...');
        const wrappedCallback = (event, ...args) => callback(...args);
        ipcRenderer.on('file-selected', wrappedCallback);
        
        // Return cleanup function
        return () => {
            ipcRenderer.removeListener('file-selected', wrappedCallback);
        };
    },
    
    // Cleanup
    removeAllListeners: (channel) => {
        console.log(`Preload: Removing all listeners for ${channel}`);
        ipcRenderer.removeAllListeners(channel);
    },
    
    // Platform info
    platform: process.platform,
    
    // App version info
    versions: {
        node: process.versions.node,
        chrome: process.versions.chrome,
        electron: process.versions.electron
    }
});

contextBridge.exposeInMainWorld('appInfo', {
    version: '1.0.0',
    name: 'Digital Twin Intelligence Platform'
});

// Log when preload script loads
console.log('Preload script loaded successfully');