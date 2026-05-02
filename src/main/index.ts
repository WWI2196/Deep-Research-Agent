import { app, BrowserWindow, dialog } from "electron";
import * as path from "path";
import { PythonManager } from "./python";

let mainWindow: BrowserWindow | null = null;
let pythonManager: PythonManager | null = null;

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    titleBarStyle: "hiddenInset",
    title: "Deep Research",
    backgroundColor: "#08080f",
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
    show: false,
  });

  mainWindow.loadFile(path.join(__dirname, "../renderer/index.html"));

  mainWindow.once("ready-to-show", () => {
    mainWindow?.show();
  });

  mainWindow.on("close", async () => {
    // Warn if research is in progress
    const result = await mainWindow?.webContents.executeJavaScript(
      "window.__researchInProgress"
    ).catch(() => false);

    if (result) {
      const { response } = await dialog.showMessageBox(mainWindow!, {
        type: "warning",
        title: "Research in progress",
        message: "A research task is still running. Are you sure you want to quit?",
        buttons: ["Cancel", "Quit"],
        defaultId: 0,
      });
      if (response === 0) {
        (mainWindow as any).__preventClose = true;
        return;
      }
    }
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.whenReady().then(async () => {
  // Start Python backend
  pythonManager = new PythonManager();
  try {
    const port = await pythonManager.start();
    console.log(`Python backend started on port ${port}`);
    // Inject port into renderer
    (global as any).__backendPort = port;
  } catch (err) {
    console.error("Failed to start Python backend:", err);
    dialog.showErrorBox(
      "Backend Error",
      "Failed to start the Python research engine.\n\n" +
      "Please make sure Python 3.12+ is installed and all dependencies are available:\n" +
      "  pip install -r requirements.txt\n\n" +
      `Error: ${err}`
    );
  }

  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("quit", () => {
  pythonManager?.stop();
});
