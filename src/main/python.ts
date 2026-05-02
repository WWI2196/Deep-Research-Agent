import { ChildProcess, spawn } from "child_process";
import * as net from "net";
import * as path from "path";
import * as fs from "fs";

export class PythonManager {
  private process: ChildProcess | null = null;
  private port: number = 0;

  async start(): Promise<number> {
    this.port = await this._findFreePort();

    const backendDir = path.join(__dirname, "../../src/backend");
    const serverPath = path.join(backendDir, "server.py");

    const env = { ...process.env, PYTHONUNBUFFERED: "1" };

    this.process = spawn("python3", [
      "-c", `
import uvicorn
import sys
sys.path.insert(0, "${backendDir.replace(/\\/g, "\\\\")}")
from server import app, set_port
set_port(${this.port})
uvicorn.run(app, host="127.0.0.1", port=${this.port}, log_level="warning")
`
    ], {
      env,
      cwd: path.join(__dirname, "../.."),
      stdio: ["ignore", "pipe", "pipe"],
    });

    this.process.stdout?.on("data", (data: Buffer) => {
      console.log("[python]", data.toString().trim());
    });

    this.process.stderr?.on("data", (data: Buffer) => {
      console.error("[python]", data.toString().trim());
    });

    this.process.on("exit", (code: number | null) => {
      console.log(`Python backend exited with code ${code}`);
    });

    // Wait for server to be ready
    await this._waitForReady();
    return this.port;
  }

  stop(): void {
    if (this.process) {
      this.process.kill();
      this.process = null;
    }
  }

  getPort(): number {
    return this.port;
  }

  private _findFreePort(): Promise<number> {
    return new Promise((resolve, reject) => {
      const server = net.createServer();
      server.listen(0, "127.0.0.1", () => {
        const address = server.address();
        if (address && typeof address === "object") {
          const port = address.port;
          server.close(() => resolve(port));
        } else {
          reject(new Error("Failed to find free port"));
        }
      });
      server.on("error", reject);
    });
  }

  private async _waitForReady(maxAttempts: number = 30): Promise<void> {
    for (let i = 0; i < maxAttempts; i++) {
      try {
        const response = await fetch(`http://127.0.0.1:${this.port}/api/health`);
        if (response.ok) return;
      } catch {
        // not ready yet
      }
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    throw new Error("Python backend did not become ready in time");
  }
}
