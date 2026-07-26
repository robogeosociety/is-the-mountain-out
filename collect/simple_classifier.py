import http.server
import socketserver
import json
import yaml
from pathlib import Path
from typing import Optional
import urllib.parse

import typer

# Minimal HTML for the classifier
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Mountain Classifier</title>
    <style>
        body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; background: #f0f0f0; margin: 0; padding: 20px; }
        .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 900px; width: 100%; }
        .image-container { width: 100%; height: 500px; display: flex; justify-content: center; align-items: center; background: #eee; margin-bottom: 20px; overflow: hidden; }
        img { max-width: 100%; max-height: 100%; object-fit: contain; }
        .controls { display: flex; gap: 10px; justify-content: center; }
        button { padding: 15px 30px; font-size: 18px; cursor: pointer; border: none; border-radius: 4px; color: white; font-weight: bold; }
        .btn-out { background: #28a745; }
        .btn-not { background: #dc3545; }
        .btn-skip { background: #ffc107; color: black; }
        .stats { margin-bottom: 10px; color: #666; display: flex; justify-content: space-between; }
        .path { font-size: 12px; color: #999; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="stats">
            <div id="count-stat">Loading...</div>
            <div id="label-stat"></div>
        </div>
        <div class="image-container">
            <img id="display-image" src="" alt="Capture">
        </div>
        <div class="controls">
            <button class="btn-out" onclick="classify(1)">MOUNTAIN IS OUT (1)</button>
            <button class="btn-not" onclick="classify(0)">NOT OUT (0)</button>
            <button class="btn-skip" onclick="classify(null)">SKIP</button>
        </div>
        <div class="path" id="current-path"></div>
    </div>

    <script>
        let currentIdx = 0;
        let images = [];
        let labels = {};

        async function loadData() {
            const resp = await fetch('/api/data');
            const data = await resp.json();
            images = data.images;
            labels = data.labels;
            render();
        }

        function render() {
            if (currentIdx >= images.length) {
                document.querySelector('.container').innerHTML = "<h1>🎉 All labeled!</h1><p>You can close this tab and the terminal.</p>";
                return;
            }
            const img = images[currentIdx];
            document.getElementById('display-image').src = '/img/' + encodeURIComponent(img.path);
            document.getElementById('current-path').innerText = img.path;
            document.getElementById('count-stat').innerText = `Image ${currentIdx + 1} of ${images.length}`;
            
            const outCount = Object.values(labels).filter(v => v === 1).length;
            const notCount = Object.values(labels).filter(v => v === 0).length;
            document.getElementById('label-stat').innerText = `Out: ${outCount} | Not Out: ${notCount}`;
        }

        async function classify(label) {
            const img = images[currentIdx];
            if (label !== null) {
                labels[img.path] = label;
                await fetch('/api/label', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: img.path, label: label })
                });
            }
            currentIdx++;
            render();
        }

        loadData();
    </script>
</body>
</html>
"""


class ClassifierHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        return  # Silent

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())

        elif self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            # Scan for images
            data_root = Path(self.server.data_root)
            all_imgs = []
            for img_p in sorted(data_root.rglob("*.jpg")):
                rel = str(img_p.relative_to(data_root))
                all_imgs.append({"path": rel})

            # Load existing labels
            labels = {}
            labels_p = data_root / "labels.yaml"
            if labels_p.exists():
                with open(labels_p, "r") as f:
                    labels = yaml.safe_load(f) or {}

            # Filter out already labeled
            unlabeled = [img for img in all_imgs if img["path"] not in labels]

            self.wfile.write(
                json.dumps({"images": unlabeled, "labels": labels}).encode()
            )

        elif self.path.startswith("/img/"):
            rel_path = urllib.parse.unquote(self.path[5:])
            abs_path = Path(self.server.data_root) / rel_path

            if abs_path.exists():
                self.send_response(200)
                self.send_header("Content-type", "image/jpeg")
                self.end_headers()
                with open(abs_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/label":
            content_length = int(self.headers["Content-Length"])
            post_data = json.loads(self.rfile.read(content_length))

            data_root = Path(self.server.data_root)
            labels_p = data_root / "labels.yaml"

            labels = {}
            if labels_p.exists():
                with open(labels_p, "r") as f:
                    labels = yaml.safe_load(f) or {}

            labels[post_data["path"]] = post_data["label"]

            with open(labels_p, "w") as f:
                yaml.safe_dump(labels, f)

            self.send_response(200)
            self.end_headers()


def run_classifier(data_root, port=9999):
    Handler = ClassifierHandler
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        httpd.data_root = data_root
        print(f"🚀 Classifier running at http://localhost:{port}")
        print(f"📁 Source: {data_root}")
        print("Press Ctrl+C to stop.")
        try:
            # Auto-open browser
            import subprocess

            subprocess.run(["open", f"http://localhost:{port}"])
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping...")
            httpd.shutdown()


cli = typer.Typer()


def get_folder_via_picker(title="Select Data Root"):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.focus_force()
    print("Opening folder picker...")
    selected_dir = filedialog.askdirectory(
        title=title, initialdir=str(Path.cwd() / "data")
    )
    root.destroy()
    return selected_dir


@cli.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    folder: Optional[str] = typer.Argument(None),
    port: int = 9999,
    f: Optional[str] = typer.Option(None, "-f"),
):
    if ctx.invoked_subcommand:
        return

    data_root = folder or f
    if not data_root:
        data_root = get_folder_via_picker()
        if not data_root:
            return

    run_classifier(data_root, port)


if __name__ == "__main__":
    cli()
