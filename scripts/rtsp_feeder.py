"""
High-Performance Zero-Latency Async RTSP Substream Feeder for Talangmas AI-Recognition (C Edition).
Features:
  - Threaded real-time RTSP frame grabber with 0-buffer lag (drops stale frames).
  - Asynchronous / multi-threaded SCRFD + ArcFace face detection.
  - Sliced RTSP substream URL configuration support.
  - Dedicated binary pipe isolation (immune to stdout text contamination).
"""

import sys
import os
import time
from datetime import datetime
import struct
import threading
import numpy as np

# Duplicate real binary stdout fd before any library imports or prints
REAL_STDOUT_FD = os.dup(1)

# Redirect standard stdout (fd 1) to stderr (fd 2) to prevent any print/log corruption
os.dup2(2, 1)
sys.stdout = sys.stderr

# Set binary mode on Windows for binary stream descriptor
if sys.platform == "win32":
    import msvcrt
    msvcrt.setmode(REAL_STDOUT_FD, os.O_BINARY)

binary_stream = os.fdopen(REAL_STDOUT_FD, "wb", buffering=0)

MAGIC = 0x53414D54  # 'TMAS' in little endian

class ZeroLatencyRTSPCapture:
    """Threaded RTSP frame grabber that eliminates buffer accumulation and lag."""
    def __init__(self, source):
        import cv2
        self.cv2 = cv2
        self.source = int(source) if isinstance(source, str) and source.isdigit() else source
        
        # Aggressive low-latency flags for FFmpeg RTSP backend
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|probesize;32|analyzeduration;0"
        )
        
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.frame_seq = 0
        self.running = True
        self.is_connected = False
        
        self._connect()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _connect(self):
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            
            is_rtsp = isinstance(self.source, str) and self.source.startswith("rtsp")
            if is_rtsp:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;500000"
            backend = self.cv2.CAP_FFMPEG if is_rtsp else self.cv2.CAP_ANY
            self.cap = self.cv2.VideoCapture(self.source, backend)
            self.cap.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.cap.isOpened():
                self.is_connected = True
                sys.stderr.write(f"[Feeder] Video source connected: {self.source}\n")
            else:
                self.is_connected = False
                sys.stderr.write(f"[Feeder] Warning: Connection failed to {self.source}\n")
        except Exception as e:
            self.is_connected = False
            sys.stderr.write(f"[Feeder] Connection error: {e}\n")

    def _capture_loop(self):
        backoff = 0.5
        while self.running:
            if not self.is_connected or self.cap is None or not self.cap.isOpened():
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 5.0)
                self._connect()
                continue
            
            # Drain internal buffers with grab() to guarantee zero-latency fresh frames
            try:
                grabbed = self.cap.grab()
                if grabbed:
                    backoff = 0.5
                    ret, frame = self.cap.retrieve()
                    if ret and frame is not None:
                        with self.lock:
                            self.latest_frame = frame
                            self.frame_seq += 1
                else:
                    time.sleep(0.005)
                    if self.cap is None or not self.cap.isOpened():
                        self.is_connected = False
            except Exception:
                self.is_connected = False
                time.sleep(0.05)

    def read_fresh(self, last_seq=0):
        with self.lock:
            if self.latest_frame is not None and self.frame_seq != last_seq:
                return True, self.latest_frame.copy(), self.frame_seq
            return False, None, last_seq

    def release(self):
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

class AsyncFaceDetector:
    """Decoupled background face detector to maintain 30+ FPS stream throughput."""
    def __init__(self, face_engine):
        self.engine = face_engine
        self.pending_frame = None
        self.latest_faces = []
        self.lock = threading.Lock()
        self.running = True
        
        if self.engine is not None:
            self.thread = threading.Thread(target=self._detect_loop, daemon=True)
            self.thread.start()

    def submit_frame(self, frame):
        with self.lock:
            self.pending_frame = frame

    def get_faces(self):
        with self.lock:
            return list(self.latest_faces)

    def _detect_loop(self):
        while self.running:
            frame_to_process = None
            with self.lock:
                if self.pending_frame is not None:
                    frame_to_process = self.pending_frame
                    self.pending_frame = None
            
            if frame_to_process is not None and self.engine is not None:
                try:
                    detected = self.engine.detect(frame_to_process)
                    with self.lock:
                        self.latest_faces = detected
                except Exception:
                    pass
            else:
                time.sleep(0.005)

# Global stream buffers and state for Web and Mobile HTTP clients
latest_jpeg_frame = None
jpeg_lock = threading.Lock()
jpeg_cond = threading.Condition(jpeg_lock)
stream_stats = {
    "fps": 0.0,
    "faces_detected": 0,
    "active_names": [],
    "camera_source": "",
    "resolution": ""
}

global_face_engine = None

# In-memory cache for employee profile thumbnails
profile_cache = {}

def get_profile_thumbnail(person_id, size=(75, 75)):
    if not person_id or person_id == "UNKNOWN":
        return None
    if person_id in profile_cache:
        return profile_cache[person_id]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    enroll_dirs = [
        os.path.join(base_dir, "data", "enroll", person_id),
        os.path.join(base_dir, "data", "enroll", person_id.replace(" ", "")),
        os.path.join(base_dir, "data", "enroll", person_id.lower()),
    ]
    img = None
    import cv2
    for d in enroll_dirs:
        if os.path.isdir(d):
            for fname in os.listdir(d):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    fpath = os.path.join(d, fname)
                    img = cv2.imread(fpath)
                    if img is not None:
                        break
        if img is not None:
            break

    if img is not None:
        h, w = img.shape[:2]
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        cropped = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
        resized = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)
        profile_cache[person_id] = resized
        return resized
    else:
        profile_cache[person_id] = None
        return None

def format_display_name(raw_name):
    if not raw_name or raw_name == "UNKNOWN":
        return "Unknown"
    if " " in raw_name:
        return raw_name
    import re
    s = re.sub(r'([A-Z])', r' \1', raw_name).strip()
    return s.title()

from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class MJPEGStreamHandler(BaseHTTPRequestHandler):
    """Serve Live MJPEG stream, snapshot JPEG, telemetry status, and employee sync endpoint."""
    def log_message(self, format, *args):
        return  # Suppress noisy HTTP request logging

    def do_DELETE(self):
        global global_face_engine
        if self.path in ["/register", "/api/v1/employees/sync-ml", "/delete"]:
            try:
                content_length = int(self.headers.get("content-length", 0))
                body_bytes = self.rfile.read(content_length) if content_length > 0 else b""
                name = ""
                old_name = ""
                employee_id = ""
                if body_bytes:
                    import json
                    try:
                        data = json.loads(body_bytes.decode("utf-8"))
                        employee_id = data.get("employeeId", "")
                        name = data.get("name", "")
                        old_name = data.get("oldName", "")
                    except Exception:
                        pass
                
                target_name = name if name else employee_id
                if not target_name and not old_name:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"success": false, "message": "Missing name or employeeId"}')
                    return

                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                enroll_base = os.path.join(base_dir, "data", "enroll")
                
                import shutil
                targets_to_purge = set([t for t in [target_name, old_name, employee_id] if t])
                deleted_any = False
                
                if os.path.exists(enroll_base):
                    for folder in os.listdir(enroll_base):
                        folder_path = os.path.join(enroll_base, folder)
                        if os.path.isdir(folder_path):
                            should_purge = False
                            meta_file = os.path.join(folder_path, "metadata.json")
                            if os.path.exists(meta_file):
                                try:
                                    import json
                                    with open(meta_file, "r") as mf:
                                        mdata = json.load(mf)
                                        if employee_id and mdata.get("employeeId") == employee_id:
                                            should_purge = True
                                except Exception:
                                    pass
                            for target in targets_to_purge:
                                if folder.lower() == target.lower() or target.lower() in folder.lower():
                                    should_purge = True

                            if should_purge:
                                shutil.rmtree(folder_path, ignore_errors=True)
                                sys.stderr.write(f"[Feeder] Deleted enroll directory: {folder_path}\n")
                                deleted_any = True

                # Re-generate embeddings.bin
                try:
                    from scripts.re_enroll_local_insightface import enroll_local
                    enroll_local()
                except Exception:
                    try:
                        import re_enroll_local_insightface
                        re_enroll_local_insightface.enroll_local()
                    except Exception as err:
                        sys.stderr.write(f"[Feeder] Re-enroll error after delete: {err}\n")

                if global_face_engine is not None:
                    global_face_engine.reload_embeddings()

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                resp_str = f'{{"success": true, "message": "Deleted employee from ML successfully", "name": "{target_name}"}}'
                self.wfile.write(resp_str.encode("utf-8"))
            except Exception as err:
                sys.stderr.write(f"[Feeder] Error in DELETE /sync-ml: {err}\n")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(f'{{"success": false, "error": "{err}"}}'.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global global_face_engine
        if self.path in ["/register", "/api/v1/employees/sync-ml"]:
            try:
                content_type = self.headers.get("content-type", "")
                content_length = int(self.headers.get("content-length", 0))
                body_bytes = self.rfile.read(content_length) if content_length > 0 else b""

                name = ""
                old_name = ""
                employee_id = ""
                photo_urls = []
                uploaded_files = []

                if "application/json" in content_type:
                    import json
                    data = json.loads(body_bytes.decode("utf-8"))
                    employee_id = data.get("employeeId", "")
                    name = data.get("name", "")
                    old_name = data.get("oldName", "")
                    photo_urls = data.get("photos", [])
                elif "multipart/form-data" in content_type:
                    boundary = content_type.split("boundary=")[-1].encode("utf-8")
                    parts = body_bytes.split(b"--" + boundary)
                    for part in parts:
                        if b"Content-Disposition" in part:
                            headers_part, body_part = part.split(b"\r\n\r\n", 1)
                            body_part = body_part.rstrip(b"\r\n--")
                            headers_str = headers_part.decode("utf-8", errors="ignore")
                            if 'name="name"' in headers_str:
                                name = body_part.decode("utf-8", errors="ignore").strip()
                            elif 'name="oldName"' in headers_str:
                                old_name = body_part.decode("utf-8", errors="ignore").strip()
                            elif 'name="employeeId"' in headers_str:
                                employee_id = body_part.decode("utf-8", errors="ignore").strip()
                            elif 'name="photos"' in headers_str:
                                uploaded_files.append(body_part)

                target_name = name if name else employee_id
                if not target_name:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"success": false, "message": "Missing name or employeeId"}')
                    return

                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                enroll_base = os.path.join(base_dir, "data", "enroll")
                
                import shutil
                # Purge old_name, matching employeeId from metadata.json, or case-insensitive folders
                targets_to_purge = set([t for t in [target_name, old_name] if t])
                if os.path.exists(enroll_base):
                    for folder in os.listdir(enroll_base):
                        folder_path = os.path.join(enroll_base, folder)
                        if os.path.isdir(folder_path):
                            should_purge = False
                            meta_file = os.path.join(folder_path, "metadata.json")
                            if os.path.exists(meta_file):
                                try:
                                    import json
                                    with open(meta_file, "r") as mf:
                                        mdata = json.load(mf)
                                        if employee_id and mdata.get("employeeId") == employee_id:
                                            should_purge = True
                                except Exception:
                                    pass
                            for target in targets_to_purge:
                                if folder.lower() == target.lower():
                                    should_purge = True

                            if should_purge:
                                shutil.rmtree(folder_path, ignore_errors=True)

                enroll_dir = os.path.join(enroll_base, target_name)
                os.makedirs(enroll_dir, exist_ok=True)

                # Write metadata.json binding employeeId <-> folder name for reliable rename tracking
                if employee_id:
                    import json
                    meta_path = os.path.join(enroll_dir, "metadata.json")
                    with open(meta_path, "w") as mf:
                        json.dump({"employeeId": employee_id, "name": target_name}, mf)

                saved_count = 0
                for idx, img_bytes in enumerate(uploaded_files):
                    if len(img_bytes) > 0:
                        out_path = os.path.join(enroll_dir, f"photo_{idx+1}.jpg")
                        with open(out_path, "wb") as f:
                            f.write(img_bytes)
                        saved_count += 1

                import urllib.request
                for idx, url in enumerate(photo_urls):
                    try:
                        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, timeout=10) as resp:
                            img_data = resp.read()
                            out_path = os.path.join(enroll_dir, f"photo_url_{idx+1}.jpg")
                            with open(out_path, "wb") as f:
                                f.write(img_data)
                            saved_count += 1
                    except Exception as err:
                        sys.stderr.write(f"[Feeder] Failed to download photo URL {url}: {err}\n")

                sys.stderr.write(f"[Feeder] Enrolled photos saved for '{target_name}' ({saved_count} photos).\n")

                # Re-generate embeddings.bin
                try:
                    from scripts.re_enroll_local_insightface import enroll_local
                    enroll_local()
                except Exception:
                    try:
                        import re_enroll_local_insightface
                        re_enroll_local_insightface.enroll_local()
                    except Exception as err:
                        sys.stderr.write(f"[Feeder] Re-enroll error: {err}\n")

                if global_face_engine is not None:
                    global_face_engine.reload_embeddings()

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                resp_str = f'{{"success": true, "message": "Enrolled employee successfully", "name": "{target_name}", "photos_saved": {saved_count}}}'
                self.wfile.write(resp_str.encode("utf-8"))
            except Exception as err:
                sys.stderr.write(f"[Feeder] Error in POST /sync-ml: {err}\n")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(f'{{"success": false, "error": "{err}"}}'.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        global latest_jpeg_frame, stream_stats

        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()

            last_frame_bytes = None
            first_frame = True
            while True:
                with jpeg_cond:
                    if latest_jpeg_frame is last_frame_bytes or latest_jpeg_frame is None:
                        jpeg_cond.wait(timeout=0.033)
                    frame_bytes = latest_jpeg_frame

                if frame_bytes is not None and frame_bytes is not last_frame_bytes:
                    try:
                        prefix = b"--frame\r\n" if first_frame else b"\r\n--frame\r\n"
                        first_frame = False
                        header = prefix + f"Content-Type: image/jpeg\r\nContent-Length: {len(frame_bytes)}\r\n\r\n".encode("ascii")
                        self.wfile.write(header + frame_bytes)
                        self.wfile.flush()
                        last_frame_bytes = frame_bytes
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                        break

        elif self.path == "/snapshot":
            with jpeg_lock:
                frame_bytes = latest_jpeg_frame
            if frame_bytes is not None:
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame_bytes)))
                self.end_headers()
                self.wfile.write(frame_bytes)
            else:
                self.send_response(503)
                self.end_headers()

        elif self.path == "/status":
            import json
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(stream_stats).encode("utf-8"))

        elif self.path == "/detect":
            import json
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            detect_data = {
                "active_names": stream_stats.get("active_names", []),
                "details": stream_stats.get("detect_details", [])
            }
            self.wfile.write(json.dumps(detect_data).encode("utf-8"))

        elif self.path == "/detect-ui":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head>
    <title>Live Detection Data</title>
    <style>
        body { font-family: monospace; background: #1e1e1e; color: #d4d4d4; padding: 20px; }
        pre { white-space: pre-wrap; word-wrap: break-word; }
    </style>
</head>
<body>
    <pre id="json-view">Loading...</pre>
    <script>
        function fetchData() {
            fetch('/detect')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('json-view').innerText = JSON.stringify(data, null, 2);
                })
                .catch(err => {
                    document.getElementById('json-view').innerText = 'Error fetching data: ' + err;
                });
        }
        fetchData();
        setInterval(fetchData, 1000);
    </script>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))

        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Talangmas AI Recognition - Camera Stream (C Edition)</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #090d16;
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 16px;
        }}
        .window {{
            background: #111827;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #1f293d;
            max-width: 1000px;
            width: 100%;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
        }}
        .titlebar {{
            background: #0f172a;
            padding: 10px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #1e293b;
        }}
        .titlebar-left {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 14px;
            font-weight: 600;
            color: #e2e8f0;
        }}
        .dot {{ width: 10px; height: 10px; border-radius: 50%; background: #10b981; display: inline-block; box-shadow: 0 0 8px #10b981; }}
        .badge {{ background: #1e293b; color: #38bdf8; border: 1px solid #334155; padding: 3px 8px; border-radius: 6px; font-size: 12px; font-weight: 500; }}
        .video-container {{
            width: 100%;
            background: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
        }}
        img {{ width: 100%; height: auto; display: block; }}
        .footer {{
            padding: 10px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: #64748b;
            background: #0f172a;
            border-top: 1px solid #1e293b;
        }}
        .overlay {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.6);
            padding: 10px;
            border-radius: 8px;
            color: white;
            font-size: 14px;
            z-index: 10;
        }}
    </style>
</head>
<body>
    <div class="window">
        <div class="titlebar">
            <div class="titlebar-left">
                <span class="dot"></span>
                <span>Talangmas AI Recognition - Camera Stream (C Edition)</span>
            </div>
            <span class="badge">ONNXRuntime Engine</span>
        </div>
        <div class="video-container">
            <img src="/stream" alt="Live AI Camera Feed" />
            <div id="detection-overlay" class="overlay">
                <div>FPS: <span id="fps">0.0</span></div>
                <div>Detected: <span id="active-names">None</span></div>
            </div>
        </div>
        <div class="footer">
            <span>Unified Live Feed &bull; Port {stream_stats.get('port', 8088)}</span>
            <span>Talangmas AI Surveillance Gateway</span>
        </div>
    </div>
    <script>
        // Auto update every 2 seconds to fetch detection data
        setInterval(() => {{
            fetch('http://192.168.77.171:8088/detect')
                .then(res => res.json())
                .then(data => {{
                    document.getElementById('fps').innerText = data.fps || 0;
                    if (data.active_names && data.active_names.length > 0) {{
                        document.getElementById('active-names').innerText = data.active_names.join(', ');
                    }} else {{
                        document.getElementById('active-names').innerText = 'None';
                    }}
                    console.log("Detection data updated:", data);
                }})
                .catch(err => console.error('Error fetching /detect:', err));
        }}, 2000);
    </script>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))

def main():
    global latest_jpeg_frame, stream_stats

    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python rtsp_feeder.py <source> [width] [height] [stream_port]\n")
        sys.exit(1)

    source = sys.argv[1]
    target_width = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    target_height = int(sys.argv[3]) if len(sys.argv) > 3 else 480
    stream_port = int(sys.argv[4]) if len(sys.argv) > 4 else int(os.environ.get("STREAM_PORT", 8088))
    stream_stats["port"] = stream_port

    try:
        import cv2
    except ImportError:
        sys.stderr.write("[Feeder] Error: cv2 not found.\n")
        sys.exit(1)

    # Initialize Face Engine (InsightFace buffalo_l ONNX)
    face_engine = None
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)

        from insightface.app import FaceAnalysis
        from insightface.app.common import Face

        def compute_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

        class StandaloneInsightFaceEngine:
            def __init__(self, device="CPU"):
                self.lock = threading.Lock()
                provider_opts = {"intra_op_num_threads": 2, "inter_op_num_threads": 1}
                self.app = FaceAnalysis(name="buffalo_l", providers=[("CPUExecutionProvider", provider_opts)])
                self.app.prepare(ctx_id=0, det_size=(320, 320))
                self.track_cache = []
                self.emp_ids = []
                self.templates = []
                bin_path = os.path.join(base_dir, "data", "embeddings.bin")
                if os.path.exists(bin_path):
                    with open(bin_path, "rb") as f:
                        magic = f.read(8)
                        if magic == b"FACES1\x00\x00":
                            num_faces, dim = struct.unpack("<ii", f.read(8))
                            for _ in range(num_faces):
                                name_bytes = f.read(128)
                                name = name_bytes.rstrip(b'\x00').decode('utf-8', errors='ignore')
                                vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
                                self.emp_ids.append(name)
                                self.templates.append(vec)
                    sys.stderr.write(f"[Feeder] Loaded {len(self.emp_ids)} faces for /detect API.\n")

            def reload_embeddings(self):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                bin_path = os.path.join(base_dir, "data", "embeddings.bin")
                new_emp_ids = []
                new_templates = []
                if os.path.exists(bin_path):
                    with open(bin_path, "rb") as f:
                        magic = f.read(8)
                        if magic == b"FACES1\x00\x00":
                            num_faces, dim = struct.unpack("<ii", f.read(8))
                            for _ in range(num_faces):
                                name_bytes = f.read(128)
                                name = name_bytes.rstrip(b'\x00').decode('utf-8', errors='ignore')
                                vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
                                new_emp_ids.append(name)
                                new_templates.append(vec)
                with self.lock:
                    self.emp_ids = new_emp_ids
                    self.templates = new_templates
                sys.stderr.write(f"[Feeder] Memory reloaded: {len(self.emp_ids)} faces active for /detect API.\n")

            def match(self, emb, threshold=0.45):
                with self.lock:
                    templates = list(self.templates)
                    emp_ids = list(self.emp_ids)

                if not templates or not emp_ids or emb is None or len(templates) != len(emp_ids):
                    return "", None, 0.0, False
                try:
                    scores = np.dot(templates, emb)
                    if scores is None or len(scores) == 0:
                        return "", None, 0.0, False
                    best_idx = int(np.argmax(scores))
                    if 0 <= best_idx < len(emp_ids) and scores[best_idx] > threshold:
                        return emp_ids[best_idx], emp_ids[best_idx], float(scores[best_idx]), True
                    return "", None, float(scores[best_idx]) if 0 <= best_idx < len(scores) else 0.0, False
                except Exception:
                    return "", None, 0.0, False

            def detect(self, frame):
                bboxes, kpss = self.app.models['detection'].detect(frame, max_num=0, metric='default')
                if bboxes.shape[0] == 0:
                    self.track_cache = []
                    return []

                results = []
                new_cache = []
                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i, 0:4]
                    det_score = bboxes[i, 4]
                    kps = kpss[i] if kpss is not None else None
                    
                    # Find best match in track cache using IoU
                    best_iou = 0
                    best_cached = None
                    for cached in self.track_cache:
                        iou = compute_iou(bbox, cached['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_cached = cached
                            
                    class FaceObj:
                        pass
                    f = FaceObj()
                    f.bbox = bbox
                    f.detection_score = float(det_score)
                    f.blur_score = 50.0
                    f.landmarks = kps
                    
                    # If same face detected, skip heavy ArcFace embedding and reuse previous identity seamlessly
                    if best_iou > 0.4:
                        f.embedding = best_cached['embedding']
                        f.person_id = best_cached['name']
                        f.person_name = best_cached['name']
                        f.similarity = best_cached['sim']
                        
                        new_cache.append({
                            'bbox': bbox,
                            'embedding': f.embedding,
                            'name': f.person_name,
                            'sim': f.similarity,
                            'age': best_cached['age'] + 1
                        })
                    else:
                        # Extract deep features (ArcFace) only for new faces or refresh every 30 frames
                        face_info = Face(bbox=bbox, kps=kps, det_score=det_score)
                        for taskname, model in self.app.models.items():
                            if taskname == 'detection': continue
                            model.get(frame, face_info)
                            
                        f.embedding = face_info.embedding if face_info.embedding is not None else np.zeros(512, dtype=np.float32)
                        pid, name, sim, is_match = self.match(f.embedding)
                        f.person_id = pid
                        f.person_name = name
                        f.similarity = sim
                        
                        new_cache.append({
                            'bbox': bbox,
                            'embedding': f.embedding,
                            'name': name,
                            'sim': sim,
                            'age': 0
                        })
                    results.append(f)
                    
                self.track_cache = new_cache
                return results

        face_engine = StandaloneInsightFaceEngine(device="CPU")
        global global_face_engine
        global_face_engine = face_engine
        sys.stderr.write("[Feeder] Standalone InsightFace Engine initialized successfully.\n")
    except Exception as e:
        sys.stderr.write(f"[Feeder] FaceEngine error: {e}\n")


    # Start HTTP Streaming Server for Frontend & Mobile
    try:
        http_server = ThreadedHTTPServer(("0.0.0.0", stream_port), MJPEGStreamHandler)
        http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
        http_thread.start()
        sys.stderr.write(f"[Stream Server] Live HTTP MJPEG Stream active on http://0.0.0.0:{stream_port}/stream\n")
    except Exception as e:
        sys.stderr.write(f"[Stream Server] Warning starting HTTP server: {e}\n")

    is_piped = not os.isatty(REAL_STDOUT_FD)

    capture = ZeroLatencyRTSPCapture(source)
    detector = AsyncFaceDetector(face_engine)

    stream_stats["camera_source"] = str(source)
    stream_stats["resolution"] = f"{target_width + 340}x{target_height}"

    frame_counter = 0
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0.0

    # Background worker for Shared Memory reading and MJPEG JPEG encoding (Decoupled from feeder loop)
    latest_camera_frame = None
    frame_lock = threading.Lock()

    def mjpeg_encoder_worker():
        nonlocal latest_camera_frame
        global latest_jpeg_frame, jpeg_cond
        shm_local = None
        shm_mutex = None
        last_valid_frame = None
        
        while True:
            frame_to_serve = None

            if shm_local is None:
                try:
                    import mmap
                    shm_local = mmap.mmap(-1, 16 * 1024 * 1024, tagname="Local\\TMAS_PREVIEW_SHM", access=mmap.ACCESS_READ)
                except Exception:
                    shm_local = None

            if shm_local is not None:
                acquired = False
                if shm_mutex is None:
                    try:
                        import ctypes
                        # OpenMutexA(SYNCHRONIZE=0x00100000, False, "Local\\TMAS_SHM_MUTEX")
                        shm_mutex = ctypes.windll.kernel32.OpenMutexA(0x00100000, False, b"Local\\TMAS_SHM_MUTEX")
                    except Exception:
                        shm_mutex = None

                if shm_mutex:
                    try:
                        import ctypes
                        # WaitForSingleObject(hMutex, 10ms)
                        res = ctypes.windll.kernel32.WaitForSingleObject(shm_mutex, 10)
                        if res == 0 or res == 0x00000080: # WAIT_OBJECT_0 (0) or WAIT_ABANDONED (0x80)
                            acquired = True
                    except Exception:
                        acquired = False

                try:
                    shm_local.seek(0)
                    hdr = shm_local.read(24)
                    if len(hdr) == 24:
                        magic, shm_w, shm_h, shm_c, seq = struct.unpack("<IIIIQ", hdr)
                        if magic == 0x53414D54 and 0 < shm_w <= 3840 and 0 < shm_h <= 2160 and shm_c == 3:
                            raw_bytes = shm_local.read(shm_w * shm_h * 3)
                            if len(raw_bytes) == shm_w * shm_h * 3:
                                frame_to_serve = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((shm_h, shm_w, 3))
                                last_valid_frame = frame_to_serve
                except Exception:
                    shm_local = None
                finally:
                    if acquired and shm_mutex:
                        try:
                            import ctypes
                            ctypes.windll.kernel32.ReleaseMutex(shm_mutex)
                        except Exception:
                            pass

            if frame_to_serve is None:
                if last_valid_frame is not None:
                    frame_to_serve = last_valid_frame
                else:
                    with frame_lock:
                        if latest_camera_frame is not None:
                            frame_to_serve = latest_camera_frame.copy()
                            last_valid_frame = frame_to_serve

            if frame_to_serve is not None:
                try:
                    ret_enc, jpeg_bytes = cv2.imencode(".jpg", frame_to_serve, [cv2.IMWRITE_JPEG_QUALITY, 65])
                    if ret_enc:
                        with jpeg_cond:
                            latest_jpeg_frame = jpeg_bytes.tobytes()
                            jpeg_cond.notify_all()
                except Exception:
                    pass

            time.sleep(0.033)

    encoder_thread = threading.Thread(target=mjpeg_encoder_worker, daemon=True)
    encoder_thread.start()

    last_camera_seq = 0
    while True:
        ret, frame, last_camera_seq = capture.read_fresh(last_camera_seq)
        if not ret or frame is None:
            time.sleep(0.002)
            continue

        frame_counter += 1
        fps_counter += 1

        if frame.shape[1] != target_width or frame.shape[0] != target_height:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        with frame_lock:
            latest_camera_frame = frame

        # Asynchronous face detection to maintain 30+ FPS stream throughput
        try:
            detector.submit_frame(frame)
            faces = detector.get_faces()
        except Exception as err:
            sys.stderr.write(f"[Feeder] Error in face detection: {err}\n")
            faces = []
        num_faces = min(len(faces), 16)

        # Calculate FPS
        now = time.time()
        if now - fps_start_time >= 1.0:
            current_fps = fps_counter / (now - fps_start_time)
            fps_counter = 0
            fps_start_time = now
            stream_stats["fps"] = round(current_fps, 1)
            stream_stats["faces_detected"] = num_faces
            stream_stats["active_names"] = [
                getattr(f, 'person_name', 'Unknown') for f in faces if getattr(f, 'person_name', None)
            ]
            stream_stats["detect_details"] = [
                {
                    "name": getattr(f, 'person_name', 'Unknown'),
                    "similarity": getattr(f, 'similarity', 0.0),
                    "bbox": [int(b) for b in getattr(f, 'bbox', [0,0,0,0])]
                } for f in faces if getattr(f, 'person_name', None)
            ]
        
        # Write binary stream to C engine if pipe is attached
        if is_piped:
            try:
                # Build packet header: uint32 magic, uint32 width, uint32 height, uint32 num_faces
                header = struct.pack("<IIII", MAGIC, target_width, target_height, num_faces)
                binary_stream.write(header)

                # Write detected faces metadata & ArcFace embeddings to C Engine
                for i in range(num_faces):
                    f = faces[i]
                    bbox = f.bbox
                    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    det_score = float(f.detection_score)
                    blur_score = float(f.blur_score)
                    
                    if hasattr(f, 'landmarks') and f.landmarks is not None and len(f.landmarks) == 5:
                        lm_x = [float(f.landmarks[k][0]) for k in range(5)]
                        lm_y = [float(f.landmarks[k][1]) for k in range(5)]
                    else:
                        lm_x = [x1 + (x2 - x1) * 0.3, x1 + (x2 - x1) * 0.7, x1 + (x2 - x1) * 0.5, x1 + (x2 - x1) * 0.35, x1 + (x2 - x1) * 0.65]
                        lm_y = [y1 + (y2 - y1) * 0.35, y1 + (y2 - y1) * 0.35, y1 + (y2 - y1) * 0.55, y1 + (y2 - y1) * 0.75, y1 + (y2 - y1) * 0.75]
                    
                    face_meta = struct.pack("<ffffff5f5f", x1, y1, x2, y2, det_score, blur_score, *lm_x, *lm_y)
                    binary_stream.write(face_meta)

                    emb = f.embedding.astype(np.float32)
                    if len(emb) == 512:
                        binary_stream.write(emb.tobytes())
                    else:
                        binary_stream.write(b'\x00' * (512 * 4))

                # Write raw uncompressed BGR image bytes to C renderer
                binary_stream.write(frame.tobytes())
                binary_stream.flush()
            except (BrokenPipeError, OSError):
                is_piped = False

    capture.release()

if __name__ == "__main__":
    main()
