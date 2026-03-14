#!/usr/bin/env python3
# core/processor.py

import os
import platform
import subprocess
import cv2
import numpy as np
import sys
import time
import warnings
import logging

# Potlačení logování pro čistý CLI výstup
os.environ["MPLCONFIGDIR"] = "/tmp"
os.environ["MPLBACKEND"] = "Agg"
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("insightface").setLevel(logging.ERROR)
os.environ["INSIGHTFACE_LOG_LEVEL"] = "ERROR"
os.environ["ORT_LOGGING_LEVEL"] = "3"

from requests.exceptions import RequestsDependencyWarning
warnings.filterwarnings("ignore", category=RequestsDependencyWarning)

from ultralytics.utils import LOGGER
LOGGER.setLevel("ERROR")
warnings.filterwarnings("ignore", category=FutureWarning)

def resource_path(relative_path):
    """Cesta k prostředkům pro PyInstaller bundle."""
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)

class VideoProcessor:
    def __init__(self, config=None):
        self.config = config or {}
        
        self.yolo_people = None
        self.yolo_lp = None
        self.face_model = None

        # Cesta k FFmpeg (automatická přípona pro Windows)
        self.ffmpeg_path = resource_path("bin/ffmpeg")
        if platform.system() == "Windows":
            self.ffmpeg_path += ".exe"

        # Načtení konfigurace z CLI
        self.detect_interval = self.config.get("DETECT_INTERVAL", 5)
        self.track_buffer = self.config.get("TRACK_BUFFER", 40)
        self.enable_faces = self.config.get("FACES", True)
        self.enable_plates = self.config.get("PLATES", True)
        self.enable_people = self.config.get("PEOPLE", False)
        self.blur_type = self.config.get("BLUR_TYPE", "blur")
        self.device = self.config.get("DEVICE", "cpu")

    # ---------- POMOCNÉ FUNKCE ----------

    def get_best_encoder(self):
        """Vybere HW akcelerovaný enkodér podle platformy."""
        curr_os = platform.system()
        if curr_os == "Darwin": return "h264_videotoolbox"
        if self.device == "cuda": return "h264_nvenc"
        return "libx264"

    def load_people_model(self):
        if self.yolo_people is None:
            from ultralytics import YOLO
            self.yolo_people = YOLO(resource_path("model/yolov8n.pt")).to(self.device)

    def load_lp_model(self):
        if self.yolo_lp is None:
            from ultralytics import YOLO
            m_path = self.config.get("MODEL_PATH", resource_path("model/yolov8_lp.pt"))
            self.yolo_lp = YOLO(m_path).to(self.device)

    def load_face_model(self):
        if self.face_model is None:
            import insightface
            prov = ["CUDAExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            self.face_model = insightface.app.FaceAnalysis(name="buffalo_s", providers=prov)
            self.face_model.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.35)

    @staticmethod
    def iou(boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0

    class FaceTrack:
        def __init__(self, bbox, kps):
            self.bbox = bbox
            self.kps = kps
            self.misses = 0

    # ---------- EFEKTY A KRESLENÍ ----------

    def apply_effect(self, roi):
        if roi.size == 0: return roi
        if self.blur_type == "blackbox": return np.zeros_like(roi)
        if self.blur_type == "pixelate":
            h, w = roi.shape[:2]
            small = cv2.resize(roi, (max(1, w//16), max(1, h//16)), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return cv2.blur(roi, (31, 31))

    def draw_face_blur(self, frame, bbox, kps):
        """Eliptické rozmazání pro obličeje."""
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        pad_x, pad_y = int((x2 - x1) * 0.3), int((y2 - y1) * 0.4)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(W - 1, x2 + pad_x), min(H - 1, y2 + pad_y)
        
        if x2 <= x1 or y2 <= y1: return
        roi = frame[y1:y2, x1:x2].copy()
        processed = self.apply_effect(roi)
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)

        # Výpočet elipsy podle landmarků
        if kps is not None:
            l_eye, r_eye = kps[0], kps[1]
            center = (int((l_eye[0] + r_eye[0]) / 2) - x1, int((l_eye[1] + r_eye[1]) / 2) - y1)
            dist = np.linalg.norm(r_eye - l_eye)
            axes = (int(dist * 2.2), int(dist * 3.0))
            angle = np.degrees(np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))
        else:
            center, axes, angle = (roi.shape[1]//2, roi.shape[0]//2), (int(roi.shape[1]*0.5), int(roi.shape[0]*0.7)), 0

        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (41, 41), 0)
        alpha = (mask.astype(float) / 255.0)[:, :, None]
        frame[y1:y2, x1:x2] = (alpha * processed + (1 - alpha) * roi).astype(np.uint8)

    def draw_box_mask(self, frame, bbox):
        """Obdélníkové rozmazání pro SPZ a postavy (původní chování)."""
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = self.apply_effect(frame[y1:y2, x1:x2])

    # ---------- HLAVNÍ PROCES ----------

    def process_video(self, input_path, output_path, progress_callback=None):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): raise RuntimeError(f"Nelze otevřít {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.enable_faces: self.load_face_model()
        if self.enable_plates: self.load_lp_model()
        if self.enable_people: self.load_people_model()

        temp_video = os.path.join(os.path.dirname(output_path), "temp_blur.mp4")
        encoder = self.get_best_encoder()

        ffmpeg_cmd = [
            self.ffmpeg_path, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", encoder, "-pix_fmt", "yuv420p", temp_video
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        # Trackery (Paměť pro plynulost)
        face_tracks = []
        plate_tracks = []
        people_tracks = []

        frame_idx = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            
            run_detection = (frame_idx == 1 or frame_idx % self.detect_interval == 0)

            # 1. FACES
            if self.enable_faces and run_detection:
                faces = self.face_model.get(frame)
                assigned = set()
                for f in faces:
                    bbox, kps = f.bbox.astype(int), f.kps
                    best_tr, max_iou = None, 0.3
                    for tr in face_tracks:
                        v = self.iou(bbox, tr.bbox)
                        if v > max_iou: max_iou, best_tr = v, tr
                    if best_tr:
                        best_tr.bbox, best_tr.kps, best_tr.misses = bbox, kps, 0
                        assigned.add(best_tr)
                    else:
                        face_tracks.append(self.FaceTrack(bbox, kps))
                for tr in face_tracks:
                    if tr not in assigned: tr.misses += 1
                face_tracks = [t for t in face_tracks if t.misses <= 15]

            # 2. PLATES (YOLO + IOU Memory)
            if self.enable_plates and run_detection:
                res = self.yolo_lp(frame, conf=0.05, imgsz=960, verbose=False)[0]
                for box in res.boxes.xyxy.cpu().numpy():
                    found = False
                    for tr in plate_tracks:
                        if self.iou(box, tr["bbox"]) > 0.5:
                            tr["bbox"], tr["last_seen"] = box, frame_idx
                            found = True
                            break
                    if not found: plate_tracks.append({"bbox": box, "last_seen": frame_idx})

            # 3. PEOPLE (YOLO + IOU Memory)
            if self.enable_people and run_detection:
                res = self.yolo_people(frame, conf=0.25, verbose=False)[0]
                for box, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                    if int(cls) == 0:
                        found = False
                        for tr in people_tracks:
                            if self.iou(box, tr["bbox"]) > 0.5:
                                tr["bbox"], tr["last_seen"] = box, frame_idx
                                found = True
                                break
                        if not found: people_tracks.append({"bbox": box, "last_seen": frame_idx})

            # --- APLIKACE MASEK ---
            
            # SPZ (Obdélníky)
            plate_tracks = [tr for tr in plate_tracks if frame_idx - tr["last_seen"] <= self.track_buffer]
            for tr in plate_tracks:
                self.draw_box_mask(frame, tr["bbox"])

            # POSTAVY (Obdélníky)
            people_tracks = [tr for tr in people_tracks if frame_idx - tr["last_seen"] <= self.track_buffer]
            for tr in people_tracks:
                self.draw_box_mask(frame, tr["bbox"])

            # OBLIČEJE (Elipsy)
            if self.enable_faces:
                for tr in face_tracks:
                    self.draw_face_blur(frame, tr.bbox, tr.kps)

            proc.stdin.write(frame.tobytes())

            if progress_callback and frame_idx % 10 == 0:
                elapsed = time.time() - start_time
                fps_curr = frame_idx / elapsed
                progress_callback(frame_idx, (total_frames - frame_idx) / fps_curr)

        cap.release()
        proc.stdin.close()
        proc.wait()

        # Final Remux (přidání zvuku a uložení)
        subprocess.run([
            self.ffmpeg_path, "-y", "-loglevel", "error",
            "-i", temp_video, "-i", input_path, "-map", "0:v", "-map", "1:a?", "-c", "copy", output_path
        ], check=True)
        if os.path.exists(temp_video): os.remove(temp_video)
