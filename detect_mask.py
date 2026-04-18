import cv2
import numpy as np
import sqlite3
import os
import time
import logging
import hashlib
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from tensorflow.keras.models import load_model
import face_recognition


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("detection.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    BASE_DIR:          Path  = Path(__file__).parent
    MODEL_PATH:        Path  = field(init=False)
    KNOWN_FACES_DIR:   Path  = field(init=False)
    VIOLATIONS_DIR:    Path  = field(init=False)
    DB_PATH:           Path  = field(init=False)
    LOG_PATH:          Path  = field(init=False)

    # Detection thresholds
    MASK_CONFIDENCE:   float = 0.75    # min confidence to label as "Mask"
    FACE_SCALE:        float = 1.3
    FACE_NEIGHBORS:    int   = 5
    FACE_MIN_SIZE:     tuple = (60, 60)

    # Deduplication
    COOLDOWN_KNOWN:    int   = 60      # seconds between saves for known person
    COOLDOWN_UNKNOWN:  int   = 30      # seconds between saves for unknown face
    ENCODING_ROUND:    int   = 1       # decimal places for hash stability

    # Input size for mask model
    MODEL_INPUT:       tuple = (128, 128)

    # Camera
    CAMERA_INDEX:      int   = 0

    # Alert — beep on violation (optional)
    ALERT_ON_VIOLATION: bool = True

    def __post_init__(self):
        self.MODEL_PATH      = self.BASE_DIR / "mask_model.h5"
        self.KNOWN_FACES_DIR = self.BASE_DIR / "known_faces"
        self.VIOLATIONS_DIR  = self.BASE_DIR / "violations"
        self.DB_PATH         = self.BASE_DIR / "mask_detection.db"
        self.LOG_PATH        = self.BASE_DIR / "detection.log"
        self.VIOLATIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)


CFG = Config()


# ──────────────────────────────────────────────────────────────────────────────
# DATABASE MANAGER
# ──────────────────────────────────────────────────────────────────────────────
class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        log.info(f"Database connected: {db_path}")

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS detections (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name   TEXT    NOT NULL DEFAULT 'Unknown',
                mask_status   TEXT    NOT NULL CHECK(mask_status IN ('Mask','No Mask')),
                confidence    REAL    NOT NULL DEFAULT 0.0,
                image_path    TEXT    DEFAULT '',
                face_key      TEXT    DEFAULT '',
                camera_id     INTEGER DEFAULT 0,
                timestamp     TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at    TEXT    NOT NULL,
                ended_at      TEXT,
                total_frames  INTEGER DEFAULT 0,
                total_faces   INTEGER DEFAULT 0,
                violations    INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_timestamp    ON detections(timestamp);
            CREATE INDEX IF NOT EXISTS idx_person       ON detections(person_name);
            CREATE INDEX IF NOT EXISTS idx_mask_status  ON detections(mask_status);
        """)
        self.conn.commit()

    def insert_detection(self, person_name: str, mask_status: str,
                         confidence: float, image_path: str,
                         face_key: str, camera_id: int = 0) -> int:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur = self.conn.execute(
            """INSERT INTO detections
               (person_name, mask_status, confidence, image_path, face_key, camera_id, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (person_name, mask_status, round(confidence, 4),
             image_path, face_key, camera_id, ts)
        )
        self.conn.commit()
        return cur.lastrowid

    def start_session(self) -> int:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur = self.conn.execute(
            "INSERT INTO sessions (started_at) VALUES (?)", (ts,)
        )
        self.conn.commit()
        return cur.lastrowid

    def end_session(self, session_id: int, frames: int, faces: int, violations: int):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conn.execute(
            """UPDATE sessions SET ended_at=?, total_frames=?,
               total_faces=?, violations=? WHERE id=?""",
            (ts, frames, faces, violations, session_id)
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
        log.info("Database connection closed.")


# ──────────────────────────────────────────────────────────────────────────────
# FACE REGISTRY  — loads & stores known face encodings
# ──────────────────────────────────────────────────────────────────────────────
class FaceRegistry:
    def __init__(self, known_faces_dir: Path):
        self.dir = known_faces_dir
        self.encodings: list = []
        self.names:     list = []
        self._load()

    def _load(self):
        loaded = 0
        for fpath in self.dir.iterdir():
            if fpath.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = face_recognition.load_image_file(str(fpath))
            encs = face_recognition.face_encodings(img)
            if encs:
                self.encodings.append(encs[0])
                self.names.append(fpath.stem)
                loaded += 1
            else:
                log.warning(f"No face found in {fpath.name} — skipped")
        log.info(f"Loaded {loaded} known face(s): {self.names}")

    def identify(self, encoding: np.ndarray,
                 tolerance: float = 0.5) -> tuple[str, float]:
        """Return (name, confidence). confidence = 1 - face_distance."""
        if not self.encodings:
            return "Unknown", 0.0

        distances = face_recognition.face_distance(self.encodings, encoding)
        best_idx  = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        confidence = max(0.0, 1.0 - best_dist)

        if best_dist <= tolerance:
            return self.names[best_idx], confidence
        return "Unknown", confidence


# ──────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION MANAGER
# ──────────────────────────────────────────────────────────────────────────────
class DeduplicationManager:
    def __init__(self, cooldown_known: int, cooldown_unknown: int,
                 encoding_round: int):
        self._last: dict[str, float] = {}
        self.cooldown_known   = cooldown_known
        self.cooldown_unknown = cooldown_unknown
        self.encoding_round   = encoding_round

    def get_key(self, name: str, encoding: np.ndarray) -> str:
        if name != "Unknown":
            return f"known::{name}"
        rounded   = np.round(encoding, self.encoding_round)
        face_hash = hashlib.md5(rounded.tobytes()).hexdigest()[:16]
        return f"unknown::{face_hash}"

    def should_save(self, key: str, name: str) -> bool:
        cooldown = self.cooldown_known if name != "Unknown" else self.cooldown_unknown
        last_ts  = self._last.get(key, 0.0)
        if time.time() - last_ts > cooldown:
            self._last[key] = time.time()
            return True
        return False

    def force_save(self, key: str):
        self._last[key] = time.time()


# ──────────────────────────────────────────────────────────────────────────────
# VIOLATION SAVER
# ──────────────────────────────────────────────────────────────────────────────
class ViolationSaver:
    def __init__(self, violations_dir: Path):
        self.dir = violations_dir

    def save(self, face_img: np.ndarray, name: str) -> str:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{ts}_{name}.jpg"
        path     = self.dir / filename
        cv2.imwrite(str(path), face_img)
        return str(path)


# ──────────────────────────────────────────────────────────────────────────────
# MASK DETECTOR
# ──────────────────────────────────────────────────────────────────────────────
class MaskDetector:
    def __init__(self, model_path: Path, input_size: tuple):
        self.model      = load_model(str(model_path))
        self.input_size = input_size
        log.info(f"Mask model loaded from {model_path}")

    def predict(self, face_img: np.ndarray) -> tuple[str, float]:
        """Returns (label, confidence)."""
        resized = cv2.resize(face_img, self.input_size)
        normed  = resized / 255.0
        inp     = np.reshape(normed, (1, *self.input_size, 3))
        pred    = self.model.predict(inp, verbose=0)
        conf    = float(pred[0][0])
        label   = "Mask" if conf > CFG.MASK_CONFIDENCE else "No Mask"
        return label, conf


# ──────────────────────────────────────────────────────────────────────────────
# OVERLAY RENDERER
# ──────────────────────────────────────────────────────────────────────────────
class OverlayRenderer:
    MASK_COLOR    = (34,  197,  94)    # green
    NOMASK_COLOR  = (239,  68,  68)    # red
    UNKNOWN_COLOR = (234, 179,   8)    # amber
    TEXT_BG       = (15,  23,  42)     # dark navy
    FONT          = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def draw_face(frame: np.ndarray, x: int, y: int, w: int, h: int,
                  name: str, label: str, confidence: float):
        color = (OverlayRenderer.MASK_COLOR
                 if label == "Mask"
                 else OverlayRenderer.NOMASK_COLOR)

        # Bounding box — thick outer + thin inner for depth effect
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.rectangle(frame, (x+2, y+2), (x+w-2, y+h-2), color, 1)

        # Corner accents
        OverlayRenderer._corner_accents(frame, x, y, w, h, color)

        # Label background pill
        text  = f"{name}  |  {label}  {confidence*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, OverlayRenderer.FONT, 0.55, 1)
        pad = 6
        cv2.rectangle(frame,
                      (x, y - th - 2*pad - 4),
                      (x + tw + 2*pad, y),
                      OverlayRenderer.TEXT_BG, -1)
        cv2.rectangle(frame,
                      (x, y - th - 2*pad - 4),
                      (x + tw + 2*pad, y),
                      color, 1)
        cv2.putText(frame, text,
                    (x + pad, y - pad - 2),
                    OverlayRenderer.FONT, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)

    @staticmethod
    def _corner_accents(frame, x, y, w, h, color, length=18, thickness=3):
        pts = [
            ((x, y),       (x+length, y),       (x, y+length)),
            ((x+w, y),     (x+w-length, y),     (x+w, y+length)),
            ((x, y+h),     (x+length, y+h),     (x, y+h-length)),
            ((x+w, y+h),   (x+w-length, y+h),   (x+w, y+h-length)),
        ]
        for corner, h_end, v_end in pts:
            cv2.line(frame, corner, h_end, color, thickness)
            cv2.line(frame, corner, v_end, color, thickness)

    @staticmethod
    def draw_hud(frame: np.ndarray, session_stats: dict):
        """Draw HUD overlay at top-right corner."""
        h_frame, w_frame = frame.shape[:2]
        lines = [
            f"Detections : {session_stats.get('total', 0)}",
            f"Violations : {session_stats.get('violations', 0)}",
            f"Frames     : {session_stats.get('frames', 0)}",
            f"FPS        : {session_stats.get('fps', 0):.1f}",
            datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
        ]
        pad = 10
        lh  = 22
        box_w = 230
        box_h = len(lines) * lh + 2 * pad
        x0 = w_frame - box_w - 10
        y0 = 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0+box_w, y0+box_h),
                      (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0+box_w, y0+box_h),
                      (99, 102, 241), 1)

        for i, line in enumerate(lines):
            color = (239, 68, 68) if "Violations" in line else (200, 200, 200)
            cv2.putText(frame, line,
                        (x0 + pad, y0 + pad + i * lh + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1,
                        cv2.LINE_AA)

    @staticmethod
    def draw_alert_banner(frame: np.ndarray, name: str):
        """Flash red banner on new violation."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (200, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        msg = f"⚠  VIOLATION DETECTED  —  {name}"
        cv2.putText(frame, msg, (20, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
                    cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN DETECTION PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
class MaskDetectionSystem:
    def __init__(self):
        log.info("Initialising Mask Detection System …")
        self.cfg        = CFG
        self.db         = DatabaseManager(CFG.DB_PATH)
        self.registry   = FaceRegistry(CFG.KNOWN_FACES_DIR)
        self.detector   = MaskDetector(CFG.MODEL_PATH, CFG.MODEL_INPUT)
        self.dedup      = DeduplicationManager(CFG.COOLDOWN_KNOWN,
                                               CFG.COOLDOWN_UNKNOWN,
                                               CFG.ENCODING_ROUND)
        self.saver      = ViolationSaver(CFG.VIOLATIONS_DIR)
        self.renderer   = OverlayRenderer()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.session_id = self.db.start_session()
        self.stats = {"frames": 0, "total": 0, "violations": 0,
                      "fps": 0.0, "_t": time.time()}
        log.info("System ready.")

    def _update_fps(self):
        now = time.time()
        elapsed = now - self.stats["_t"]
        if elapsed >= 1.0:
            self.stats["fps"] = self.stats["frames"] / elapsed
            self.stats["frames"] = 0
            self.stats["_t"] = now

    def _process_face(self, frame: np.ndarray,
                      x: int, y: int, w: int, h: int,
                      show_alert: list):
        face_bgr = frame[y:y+h, x:x+w]
        if face_bgr.size == 0:
            return

        # 1. Mask prediction
        label, confidence = self.detector.predict(face_bgr)

        # 2. Face recognition — skip if encoding fails
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(face_rgb)
        if not encodings:
            # Draw minimal box without saving
            self.renderer.draw_face(frame, x, y, w, h,
                                    "?", label, confidence)
            return

        name, id_conf = self.registry.identify(encodings[0])

        # 3. Deduplication
        face_key = self.dedup.get_key(name, encodings[0])
        if self.dedup.should_save(face_key, name):
            image_path = ""
            if label == "No Mask":
                image_path = self.saver.save(face_bgr, name)
                self.stats["violations"] += 1
                show_alert.append(name)

            self.db.insert_detection(
                person_name=name,
                mask_status=label,
                confidence=confidence,
                image_path=image_path,
                face_key=face_key,
                camera_id=self.cfg.CAMERA_INDEX,
            )
            self.stats["total"] += 1
            log.info(f"Saved  [{label}]  {name}  conf={confidence:.2f}  key={face_key[:20]}")

        # 4. Draw
        self.renderer.draw_face(frame, x, y, w, h, name, label, confidence)

    def run(self):
        cap = cv2.VideoCapture(self.cfg.CAMERA_INDEX)
        if not cap.isOpened():
            log.error(f"Cannot open camera index {self.cfg.CAMERA_INDEX}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        log.info("Camera opened. Press Q to quit.")

        alert_until = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Frame read failed — retrying …")
                    time.sleep(0.05)
                    continue

                self.stats["frames"] += 1
                self._update_fps()

                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.cfg.FACE_SCALE,
                    minNeighbors=self.cfg.FACE_NEIGHBORS,
                    minSize=self.cfg.FACE_MIN_SIZE,
                )

                show_alert: list[str] = []

                for (x, y, w, h) in faces:
                    self._process_face(frame, x, y, w, h, show_alert)

                if show_alert:
                    alert_until = time.time() + 2.5

                if time.time() < alert_until:
                    self.renderer.draw_alert_banner(frame, show_alert[0] if show_alert else "")

                self.renderer.draw_hud(frame, self.stats)

                cv2.imshow("Mask Detection System  —  Press Q to quit", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    log.info("Quit signal received.")
                    break

        except KeyboardInterrupt:
            log.info("Interrupted.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.db.end_session(
                self.session_id,
                self.stats["frames"],
                self.stats["total"],
                self.stats["violations"],
            )
            self.db.close()
            log.info("System shutdown complete.")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    system = MaskDetectionSystem()
    system.run()