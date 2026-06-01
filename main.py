"""
MotoLens Garage
===============

A single-file motorcycle maintenance companion prototype.

The app keeps a private garage, walks a rider through a detailed baseline
inspection, stores mileage and delivery trips, creates local health reports,
and optionally uses OpenAI for maintenance research, image understanding, and
generated bike artwork.

Production notes:
* Motorcycle service specifications differ by model and year. This app never
  invents torque values or service limits. The owner's manual and service manual
  remain authoritative.
* AES-GCM protects sensitive text fields. Full database-at-rest encryption needs
  a SQLCipher build or a platform database encryption layer.
* OPENAI_API_KEY support is for local development. A shipped mobile app should
  call a controlled backend and keep provider credentials off the device.
* Camera, GPS, and notification hooks use Plyer when it is installed. Their
  Android/iOS permissions still need to be declared in the native package.

Current official OpenAI docs used for the optional integration:
https://developers.openai.com/api/docs/models/gpt-5.5
https://developers.openai.com/api/docs/models/gpt-image-2
https://developers.openai.com/api/docs/guides/tools-web-search
https://developers.openai.com/api/docs/guides/images-vision
https://developers.openai.com/api/docs/guides/image-generation
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import math
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


APP_NAME = "MotoLens Garage"
APP_VERSION = "1.0.0"
OPENAI_REASONING_MODEL = "gpt-5.5"
OPENAI_IMAGE_MODEL = "gpt-image-2"
DEFAULT_DATA_DIR = Path(
    os.environ.get("MOTOLENS_DATA_DIR", "~/.local/share/motolens")
).expanduser()

STATUS_OPEN = "OPEN"
STATUS_PASS = "PASS"
STATUS_MONITOR = "MONITOR"
STATUS_SERVICE = "SERVICE"
STATUS_SKIP = "SKIP"
DONE_STATUSES = {STATUS_PASS, STATUS_MONITOR, STATUS_SERVICE, STATUS_SKIP}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_mileage(raw_value: Any) -> int:
    value = str(raw_value).replace(",", "").strip()
    if not value:
        raise ValueError("Mileage is required.")
    mileage = int(float(value))
    if mileage < 0:
        raise ValueError("Mileage cannot be negative.")
    return mileage


def haversine_miles(
    lat_a: float, lon_a: float, lat_b: float, lon_b: float
) -> float:
    radius_miles = 3958.7613
    lat_1, lat_2 = math.radians(lat_a), math.radians(lat_b)
    delta_lat = math.radians(lat_b - lat_a)
    delta_lon = math.radians(lon_b - lon_a)
    root = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_1) * math.cos(lat_2) * math.sin(delta_lon / 2.0) ** 2
    )
    return radius_miles * 2.0 * math.atan2(math.sqrt(root), math.sqrt(1.0 - root))


@dataclass(frozen=True)
class Bike:
    bike_id: str
    year: int
    make: str
    model: str
    trim: str
    mileage: int
    notes: str
    nickname: str
    image_path: str
    state: str
    created_at: str
    updated_at: str

    @property
    def display_name(self) -> str:
        trim = f" {self.trim}" if self.trim else ""
        return f"{self.year} {self.make} {self.model}{trim}".strip()


@dataclass(frozen=True)
class InspectionTemplate:
    item_key: str
    category: str
    title: str
    short_label: str
    guide: str
    photo_required: bool = False
    safety_critical: bool = False


@dataclass(frozen=True)
class InspectionItem:
    item_id: str
    session_id: str
    item_key: str
    category: str
    title: str
    guide: str
    photo_required: bool
    safety_critical: bool
    status: str
    photo_path: str
    notes: str
    measured_value: str


INSPECTION_TEMPLATES: Tuple[InspectionTemplate, ...] = (
    InspectionTemplate(
        "torque_controls",
        "FASTENERS",
        "Critical bolt torque check",
        "Torque wrench",
        "Use the correct service manual torque chart and a calibrated torque "
        "wrench. Check controls, steering, axles, brake hardware, chassis, and "
        "other critical fasteners specified by your manufacturer. Never guess a "
        "torque value and never use this app as the specification source.",
        safety_critical=True,
    ),
    InspectionTemplate(
        "chain",
        "DRIVE",
        "Chain and sprockets",
        "Chain",
        "Photograph a clean side view of the chain and rear sprocket in bright "
        "light. With the motorcycle off and secured, inspect lubrication, tight "
        "spots, damaged links, sprocket tooth wear, and slack only as described "
        "by the service manual.",
        photo_required=True,
        safety_critical=True,
    ),
    InspectionTemplate(
        "front_tire",
        "TIRES",
        "Front tire tread and sidewall",
        "Front tire",
        "Use a tread-depth gauge when available. For a visual reference photo, "
        "place a penny into a center groove and secure it temporarily with clear "
        "tape. Photograph tread and sidewall, then remove the tape and penny "
        "before riding. Check pressure cold with a real gauge.",
        photo_required=True,
        safety_critical=True,
    ),
    InspectionTemplate(
        "rear_tire",
        "TIRES",
        "Rear tire tread and sidewall",
        "Rear tire",
        "Use a tread-depth gauge when available. For a visual reference photo, "
        "place a penny into a center groove and secure it temporarily with clear "
        "tape. Photograph tread and sidewall, then remove the tape and penny "
        "before riding. Check for flat spots, cracking, punctures, and cold PSI.",
        photo_required=True,
        safety_critical=True,
    ),
    InspectionTemplate(
        "front_brake_pad",
        "BRAKES",
        "Front brake pads",
        "Front pads",
        "Secure the motorcycle, keep hands away from hot components, and point "
        "the camera through the caliper inspection opening toward the pad "
        "material. Do not disassemble the brake, touch friction surfaces, or "
        "apply products. Use your service manual for the wear limit.",
        photo_required=True,
        safety_critical=True,
    ),
    InspectionTemplate(
        "front_brake_rotor",
        "BRAKES",
        "Front brake rotor",
        "Front rotor",
        "Photograph the full rotor face and a close angled view in bright light. "
        "Look for scoring, cracks, discoloration, and obvious edge lips. Rotor "
        "thickness requires a suitable measuring tool and the model-specific "
        "service limit.",
        photo_required=True,
        safety_critical=True,
    ),
    InspectionTemplate(
        "rear_brake_pad",
        "BRAKES",
        "Rear brake pads",
        "Rear pads",
        "Point the camera through the rear caliper inspection opening toward the "
        "pad material. Keep the motorcycle secured and do not touch friction "
        "surfaces. Compare measurements with your service manual.",
        photo_required=True,
        safety_critical=True,
    ),
    InspectionTemplate(
        "rear_brake_rotor",
        "BRAKES",
        "Rear brake rotor",
        "Rear rotor",
        "Photograph the rotor face and a close angled view. Look for scoring, "
        "cracks, discoloration, and an obvious edge lip. A photo cannot replace "
        "a thickness measurement.",
        photo_required=True,
        safety_critical=True,
    ),
    InspectionTemplate(
        "fluids_leaks",
        "FLUIDS",
        "Fluids and leak sweep",
        "Fluids",
        "Inspect engine oil level according to the manual, coolant where "
        "applicable, brake fluid reservoirs, fork seals, hoses, and the parking "
        "area for fresh drips. Escalate active leaks to a qualified mechanic.",
        photo_required=True,
        safety_critical=True,
    ),
    InspectionTemplate(
        "controls",
        "CONTROLS",
        "Controls and steering",
        "Controls",
        "With the engine off, inspect throttle return, levers, cables, steering "
        "movement, mirrors, bars, and grips. Stop if steering binds or controls "
        "do not return normally.",
        safety_critical=True,
    ),
    InspectionTemplate(
        "lights",
        "ELECTRICAL",
        "Lights, horn, and battery",
        "Electrical",
        "Verify headlight, high beam, brake-light activation from both controls, "
        "turn signals, hazards where fitted, license light, horn, and visible "
        "battery condition.",
        safety_critical=True,
    ),
    InspectionTemplate(
        "suspension",
        "CHASSIS",
        "Suspension and wheels",
        "Chassis",
        "Inspect forks, shock, wheel condition, visible axle hardware, and any "
        "play or unusual movement. Wheel bearings and suspension issues require "
        "hands-on diagnosis if anything feels abnormal.",
        safety_critical=True,
    ),
)


LOCAL_SERVICE_LIBRARY: Tuple[Tuple[str, str, int, int, str], ...] = (
    ("Pre-ride safety check", "SAFETY", 0, 0, "Before every ride"),
    ("Tire pressure cold check", "TIRES", 250, 1, "Use manufacturer cold PSI"),
    ("Chain clean, lubricate, and inspect", "DRIVE", 500, 1, "Adjust only to manual spec"),
    ("Engine oil and filter review", "ENGINE", 3000, 6, "Confirm your model interval"),
    ("Brake system inspection", "BRAKES", 3000, 6, "Measure against service limits"),
    ("Battery and charging review", "ELECTRICAL", 6000, 12, "Inspect sooner after storage"),
    ("Full fastener torque inspection", "FASTENERS", 6000, 12, "Use model torque chart"),
)


class VaultCipher:
    """Small field-encryption envelope with an authenticated fallback for tests."""

    def __init__(self, key_path: Path):
        self.key_path = Path(key_path)
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self.key = self._load_or_create_key()
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            self._aesgcm_class = AESGCM
        except ImportError:
            self._aesgcm_class = None

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            key = self.key_path.read_bytes()
            if len(key) != 32:
                raise ValueError("MotoLens vault key has an invalid length.")
            return key
        key = os.urandom(32)
        self.key_path.write_bytes(key)
        try:
            os.chmod(self.key_path, 0o600)
        except OSError:
            pass
        return key

    def seal(self, plaintext: str) -> str:
        raw = plaintext.encode("utf-8")
        if not raw:
            return ""
        if self._aesgcm_class:
            nonce = os.urandom(12)
            payload = nonce + self._aesgcm_class(self.key).encrypt(nonce, raw, None)
            return "aes1:" + base64.urlsafe_b64encode(payload).decode("ascii")
        nonce = os.urandom(16)
        encrypted = self._xor_stream(raw, nonce)
        signature = hmac.new(self.key, nonce + encrypted, hashlib.sha256).digest()
        return "dev1:" + base64.urlsafe_b64encode(nonce + signature + encrypted).decode("ascii")

    def open(self, envelope: str) -> str:
        if not envelope:
            return ""
        prefix, encoded = envelope.split(":", 1)
        payload = base64.urlsafe_b64decode(encoded.encode("ascii"))
        if prefix == "aes1":
            nonce, encrypted = payload[:12], payload[12:]
            if not self._aesgcm_class:
                raise RuntimeError("Install cryptography to open this AES-GCM field.")
            raw = self._aesgcm_class(self.key).decrypt(nonce, encrypted, None)
            return raw.decode("utf-8")
        if prefix == "dev1":
            nonce, signature, encrypted = payload[:16], payload[16:48], payload[48:]
            expected = hmac.new(self.key, nonce + encrypted, hashlib.sha256).digest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError("Vault field authentication failed.")
            return self._xor_stream(encrypted, nonce).decode("utf-8")
        raise ValueError("Unknown encrypted field format.")

    def _xor_stream(self, payload: bytes, nonce: bytes) -> bytes:
        output = bytearray()
        offset = 0
        counter = 0
        while offset < len(payload):
            block = hashlib.sha256(self.key + nonce + counter.to_bytes(4, "big")).digest()
            segment = payload[offset : offset + len(block)]
            output.extend(a ^ b for a, b in zip(segment, block))
            offset += len(segment)
            counter += 1
        return bytes(output)


class MotoRepository:
    """SQLite-backed garage with transactional writes and rolling backups."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.data_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.backups_dir = self.data_dir / "backups"
        self.backups_dir.mkdir(exist_ok=True)
        self.cipher = VaultCipher(self.data_dir / ".vault.key")
        self.db_path = self.data_dir / "motolens.db"
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self._create_schema()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    @contextmanager
    def transaction(self):
        with self._lock:
            try:
                yield self.conn
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def _create_schema(self) -> None:
        with self.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    setting_key TEXT PRIMARY KEY,
                    setting_value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS bikes (
                    bike_id TEXT PRIMARY KEY,
                    year INTEGER NOT NULL,
                    make TEXT NOT NULL,
                    model TEXT NOT NULL,
                    trim TEXT NOT NULL DEFAULT '',
                    mileage INTEGER NOT NULL DEFAULT 0,
                    encrypted_notes TEXT NOT NULL DEFAULT '',
                    nickname TEXT NOT NULL DEFAULT '',
                    image_path TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'INSPECTION REQUIRED',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS inspection_sessions (
                    session_id TEXT PRIMARY KEY,
                    bike_id TEXT NOT NULL REFERENCES bikes(bike_id) ON DELETE CASCADE,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS inspection_items (
                    item_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES inspection_sessions(session_id) ON DELETE CASCADE,
                    item_key TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    guide TEXT NOT NULL,
                    photo_required INTEGER NOT NULL DEFAULT 0,
                    safety_critical INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    photo_path TEXT NOT NULL DEFAULT '',
                    encrypted_notes TEXT NOT NULL DEFAULT '',
                    measured_value TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    UNIQUE(session_id, item_key)
                );
                CREATE TABLE IF NOT EXISTS health_reports (
                    report_id TEXT PRIMARY KEY,
                    bike_id TEXT NOT NULL REFERENCES bikes(bike_id) ON DELETE CASCADE,
                    session_id TEXT NOT NULL REFERENCES inspection_sessions(session_id) ON DELETE CASCADE,
                    health_score INTEGER NOT NULL,
                    summary_json TEXT NOT NULL,
                    hero_image_path TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS maintenance_tasks (
                    task_id TEXT PRIMARY KEY,
                    bike_id TEXT NOT NULL REFERENCES bikes(bike_id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    due_mileage INTEGER NOT NULL DEFAULT 0,
                    due_date TEXT NOT NULL DEFAULT '',
                    priority TEXT NOT NULL DEFAULT 'ROUTINE',
                    source_url TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'UPCOMING',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rides (
                    ride_id TEXT PRIMARY KEY,
                    bike_id TEXT NOT NULL REFERENCES bikes(bike_id) ON DELETE CASCADE,
                    purpose TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL DEFAULT '',
                    distance_miles REAL NOT NULL DEFAULT 0,
                    encrypted_route TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'ACTIVE'
                );
                """
            )

    def set_setting(self, key: str, value: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO settings(setting_key, setting_value) VALUES (?, ?)
                ON CONFLICT(setting_key) DO UPDATE SET setting_value=excluded.setting_value
                """,
                (key, value),
            )

    def get_setting(self, key: str, default: str = "") -> str:
        row = self.conn.execute(
            "SELECT setting_value FROM settings WHERE setting_key=?", (key,)
        ).fetchone()
        return row["setting_value"] if row else default

    def create_bike(
        self,
        year: int,
        make: str,
        model: str,
        mileage: int,
        notes: str = "",
        trim: str = "",
        nickname: str = "",
    ) -> Bike:
        if not make.strip() or not model.strip():
            raise ValueError("Bike make and model are required.")
        if int(year) < 1900 or int(year) > date.today().year + 1:
            raise ValueError("Enter a valid model year.")
        bike_id = str(uuid.uuid4())
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO bikes(
                    bike_id, year, make, model, trim, mileage, encrypted_notes,
                    nickname, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'INSPECTION REQUIRED', ?, ?)
                """,
                (
                    bike_id,
                    int(year),
                    make.strip(),
                    model.strip(),
                    trim.strip(),
                    int(mileage),
                    self.cipher.seal(notes.strip()),
                    nickname.strip(),
                    now,
                    now,
                ),
            )
        self.seed_maintenance_plan(bike_id)
        return self.get_bike(bike_id)

    def get_bike(self, bike_id: str) -> Bike:
        row = self.conn.execute("SELECT * FROM bikes WHERE bike_id=?", (bike_id,)).fetchone()
        if not row:
            raise KeyError(f"Unknown bike: {bike_id}")
        return self._row_to_bike(row)

    def list_bikes(self) -> List[Bike]:
        rows = self.conn.execute(
            "SELECT * FROM bikes ORDER BY updated_at DESC"
        ).fetchall()
        return [self._row_to_bike(row) for row in rows]

    def _row_to_bike(self, row: sqlite3.Row) -> Bike:
        return Bike(
            bike_id=row["bike_id"],
            year=row["year"],
            make=row["make"],
            model=row["model"],
            trim=row["trim"],
            mileage=row["mileage"],
            notes=self.cipher.open(row["encrypted_notes"]),
            nickname=row["nickname"],
            image_path=row["image_path"],
            state=row["state"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update_bike_image(self, bike_id: str, image_path: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE bikes SET image_path=?, updated_at=? WHERE bike_id=?",
                (image_path, utc_now(), bike_id),
            )

    def update_mileage(self, bike_id: str, mileage: int) -> None:
        bike = self.get_bike(bike_id)
        if mileage < bike.mileage:
            raise ValueError("Mileage cannot move backwards.")
        with self.transaction() as conn:
            conn.execute(
                "UPDATE bikes SET mileage=?, updated_at=? WHERE bike_id=?",
                (int(mileage), utc_now(), bike_id),
            )

    def start_inspection(self, bike_id: str) -> str:
        active = self.conn.execute(
            """
            SELECT session_id FROM inspection_sessions
            WHERE bike_id=? AND status='ACTIVE'
            ORDER BY created_at DESC LIMIT 1
            """,
            (bike_id,),
        ).fetchone()
        if active:
            return active["session_id"]
        session_id = str(uuid.uuid4())
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO inspection_sessions(session_id, bike_id, status, created_at)
                VALUES (?, ?, 'ACTIVE', ?)
                """,
                (session_id, bike_id, now),
            )
            conn.executemany(
                """
                INSERT INTO inspection_items(
                    item_id, session_id, item_key, category, title, guide,
                    photo_required, safety_critical, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(uuid.uuid4()),
                        session_id,
                        item.item_key,
                        item.category,
                        item.title,
                        item.guide,
                        int(item.photo_required),
                        int(item.safety_critical),
                        now,
                    )
                    for item in INSPECTION_TEMPLATES
                ],
            )
        return session_id

    def current_inspection(self, bike_id: str) -> str:
        return self.find_active_inspection(bike_id) or self.start_inspection(bike_id)

    def find_active_inspection(self, bike_id: str) -> str:
        row = self.conn.execute(
            """
            SELECT session_id FROM inspection_sessions
            WHERE bike_id=? AND status='ACTIVE'
            ORDER BY created_at DESC LIMIT 1
            """,
            (bike_id,),
        ).fetchone()
        return row["session_id"] if row else ""

    def list_inspection_items(self, session_id: str) -> List[InspectionItem]:
        rows = self.conn.execute(
            """
            SELECT * FROM inspection_items
            WHERE session_id=?
            ORDER BY rowid
            """,
            (session_id,),
        ).fetchall()
        return [
            InspectionItem(
                item_id=row["item_id"],
                session_id=row["session_id"],
                item_key=row["item_key"],
                category=row["category"],
                title=row["title"],
                guide=row["guide"],
                photo_required=bool(row["photo_required"]),
                safety_critical=bool(row["safety_critical"]),
                status=row["status"],
                photo_path=row["photo_path"],
                notes=self.cipher.open(row["encrypted_notes"]),
                measured_value=row["measured_value"],
            )
            for row in rows
        ]

    def update_inspection_item(
        self,
        item_id: str,
        status: str,
        photo_path: str = "",
        notes: str = "",
        measured_value: str = "",
    ) -> None:
        if status not in {STATUS_OPEN, *DONE_STATUSES}:
            raise ValueError(f"Unknown inspection status: {status}")
        current = self.conn.execute(
            "SELECT photo_path FROM inspection_items WHERE item_id=?", (item_id,)
        ).fetchone()
        if not current:
            raise KeyError(f"Unknown inspection item: {item_id}")
        saved_photo = photo_path or current["photo_path"]
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE inspection_items
                SET status=?, photo_path=?, encrypted_notes=?, measured_value=?, updated_at=?
                WHERE item_id=?
                """,
                (
                    status,
                    saved_photo,
                    self.cipher.seal(notes.strip()),
                    measured_value.strip(),
                    utc_now(),
                    item_id,
                ),
            )

    def attach_photo(self, item_id: str, photo_path: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE inspection_items SET photo_path=?, updated_at=? WHERE item_id=?",
                (photo_path, utc_now(), item_id),
            )

    def inspection_progress(self, session_id: str) -> Tuple[int, int]:
        row = self.conn.execute(
            """
            SELECT SUM(CASE WHEN status != 'OPEN' THEN 1 ELSE 0 END) AS done,
                   COUNT(*) AS total
            FROM inspection_items WHERE session_id=?
            """,
            (session_id,),
        ).fetchone()
        return int(row["done"] or 0), int(row["total"] or 0)

    def finalize_inspection(self, session_id: str, ai_summary: str = "") -> Dict[str, Any]:
        session = self.conn.execute(
            "SELECT * FROM inspection_sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if not session:
            raise KeyError(f"Unknown inspection: {session_id}")
        items = self.list_inspection_items(session_id)
        open_items = [item for item in items if item.status == STATUS_OPEN]
        if open_items:
            raise ValueError(f"Finish all {len(open_items)} remaining inspection items first.")
        photo_gaps = [
            item.title for item in items
            if item.photo_required and not item.photo_path and item.status != STATUS_SKIP
        ]
        service_items = [item.title for item in items if item.status == STATUS_SERVICE]
        monitor_items = [item.title for item in items if item.status == STATUS_MONITOR]
        skip_items = [item.title for item in items if item.status == STATUS_SKIP]
        score = int(
            clamp(
                100
                - len(service_items) * 18
                - len(monitor_items) * 7
                - len(skip_items) * 5
                - len(photo_gaps) * 3,
                0,
                100,
            )
        )
        summary = {
            "health_score": score,
            "service_now": service_items,
            "monitor": monitor_items,
            "skipped": skip_items,
            "photo_gaps": photo_gaps,
            "ai_summary": ai_summary.strip(),
            "notice": (
                "Photo-based guidance is informational. Use the official service "
                "manual and a qualified mechanic for safety-critical decisions."
            ),
        }
        report_id = str(uuid.uuid4())
        now = utc_now()
        bike_state = "SERVICE REQUIRED" if service_items else "READY"
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO health_reports(
                    report_id, bike_id, session_id, health_score, summary_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    session["bike_id"],
                    session_id,
                    score,
                    json.dumps(summary, sort_keys=True),
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE inspection_sessions
                SET status='COMPLETE', completed_at=?
                WHERE session_id=?
                """,
                (now, session_id),
            )
            conn.execute(
                "UPDATE bikes SET state=?, updated_at=? WHERE bike_id=?",
                (bike_state, now, session["bike_id"]),
            )
        self.backup_database()
        return {"report_id": report_id, **summary}

    def latest_report(self, bike_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            """
            SELECT * FROM health_reports
            WHERE bike_id=? ORDER BY created_at DESC LIMIT 1
            """,
            (bike_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "report_id": row["report_id"],
            "health_score": row["health_score"],
            "hero_image_path": row["hero_image_path"],
            "created_at": row["created_at"],
            **json.loads(row["summary_json"]),
        }

    def update_report_image(self, report_id: str, image_path: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE health_reports SET hero_image_path=? WHERE report_id=?",
                (image_path, report_id),
            )

    def seed_maintenance_plan(self, bike_id: str) -> None:
        bike = self.get_bike(bike_id)
        now = utc_now()
        with self.transaction() as conn:
            for title, category, miles, months, notes in LOCAL_SERVICE_LIBRARY:
                due_mileage = bike.mileage + miles if miles else bike.mileage
                due_date = (
                    date.today() + timedelta(days=months * 30)
                ).isoformat() if months else date.today().isoformat()
                conn.execute(
                    """
                    INSERT INTO maintenance_tasks(
                        task_id, bike_id, title, category, due_mileage, due_date,
                        priority, notes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        bike_id,
                        title,
                        category,
                        due_mileage,
                        due_date,
                        "SAFETY" if category in {"SAFETY", "TIRES", "BRAKES"} else "ROUTINE",
                        notes,
                        now,
                    ),
                )

    def list_maintenance_tasks(self, bike_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM maintenance_tasks WHERE bike_id=?
            ORDER BY CASE priority WHEN 'SAFETY' THEN 0 ELSE 1 END, due_mileage, due_date
            """,
            (bike_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def add_research_note(self, bike_id: str, research: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO maintenance_tasks(
                    task_id, bike_id, title, category, due_mileage, due_date,
                    priority, source_url, notes, created_at
                ) VALUES (?, ?, ?, 'RESEARCH', 0, '', 'REFERENCE', '', ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    bike_id,
                    "AI maintenance research brief",
                    research,
                    utc_now(),
                ),
            )

    def start_ride(self, bike_id: str, purpose: str) -> str:
        ride_id = str(uuid.uuid4())
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO rides(ride_id, bike_id, purpose, started_at)
                VALUES (?, ?, ?, ?)
                """,
                (ride_id, bike_id, purpose.strip() or "Personal", utc_now()),
            )
        return ride_id

    def finish_ride(
        self, ride_id: str, distance_miles: float, route_points: Sequence[Tuple[float, float]]
    ) -> None:
        row = self.conn.execute(
            "SELECT bike_id FROM rides WHERE ride_id=?", (ride_id,)
        ).fetchone()
        if not row:
            raise KeyError(f"Unknown ride: {ride_id}")
        distance = max(0.0, float(distance_miles))
        bike = self.get_bike(row["bike_id"])
        updated_mileage = bike.mileage + int(round(distance))
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE rides
                SET ended_at=?, distance_miles=?, encrypted_route=?, state='COMPLETE'
                WHERE ride_id=?
                """,
                (
                    utc_now(),
                    distance,
                    self.cipher.seal(json.dumps(route_points)),
                    ride_id,
                ),
            )
            conn.execute(
                "UPDATE bikes SET mileage=?, updated_at=? WHERE bike_id=?",
                (updated_mileage, utc_now(), row["bike_id"]),
            )

    def ride_summary(self, bike_id: str) -> Dict[str, Any]:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS count, COALESCE(SUM(distance_miles), 0) AS miles
            FROM rides WHERE bike_id=? AND state='COMPLETE'
            """,
            (bike_id,),
        ).fetchone()
        return {"count": int(row["count"]), "miles": float(row["miles"])}

    def backup_database(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        target = self.backups_dir / f"motolens-{timestamp}.db"
        with self._lock:
            destination = sqlite3.connect(str(target))
            try:
                self.conn.backup(destination)
            finally:
                destination.close()
        backups = sorted(self.backups_dir.glob("motolens-*.db"), reverse=True)
        for stale in backups[5:]:
            stale.unlink(missing_ok=True)
        return target


class OpenAICoPilot:
    """Optional development integration for research, vision, and generated art."""

    def __init__(self, images_dir: Path):
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.client = None
        self.disabled_reason = ""
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            self.disabled_reason = "OPENAI_API_KEY is not configured."
            return
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key)
        except ImportError:
            self.disabled_reason = "Install the openai package to enable cloud AI."

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def research_maintenance(self, bike: Bike) -> str:
        if not self.client:
            return self.disabled_reason
        response = self.client.responses.create(
            model=OPENAI_REASONING_MODEL,
            tools=[{"type": "web_search"}],
            input=(
                f"Research maintenance information for a {bike.display_name}. "
                "Prioritize official manufacturer documentation and clearly identify "
                "sources. Produce a compact maintenance brief. Do not invent torque "
                "values, wear limits, or intervals when an official source is unavailable."
            ),
        )
        return response.output_text

    def generate_bike_portrait(self, bike: Bike) -> str:
        if not self.client:
            return ""
        response = self.client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=(
                f"A premium studio profile portrait of a {bike.display_name} "
                "motorcycle, exact vehicle proportions, dark graphite seamless "
                "background, soft rim lighting, no text, no watermark, landscape format."
            ),
            size="1536x1024",
            quality="high",
        )
        encoded = response.data[0].b64_json
        target = self.images_dir / f"bike-{bike.bike_id}.png"
        target.write_bytes(base64.b64decode(encoded))
        return str(target)

    def inspect_photos(self, bike: Bike, items: Sequence[InspectionItem]) -> str:
        if not self.client:
            return ""
        photo_items = [item for item in items if item.photo_path and Path(item.photo_path).exists()]
        if not photo_items:
            return ""
        content: List[Dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    f"Review inspection images for this {bike.display_name}. "
                    "Return a concise safety-first summary. Explain visible issues and "
                    "uncertainty. A photo is not a measurement. Do not invent service "
                    "limits or torque values. Recommend a qualified mechanic for any "
                    "safety-critical concern."
                ),
            }
        ]
        for item in photo_items:
            encoded = base64.b64encode(Path(item.photo_path).read_bytes()).decode("ascii")
            content.append({"type": "input_text", "text": f"Inspection area: {item.title}"})
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encoded}",
                    "detail": "high",
                }
            )
        response = self.client.responses.create(
            model=OPENAI_REASONING_MODEL,
            input=[{"role": "user", "content": content}],
        )
        return response.output_text

    def generate_report_art(self, bike: Bike, report: Dict[str, Any]) -> str:
        if not self.client:
            return ""
        flagged = report.get("service_now") or report.get("monitor") or ["baseline inspection"]
        response = self.client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=(
                f"Premium cinematic service-bay portrait of a {bike.display_name} "
                f"motorcycle after a detailed inspection. Visual focus areas: {', '.join(flagged[:3])}. "
                "Dark graphite workshop, refined teal diagnostic light, precise realistic "
                "motorcycle proportions, clean editorial composition, no labels, no text, "
                "no watermark, landscape format."
            ),
            size="1536x1024",
            quality="high",
        )
        target = self.images_dir / f"report-{report['report_id']}.png"
        target.write_bytes(base64.b64decode(response.data[0].b64_json))
        return str(target)


class CameraBridge:
    def __init__(self, images_dir: Path):
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        try:
            from plyer import camera

            self.camera = camera
        except ImportError:
            self.camera = None

    def capture(self, item_key: str, callback: Callable[[str], None]) -> str:
        target = self.images_dir / f"{item_key}-{int(time.time())}.jpg"
        if not self.camera:
            raise RuntimeError("Camera bridge unavailable. Install Plyer in the mobile build.")

        def finished(path: str) -> None:
            callback(path if path else str(target))

        self.camera.take_picture(filename=str(target), on_complete=finished)
        return str(target)


class NotificationBridge:
    def __init__(self):
        try:
            from plyer import notification

            self.notification = notification
        except ImportError:
            self.notification = None

    def send(self, title: str, message: str) -> None:
        if self.notification:
            self.notification.notify(title=title, message=message, app_name=APP_NAME)


class RideTracker:
    """GPS trip tracker with encrypted persistence delegated to MotoRepository."""

    def __init__(self, repository: MotoRepository):
        self.repository = repository
        self.ride_id = ""
        self.bike_id = ""
        self.points: List[Tuple[float, float]] = []
        self.distance_miles = 0.0
        try:
            from plyer import gps

            self.gps = gps
        except ImportError:
            self.gps = None

    @property
    def active(self) -> bool:
        return bool(self.ride_id)

    def start(self, bike_id: str, purpose: str) -> None:
        if self.active:
            raise RuntimeError("A mileage trip is already active.")
        self.ride_id = self.repository.start_ride(bike_id, purpose)
        self.bike_id = bike_id
        self.points = []
        self.distance_miles = 0.0
        if self.gps:
            self.gps.configure(on_location=self._on_location)
            self.gps.start(minTime=1000, minDistance=5)

    def _on_location(self, **kwargs: Any) -> None:
        lat = float(kwargs.get("lat", 0))
        lon = float(kwargs.get("lon", 0))
        self.add_point(lat, lon)

    def add_point(self, lat: float, lon: float) -> None:
        point = (float(lat), float(lon))
        if self.points:
            self.distance_miles += haversine_miles(*self.points[-1], *point)
        self.points.append(point)

    def stop(self) -> float:
        if not self.active:
            raise RuntimeError("No mileage trip is active.")
        if self.gps:
            try:
                self.gps.stop()
            except Exception:
                pass
        distance = self.distance_miles
        self.repository.finish_ride(self.ride_id, distance, self.points)
        self.ride_id = ""
        self.bike_id = ""
        self.points = []
        self.distance_miles = 0.0
        return distance


# Optional GUI layer. Services and tests remain importable without Kivy.
HAS_GUI = False
try:
    from kivy.clock import Clock
    from kivy.graphics import Color, RoundedRectangle
    from kivy.lang import Builder
    from kivy.metrics import dp
    from kivy.properties import (
        BooleanProperty,
        ColorProperty,
        ListProperty,
        NumericProperty,
        ObjectProperty,
        StringProperty,
    )
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.button import Button
    from kivy.uix.label import Label
    from kivy.uix.popup import Popup
    from kivy.uix.progressbar import ProgressBar
    from kivy.uix.scrollview import ScrollView
    from kivy.uix.screenmanager import NoTransition, Screen, ScreenManager
    from kivy.uix.textinput import TextInput

    try:
        from kivymd.app import MDApp
    except ImportError:
        from kivy.app import App as MDApp

    HAS_GUI = True
except ImportError:
    Clock = None
    Builder = None
    MDApp = object
    MotoBoxLayout = object
    Screen = object


if HAS_GUI:
    class MotoBoxLayout(BoxLayout):
        """Version-neutral layout with KivyMD-like convenience properties."""

        adaptive_height = BooleanProperty(False)
        md_bg_color = ColorProperty([0, 0, 0, 0])
        radius = ListProperty([0, 0, 0, 0])
        elevation = NumericProperty(0)

        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            with self.canvas.before:
                self._background_color = Color(rgba=self.md_bg_color)
                self._background_shape = RoundedRectangle(
                    pos=self.pos, size=self.size, radius=self._rounded_radius()
                )
            self.bind(
                md_bg_color=self._sync_background,
                pos=self._sync_background,
                size=self._sync_background,
                radius=self._sync_background,
                adaptive_height=self._sync_adaptive_height,
            )
            self._sync_adaptive_height()

        def _rounded_radius(self) -> List[Tuple[float, float]]:
            values = list(self.radius) or [0]
            values = (values * 4)[:4]
            return [(float(value), float(value)) for value in values]

        def _sync_background(self, *args: Any) -> None:
            self._background_color.rgba = self.md_bg_color
            self._background_shape.pos = self.pos
            self._background_shape.size = self.size
            self._background_shape.radius = self._rounded_radius()

        def _sync_adaptive_height(self, *args: Any) -> None:
            if self.adaptive_height:
                self.size_hint_y = None
                self.bind(minimum_height=self.setter("height"))


    class MotoCard(MotoBoxLayout):
        elevation = NumericProperty(0)


    class MotoLabel(Label):
        """Small styled label preserving the 1.x KV surface across KivyMD versions."""

        adaptive_height = BooleanProperty(False)
        theme_text_color = StringProperty("Custom")
        text_color = ColorProperty([1, 1, 1, 1])
        font_style = StringProperty("Body1")

        _font_sizes = {
            "Caption": 12,
            "Button": 14,
            "Body1": 16,
            "H5": 22,
            "H4": 30,
            "H2": 48,
        }

        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            self.color = self.text_color
            self.bind(
                text_color=self._sync_text_color,
                font_style=self._sync_font_style,
                adaptive_height=self._sync_adaptive_height,
                width=self._sync_text_width,
                texture_size=self._sync_texture_height,
            )
            self._sync_font_style()
            self._sync_adaptive_height()

        def _sync_text_color(self, *args: Any) -> None:
            self.color = self.text_color

        def _sync_font_style(self, *args: Any) -> None:
            self.font_size = dp(self._font_sizes.get(self.font_style, 16))

        def _sync_adaptive_height(self, *args: Any) -> None:
            if self.adaptive_height:
                self.size_hint_y = None
                self._sync_texture_height()

        def _sync_text_width(self, *args: Any) -> None:
            self.text_size = (self.width, None)

        def _sync_texture_height(self, *args: Any) -> None:
            if self.adaptive_height:
                self.height = max(dp(18), self.texture_size[1] + dp(4))


    class _MotoButton(Button):
        md_bg_color = ColorProperty([0.10, 0.13, 0.18, 1])
        text_color = ColorProperty([0.93, 0.96, 1.0, 1])
        font_style = StringProperty("Button")

        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            self.background_normal = ""
            self.background_down = ""
            self.size_hint_y = None
            self.height = dp(46)
            self.background_color = self.md_bg_color
            self.color = self.text_color
            self.bind(
                md_bg_color=self._sync_background_color,
                text_color=self._sync_text_color,
            )

        def _sync_background_color(self, *args: Any) -> None:
            self.background_color = self.md_bg_color

        def _sync_text_color(self, *args: Any) -> None:
            self.color = self.text_color


    class MotoRaisedButton(_MotoButton):
        pass


    class MotoFlatButton(_MotoButton):
        def __init__(self, **kwargs: Any):
            kwargs.setdefault("md_bg_color", [0, 0, 0, 0])
            super().__init__(**kwargs)


    class MotoScrollView(ScrollView):
        """Touch-friendly scrolling with a visible fallback drag handle."""

        def __init__(self, **kwargs: Any):
            kwargs.setdefault("do_scroll_x", False)
            kwargs.setdefault("scroll_type", ["bars", "content"])
            kwargs.setdefault("bar_width", dp(8))
            kwargs.setdefault("bar_margin", dp(4))
            kwargs.setdefault("scroll_distance", dp(6))
            kwargs.setdefault("scroll_timeout", 120)
            super().__init__(**kwargs)
            self.bar_color = [0.16, 0.89, 0.77, 0.76]
            self.bar_inactive_color = [0.16, 0.89, 0.77, 0.28]

        def reveal(self, widget: Any) -> None:
            Clock.schedule_once(lambda dt: self.scroll_to(widget, padding=dp(20)), 0.05)


    class MotoTextField(TextInput):
        line_color_focus = ColorProperty([0.16, 0.89, 0.77, 1])
        text_color_focus = ColorProperty([0.93, 0.96, 1.0, 1])
        current_hint_text_color = ColorProperty([0.56, 0.64, 0.74, 1])

        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            self.size_hint_y = None
            self.height = dp(48) if not self.multiline else dp(88)
            self.background_normal = ""
            self.background_active = ""
            self.background_color = [0.075, 0.095, 0.135, 1]
            self.foreground_color = [0.93, 0.96, 1.0, 1]
            self.hint_text_color = [0.56, 0.64, 0.74, 1]
            self.cursor_color = [0.16, 0.89, 0.77, 1]
            self.padding = [dp(12), dp(12)]
            self.bind(focus=self._reveal_when_focused)

        def _reveal_when_focused(self, instance: Any, focused: bool) -> None:
            if not focused:
                return
            parent = self.parent
            while parent:
                if isinstance(parent, MotoScrollView):
                    parent.reveal(self)
                    return
                parent = parent.parent


    class MotoProgressBar(ProgressBar):
        color = ColorProperty([0.16, 0.89, 0.77, 1])

        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            self.size_hint_y = None
            self.height = dp(8)


    class MotoTopAppBar(MotoBoxLayout):
        title = StringProperty("")
        specific_text_color = ColorProperty([1, 1, 1, 1])
        right_action_items = ListProperty([])

        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            self.size_hint_y = None
            self.height = dp(58)
            self.padding = [dp(18), dp(6)]
            self.bind(
                title=self._render,
                specific_text_color=self._render,
                right_action_items=self._render,
            )
            Clock.schedule_once(self._render, 0)

        def _render(self, *args: Any) -> None:
            self.clear_widgets()
            self.add_widget(
                MotoLabel(
                    text=self.title,
                    bold=True,
                    font_style="H5",
                    text_color=self.specific_text_color,
                )
            )
            for action in self.right_action_items:
                callback = action[1] if len(action) > 1 else None
                button = MotoFlatButton(text="PRIVACY", size_hint_x=None, width=dp(92))
                if callback:
                    button.bind(on_release=callback)
                self.add_widget(button)


    class MotoDialog:
        def __init__(
            self, title: str = "", text: str = "", buttons: Optional[List[Any]] = None
        ):
            content = MotoBoxLayout(
                orientation="vertical",
                spacing=dp(12),
                padding=dp(18),
                md_bg_color=[0.055, 0.072, 0.105, 1],
            )
            content.add_widget(
                MotoLabel(
                    text=title,
                    bold=True,
                    font_style="H5",
                    adaptive_height=True,
                    text_color=[0.93, 0.96, 1, 1],
                )
            )
            content.add_widget(
                MotoLabel(
                    text=text,
                    adaptive_height=True,
                    text_color=[0.70, 0.77, 0.86, 1],
                )
            )
            row = MotoBoxLayout(spacing=dp(8), size_hint_y=None, height=dp(48))
            for button in buttons or []:
                row.add_widget(button)
            content.add_widget(row)
            self._popup = Popup(
                title="",
                content=content,
                size_hint=(0.86, None),
                height=dp(340),
                separator_height=0,
                background="",
                background_color=[0.025, 0.035, 0.055, 1],
            )

        def open(self) -> None:
            self._popup.open()

        def dismiss(self) -> None:
            self._popup.dismiss()


    class MotoSnackbar:
        def __init__(self, text: str, duration: float = 2.5):
            self.duration = duration
            self._popup = Popup(
                title="",
                content=MotoLabel(
                    text=text,
                    adaptive_height=True,
                    text_color=[0.93, 0.96, 1, 1],
                ),
                size_hint=(0.88, None),
                height=dp(82),
                separator_height=0,
                background="",
                background_color=[0.10, 0.13, 0.18, 1],
            )

        def open(self) -> None:
            self._popup.open()
            Clock.schedule_once(lambda dt: self._popup.dismiss(), self.duration)


    KV = r"""
#:import dp kivy.metrics.dp
#:import NoTransition kivy.uix.screenmanager.NoTransition

<SurfaceCard@MotoCard>:
    orientation: "vertical"
    padding: dp(18)
    spacing: dp(8)
    radius: [dp(18), dp(18), dp(18), dp(18)]
    md_bg_color: app.colors["surface"]
    elevation: 0

<MutedLabel@MotoLabel>:
    theme_text_color: "Custom"
    text_color: app.colors["muted"]
    font_style: "Caption"

<GarageScreen>:
    name: "garage"
    MotoScrollView:
        MotoBoxLayout:
            orientation: "vertical"
            padding: dp(20)
            spacing: dp(14)
            adaptive_height: True
            MotoLabel:
                text: "GARAGE"
                font_style: "H4"
                bold: True
                theme_text_color: "Custom"
                text_color: app.colors["text"]
                adaptive_height: True
            MutedLabel:
                text: "Your motorcycle, its health, and the next right action."
                adaptive_height: True
            SurfaceCard:
                adaptive_height: True
                AsyncImage:
                    id: bike_art
                    source: ""
                    size_hint_y: None
                    height: dp(170) if self.source else 0
                MotoLabel:
                    id: bike_name
                    text: "No motorcycle added"
                    font_style: "H5"
                    bold: True
                    theme_text_color: "Custom"
                    text_color: app.colors["text"]
                    adaptive_height: True
                MutedLabel:
                    id: bike_meta
                    text: "Run setup to begin."
                    adaptive_height: True
                MotoLabel:
                    id: health_score
                    text: "--"
                    font_style: "H2"
                    bold: True
                    theme_text_color: "Custom"
                    text_color: app.colors["accent"]
                    adaptive_height: True
                MutedLabel:
                    text: "VEHICLE HEALTH SCORE"
                    adaptive_height: True
                MotoLabel:
                    id: bike_state
                    text: "INSPECTION REQUIRED"
                    font_style: "Button"
                    bold: True
                    theme_text_color: "Custom"
                    text_color: app.colors["amber"]
                    adaptive_height: True
            SurfaceCard:
                adaptive_height: True
                MotoLabel:
                    text: "Baseline inspection"
                    bold: True
                    theme_text_color: "Custom"
                    text_color: app.colors["text"]
                    adaptive_height: True
                MutedLabel:
                    id: inspection_summary
                    text: "Add a motorcycle to unlock the inspection."
                    adaptive_height: True
                MotoProgressBar:
                    id: garage_progress
                    value: 0
                    color: app.colors["accent"]
                MotoRaisedButton:
                    text: "CONTINUE FULL INSPECTION"
                    md_bg_color: app.colors["accent"]
                    text_color: 0.01, 0.04, 0.04, 1
                    on_release: app.open_inspection()
            MotoRaisedButton:
                text: "ADD ANOTHER MOTORCYCLE"
                md_bg_color: app.colors["surface_high"]
                on_release: app.open_onboarding()

<OnboardingScreen>:
    name: "onboarding"
    MotoBoxLayout:
        orientation: "vertical"
        MotoScrollView:
            MotoBoxLayout:
                orientation: "vertical"
                padding: dp(20), dp(20), dp(20), dp(10)
                spacing: dp(14)
                adaptive_height: True
                MotoLabel:
                    text: "BUILD YOUR GARAGE"
                    font_style: "H4"
                    bold: True
                    theme_text_color: "Custom"
                    text_color: app.colors["text"]
                    adaptive_height: True
                MutedLabel:
                    text: "Tell MotoLens what you ride. Your baseline inspection starts next."
                    adaptive_height: True
                SurfaceCard:
                    adaptive_height: True
                    MotoTextField:
                        id: year
                        hint_text: "Model year"
                        input_filter: "int"
                    MotoTextField:
                        id: make
                        hint_text: "Make, for example Yamaha"
                    MotoTextField:
                        id: model
                        hint_text: "Model, for example MT-07"
                    MotoTextField:
                        id: trim
                        hint_text: "Trim or package (optional)"
                    MotoTextField:
                        id: mileage
                        hint_text: "Current odometer mileage"
                        input_filter: "float"
                    MotoTextField:
                        id: nickname
                        hint_text: "Nickname (optional)"
                    MotoTextField:
                        id: notes
                        hint_text: "Notes: recent service, mods, concerns"
                        multiline: True
        MotoBoxLayout:
            orientation: "vertical"
            spacing: dp(6)
            padding: dp(20), dp(8), dp(20), dp(10)
            size_hint_y: None
            height: dp(110)
            MotoRaisedButton:
                text: "CREATE BIKE + START INSPECTION"
                md_bg_color: app.colors["accent"]
                text_color: 0.01, 0.04, 0.04, 1
                on_release: app.submit_bike_setup()
            MotoFlatButton:
                text: "BACK TO GARAGE"
                text_color: app.colors["muted"]
                on_release: app.show_screen("garage")

<InspectionScreen>:
    name: "inspection"
    MotoBoxLayout:
        orientation: "vertical"
        MotoScrollView:
            MotoBoxLayout:
                orientation: "vertical"
                padding: dp(20)
                spacing: dp(14)
                adaptive_height: True
                MotoLabel:
                    text: "FULL INSPECTION"
                    font_style: "H4"
                    bold: True
                    theme_text_color: "Custom"
                    text_color: app.colors["text"]
                    adaptive_height: True
                MutedLabel:
                    id: inspection_counter
                    text: "0 / 0 AREAS COMPLETE"
                    adaptive_height: True
                MotoProgressBar:
                    id: progress
                    value: 0
                    color: app.colors["accent"]
                SurfaceCard:
                    adaptive_height: True
                    MutedLabel:
                        id: category
                        text: "SAFETY"
                        adaptive_height: True
                    MotoLabel:
                        id: title
                        text: "Select a motorcycle to begin."
                        font_style: "H5"
                        bold: True
                        theme_text_color: "Custom"
                        text_color: app.colors["text"]
                        adaptive_height: True
                    MotoLabel:
                        id: guide
                        text: ""
                        theme_text_color: "Custom"
                        text_color: app.colors["muted"]
                        adaptive_height: True
                    MutedLabel:
                        id: evidence
                        text: ""
                        adaptive_height: True
                    MotoRaisedButton:
                        id: camera_button
                        text: "CAPTURE GUIDED PHOTO"
                        md_bg_color: app.colors["surface_high"]
                        on_release: app.capture_current_photo()
                MotoBoxLayout:
                    spacing: dp(8)
                    adaptive_height: True
                    MotoRaisedButton:
                        text: "PASS"
                        md_bg_color: app.colors["accent"]
                        text_color: 0.01, 0.04, 0.04, 1
                        on_release: app.mark_current_item("PASS")
                    MotoRaisedButton:
                        text: "MONITOR"
                        md_bg_color: app.colors["amber"]
                        text_color: 0.06, 0.04, 0.01, 1
                        on_release: app.mark_current_item("MONITOR")
                    MotoRaisedButton:
                        text: "SERVICE"
                        md_bg_color: app.colors["red"]
                        on_release: app.mark_current_item("SERVICE")
                MotoBoxLayout:
                    spacing: dp(8)
                    adaptive_height: True
                    MotoFlatButton:
                        text: "PREVIOUS"
                        text_color: app.colors["muted"]
                        on_release: app.previous_inspection_item()
                    MotoFlatButton:
                        text: "SKIP WITH FLAG"
                        text_color: app.colors["muted"]
                        on_release: app.mark_current_item("SKIP")
                    MotoFlatButton:
                        text: "NEXT"
                        text_color: app.colors["text"]
                        on_release: app.next_inspection_item()
                MotoRaisedButton:
                    text: "GENERATE VEHICLE HEALTH REPORT"
                    md_bg_color: app.colors["surface_high"]
                    on_release: app.finalize_current_inspection()

<RideScreen>:
    name: "ride"
    MotoScrollView:
        MotoBoxLayout:
            orientation: "vertical"
            padding: dp(20)
            spacing: dp(14)
            adaptive_height: True
            MotoLabel:
                text: "RIDE TRACKER"
                font_style: "H4"
                bold: True
                theme_text_color: "Custom"
                text_color: app.colors["text"]
                adaptive_height: True
            MutedLabel:
                text: "GPS mileage for personal rides, DoorDash, and Uber Eats."
                adaptive_height: True
            SurfaceCard:
                adaptive_height: True
                MutedLabel:
                    text: "CURRENT TRIP"
                    adaptive_height: True
                MotoLabel:
                    id: live_distance
                    text: "0.0 mi"
                    font_style: "H2"
                    bold: True
                    theme_text_color: "Custom"
                    text_color: app.colors["accent"]
                    adaptive_height: True
                MutedLabel:
                    id: tracker_state
                    text: "READY TO TRACK"
                    adaptive_height: True
                MotoTextField:
                    id: purpose
                    hint_text: "Purpose: Personal, DoorDash, Uber Eats"
                    text: "DoorDash"
                MotoRaisedButton:
                    id: tracker_button
                    text: "START GPS MILEAGE"
                    md_bg_color: app.colors["accent"]
                    text_color: 0.01, 0.04, 0.04, 1
                    on_release: app.toggle_ride_tracking()
            SurfaceCard:
                adaptive_height: True
                MutedLabel:
                    text: "DELIVERY + RIDE HISTORY"
                    adaptive_height: True
                MotoLabel:
                    id: ride_history
                    text: "No completed rides yet."
                    theme_text_color: "Custom"
                    text_color: app.colors["text"]
                    adaptive_height: True

<ServiceScreen>:
    name: "service"
    MotoScrollView:
        MotoBoxLayout:
            orientation: "vertical"
            padding: dp(20)
            spacing: dp(14)
            adaptive_height: True
            MotoLabel:
                text: "SERVICE PLAN"
                font_style: "H4"
                bold: True
                theme_text_color: "Custom"
                text_color: app.colors["text"]
                adaptive_height: True
            MutedLabel:
                text: "Local reminders plus researched model-specific reference notes."
                adaptive_height: True
            SurfaceCard:
                adaptive_height: True
                MotoLabel:
                    id: task_list
                    text: "Add a motorcycle to build its maintenance plan."
                    theme_text_color: "Custom"
                    text_color: app.colors["text"]
                    adaptive_height: True
            MotoRaisedButton:
                text: "RESEARCH MY BIKE ONLINE"
                md_bg_color: app.colors["surface_high"]
                on_release: app.research_active_bike()
            MutedLabel:
                id: ai_state
                text: "AI research uses GPT-5.5 web search when a development key is configured."
                adaptive_height: True

<AppShell>:
    orientation: "vertical"
    md_bg_color: app.colors["background"]
    MotoTopAppBar:
        title: "MOTOLENS"
        elevation: 0
        md_bg_color: app.colors["background"]
        specific_text_color: app.colors["text"]
        right_action_items: [["shield-check-outline", lambda x: app.show_privacy_info()]]
    ScreenManager:
        id: workspace
        transition: NoTransition()
        GarageScreen:
        OnboardingScreen:
        InspectionScreen:
        RideScreen:
        ServiceScreen:
    MotoBoxLayout:
        adaptive_height: True
        padding: dp(4), dp(2), dp(4), dp(6)
        spacing: dp(2)
        md_bg_color: app.colors["surface"]
        MotoFlatButton:
            text: "GARAGE"
            text_color: app.colors["text"]
            on_release: app.show_screen("garage")
        MotoFlatButton:
            text: "INSPECT"
            text_color: app.colors["text"]
            on_release: app.open_inspection()
        MotoFlatButton:
            text: "RIDE"
            text_color: app.colors["text"]
            on_release: app.show_screen("ride")
        MotoFlatButton:
            text: "SERVICE"
            text_color: app.colors["text"]
            on_release: app.show_screen("service")
"""

    class GarageScreen(Screen):
        pass

    class OnboardingScreen(Screen):
        pass

    class InspectionScreen(Screen):
        pass

    class RideScreen(Screen):
        pass

    class ServiceScreen(Screen):
        pass

    class AppShell(MotoBoxLayout):
        pass


    class MotoLensApp(MDApp):
        """KivyMD mobile shell for the single-file MotoLens prototype."""

        def __init__(self, data_dir: Path = DEFAULT_DATA_DIR, **kwargs: Any):
            super().__init__(**kwargs)
            self.title = APP_NAME
            self.repository = MotoRepository(data_dir)
            self.ai = OpenAICoPilot(self.repository.images_dir)
            self.camera = CameraBridge(self.repository.images_dir)
            self.notifications = NotificationBridge()
            self.tracker = RideTracker(self.repository)
            self.active_bike_id = ""
            self.active_session_id = ""
            self.inspection_index = 0
            self._report_processing = False
            self.colors = {
                "background": [0.025, 0.035, 0.055, 1],
                "surface": [0.055, 0.072, 0.105, 1],
                "surface_high": [0.10, 0.13, 0.18, 1],
                "text": [0.93, 0.96, 1.0, 1],
                "muted": [0.56, 0.64, 0.74, 1],
                "accent": [0.16, 0.89, 0.77, 1],
                "amber": [1.0, 0.68, 0.23, 1],
                "red": [0.96, 0.31, 0.35, 1],
            }
            self._dialog: Optional[MotoDialog] = None

        def build(self) -> AppShell:
            if hasattr(self, "theme_cls"):
                self.theme_cls.theme_style = "Dark"
                self.theme_cls.primary_palette = "Teal"
            Builder.load_string(KV)
            return AppShell()

        def on_start(self) -> None:
            bikes = self.repository.list_bikes()
            if bikes:
                self.active_bike_id = bikes[0].bike_id
                self.active_session_id = self.repository.find_active_inspection(self.active_bike_id)
                self.show_screen("garage")
                Clock.schedule_once(lambda dt: self.show_torque_reminder(), 0.4)
            else:
                self.show_screen("onboarding")
                Clock.schedule_once(lambda dt: self.show_how_to(), 0.4)
            self.refresh_all()

        def on_stop(self) -> None:
            if self.tracker.active:
                self.tracker.stop()
            self.repository.backup_database()
            self.repository.close()

        def screen(self, name: str) -> Screen:
            return self.root.ids.workspace.get_screen(name)

        def show_screen(self, name: str) -> None:
            self.root.ids.workspace.current = name
            self.refresh_all()

        def notify(self, text: str) -> None:
            MotoSnackbar(text=text, duration=2.5).open()

        def show_dialog(self, title: str, text: str, buttons: Optional[List[Any]] = None) -> None:
            if self._dialog:
                self._dialog.dismiss()
            dialog = MotoDialog(title=title, text=text, buttons=buttons or [])
            self._dialog = dialog
            dialog.open()

        def dismiss_dialog(self) -> None:
            if self._dialog:
                self._dialog.dismiss()
                self._dialog = None

        def show_how_to(self) -> None:
            self.show_dialog(
                "Your first garage",
                "Add your motorcycle and current mileage. MotoLens will build a "
                "baseline inspection, guide your photos, prepare maintenance "
                "reference tasks, and track ride mileage. Safety-critical work "
                "still follows your official service manual.",
                [MotoFlatButton(text="START SETUP", on_release=lambda x: self.dismiss_dialog())],
            )

        def show_torque_reminder(self) -> None:
            if not self.active_bike_id:
                return
            self.show_dialog(
                "Pre-ride fastener reminder",
                "Before riding, check the critical fasteners required by your "
                "motorcycle's service manual with the correct calibrated torque "
                "wrench. MotoLens will never guess a torque value.",
                [
                    MotoFlatButton(text="OPEN INSPECTION", on_release=lambda x: self._dialog_to_inspection()),
                    MotoFlatButton(text="ACKNOWLEDGE", on_release=lambda x: self.dismiss_dialog()),
                ],
            )

        def _dialog_to_inspection(self) -> None:
            self.dismiss_dialog()
            self.open_inspection()

        def show_privacy_info(self) -> None:
            self.show_dialog(
                "Private by design",
                "Sensitive notes and recorded routes are encrypted before SQLite "
                "storage. Reports trigger timestamped local backups. A production "
                "build should add SQLCipher or platform storage encryption for the "
                "whole database and proxy AI requests through your backend.",
                [MotoFlatButton(text="CLOSE", on_release=lambda x: self.dismiss_dialog())],
            )

        def open_onboarding(self) -> None:
            self.show_screen("onboarding")

        def submit_bike_setup(self) -> None:
            ids = self.screen("onboarding").ids
            try:
                bike = self.repository.create_bike(
                    year=int(ids.year.text.strip()),
                    make=ids.make.text,
                    model=ids.model.text,
                    trim=ids.trim.text,
                    mileage=parse_mileage(ids.mileage.text),
                    nickname=ids.nickname.text,
                    notes=ids.notes.text,
                )
            except Exception as exc:
                self.notify(str(exc))
                return
            self.active_bike_id = bike.bike_id
            self.active_session_id = self.repository.start_inspection(bike.bike_id)
            self.inspection_index = 0
            self.refresh_all()
            self.show_screen("inspection")
            self.show_torque_reminder()
            threading.Thread(
                target=self._generate_bike_portrait_background,
                args=(bike,),
                daemon=True,
            ).start()
            if self.ai.enabled:
                threading.Thread(
                    target=self._research_background,
                    args=(bike, False),
                    daemon=True,
                ).start()

        def _generate_bike_portrait_background(self, bike: Bike) -> None:
            try:
                path = self.ai.generate_bike_portrait(bike)
                if path:
                    self.repository.update_bike_image(bike.bike_id, path)
            except Exception:
                return

        def active_bike(self) -> Optional[Bike]:
            if not self.active_bike_id:
                bikes = self.repository.list_bikes()
                if not bikes:
                    return None
                self.active_bike_id = bikes[0].bike_id
            return self.repository.get_bike(self.active_bike_id)

        def inspection_items(self, create: bool = True) -> List[InspectionItem]:
            bike = self.active_bike()
            if not bike:
                return []
            self.active_session_id = (
                self.active_session_id
                or self.repository.find_active_inspection(bike.bike_id)
            )
            if not self.active_session_id and create:
                self.active_session_id = self.repository.start_inspection(bike.bike_id)
            if not self.active_session_id:
                return []
            return self.repository.list_inspection_items(self.active_session_id)

        def open_inspection(self) -> None:
            if not self.active_bike():
                self.open_onboarding()
                return
            if not self.active_session_id:
                self.active_session_id = self.repository.start_inspection(self.active_bike_id)
            self.show_screen("inspection")
            self.refresh_inspection()

        def current_inspection_item(self) -> Optional[InspectionItem]:
            items = self.inspection_items()
            if not items:
                return None
            self.inspection_index %= len(items)
            return items[self.inspection_index]

        def previous_inspection_item(self) -> None:
            self.inspection_index -= 1
            self.refresh_inspection()

        def next_inspection_item(self) -> None:
            self.inspection_index += 1
            self.refresh_inspection()

        def capture_current_photo(self) -> None:
            item = self.current_inspection_item()
            if not item:
                return
            try:
                self.camera.capture(item.item_key, lambda path: self._photo_complete(item.item_id, path))
            except Exception as exc:
                self.show_dialog(
                    "Guided camera",
                    f"{item.guide}\n\nCamera status: {exc}",
                    [MotoFlatButton(text="CLOSE", on_release=lambda x: self.dismiss_dialog())],
                )

        def _photo_complete(self, item_id: str, path: str) -> None:
            self.repository.attach_photo(item_id, path)
            Clock.schedule_once(lambda dt: self.refresh_inspection(), 0)

        def mark_current_item(self, status: str) -> None:
            item = self.current_inspection_item()
            if not item:
                return
            if item.photo_required and not item.photo_path and status != STATUS_SKIP:
                self.notify("Capture the guided photo first, or use SKIP WITH FLAG.")
                return
            self.repository.update_inspection_item(item.item_id, status=status)
            self.inspection_index += 1
            self.refresh_all()

        def finalize_current_inspection(self) -> None:
            if not self.active_session_id:
                return
            if self._report_processing:
                self.notify("Your vehicle health report is already processing.")
                return
            items = self.repository.list_inspection_items(self.active_session_id)
            open_items = [item for item in items if item.status == STATUS_OPEN]
            if open_items:
                self.notify(f"Finish all {len(open_items)} remaining inspection items first.")
                return
            self._report_processing = True
            self.notify("Processing inspection photos and building your report...")
            threading.Thread(
                target=self._finalize_inspection_background,
                args=(self.active_session_id,),
                daemon=True,
            ).start()

        def _finalize_inspection_background(self, session_id: str) -> None:
            try:
                items = self.repository.list_inspection_items(session_id)
                bike = self.active_bike()
                ai_summary = self.ai.inspect_photos(bike, items) if bike else ""
                report = self.repository.finalize_inspection(session_id, ai_summary)
            except Exception as exc:
                Clock.schedule_once(
                    lambda dt, message=str(exc): self._finalize_inspection_failed(message),
                    0,
                )
                return
            Clock.schedule_once(
                lambda dt: self._finalize_inspection_complete(bike, report),
                0,
            )

        def _finalize_inspection_failed(self, message: str) -> None:
            self._report_processing = False
            self.notify(message)

        def _finalize_inspection_complete(
            self, bike: Optional[Bike], report: Dict[str, Any]
        ) -> None:
            self._report_processing = False
            self.notifications.send(
                "MotoLens report ready",
                f"Vehicle health score: {report['health_score']}/100",
            )
            if bike and self.ai.enabled:
                threading.Thread(
                    target=self._generate_report_art_background,
                    args=(bike, report),
                    daemon=True,
                ).start()
            self.notify(f"Vehicle health report ready: {report['health_score']}/100")
            self.active_session_id = ""
            self.show_screen("garage")

        def _generate_report_art_background(self, bike: Bike, report: Dict[str, Any]) -> None:
            try:
                path = self.ai.generate_report_art(bike, report)
                if path:
                    self.repository.update_report_image(report["report_id"], path)
                    Clock.schedule_once(lambda dt: self.refresh_garage(), 0)
            except Exception:
                return

        def toggle_ride_tracking(self) -> None:
            bike = self.active_bike()
            if not bike:
                self.open_onboarding()
                return
            try:
                if self.tracker.active:
                    distance = self.tracker.stop()
                    self.notifications.send(
                        "Mileage recorded", f"Trip saved: {distance:.1f} miles"
                    )
                    self._send_due_service_reminder()
                else:
                    purpose = self.screen("ride").ids.purpose.text
                    self.tracker.start(bike.bike_id, purpose)
            except Exception as exc:
                self.notify(str(exc))
            self.refresh_ride()

        def _send_due_service_reminder(self) -> None:
            bike = self.active_bike()
            if not bike:
                return
            due = [
                task for task in self.repository.list_maintenance_tasks(bike.bike_id)
                if task["due_mileage"] and task["due_mileage"] <= bike.mileage
            ]
            if due:
                self.notifications.send(
                    "MotoLens service reminder",
                    f"{due[0]['title']} is due after your recorded route.",
                )

        def research_active_bike(self) -> None:
            bike = self.active_bike()
            if not bike:
                self.open_onboarding()
                return
            if not self.ai.enabled:
                self.notify(self.ai.disabled_reason)
                return
            self.screen("service").ids.ai_state.text = "Researching official sources..."
            threading.Thread(
                target=self._research_background, args=(bike,), daemon=True
            ).start()

        def _research_background(self, bike: Bike, announce: bool = True) -> None:
            try:
                research = self.ai.research_maintenance(bike)
                self.repository.add_research_note(bike.bike_id, research)
                message = "Research brief saved to your service plan."
            except Exception as exc:
                message = f"Research failed: {exc}"
            if announce:
                Clock.schedule_once(lambda dt: self._research_complete(message), 0)

        def _research_complete(self, message: str) -> None:
            self.screen("service").ids.ai_state.text = message
            self.refresh_service()
            self.notify(message)

        def refresh_all(self) -> None:
            if not self.root:
                return
            self.refresh_garage()
            self.refresh_inspection()
            self.refresh_ride()
            self.refresh_service()

        def refresh_garage(self) -> None:
            view = self.screen("garage")
            bike = self.active_bike()
            if not bike:
                return
            report = self.repository.latest_report(bike.bike_id)
            session = self.repository.find_active_inspection(bike.bike_id)
            if session:
                done, total = self.repository.inspection_progress(session)
            elif report:
                done = total = len(INSPECTION_TEMPLATES)
            else:
                done, total = 0, len(INSPECTION_TEMPLATES)
            view.ids.bike_name.text = bike.nickname or bike.display_name
            view.ids.bike_meta.text = f"{bike.display_name}  |  {bike.mileage:,} mi"
            view.ids.bike_state.text = bike.state
            view.ids.health_score.text = str(report["health_score"]) if report else "--"
            view.ids.inspection_summary.text = f"{done} of {total} areas documented"
            view.ids.garage_progress.value = 100 * done / total if total else 0
            view.ids.bike_art.source = (report or {}).get("hero_image_path", "") or bike.image_path

        def refresh_inspection(self) -> None:
            view = self.screen("inspection")
            create = self.root.ids.workspace.current == "inspection"
            items = self.inspection_items(create=create)
            if not items:
                return
            self.inspection_index %= len(items)
            item = items[self.inspection_index]
            done, total = self.repository.inspection_progress(item.session_id)
            view.ids.inspection_counter.text = f"{done} / {total} AREAS COMPLETE"
            view.ids.progress.value = 100 * done / total if total else 0
            view.ids.category.text = f"{item.category}  |  {item.status}"
            view.ids.title.text = item.title
            view.ids.guide.text = item.guide
            evidence = "PHOTO REQUIRED" if item.photo_required else "VISUAL + MANUAL CHECK"
            if item.photo_path:
                evidence += "  |  PHOTO ATTACHED"
            if item.safety_critical:
                evidence += "  |  SAFETY CRITICAL"
            view.ids.evidence.text = evidence
            view.ids.camera_button.disabled = not item.photo_required

        def refresh_ride(self) -> None:
            view = self.screen("ride")
            bike = self.active_bike()
            summary = self.repository.ride_summary(bike.bike_id) if bike else {"count": 0, "miles": 0}
            view.ids.live_distance.text = f"{self.tracker.distance_miles:.1f} mi"
            view.ids.tracker_state.text = "TRACKING GPS" if self.tracker.active else "READY TO TRACK"
            view.ids.tracker_button.text = "STOP + SAVE TRIP" if self.tracker.active else "START GPS MILEAGE"
            view.ids.ride_history.text = (
                f"{summary['miles']:.1f} tracked miles across {summary['count']} completed rides."
            )

        def refresh_service(self) -> None:
            view = self.screen("service")
            bike = self.active_bike()
            if not bike:
                return
            tasks = self.repository.list_maintenance_tasks(bike.bike_id)
            lines = []
            for task in tasks[:10]:
                due = f"{task['due_mileage']:,} mi" if task["due_mileage"] else "reference"
                lines.append(f"{task['category']}  |  {task['title']}\n{due}  |  {task['notes'][:90]}")
            view.ids.task_list.text = "\n\n".join(lines)


else:
    class MotoLensApp:  # pragma: no cover - only used to explain missing GUI dependencies.
        def run(self) -> None:
            raise RuntimeError(
                "Kivy and KivyMD are required for the visual app. "
                "Run `python3 main.py --test` for the headless service suite."
            )


class MotoLensTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="motolens-tests-"))
        self.repository = MotoRepository(self.temp_dir)
        self.bike = self.repository.create_bike(
            year=2024,
            make="Yamaha",
            model="MT-07",
            mileage=1200,
            notes="Needs a baseline check",
        )

    def tearDown(self) -> None:
        self.repository.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sensitive_fields_are_encrypted_at_rest(self) -> None:
        raw = self.repository.conn.execute(
            "SELECT encrypted_notes FROM bikes WHERE bike_id=?", (self.bike.bike_id,)
        ).fetchone()["encrypted_notes"]
        self.assertNotIn("baseline", raw)
        self.assertEqual(self.repository.get_bike(self.bike.bike_id).notes, "Needs a baseline check")

    def test_baseline_inspection_creates_every_area_and_report(self) -> None:
        session = self.repository.start_inspection(self.bike.bike_id)
        items = self.repository.list_inspection_items(session)
        self.assertEqual(len(items), len(INSPECTION_TEMPLATES))
        for item in items:
            if item.photo_required:
                photo = self.temp_dir / f"{item.item_key}.jpg"
                photo.write_bytes(b"inspection evidence")
                self.repository.attach_photo(item.item_id, str(photo))
            self.repository.update_inspection_item(item.item_id, STATUS_PASS)
        report = self.repository.finalize_inspection(session)
        self.assertEqual(report["health_score"], 100)
        self.assertEqual(self.repository.get_bike(self.bike.bike_id).state, "READY")
        self.assertTrue(list(self.repository.backups_dir.glob("motolens-*.db")))

    def test_report_requires_completed_checklist(self) -> None:
        session = self.repository.start_inspection(self.bike.bike_id)
        with self.assertRaisesRegex(ValueError, "Finish all"):
            self.repository.finalize_inspection(session)

    def test_gps_distance_and_trip_mileage(self) -> None:
        tracker = RideTracker(self.repository)
        tracker.gps = None
        tracker.start(self.bike.bike_id, "DoorDash")
        tracker.add_point(40.7128, -74.0060)
        tracker.add_point(40.7228, -74.0060)
        distance = tracker.stop()
        self.assertGreater(distance, 0.5)
        self.assertGreaterEqual(self.repository.get_bike(self.bike.bike_id).mileage, 1201)
        self.assertEqual(self.repository.ride_summary(self.bike.bike_id)["count"], 1)

    def test_rotating_backup_limit(self) -> None:
        for _ in range(7):
            self.repository.backup_database()
        self.assertEqual(len(list(self.repository.backups_dir.glob("motolens-*.db"))), 5)


def execute_motolens_test_suite() -> unittest.result.TestResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MotoLensTests)
    return unittest.TextTestRunner(verbosity=2).run(suite)


def main() -> int:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--test", action="store_true", help="run the headless test suite")
    args = parser.parse_args()
    if args.test:
        return 0 if execute_motolens_test_suite().wasSuccessful() else 1
    if not HAS_GUI:
        print(
            "MotoLens services are ready, but Kivy/KivyMD are not installed.\n"
            "Run `python3 main.py --test` for the headless suite or install the GUI dependencies."
        )
        return 0
    MotoLensApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
