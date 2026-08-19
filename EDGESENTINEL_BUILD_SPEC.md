# EdgeSentinel AI — Locked Build Spec (v1)

> **Purpose of this file.** This is the frozen architectural contract agreed before implementation.
> Drop it in the repo root (or `docs/`) so any agent session opened against this codebase inherits
> the full decision history without re-litigating it. If you change a decision, change it *here first*.

**Status:** M1 complete and hardware-verified. M2 in progress.
**Target:** Local repository, NVIDIA RTX + CUDA, fully offline-capable.

### Amendments since v1

| # | Date | Change | Reason |
|---|---|---|---|
| A1 | 2026-08-19 | Class set is **25 classes**, not 10 | Roboflow dataset v30 adds vehicle/equipment subtypes. The 5 acted-on classes are unchanged; the other 20 map to ignore |
| A2 | 2026-08-19 | Incidents carry a **`violations` array**, not a single `violation_type` | Simultaneous violations on one `track_id` are batched into one incident. Halves LLM calls per non-compliant worker and keeps the bilingual report unified |
| A3 | 2026-08-19 | `violation.track_lost_seconds` added to `config.yaml` | §8 requires it; §3 omitted it |
| A4 | 2026-08-19 | `lapx` added to requirements | ByteTrack imports `lap`; ultralytics does not install it |
| A5 | 2026-08-19 | **LLM is Gemini, not Claude.** `gemini-3.7-flash` default, `gemini-3.6-flash` fallback (2.5-flash is retired — 404 for new users) | Provider decision by the project owner. §7's structured-output guarantee is preserved via `response_schema` + `response.parsed`, which is a stronger contract than forced tool-use: the SDK returns a validated Pydantic instance, so there is still no free-text JSON parsing |

| A6 | 2026-08-19 | **Cross-provider fallback.** Route chain is `gemini-3.7-flash` → `gemini-3.6-flash` → `claude-sonnet-5` | The Gemini free tier is 20 requests/day/model, far below §7's 200 incidents/day. A second vendor turns budget exhaustion into a slower report instead of no report. Anthropic uses forced tool-use, exactly as the original §7 specified — both providers still validate through the same Pydantic model |

Confirmed unchanged: `conf_threshold` stays **0.75** (real-camera confidences are 0.86–0.98),
`display_timezone` stays **Asia/Riyadh** for LEAP, `blur_faces` stays **false** for the demo.

---

## 1. Locked decisions

| Area | Decision | Notes |
|---|---|---|
| CV model | **Pretrained YOLOv8 PPE weights** (Roboflow *Construction Site Safety* class set) | Fine-tune later only if site accuracy is poor |
| Classes | `Person`, `Hardhat`, `Safety Vest`, `NO-Hardhat`, `NO-Safety Vest` | Roboflow set also ships `Mask`, `NO-Mask`, `machinery`, `vehicle` — map to ignore |
| Confidence | `> 0.75` | Configurable per class |
| Tracking | **ByteTrack** via `model.track(persist=True, tracker="bytetrack.yaml")` | Mandatory — the 2s rule is meaningless without stable IDs |
| Violation rule | Non-compliant state held **> 2.0s** on a single `track_id` | Per-track timer, not per-frame |
| Runtime | NVIDIA RTX, CUDA | `torch` + `torchvision` from the CUDA wheel index |
| Video transport | **MJPEG over HTTP** `/video_feed` @ 720p / 15fps / JPEG q70 | Pixels only |
| Event transport | **WebSocket JSON** `/ws/events` | Detections + incidents only. Never carries frames |
| LLM | `claude-sonnet-5` default, `claude-haiku-4-5` fallback flag | `claude-3-5-sonnet` from the original PRD is legacy — do not use |
| LLM coupling | **Fully async.** Detection loop never awaits the API | `asyncio.Queue` → background worker pool |
| Persistence | **SQLite** + `evidence/*.jpg` | Incident row written *before* the LLM call |
| Hardware | **ESP32-S3 = alarm output only** | MQTT → local Mosquitto `127.0.0.1:1883`; USB serial fallback |
| Cloud deps | **Zero** for the hardware trigger loop | Must work on exhibition Wi-Fi with no internet |
| Zones | One static `zone_id` per camera, e.g. `ZONE-01-MAIN-ENTRANCE` | No ROI polygons in v1 |
| Time | Store **UTC ISO-8601**, render **Asia/Riyadh** (UTC+3) | `zoneinfo`, never naive datetimes |
| Arabic | `dir="rtl"` + `lang="ar"` on the Arabic block **only**, Noto Naskh Arabic | Do not set RTL page-wide |
| Report ID | `INC-{YYYYMMDD}-{ZONE}-{SEQ:04d}` | e.g. `INC-20260818-ZONE01-0007` |

### ⚠ CUDA wheel caveat
`cu121` is correct for RTX 20/30/40-series. **RTX 50-series (Blackwell) will not run on `cu121`** — it needs
`cu128` or newer. The setup script must detect this rather than hardcode:

```bash
# RTX 20/30/40
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# RTX 50 (Blackwell)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

M1 must print `torch.cuda.is_available()`, `torch.cuda.get_device_name(0)` and the compiled CUDA
version at startup and **hard-fail loudly** if CUDA is missing, rather than silently falling back to CPU.

---

## 2. Repository layout

```
edgesentinel/
├── .env.example
├── .env                        # gitignored
├── .gitignore
├── requirements.txt
├── config.yaml
├── README.md
├── EDGESENTINEL_BUILD_SPEC.md  # this file
│
├── models/
│   └── ppe_yolov8.pt           # gitignored (LFS or download script)
│
├── evidence/                   # gitignored — annotated violation JPEGs
├── data/
│   └── edgesentinel.db         # gitignored
│
├── vision/
│   ├── __init__.py
│   ├── capture.py              # camera source abstraction (webcam / RTSP / file)
│   ├── detector.py             # YOLOv8 + ByteTrack wrapper
│   ├── associator.py           # PPE box → Person track binding
│   ├── state_machine.py        # per-track violation timers + cooldown
│   ├── annotate.py             # bounding box overlay rendering
│   └── run.py                  # M1 standalone CLI entrypoint
│
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app + lifespan
│   ├── schemas.py              # Pydantic models (events, incidents, LLM output)
│   ├── db.py                   # SQLite schema + async access
│   ├── ws.py                   # ConnectionManager / broadcast
│   ├── stream.py               # MJPEG generator
│   ├── routes.py               # REST: /api/incidents, /api/health
│   └── pipeline.py             # vision → queue → workers wiring
│
├── agent/
│   ├── __init__.py
│   ├── client.py               # Anthropic client, retry/backoff
│   ├── prompts.py              # system prompt + few-shot
│   ├── tools.py                # structured output tool schema
│   └── worker.py               # queue consumer
│
├── hardware/
│   ├── mqtt_publisher.py
│   ├── serial_bridge.py        # USB fallback
│   └── esp32_s3/
│       └── edgesentinel_alarm.ino
│
└── frontend/
    ├── index.html              # Tailwind CDN, single file
    └── static/
        └── fonts/              # Noto Naskh Arabic (bundle locally — no CDN offline)
```

**Offline note:** Tailwind CDN and Google Fonts will **not** load on exhibition Wi-Fi with no internet.
Vendor both into `frontend/static/` before the demo. This is a real failure mode, not a nicety.

---

## 3. `config.yaml`

```yaml
cameras:
  - camera_id: CAM-01
    zone_id: ZONE-01-MAIN-ENTRANCE
    zone_label_en: "Main Entrance"
    zone_label_ar: "المدخل الرئيسي"
    source: 0                  # int = webcam index, str = RTSP/file path
    capture_width: 1920
    capture_height: 1080
    capture_fps: 30

detection:
  weights: models/ppe_yolov8.pt
  device: cuda:0
  imgsz: 640
  conf_threshold: 0.75
  iou_threshold: 0.45
  tracker: bytetrack.yaml
  monitored_violations: [NO-Hardhat, NO-Safety Vest]

violation:
  persist_seconds: 2.0         # must hold this long to fire
  clear_seconds: 1.5           # must be clean this long to reset
  cooldown_seconds: 120        # per (track_id, violation_type)
  max_incidents_per_minute: 10 # global circuit breaker

stream:
  width: 1280
  height: 720
  fps: 15
  jpeg_quality: 70

agent:
  model: claude-sonnet-5
  fallback_model: claude-haiku-4-5
  use_fallback: false
  max_tokens: 1500
  worker_count: 2
  max_retries: 4
  backoff_base_seconds: 2

hardware:
  mqtt_enabled: true
  mqtt_host: 127.0.0.1
  mqtt_port: 1883
  serial_fallback: true
  serial_port: /dev/ttyUSB0
  serial_baud: 115200

storage:
  db_path: data/edgesentinel.db
  evidence_dir: evidence
  retention_days: 30
  blur_faces: false

locale:
  display_timezone: Asia/Riyadh
```

---

## 4. `.env.example`

```
ANTHROPIC_API_KEY=sk-ant-...
EDGESENTINEL_AUTH_TOKEN=change-me-before-demo
LOG_LEVEL=INFO
```

Loaded with `python-dotenv`. The API key is **server-side only** — the browser must never see it,
and the dashboard must never call the Anthropic API directly.

---

## 5. SQLite schema

```sql
CREATE TABLE IF NOT EXISTS incidents (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id         TEXT UNIQUE NOT NULL,      -- INC-20260818-ZONE01-0007
    camera_id         TEXT NOT NULL,
    zone_id           TEXT NOT NULL,
    track_id          INTEGER NOT NULL,
    violations        TEXT NOT NULL,             -- A2: JSON array, e.g. ["NO-Hardhat","NO-Safety Vest"]
    confidence        REAL NOT NULL,             -- max across the batched violations
    duration_seconds  REAL NOT NULL,
    detected_at_utc   TEXT NOT NULL,             -- ISO-8601 UTC
    evidence_path     TEXT,

    status            TEXT NOT NULL DEFAULT 'pending',
                      -- pending | enriched | failed
    attempts          INTEGER NOT NULL DEFAULT 0,
    last_error        TEXT,

    risk_level        TEXT,                      -- Low|Medium|High|Critical
    recommended_protocol TEXT,
    summary_en        TEXT,
    report_ar         TEXT,
    model_used        TEXT,
    enriched_at_utc   TEXT
);

CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_incidents_time   ON incidents(detected_at_utc DESC);
```

**Critical ordering:** the row is INSERTed with `status='pending'` the moment the violation fires and
the evidence JPEG is written. The Claude call is a later UPDATE. The audit trail must never depend on
the API being reachable — that is what makes this defensible as an HSE record.

A startup task re-queues any rows still `pending` from a previous run.

---

## 6. Event contracts (WebSocket JSON)

All messages: `{"type": "...", "data": {...}}`

**`detection.frame`** — high frequency, throttled to ~5/s:
```json
{ "type": "detection.frame", "data": {
    "camera_id": "CAM-01",
    "ts_utc": "2026-08-18T09:14:22.412Z",
    "compliant": false,
    "tracks": [
      {"track_id": 7, "bbox": [120,88,340,610], "label": "Person",
       "conf": 0.93, "violations": ["NO-Hardhat"], "violation_elapsed": 1.4}
    ]
}}
```

**`incident.created`** — fires instantly on violation, before the LLM:
```json
{ "type": "incident.created", "data": {
    "report_id": "INC-20260818-ZONE01-0007",
    "zone_id": "ZONE-01-MAIN-ENTRANCE",
    "violations": ["NO-Hardhat", "NO-Safety Vest"],
    "confidence": 0.91,
    "duration_seconds": 2.3,
    "detected_at_utc": "2026-08-18T09:14:24.700Z",
    "evidence_url": "/evidence/INC-20260818-ZONE01-0007.jpg",
    "status": "pending"
}}
```

**`incident.enriched`** — arrives seconds later, same `report_id`, dashboard patches the existing card:
```json
{ "type": "incident.enriched", "data": {
    "report_id": "INC-20260818-ZONE01-0007",
    "status": "enriched",
    "risk_level": "High",
    "recommended_protocol": "Halt work in the entrance corridor and escort the worker to the PPE station before re-entry.",
    "summary_en": "A worker entered the main entrance zone without a hardhat and remained non-compliant for 2.3 seconds...",
    "report_ar": "تقرير حادثة سلامة مهنية ...",
    "model_used": "claude-sonnet-5"
}}
```

**`system.status`** — heartbeat: FPS, GPU name, queue depth, MQTT connected, ESP32 last-seen.

---

## 7. Claude agent contract

Structured output enforced via a tool schema — **no free-text JSON parsing.**

```python
INCIDENT_REPORT_TOOL = {
    "name": "emit_hse_incident_report",
    "description": "Emit the validated bilingual HSE incident report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "risk_level": {
                "type": "string",
                "enum": ["Low", "Medium", "High", "Critical"]
            },
            "recommended_protocol": {
                "type": "string",
                "description": "One specific, immediately actionable on-site instruction for the supervisor."
            },
            "summary_en": {
                "type": "string",
                "description": "Executive-level incident description, 2-4 sentences, factual, no speculation."
            },
            "report_ar": {
                "type": "string",
                "description": "Formal Arabic HSE incident report in Modern Standard Arabic, suitable for a site supervisor. Must be complete Arabic prose, not a transliteration."
            }
        },
        "required": ["risk_level", "recommended_protocol", "summary_en", "report_ar"]
    }
}
```

Call with `tool_choice={"type": "tool", "name": "emit_hse_incident_report"}` to force the shape.
Validate the result through a Pydantic model server-side; on validation failure, one re-ask, then
mark `status='failed'` with `last_error`.

**Guardrails for the prompt:**
- Feed only the structured event fields (zone, violation type, confidence, duration, timestamp). Do **not** send the image in v1 — it triples cost and latency for marginal gain.
- Instruct the model to describe only what the payload states. No inventing worker names, injuries, or causes.
- The Arabic is a *parallel formal report*, not a translation of the English summary — different register, different audience.

**Cost envelope:** ~1.5k in / 800 out per report on `claude-sonnet-5` ≈ **$0.011/incident**.
200 incidents/day ≈ $2.20/day. The 120s per-track cooldown is what keeps this true — without it a single
stationary non-compliant worker generates hundreds of calls.

---

## 8. Violation state machine

Per `track_id`, per violation type:

```
IDLE
  └─ violation observed ──────────────► ARMING (start timer)

ARMING
  ├─ still violating, elapsed > 2.0s ─► FIRED  (emit incident, start cooldown)
  ├─ clean for > 1.5s ────────────────► IDLE   (reset timer)
  └─ track lost > 1.0s ───────────────► IDLE   (discard)

FIRED
  ├─ cooldown active (120s) ──────────► SUPPRESSED (count, do not emit)
  └─ cooldown expired + still bad ────► FIRED again
```

Track loss handling matters: ByteTrack will drop and re-issue IDs on occlusion. A re-issued ID
restarts the timer — accept this in v1, but log re-ID churn so it's visible if it becomes a false-negative source.

---

## 9. MQTT contract (ESP32-S3 alarm)

| Topic | Direction | Payload |
|---|---|---|
| `edgesentinel/alarm` | backend → ESP32 | `{"state":"HAZARD","risk":"High","zone":"ZONE-01-MAIN-ENTRANCE","ttl":15}` |
| `edgesentinel/alarm` | backend → ESP32 | `{"state":"CLEAR"}` |
| `edgesentinel/heartbeat` | ESP32 → backend | `{"device":"esp32-s3-01","uptime":8123,"rssi":-54}` every 5s |

Backend marks the device **offline** if no heartbeat for 15s and surfaces that on the dashboard —
a silent alarm module that everyone assumes is working is worse than no alarm module.

Beacon behaviour: `Critical`/`High` → solid red + buzzer. `Medium` → slow blink, no buzzer.
`Low` → single chirp. Auto-clears after `ttl` seconds if no follow-up message (so a backend crash
doesn't leave the siren stuck on).

---

## 10. Milestones & acceptance criteria

### M1 · Vision core — standalone, no server
`python -m vision.run --camera CAM-01`

- Prints CUDA device name and torch/CUDA versions; hard-fails if CUDA unavailable
- Opens the configured source, runs YOLOv8 + ByteTrack, applies 0.75 threshold
- Runs the state machine and prints event JSON to stdout
- Saves annotated JPEGs to `evidence/`
- Optional `--show` window with overlays

**Accept when:** walking in front of the camera without a hardhat for >2s prints exactly one event,
and standing there for 60s prints exactly one more (cooldown proven). No FastAPI, no Claude in this milestone —
if the CV is wrong nothing downstream matters.

### M2 · Backend + persistence
- FastAPI + lifespan, SQLite schema created on boot
- Vision loop as an async background task in a thread executor
- `/video_feed` MJPEG, `/ws/events`, `/api/incidents`, `/api/health`
- Incidents persist as `pending`; pending rows re-queued on restart
- Bearer-token auth on WS + REST

**Accept when:** violations appear as rows in SQLite with evidence files, `/video_feed` renders in a browser,
and a `wscat` client receives `incident.created` within ~100ms of the violation.

### M3 · Claude HSE agent
- Structured-output call, Pydantic validation, retry with exponential backoff
- Worker pool draining the queue, cooldown + dedupe enforced
- Report written back, `incident.enriched` broadcast
- Fallback model flag exercised

**Accept when:** pulling the network cable mid-run still records incidents, and reconnecting drains
the pending backlog into complete bilingual reports.

### M4 · Dashboard + ESP32 alarm
- Tailwind UI, live feed with overlays, `COMPLIANT` / `HAZARD DETECTED` badge
- Streaming bilingual log, correct RTL rendering, history replay on connect
- MQTT publish + ESP32 sketch + heartbeat/offline indicator
- Assets vendored locally for offline operation

**Accept when:** the full loop runs end-to-end with the machine's internet disabled except for the
Anthropic API, and the beacon fires within ~1s of the badge turning red.

---

## 11. Known risks

1. **Re-ID churn under occlusion** — ByteTrack reassigns IDs, restarting timers. Mitigate by logging churn rate; consider a lightweight ReID embedding only if it proves to be a real false-negative source.
2. **Class imbalance in the pretrained weights** — `NO-Safety Vest` is typically weaker than `NO-Hardhat`. Validate per-class precision on your actual lighting before trusting a single global 0.75 threshold.
3. **Exhibition Wi-Fi** — assume it fails. Vendor every frontend asset, run Mosquitto locally, keep the serial bridge tested as a real fallback, not a theoretical one.
4. **Face privacy in evidence JPEGs** — `blur_faces` is off by default. Turn it on before any real site deployment (UAE PDPL / KSA PDPL).
5. **Demo failure mode** — if the Anthropic API is unreachable at the booth, the dashboard must still look alive: incidents appear, badges flip, alarm fires, and report cards show a clear "generating…" state rather than a broken card.
