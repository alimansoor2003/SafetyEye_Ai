# SafetyEye AI

Industrial PPE safety monitoring. Detects hardhat / safety-vest violations on live video, holds them
through a per-track timer, and writes an auditable bilingual HSE incident record.

The frozen architectural contract is [EDGESENTINEL_BUILD_SPEC.md](EDGESENTINEL_BUILD_SPEC.md).
Change decisions there first, then in code.

**Current state:** M1 (vision core) implemented. M2–M4 not started.

---

## Environment — read before installing

This machine resolves the spec's open hardware questions as follows:

| Item | Detected | Consequence |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop (Ada) | `cu121` is correct — the Blackwell/`cu128` branch does not apply |
| Python | 3.12.10 (installed alongside 3.14.3) | Use 3.12 — PyTorch publishes no wheels for 3.14 |
| Installed stack | torch 2.5.1+cu121, ultralytics 8.3.0, numpy 1.26.4, opencv-python 4.10.0.84 | `torch.cuda.is_available()` verified `True` |

The `.venv` in this repo is already built on 3.12 with CUDA working. To recreate it from scratch:

## Setup

```bash
py -3.12 -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install -r requirements.txt
```

Then copy `.env.example` to `.env` and get the weights (below).

## Getting the weights

The spec calls for the Roboflow *Construction Site Safety* class set — 10 classes (`Hardhat`, `Mask`,
`NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`, `machinery`,
`vehicle`), CC BY 4.0. Roboflow Universe reliably distributes the *dataset*; it does not reliably
offer a downloadable `.pt`. So fetch the dataset and train locally — ~30–60 min on the RTX 4050.

Add your key from https://app.roboflow.com (Settings → API Keys) to `.env` as `ROBOFLOW_API_KEY`,
then:

```bash
pip install roboflow
```

```bash
python scripts/fetch_dataset.py
```

**Immediately repair OpenCV afterwards** — `roboflow` depends on `opencv-python-headless`, which
shadows `opencv-python` and makes `cv2.imshow` raise, silently breaking `--show`:

```bash
pip install --force-reinstall opencv-python==4.10.0.84
```

Then train:

```bash
python scripts/train.py --data datasets/construction-site-safety/data.yaml
```

This writes `models/ppe_yolov8.pt` and prints per-class precision/recall. Read that table before
trusting the single global `conf_threshold: 0.75` — spec risk #2 warns `NO-Safety Vest` is usually
the weaker class, and this is where you'd find out.

Training rather than downloading a stranger's checkpoint is also the safer call: a `.pt` is a Python
pickle that executes code on load. If you do source weights elsewhere, make sure you trust the
publisher.

## M1 — vision core

Standalone. No FastAPI, no Claude, no database.

```bash
python -m vision.run --camera CAM-01 --show
```

Prints the resolved torch/CUDA build and GPU name at startup, then emits `incident.created` JSON on
stdout and writes an annotated JPEG to `evidence/` for every violation that survives the state machine.

Useful flags:

- `--show` — preview window with overlays (`q` or `Esc` to quit)
- `--frames` — also print throttled `detection.frame` events at 5 Hz
- `--source path/to/clip.mp4` — override the configured camera source. File sources run violation
  timers on the clip's own timebase, not the wall clock, so a clip that decodes in 2s still has to
  hold a violation for 2 clip-seconds to fire
- `--device cpu` — bypass the CUDA hard-fail for logic testing only; throughput will not meet 15fps

### Verified on hardware — 2026-08-19

Live webcam run at 1920x1080, RTX 4050, `models/ppe_yolov8.pt`:

- 14.5 fps steady state (target is 15)
- `NO-Hardhat` detected at 0.97, `NO-Safety Vest` at 0.86 — both well clear of the 0.75 threshold
- Fired at exactly 2.0s hold; cooldown then suppressed 120 consecutive frames
- Stable ByteTrack ID, `reid_churn_discards=0`
- Evidence JPEG written with correct overlays

### Acceptance

Walking past the camera without a hardhat for >2s prints exactly one event. Standing there for a
further 60s prints nothing more — the 120s per-track cooldown holds. On exit the run logs
`tracks_seen`, `events_emitted`, `events_suppressed_cooldown` and `reid_churn_discards`; the last of
these is the ByteTrack re-ID churn signal called out as risk #1 in the spec.

## M2 — backend + persistence

```bash
.venv\Scripts\python.exe -m backend.main
```

Serves on `http://127.0.0.1:8000`. Auth uses `EDGESENTINEL_AUTH_TOKEN` from `.env`, supplied either
as `Authorization: Bearer <token>` or as `?token=<token>`. The query form exists because an MJPEG
`<img src>` and a browser `WebSocket` cannot set request headers — header-only auth would leave
`/video_feed` and `/ws/events` open.

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /api/health` | none | Liveness: fps, GPU, queue depth, camera state, pending count |
| `GET /api/incidents` | yes | Paged history — `limit`, `offset`, `status`, `camera_id` |
| `GET /api/incidents/{report_id}` | yes | One incident |
| `GET /video_feed` | yes | MJPEG, 720p / 15fps / q70 |
| `GET /evidence/{file}` | yes | Annotated violation JPEG (path-traversal confined) |
| `WS /ws/events` | yes | `detection.frame`, `incident.created`, `system.status`; replays recent history on connect |

### Verifying M2 by hand

With the server running, in another terminal:

```bash
curl -s http://127.0.0.1:8000/api/health
```

```bash
curl -s "http://127.0.0.1:8000/api/incidents?token=leap-demo-token&limit=5"
```

```bash
curl -s -o NUL -w "%{http_code}\n" http://127.0.0.1:8000/api/incidents
```

That last one must print `401`. To watch the live feed, open this in a browser:

```
http://127.0.0.1:8000/video_feed?token=leap-demo-token
```

Then stand in front of the camera without a hardhat for more than two seconds and re-run the
incidents query — a new row appears with `status: "pending"`.

### M2 notes

- **Performance:** annotation and JPEG encoding are throttled to `stream.fps`. Doing that work on
  every captured frame dropped the loop from ~20fps to 7.2fps for frames nobody ever sees.
- **`detections` table** only stores rows for non-compliant tracks. Persisting every clean frame at
  5Hz would add roughly 430k rows/day with no forensic value.
- **aiosqlite, not SQLAlchemy** — the access pattern is a handful of hand-written statements.
- Enrichment columns (`risk_level`, `summary_en`, `report_ar`, …) exist and stay `null` until M3.

## M3 — Gemini HSE agent

Set `GEMINI_API_KEY` in `.env`, then start the server as above. Incidents are recorded instantly
and enriched asynchronously; the detection loop never waits on the API.

### Provider chain

Amendment A6 routes each incident through `gemini-3.7-flash` → `gemini-3.6-flash` →
`claude-sonnet-5`, moving to the next only when a budget is exhausted. Both vendors produce the
same validated `HSEIncidentReport`: Gemini via `response_schema`, Anthropic via forced tool-use as
the original spec §7 specified. Neither path parses free-text JSON.

`ANTHROPIC_API_KEY` is optional. Without it the chain is Gemini-only and enrichment pauses when
that budget runs out.

### API budget discipline

The Gemini **free tier allows 20 requests/day/model**, which is far below spec §7's 200
incidents/day. Enable billing before the demo.

The agent is built so an exhausted budget degrades gracefully rather than destructively:

- A `429`/`RESOURCE_EXHAUSTED` aborts that model immediately instead of spending its remaining
  retries — retrying a declined budget only costs more budget.
- A per-**day** quota pauses enrichment for 15 minutes, not the ~48s the API suggests. That hint is
  a token-bucket refill, and honouring it literally burns the next day's allowance too.
- Quota blocks are **not** counted as incident attempts. A billing state must never push a valid
  HSE record to `failed`.
- `/api/health` reports `enrichment_paused_seconds` so the dashboard can say "waiting for API
  budget" rather than looking broken.

Measured on the real backlog of 17 pending incidents with the quota already spent: **4 API calls
total, then a clean pause**. The previous behaviour would have attempted ~136.

Rows stay `pending` throughout and enrich on the next restart once budget returns.

## Tests

50 tests, none of which need a GPU, camera, weights, or an API key:

```bash
python tests/run_all.py
```

- `test_state_machine.py` — the IDLE/ARMING/FIRED contract, including the spec's exact acceptance
  case (2s hold fires once, a further 60s fires nothing) and the circuit breaker.
- `test_associator.py` — PPE-to-person binding geometry: band gating, confidence tie-breaks between
  `Hardhat` and `NO-Hardhat`, and overlapping people.
- `test_pipeline_smoke.py` — end-to-end wiring on a synthetic clip with a scripted stand-in for
  YOLOv8. Covers capture, association, timers, evidence JPEG writing, report-ID sequencing, and both
  WebSocket payload shapes from spec §6.
- `test_backend_m2.py` — boots the real FastAPI app against a synthetic camera: violation batching,
  SQLite rows, Riyadh time rendering, auth on every protected route, path-traversal rejection,
  MJPEG framing, and WebSocket history replay.
- `test_agent_m3.py` — enrichment policy with a faked agent: the offline-survival guarantee
  (transient failure stays `pending`, unusable report goes `failed`), backlog drain on reconnect,
  the Arabic-script guard against transliteration, and quota handling parsed from the verbatim 429
  this project actually received.

Only the model forward pass is unverified — everything around it runs. These pass on Python 3.14
with `pydantic`, `pyyaml`, `numpy`, `opencv-python` and `python-dotenv` installed, so the logic can
be exercised before the 3.12 environment exists.

## Layout notes

Two additions to the spec's §2 tree:

- `config.py` — Pydantic config loader. The spec's layout has no config module, but `vision/` and
  `backend/` both need to read `config.yaml` and validate it at boot.
- `tests/` — M1 acceptance and wiring tests.
- `scripts/` — dataset fetch and training, needed to produce `models/ppe_yolov8.pt`.

`config.yaml` also gains `violation.track_lost_seconds`, which §8 of the spec requires (`track lost >
1.0s → IDLE`) but §3 omitted.
