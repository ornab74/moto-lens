# MotoLens Garage

MotoLens is a single-file Kivy/KivyMD motorcycle maintenance prototype. The full
implementation lives in `main.py`.

## Product flow

- First-launch garage setup asks for motorcycle details, mileage, and notes.
- A full baseline inspection guides chain, tire, brake, fluid, control, light,
  suspension, and model-specific torque-wrench checks.
- Sensitive notes and GPS routes are encrypted before SQLite storage.
- Reports create rolling local database backups.
- GPS trip tracking records personal, DoorDash, and Uber Eats mileage.
- The ride tracker keeps an auditable trip ledger with route-point counts and
  stable short audit IDs while encrypting route coordinates at rest.
- Model-specific online research saves source-linked service intervals per bike.
- The Manual tab discovers or accepts an authorized direct HTTPS PDF, indexes
  searchable text chunks once, and lazily renders zoomable JPEG reader pages.
- A near-fullscreen manual reader, background neighbor-page prefetch, and an
  optional collapsed text preview keep large manuals responsive.
- The AI Mechanic tab retrieves relevant local manual pages before asking the
  configured model and stores source-linked chat evidence.
- The Settings tab encrypts an optional user-supplied OpenAI API key with
  AES-GCM and scrypt after installation.
- Optional OpenAI integration uses `gpt-5.5` for web research and photo review,
  plus `gpt-image-2` for bike and report artwork.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

Run the headless service suite without GUI dependencies:

```bash
python3 main.py --test
```

The existing helper also works:

```bash
bash run_tests.sh
```

## Configuration

Use the Settings tab to opt into storing a personal OpenAI API key after
installation. MotoLens saves only an AES-GCM ciphertext envelope inside SQLite;
the unlocked key remains in memory for the active session. Desktop development
can use `OPENAI_API_KEY` instead.

OpenAI's official guidance is still to keep API keys out of browsers and mobile
apps. Local encrypted storage reduces casual disclosure risk but does not make a
key impossible to extract from a compromised device. Use a restricted key,
monitor usage, and rotate it after suspected compromise.

MotoLens calls the OpenAI Responses API directly. Its manual discovery,
service-interval research, and maintenance brief prompts enable the built-in
`web_search` tool. Manual discovery searches for an authorized direct HTTPS PDF,
indexes it locally when found, retrieves relevant manual excerpts first, and
uses online search to fill evidence gaps.

Plyer exposes the camera, GPS, and notification hooks. A packaged mobile build
must still declare native camera, location, and notification permissions.

Only download manuals that the manufacturer or another authorized source makes
publicly available. MotoLens rejects non-HTTPS links, embedded URL credentials,
literal local/private IP hosts, unsafe redirect targets, non-PDF manual URLs,
files above 80 MB, and manuals above 800 pages.

## Storage boundary

MotoLens encrypts sensitive SQLite fields with AES-GCM when `cryptography` is
installed and keeps five timestamped backups. Its credential vault uses
AES-256-GCM, a scrypt-derived key, OS CSPRNG installation seeds, atomic private
files, and supplemental `psutil` device context. The encrypted user-key envelope
is stored in SQLite with parameterized queries. On packaged Android builds, the
installation seed is wrapped by Android Keystore and MotoLens refuses to
silently downgrade if that layer cannot initialize. Full database-at-rest
encryption still requires SQLCipher or a platform storage encryption layer.

Dynamic SQLite values use bound parameters. Text entering storage, prompts,
logs, and plain UI labels passes through an `nh3`-backed sanitizer with a
conservative fallback. PDF rendering uses `PyMuPDF`. The AI Mechanic knowledge
surface visualizes real local retrieval score, bounded chat-history compaction,
and query expansion counts. It is telemetry, not a cryptographic or quantum
computation claim.

Motorcycle service specifications vary by model and year. MotoLens intentionally
does not invent torque values, service intervals, or wear limits. Use the
official owner's manual, official service manual, and a qualified mechanic for
safety-critical decisions.

