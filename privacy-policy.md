# Privacy Policy - MotoLens

**Effective date:** June 1, 2026

This Privacy Policy describes how **MotoLens** (the "App") handles information
when you use it.

## 1. Developer

**Developer:** Graylan Janulis

**Contact:** [janulisgraylan@gmail.com](mailto:janulisgraylan@gmail.com)

**Country:** United States

## 2. What MotoLens does

MotoLens is a motorcycle maintenance companion. It can store a private garage,
guide inspections, track optional GPS ride mileage, download authorized service
manual PDFs, and provide optional AI-assisted maintenance research.

## 3. Data stored on your device

MotoLens may store:

- Motorcycle year, make, model, mileage, nickname, notes, and maintenance tasks
- Inspection status, inspection notes, and photos you choose to capture
- Ride purpose, distance, timestamps, and GPS route points when you start mileage tracking
- Downloaded manual PDFs, locally rendered reader pages, and indexed manual text
- AI Mechanic chat history and source citations
- An optional OpenAI API key that you enter after installation

Sensitive notes and GPS routes are encrypted before SQLite storage. A
user-supplied OpenAI API key is encrypted with AES-GCM and a scrypt-derived key.
On packaged Android builds, MotoLens wraps the installation seed with Android
Keystore. Local backups are retained on device.

No software storage mechanism can guarantee protection on a compromised or
rooted device. Keep your phone updated and use a screen lock.

## 4. Optional network activity

MotoLens uses the network only for features that need it:

- **OpenAI API:** If you configure an OpenAI API key and use an AI feature,
  MotoLens sends the relevant prompt and selected context to OpenAI. Photo
  review sends inspection images you choose to analyze. Manual research may
  send relevant manual excerpts and use OpenAI web search.
- **Manual downloads:** If you request a PDF manual download, your device
  connects to the selected manual host. That host receives ordinary network
  metadata such as your IP address and request timestamp.

OpenAI's handling of API data is governed by OpenAI's applicable terms and
privacy documentation. External manual hosts apply their own policies.

## 5. Permissions

MotoLens may request:

- **INTERNET** for AI requests and manual downloads
- **CAMERA** for guided inspection photos
- **PRECISE OR APPROXIMATE LOCATION** for ride mileage tracking that you start
- **NOTIFICATIONS** for optional report and maintenance reminders
- **FOREGROUND SERVICE** and **WAKE LOCK** to support active ride tracking

You can revoke permissions in Android settings. Features that need a revoked
permission may stop working.

## 6. What MotoLens does not include

MotoLens does not include advertising SDKs, behavioral analytics SDKs, or data
sales. MotoLens does not start GPS mileage tracking until you request it.

## 7. Retention and deletion

MotoLens data remains on your device until you remove it through the App,
clear the App's Android storage, or uninstall the App. Data sent to OpenAI or a
manual host is subject to that service's retention rules.

## 8. Children

MotoLens is not directed to children under 13, and the developer does not
knowingly collect personal information from children.

## 9. Changes

This policy may be updated as MotoLens changes. The effective date above will be
updated when the policy changes.

## 10. Contact

For privacy questions, contact
[janulisgraylan@gmail.com](mailto:janulisgraylan@gmail.com).
