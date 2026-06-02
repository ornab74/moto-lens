[app]

title = MotoLens

# Keep the existing application ID so this remains an update to the current
# Google Play internal-testing listing. Changing either value creates a new app.
package.name = lightcal
package.domain = com.qroadscan

source.dir = .
source.main = main.py
source.include_exts = py,png,jpg,jpeg,kv,atlas,json,txt,md,pdf

version = 0.2.0
android.version_code = 1024834

python.version = 3.11

# Android package dependencies only. Desktop-only native extensions such as
# PyMuPDF, nh3, and psutil require dedicated python-for-android recipes and are
# intentionally handled by runtime fallbacks in main.py.
requirements = python3==3.11.9,hostpython3==3.11.9,kivy==2.3.1,kivymd,cryptography,plyer

orientation = portrait
fullscreen = 0

android.permissions = INTERNET,CAMERA,ACCESS_FINE_LOCATION,ACCESS_COARSE_LOCATION,POST_NOTIFICATIONS,FOREGROUND_SERVICE,WAKE_LOCK

android.sdk_path = /usr/local/lib/android/sdk
android.api = 35
android.minapi = 26
android.ndk_api = 26
android.ndk = 27c
android.build_tools_version = 35.0.0
android.archs = arm64-v8a

p4a.bootstrap = sdl2

android.logcat_filters = python:D,Python:D,ActivityManager:I,WindowManager:I
android.allow_backup = False

[buildozer]
log_level = 2
warn_on_root = 1
build_dir = .buildozer
android.accept_sdk_license = True

