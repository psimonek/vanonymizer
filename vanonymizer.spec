# vanonymizer.spec

import sys
from PyInstaller.utils.hooks import collect_all

block_cipher = None

ultra_datas, ultra_bins, ultra_hidden = collect_all("ultralytics")
insight_datas, insight_bins, insight_hidden = collect_all("insightface")

# --- FFmpeg podle OS ---
ffmpeg_bin = []

if sys.platform == "darwin":
    ffmpeg_bin.append(("bin/mac/ffmpeg", "bin"))

elif sys.platform == "linux":
    ffmpeg_bin.append(("bin/linux/ffmpeg", "bin"))

elif sys.platform == "win32":
    ffmpeg_bin.append(("bin/windows/ffmpeg.exe", "bin"))

a = Analysis(
    ["cli.py"],
    pathex=["."],

    binaries=ultra_bins + insight_bins + ffmpeg_bin,

    datas=ultra_datas + insight_datas + [
        ("model", "model"),
    ],

    hiddenimports=ultra_hidden + insight_hidden + [
        "torch",
        "torchvision",
        "skimage",
        "skimage.measure",
        "scipy",
        "numpy",
        "charset_normalizer",
        "onnxruntime",
        "matplotlib",
        "matplotlib.pyplot",
    ],

    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="vanonymizer",
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="vanonymizer",
)
