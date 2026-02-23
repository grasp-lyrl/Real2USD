"""
Legacy entrypoint: delegates to scripts/plot_overlays.py which produces
bbox-vs-GT and (when PC available) bbox+GT on point cloud, both axis-aligned.
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from scripts.plot_overlays import main

if __name__ == "__main__":
    main()
