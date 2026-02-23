"""
Thin wrapper: delegate to scripts/run_eval so that existing imports
(e.g. sweep_eval: from run_eval import evaluate, _filter_predictions_by_config, ...) keep working.
"""
from pathlib import Path
import sys
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from scripts.run_eval import (
    _filter_predictions_by_config,
    _load_eval_config,
    _read_json,
    evaluate,
    main,
)

if __name__ == "__main__":
    main()
