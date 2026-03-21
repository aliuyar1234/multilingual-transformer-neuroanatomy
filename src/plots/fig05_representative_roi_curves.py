from __future__ import annotations

import argparse

from src.plots.generate_all_figures import _generate_fig05_representative_roi_curves
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 5 representative ROI curves.")
    parser.add_argument("--config", default="conf/base.yaml", help="Path to sequel config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = _generate_fig05_representative_roi_curves(load_config(args.config))
    if outputs is None:
        raise SystemExit("Figure 5 inputs are not available.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
