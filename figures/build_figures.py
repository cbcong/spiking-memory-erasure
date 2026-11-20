"""Rebuild the manuscript and supplementary figures using CPU and local data."""
import argparse
from pathlib import Path
import subprocess
import sys

import matplotlib as mpl


ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "results" / "confirm_strong_20261119_v1" / "report.json"


def build(output_root):
    manuscript = output_root / "Manuscript" / "figs"
    supplement = output_root / "Supplementary" / "figs"
    manuscript.mkdir(parents=True, exist_ok=True)
    supplement.mkdir(parents=True, exist_ok=True)

    # The versioned plotting modules configure Matplotlib at import time.
    with mpl.rc_context():
        from .plot_manuscript_v100 import trajectory
        from .plot_workflow_v120 import module
        module(manuscript)
        trajectory(manuscript)
    with mpl.rc_context():
        from .plot_drive_distribution import main as drive_distribution
        drive_distribution(manuscript / "replication.pdf")
    with mpl.rc_context():
        from .plot_confirmation_deadlines import render as deadline_sensitivity
        deadline_sensitivity(manuscript / "deadline_sensitivity.pdf")
    with mpl.rc_context():
        from .plot_ring_scaling import fold_lines, last_spikes
        fold_lines(*last_spikes(), output_pdf=manuscript / "ring_scaling.pdf")
    with mpl.rc_context():
        from .plot_ring_raster import main as spike_raster
        spike_raster(manuscript / "ring_raster.pdf")
    subprocess.run([sys.executable, "-B", "-m", "figures.plot_supplement_completion",
                    str(REPORT), str(supplement / "paired_completion.pdf")], cwd=ROOT, check=True)
    with mpl.rc_context():
        from .plot_graph_drive_shift import render as graph_drive_shift
        graph_drive_shift(supplement / "graph_drive_shift.pdf")
    with mpl.rc_context():
        from .plot_three_module_phases import render as release_phases
        release_phases(supplement / "three_module_phases.pdf")
    from .plot_multiweak_layout import render_control, render_main
    render_main(manuscript / "multiweak_layout.pdf")
    render_control(supplement / "multiweak_size_control.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("reproduced_figures"),
                        help="Output directory (default: ./reproduced_figures).")
    build(parser.parse_args().output_root.resolve())


if __name__ == "__main__":
    main()
