# PeakLimitedErasure

This repository contains the inputs, Python scripts and saved runs for a set of spiking-circuit experiments. You can inspect the numbers from an archived run, repeat it from its original graph seeds, or redraw the figures. The commands below assume CPython 3.11 and a CPU. The graph experiments run offline once the Python dependencies are installed.

## Start with the stored results

Run these commands from the top-level directory.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -q
python -m experiments.tabulate_confirmation_methods
```

The last command prints one CSV row per schedule-selection rule. The columns give peak-amplitude totals by graph family, the total across families, mean elapsed ticks and cumulative input. For individual schedules, open [the archived report](results/confirm_strong_20261119_v1/report.json). Its companion [design file](results/confirm_strong_20261119_v1/design.json) records the seeds and checksums used for that run.

## Repeat the graph comparison

Use a fresh output directory. The first command copies the archived design there. The runner checks graph identities before it evaluates the schedules.

```sh
python -c "from pathlib import Path; import shutil; p=Path('reproduced/confirm'); p.mkdir(parents=True); shutil.copyfile('results/confirm_strong_20261119_v1/design.json',p/'design.json')"
python -m experiments.run_confirm_strong run --output reproduced/confirm
python -m experiments.tabulate_confirmation_methods --report reproduced/confirm/report.json
```

## Locate an experiment

| Input or output | Where to start |
| --- | --- |
| Register update rules | [Model source](integer_snn) |
| Same graphs at four time limits | `python -m experiments.run_confirmation_deadlines --source-root . --output reproduced/deadlines.json --workers 2` |
| Matched incoming weights | `python -m experiments.run_matched_block_mixing --output reproduced/matched.json` |
| Small non-ring graphs with exact checks | `python -m experiments.run_nonring_deadline_oracle --output reproduced/nonring.json` |
| Rings with different edge orders | `python -m experiments.run_weak_ring_spacing --report reproduced/spacing.json` |
| Individual register states | [State-sampling script](experiments/run_multiweak_legal_states.py) |
| Distance to the peak lower bound | `python -m experiments.audit_confirm_peak_gap` |
| Vector figures | `python -m figures.build_figures --output-root reproduced/figures` · [inputs for each figure](figures/README.md) |

## Optional input files

The following files are needed only for their respective experiments. Each link points to the original data provider.

- **N-MNIST.** Store the test ZIP at `data/nmnist/Test.zip` (0.170 GB). Run `python -m experiments.run_nmnist_reuse --output reproduced/nmnist.json`. The script checks the archive's byte count and checksum. To retrieve the ZIP through the script, add `--fetch`. [N-MNIST archive](https://doi.org/10.17632/468j46mzdv.1)
- **Spiking Heidelberg Digits.** Run `python -m experiments.prepare_shd` to fetch the two archives (0.169 GB combined). Their extracted files occupy another 0.348 GB. Allow additional space for computed features. Run `python -m experiments.build_shd_features`, then `python -m experiments.run_shd_query --output reproduced/shd`. Small fitted encoders are included. To refit them, use `python -m experiments.train_shd_encoder --output reproduced/encoders` and supply `--encoders reproduced/encoders` to the query script. [SHD downloads and terms](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)
- **Continuous latch.** Place the supplementary notebook from Spaeth et al. at `literature/spaeth_2020_s1.ipynb`. Create the directory first. Run `python -m experiments.run_spaeth_latch --output reproduced/spaeth.json` for the trace, or `python -m experiments.run_latch_task_comparison --output reproduced/latch.json` for the command sequence. The [archived phase observations](results/latch_task_comparison_20261119_v2/report.json) are also available directly. [Original notebook](https://doi.org/10.1371/journal.pone.0240267.s001)
