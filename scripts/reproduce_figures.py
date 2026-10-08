"""Render saved paper tables only: no inference, simulation, or statistical fit."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PLOTS = {
    'physical-response': 'plot_fig2_physical_response_v5.py',
    'joint-attribution': 'plot_fig3_joint_attribution_v4.py',
    'reduced-compute': 'plot_fig_reduced_compute_v5.py',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', choices=['all', *PLOTS], default='all')
    args = parser.parse_args()
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = str(ROOT/'src') + os.pathsep + env.get('PYTHONPATH', '')
    # Bind to the release copy, never the historical experiment workspace.
    env['WAM_WORKSPACE'] = str(ROOT/'reference/workspace')
    env['WAM_DATA'] = str(ROOT/'reference/data')
    chosen = PLOTS if args.figure == 'all' else {args.figure:PLOTS[args.figure]}
    for name, filename in chosen.items():
        print('Rendering', name, flush=True)
        subprocess.run([sys.executable, '-B', str(ROOT/'reference/workspace/FastWAM/scripts'/filename)],
                       cwd=ROOT, env=env, check=True)


if __name__ == '__main__':main()
