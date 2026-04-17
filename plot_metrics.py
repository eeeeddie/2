from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def smooth(series: pd.Series, window: int) -> pd.Series:
    if len(series) == 0:
        return series
    w = max(1, int(window))
    return series.rolling(window=w, min_periods=1).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--train_window', type=int, default=20)
    p.add_argument('--eval_window', type=int, default=5)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    train_csv = run_dir / 'train_metrics.csv'
    eval_csv = run_dir / 'eval_metrics.csv'
    out_dir = run_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    if train_csv.exists():
        train_df = pd.read_csv(train_csv)
        if len(train_df) > 0:
            # 1 train return
            plt.figure(figsize=(7,4))
            plt.plot(train_df['env_steps'], train_df['return'], alpha=0.25)
            plt.plot(train_df['env_steps'], smooth(train_df['return'], args.train_window))
            plt.xlabel('env_steps')
            plt.ylabel('return')
            plt.title('Train Return')
            plt.tight_layout()
            plt.savefig(out_dir / '01_train_return.png', dpi=160)
            plt.close()

            # 2 entropy
            if 'entropy' in train_df.columns:
                plt.figure(figsize=(7,4))
                plt.plot(train_df['env_steps'], train_df['entropy'], alpha=0.25)
                plt.plot(train_df['env_steps'], smooth(train_df['entropy'], args.train_window))
                plt.xlabel('env_steps')
                plt.ylabel('entropy')
                plt.title('Train Entropy')
                plt.tight_layout()
                plt.savefig(out_dir / '02_train_entropy.png', dpi=160)
                plt.close()

    if eval_csv.exists():
        eval_df = pd.read_csv(eval_csv)
        if len(eval_df) > 0:
            # 3 win rate
            plt.figure(figsize=(7,4))
            plt.plot(eval_df['env_steps'], eval_df['eval_win_rate'], marker='o')
            plt.plot(eval_df['env_steps'], smooth(eval_df['eval_win_rate'], args.eval_window))
            plt.xlabel('env_steps')
            plt.ylabel('eval_win_rate')
            plt.title('Eval Win Rate')
            plt.tight_layout()
            plt.savefig(out_dir / '03_eval_win_rate.png', dpi=160)
            plt.close()

            # 4 kills
            plt.figure(figsize=(7,4))
            plt.plot(eval_df['env_steps'], eval_df['eval_mean_kills'], marker='o')
            plt.plot(eval_df['env_steps'], smooth(eval_df['eval_mean_kills'], args.eval_window))
            plt.xlabel('env_steps')
            plt.ylabel('eval_mean_kills')
            plt.title('Eval Mean Kills')
            plt.tight_layout()
            plt.savefig(out_dir / '04_eval_mean_kills.png', dpi=160)
            plt.close()

    print(f'Plots saved to: {out_dir}')


if __name__ == '__main__':
    main()
