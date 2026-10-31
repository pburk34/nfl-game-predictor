"""
Generates static PNG charts from the experiment results documented in README.md, for
embedding directly in the README so it's visual on GitHub without needing to rerun anything.

Data here is transcribed directly from the sweep/experiment scripts' output (see README's
"Full experiment log" for the exact tables) rather than recomputed, since those results
are already validated and deterministic -- no need to re-run a multi-second sweep just to
plot numbers we already have. If you change the model or the data and the numbers shift,
update the lists below to match the new script output.

Requires matplotlib (only needed for this script, not for the model itself).
"""

import os
import matplotlib.pyplot as plt

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

ACCENT = "#1f6feb"        # blue -- primary series
ACCENT2 = "#d1495b"       # red -- secondary/comparison series
DEFAULT_MARK = "#f2a900"  # gold -- highlights the chosen default value


def sweep_chart(x, accuracy, brier, default_x, xlabel, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x, accuracy, marker="o", color=ACCENT, linewidth=2)
    axes[0].axvline(default_x, color=DEFAULT_MARK, linestyle="--", linewidth=1.5, alpha=0.8,
                    label=f"default ({default_x})")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=9)

    axes[1].plot(x, brier, marker="o", color=ACCENT2, linewidth=2)
    axes[1].axvline(default_x, color=DEFAULT_MARK, linestyle="--", linewidth=1.5, alpha=0.8,
                    label=f"default ({default_x})")
    axes[1].set_title("Brier score (lower = better)")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Brier score")
    axes[1].legend(fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def k_factor_chart():
    k = [5, 10, 15, 20, 25, 30, 40, 60]
    accuracy = [0.621358, 0.636751, 0.644722, 0.641699, 0.640187, 0.641424, 0.640874, 0.633452]
    brier = [0.228843, 0.223519, 0.221682, 0.221368, 0.221883, 0.222918, 0.225965, 0.234255]
    sweep_chart(k, accuracy, brier, default_x=20, xlabel="K_FACTOR",
                title="Elo K_FACTOR sweep", filename="k_factor_sweep.png")


def home_advantage_chart():
    values = [0, 15, 25, 35, 48, 55, 65, 80, 100]
    accuracy = [0.629742, 0.634827, 0.638950, 0.639225, 0.641699, 0.641974, 0.641836, 0.636888, 0.628642]
    brier = [0.225373, 0.223219, 0.222241, 0.221627, 0.221368, 0.221476, 0.221925, 0.223224, 0.226051]
    sweep_chart(values, accuracy, brier, default_x=48, xlabel="HOME_ADVANTAGE (Elo points)",
                title="Home-field advantage sweep", filename="home_advantage_sweep.png")


def season_regression_chart():
    values = [0.00, 0.10, 0.20, 0.25, 0.333, 0.40, 0.50, 0.65, 1.00]
    accuracy = [0.636751, 0.638813, 0.639362, 0.642386, 0.641699, 0.642248, 0.641424, 0.639087, 0.623145]
    brier = [0.228694, 0.225375, 0.223122, 0.222315, 0.221368, 0.220924, 0.220710, 0.221252, 0.225608]
    sweep_chart(values, accuracy, brier, default_x=0.333, xlabel="SEASON_REGRESSION",
                title="Season regression sweep", filename="season_regression_sweep.png")


def momentum_by_season_chart():
    seasons = list(range(2010, 2026))
    model = [0.621723, 0.659176, 0.621723, 0.659176, 0.685393, 0.666667, 0.651685, 0.666667,
             0.632959, 0.644195, 0.650558, 0.600000, 0.633803, 0.617544, 0.677193, 0.624561]
    elo = [0.610487, 0.644195, 0.632959, 0.659176, 0.692884, 0.674157, 0.647940, 0.670412,
           0.629213, 0.640449, 0.650558, 0.603509, 0.637324, 0.610526, 0.677193, 0.635088]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(seasons, elo, marker="o", color=ACCENT, linewidth=2, label="Pure Elo")
    ax.plot(seasons, model, marker="s", color=ACCENT2, linewidth=2, linestyle="--",
            label="Elo + Momentum")
    ax.set_title("Momentum barely moves the needle: accuracy by holdout season",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Test season")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "momentum_by_season.png"), dpi=150)
    plt.close(fig)
    print("Saved momentum_by_season.png")


def feature_comparison_chart():
    labels = ["Elo only", "+ EPA", "+ Turnovers", "+ EPA &\nTurnovers"]
    accuracy = [0.644312, 0.641082, 0.643129, 0.644031]
    brier = [0.221423, 0.219948, 0.221545, 0.220144]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    colors = [ACCENT, ACCENT2, ACCENT2, "#4c9f70"]
    axes[0].bar(labels, accuracy, color=colors)
    axes[0].set_title("Accuracy")
    axes[0].set_ylim(min(accuracy) - 0.01, max(accuracy) + 0.005)

    axes[1].bar(labels, brier, color=colors)
    axes[1].set_title("Brier score (lower = better)")
    axes[1].set_ylim(min(brier) - 0.002, max(brier) + 0.001)

    fig.suptitle("EPA per play helps; turnover margin doesn't", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "feature_comparison.png"), dpi=150)
    plt.close(fig)
    print("Saved feature_comparison.png")


if __name__ == "__main__":
    k_factor_chart()
    home_advantage_chart()
    season_regression_chart()
    momentum_by_season_chart()
    feature_comparison_chart()
    print(f"\nAll charts saved to {IMAGES_DIR}/")
