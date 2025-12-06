# Predicting Mahjong Discard Decisions with Machine Learning

This project uses real Japanese Riichi Mahjong game logs from Tenhou to build a model that predicts which tile a player will discard next, given the current game state.
from https://www.kaggle.com/datasets/hphphp123321/tenhou-4-player-riichi-mahjong-dataset

## Dataset

The data comes from a processed Tenhou log database (`datasets_positive.db`), stored as a 50GB SQLite file.  
I use the `Discard` table, where each row contains:

- A gzip-compressed JSON game state (`Data`)
- Table skill indicators (`MaxDan`, `MinDan`)

Each JSON state includes:

- Round context (round wind, honba, riichi sticks, remaining tiles)
- Player identity and seat
- Hand tiles
- Dora indicators
- Opponent melds, discards, and riichi status
- Valid actions and the actual chosen action

## Problem Definition

Given a game state, predict **which tile type (0–33)** the player will discard next.

This is formulated as a 34-class classification problem.

## Feature Engineering

For each state, I extract:

- Round and table context:  
  `round_wind`, `num_honba`, `num_riichi`, `remain_tiles`, `player_wind`, `position`
- Skill: `MaxDan`, `MinDan`
- Dora count
- Hand representation: 34-length tile count vector
- Player info: points, riichi flag
- Opponent pressure: number of riichi opponents, total meld count

The label is the tile type of the chosen discard, computed from `real_action_idx` and `valid_actions`.

## Models

I compare three models:

1. **Baseline** – always predict the most frequent discard tile.
2. **Multinomial Logistic Regression** – linear baseline with standardized features.
3. **Random Forest Classifier** – 300 trees, non-linear model for tile and context interactions.

## Results

| Model                      | Accuracy | Macro F1 |
|---------------------------|----------|----------|
| Baseline (most frequent)  | 0.049    | 0.003    |
| Logistic Regression       | 0.125    | 0.097    |
| Random Forest             | 0.416    | 0.265    |

Random Forest significantly outperforms both baseline and Logistic Regression, showing that Mahjong discard decisions contain nonlinear structure that can be captured by tree ensembles.

## Files

- `notebook.ipynb` – full analysis, modeling, and evaluation
- `report.pdf` – final project report
- `README.md` – project overview (this file)

## How to Run

1. Place `datasets_positive.db` in the project folder or update the path in the notebook.
2. Create and activate a Python environment.
## Installation

```bash
pip install -r requirements.txt
