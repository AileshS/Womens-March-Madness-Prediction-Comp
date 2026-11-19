# Women's College Basketball Win-Probability Model

Built for the **2025 Wharton High School Data Science Competition**. The task was to predict
win probabilities for a set of regional tournament games, using one season (2022) of
women's college basketball game data.

## The approach

The model works in four steps. Each one builds on the one before it.

1. **Elo ratings.** Every team starts at 1000. After each game the winner takes rating points
   from the loser, and an upset moves more points than an expected result. The season is
   replayed twice so early-season games are rated against more settled ratings.
2. **Elo gap → expected margin of victory.** A scaling constant `k` is fitted with a numerical
   optimizer (`nloptr`) so that the rating difference between two teams best predicts the
   actual point margin (it minimizes squared error).
3. **Context adjustment for rest and travel.** A generalized additive model (GAM, `mgcv`)
   asks whether teams beat or miss their expected margin depending on the difference in
   rest days and travel distance between the two teams. Its prediction nudges the
   expected margin up or down.
4. **Margin → win probability.** A logistic regression turns the adjusted margin into a
   probability that the home team wins. That model is then applied to the tournament games.

## Run it

All the code is in `elo_win_probability.R` (R, with `dplyr`, `readr`, `tidyr`, `mgcv` and `nloptr`).
Put the competition files (`games_2022.csv` and `East Regional Games to predict.csv`) in your
working directory, then `source("elo_win_probability.R")`. The script writes
`Predicted_Winning_Percentages.csv`.

The competition data isn't included in this repo.

## Limitations and what I'd improve

- **Ratings are fitted and judged on the same season.** There's no held-out test set, so I can't
  report a true out-of-sample accuracy. Next step: train on the first part of the season and
  check calibration on the rest.
- **The Elo K-factor is fixed at 30.** It wasn't tuned, and margin of victory doesn't affect
  rating updates.
- **`k` gets applied twice to the expected margin.** It's flagged with a comment in the code and
  left in, because the submitted predictions were produced with it.
- **Each piece is fitted separately.** Fitting the rest/travel adjustment and the win-probability
  model together would be cleaner.
