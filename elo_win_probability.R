# WHSDSC 2025: ELO ratings + rest/travel adjustment -> win probabilities
#
# 1. Build ELO ratings from games_2022.csv (two passes over the season)
# 2. Fit a scaling constant k so ELO difference predicts margin of victory
# 3. GAM on rest-day / travel-distance differences for contextual adjustment
# 4. Logistic regression: adjusted MOV -> win probability
# 5. Score "East Regional Games to predict.csv"
#
# Reconstructed from .Rhistory (lines 251-511).

library(dplyr)
library(readr)
library(tidyr)
library(mgcv)
library(nloptr)

setwd("C:/Users/ailes/Downloads/Wharton Sata/WHSDSC 2025")

# ---- Load data ----
data <- read_csv("games_2022.csv")

data <- data %>%
  filter(!is.na(rest_days), !is.na(travel_dist))

data <- data %>%
  group_by(game_id) %>%
  filter(n() == 2) %>%  # Keep only game_ids that appear exactly twice
  ungroup()

# Sort data by date
data <- data %>% arrange(game_date)

# ---- ELO ----
# Function to update ELO ratings
update_elo <- function(winner_elo, loser_elo, k = 30) {
  expected_win <- 1 / (1 + 10^((loser_elo - winner_elo) / 400))
  new_winner_elo <- winner_elo + k * (1 - expected_win)
  new_loser_elo <- loser_elo + k * (0 - (1 - expected_win))
  return(c(new_winner_elo, new_loser_elo))
}

# Run one pass over the season, updating elo_ratings in place (via <<-) and
# returning each team's pre-/post-game ELO for every game
run_elo_pass <- function(data) {
  elo_history <- data.frame(game_id = character(), team = character(), pre_game_elo = numeric(),
                            opponent_pre_game_elo = numeric(), elo = numeric())
  for (game in unique(data$game_id)) {
    game_data <- data %>% filter(game_id == game)
    if (nrow(game_data) == 2) {  # Ensure each game has exactly 2 teams
      team1 <- game_data$team[1]
      team2 <- game_data$team[2]
      score1 <- game_data$stat_team_score[1]
      score2 <- game_data$stat_team_score[2]

      # Get pre-game ELOs
      elo1 <- elo_ratings$elo[elo_ratings$team == team1]
      elo2 <- elo_ratings$elo[elo_ratings$team == team2]

      # Store pre-game ELOs before updating them
      elo_history <- rbind(elo_history, data.frame(game_id = game, team = team1, pre_game_elo = elo1, opponent_pre_game_elo = elo2, elo = NA))
      elo_history <- rbind(elo_history, data.frame(game_id = game, team = team2, pre_game_elo = elo2, opponent_pre_game_elo = elo1, elo = NA))

      # Update ELOs based on game result
      if (score1 > score2) {
        new_elo1 <- update_elo(elo1, elo2)[1]
        new_elo2 <- update_elo(elo1, elo2)[2]
      } else {
        new_elo2 <- update_elo(elo2, elo1)[1]
        new_elo1 <- update_elo(elo2, elo1)[2]
      }

      # Save updated ELOs
      elo_ratings$elo[elo_ratings$team == team1] <<- new_elo1
      elo_ratings$elo[elo_ratings$team == team2] <<- new_elo2

      # Store new ELOs in history
      elo_history$elo[elo_history$team == team1 & elo_history$game_id == game] <- new_elo1
      elo_history$elo[elo_history$team == team2 & elo_history$game_id == game] <- new_elo2
    }
  }
  elo_history
}

# Pass 1: everyone starts at 1000
elo_ratings <- data.frame(team = unique(data$team), elo = 1000)
elo_history <- run_elo_pass(data)

elo_ratings <- elo_ratings %>% arrange(desc(elo))
print(elo_ratings)

# Merge with original dataset for regression analysis
data <- data %>% left_join(elo_history, by = c("game_id", "team"))

first_elo_ratings <- data %>%
  select(team, pre_game_elo) %>%
  group_by(team) %>%
  summarise(first_elo = last(pre_game_elo)) %>%
  arrange(desc(first_elo))  # Sort from highest to lowest
print(first_elo_ratings)

# Pass 2: replay the season starting from the end-of-pass-1 ratings
elo_history <- run_elo_pass(data)

final_elo_ratings <- elo_ratings %>%
  select(team, elo) %>%
  rename(final_elo = elo)

data <- data %>%
  left_join(final_elo_ratings, by = "team")

# Create home and away final ELO columns
data <- data %>%
  group_by(game_id) %>%
  mutate(
    final_elo_home = case_when(
      home_away_NS == 1 ~ final_elo,
      home_away_NS == 0 ~ ifelse(home_away == "home", final_elo, coalesce(lead(final_elo), lag(final_elo))),  # Take non-NA value
      TRUE ~ NA_real_  # Default case
    ),
    final_elo_away = case_when(
      home_away_NS == -1 ~ final_elo,
      home_away_NS == 0 ~ ifelse(home_away == "away", final_elo, coalesce(lead(final_elo), lag(final_elo))),  # Take non-NA value
      TRUE ~ NA_real_  # Default case
    )
  ) %>%
  fill(final_elo_home, final_elo_away, .direction = "downup") %>%
  ungroup()

# Convert game outcomes to binary (1 if this row's team won, 0 otherwise)
data <- data %>%
  mutate(home_win = ifelse(stat_team_score > stat_opponent_team_score, 1, 0))

# ---- Fit k: ELO difference -> expected margin of victory ----
sse_function <- function(k, data) {
  data <- data %>%
    mutate(
      expected_mov = case_when(
        home_away_NS == 1 | (home_away_NS == 0 & home_away == "home") ~
          k * (coalesce(final_elo_home, 1000) - coalesce(final_elo_away, 1000)) /
          (sqrt(pmax(abs(final_elo_home - final_elo_away), 1) + 500) / 2),
        home_away_NS == -1 | (home_away_NS == 0 & home_away == "away") ~
          k * (coalesce(final_elo_away, 1000) - coalesce(final_elo_home, 1000)) /
          (sqrt(pmax(abs(final_elo_away - final_elo_home), 1) + 500) / 2),
        TRUE ~ NA_real_
      ),
      diff = (stat_team_score - stat_opponent_team_score) - expected_mov
    )
  return(sum(data$diff^2, na.rm = TRUE))  # Minimize the squared error
}

# Use optimization to find the best k
result <- nloptr(
  x0 = 1,  # Initial guess for k
  eval_f = function(k) sse_function(k, data),
  lb = 0.1, ub = 10,  # Bound k between 0.1 and 10 to prevent extreme values
  opts = list(algorithm = "NLOPT_LN_SBPLX", xtol_rel = 1e-6)
)
optimal_k <- result$solution  # Extract the best k value

data <- data %>%
  mutate(
    expected_mov = case_when(
      home_away_NS == 1 | (home_away_NS == 0 & home_away == "home") ~
        optimal_k * (coalesce(final_elo_home, 1000) - coalesce(final_elo_away, 1000)) /
        (sqrt(pmax(abs(final_elo_home - final_elo_away), 1) + 500) / 2),
      home_away_NS == -1 | (home_away_NS == 0 & home_away == "away") ~
        optimal_k * (coalesce(final_elo_away, 0) - coalesce(final_elo_home, 0)) /
        (sqrt(pmax(abs(final_elo_away - final_elo_home), 1) + 500) / 2),
      TRUE ~ NA_real_
    ),
    diff_from_expected_mov = (stat_team_score - stat_opponent_team_score) - expected_mov
  )

# NOTE: this multiplies by optimal_k a second time (it was already applied
# above). Kept as-is because the saved outputs were produced with it.
data <- data %>%
  mutate(expected_mov = expected_mov * optimal_k,
         diff_from_expected_mov = (stat_team_score - stat_opponent_team_score) - expected_mov)

# ---- Rest / travel differences ----
data <- data %>%
  arrange(game_id, home_away) %>%  # Ensure correct ordering
  group_by(game_id) %>%
  mutate(
    rest_days_Home = case_when(
      home_away == "home" ~ rest_days,
      home_away == "away" ~ coalesce(lag(rest_days), lead(rest_days)),
      TRUE ~ lead(rest_days)  # Default case for away teams
    ),
    rest_days_Away = case_when(
      home_away == "away" ~ rest_days,
      home_away == "home" ~ coalesce(lead(rest_days), lag(rest_days)),
      TRUE ~ lag(rest_days)  # Default case for home teams
    ),
    travel_dist_Home = case_when(
      home_away == "home" ~ travel_dist,
      home_away == "away" ~ coalesce(lag(travel_dist), lead(travel_dist)),
      TRUE ~ lead(travel_dist)
    ),
    travel_dist_Away = case_when(
      home_away == "away" ~ travel_dist,
      home_away == "home" ~ coalesce(lead(travel_dist), lag(travel_dist)),
      TRUE ~ lag(travel_dist)
    )
  ) %>%
  ungroup()

data <- data %>%
  mutate(
    rest_days_diff =
      case_when(home_away == "home" ~ rest_days_Home - rest_days_Away,
                home_away == "away" ~ rest_days_Away - rest_days_Home),
    travel_dist_diff =
      case_when(home_away == "home" ~ travel_dist_Home - travel_dist_Away,
                home_away == "away" ~ travel_dist_Away - travel_dist_Home)
  )

# ---- Contextual adjustment + win probability models ----
# Regression on adjusted performance
gam_model <- gam(diff_from_expected_mov ~ s(rest_days_diff) + s(travel_dist_diff) +
                   ti(rest_days_diff, travel_dist_diff), data = data)

data <- data %>%
  mutate(contextual_adjustment = predict(gam_model, newdata = data),
         # Adjusted MOV: Expected MOV + the model's predicted contextual impact (divided by 2)
         adjusted_mov = expected_mov + contextual_adjustment / 2)

# Fit a logistic regression model using adjusted margin of victory
win_prob_model <- glm(home_win ~ adjusted_mov, data = data, family = binomial)
summary(win_prob_model)

write.csv(data, "data.csv")

# ---- Predict the East Regional games ----
new_games <- read_csv("East Regional Games to predict.csv")

# Merge final ELOs for both home and away teams
new_games <- new_games %>%
  left_join(final_elo_ratings %>%
              select(team = team, final_elo),
            by = c("team_home" = "team")) %>%
  left_join(final_elo_ratings %>%
              select(team = team, final_elo),
            by = c("team_away" = "team"), suffix = c("_home", "_away"))

# Calculate expected margin of victory (MOV) based on ELO difference
new_games <- new_games %>%
  mutate(expected_mov_home = optimal_k * (final_elo_home - final_elo_away) / (sqrt(abs(final_elo_home - final_elo_away) + 500) / 2),
         expected_mov_away = optimal_k * (final_elo_away - final_elo_home) / (sqrt(abs(final_elo_away - final_elo_home) + 500) / 2))

new_games <- new_games %>%
  mutate(
    rest_days_diff = rest_days_Home - rest_days_Away,
    travel_dist_diff = travel_dist_Home - travel_dist_Away
  )

# Apply the contextual model for adjusted performance
new_games <- new_games %>%
  mutate(contextual_adjustment = predict(gam_model, newdata = new_games),
         # Adjust expected MOV using the predicted contextual adjustment
         adjusted_mov = expected_mov_home + contextual_adjustment / 2,
         adjusted_mov_away = expected_mov_away - contextual_adjustment / 2)

# Convert adjusted MOV into a home team win probability
new_games <- new_games %>%
  mutate(win_probability_home = predict(win_prob_model, newdata = new_games, type = "response"))

# predict() returns array columns; flatten them so write_csv works
new_games <- new_games %>%
  mutate(across(where(is.array), ~ sapply(., function(x) if (is.null(x)) NA else as.character(x))))

# Save the new dataset with winning percentage
write_csv(new_games, "Predicted_Winning_Percentages.csv")

plot(new_games$adjusted_mov, new_games$win_probability_home)
