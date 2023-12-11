import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Read Dataset for Training
file_path = "/Users/matthew.yough/Documents/GitHub/ncaab_game_predictor"
file_name = "cbb_norm_data.xlsx"

# Model Output Path and File Names
model_path = "/Users/matthew.yough/Documents/GitHub/ncaab_game_predictor"
model_file = "CBB_Score_Model.pkl"
#away_model_file = "Away_Team_Model.pkl"

data_df = pd.read_excel(f"{file_path}/{file_name}", index_col=0)
data_df = data_df.dropna(axis = 0).reset_index()

# Define Metrics for Input Variables
X = data_df[['total_team_points', 'total_opp_points', 'total_team_fg_att', 'total_opp_fg_att', 'total_team_fg_made', 'total_opp_fg_made', 'total_team_fg_pct', 'total_opp_fg_pct', 
                'total_team_3pt_att', 'total_opp_3pt_att', 'total_team_3pt_made', 'total_opp_3pt_made', 'total_team_3pt_pct', 'total_opp_3pt_pct', 'total_team_ft_att', 'total_opp_ft_att',
                'total_team_ft_made', 'total_opp_ft_made', 'total_team_ft_pct', 'total_opp_ft_pct', 'total_team_rebounds', 'total_opp_rebounds', 'total_team_assists', 'total_opp_assists',
                'total_team_steals', 'total_opp_steals', 'total_team_blocks', 'total_opp_blocks', 'total_team_turnovers', 'total_opp_turnovers', 'total_team_fouls', 'total_opp_fouls',
                 'venue_code', 'team_code', 'opp_code']]
X = X.reset_index(drop=True)

y_team = data_df['team_points']

# Home Linear Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y_team, test_size = 0.20)

regr = LinearRegression()
regr.fit(X_train, y_train.values.ravel())
print(regr.score(X_test, y_test))

# Away Linear Regression model

# Save Models
joblib.dump(regr, f"{model_path}/{model_file}")