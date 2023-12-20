import pandas as pd
import requests
import json
import http.client
import time
import numpy as np
from datetime import date, datetime
import joblib
from sportsipy.ncaab.teams import Teams, Team
from sportsipy.ncaab.schedule import Schedule
from sportsipy.ncaab.boxscore import Boxscores, Boxscore

out_path = '/Users/matthew.yough/Documents/GitHub/ncaab_game_predictor'
model_path = '/Users/matthew.yough/Documents/GitHub/ncaab_game_predictor'
model_file = 'CBB_Score_Model.pkl'

date_ = date.today()
year = date_.year

teams = Teams(year=year)
stats_dict = {'team': [], 'games': [], 'pace': [], 'field_goals_made': [], 'field_goal_attempts': [], 'field_goal_pct': [], '3pt_made': [], '3pt_attempts': [],
             '3pt_pct': [], 'free_throws_made': [], 'free_throw_attempts': [], 'free_throw_pct': [], 'offensive_rebounds': [],
             'defensive_rebounds': [], 'total_rebounds': [], 'assists': [], 'steals': [], 'blocks': [], 'turnovers': [], 'fouls': [], 'points': []}
for team in teams:
    stats_dict['team'].append(team.name)
    stats_dict['games'].append(team.games_played)
    stats_dict['pace'].append(team.pace)

    stats_dict['field_goals_made'].append(team.field_goals)
    stats_dict['field_goal_attempts'].append(team.field_goal_attempts)
    stats_dict['field_goal_pct'].append(team.field_goal_percentage)
    stats_dict['3pt_made'].append(team.three_point_field_goals)
    stats_dict['3pt_attempts'].append(team._three_point_field_goal_attempts)
    stats_dict['3pt_pct'].append(team.three_point_field_goal_percentage)
    stats_dict['free_throws_made'].append(team.free_throws)
    stats_dict['free_throw_attempts'].append(team.free_throw_attempts)
    stats_dict['free_throw_pct'].append(team.free_throw_percentage)
    stats_dict['offensive_rebounds'].append(team.offensive_rebounds)
    stats_dict['defensive_rebounds'].append(team.defensive_rebounds)
    stats_dict['total_rebounds'].append(team.total_rebounds)
    stats_dict['assists'].append(team.assists)
    stats_dict['steals'].append(team.steals)
    stats_dict['blocks'].append(team.blocks)
    stats_dict['turnovers'].append(team.turnovers)
    stats_dict['fouls'].append(team.personal_fouls)
    stats_dict['points'].append(team.points)

stats_df = pd.DataFrame(stats_dict)

games = Boxscores(datetime.today())
game_dict = {'date': [], 'away_team': [], 'away_rank': [], 'home_team': [], 'home_rank': []}

cur_date = date.today()
day = f"{str(cur_date.month)}-{str(cur_date.day)}-{str(cur_date.year)}"
for game in games.games[day]:


    game_dict['date'].append(day)
    game_dict['away_rank'].append(game['away_rank'])
    game_dict['home_rank'].append(game['home_rank'])
    game_dict['away_team'].append(game['away_name'])
    game_dict['home_team'].append(game['home_name'])

game_df = pd.DataFrame(game_dict)

cbb_df = pd.merge(game_df, stats_df, left_on='away_team', right_on='team')
cbb_df2 = pd.merge(cbb_df, stats_df, left_on='home_team', right_on='team')

cbb_df2.drop(columns=['team_x', 'team_y'], inplace=True)
cbb_df2.rename(columns={'games_x': 'away_games', 'pace_x': 'away_pace', 'field_goals_made_x': 'away_field_goals_made', 'field_goal_attempts_x': 'away_field_goal_attempts',
                        'field_goal_pct_x': 'away_field_goal_pct', '3pt_made_x': 'away_3pt_made', '3pt_attempts_x': 'away_3pt_attempts',
                        '3pt_pct_x': 'away_3pt_pct', 'free_throws_made_x': 'away_free_throws_made', 'free_throw_attempts_x': 'away_free_throw_attempts',
                        'free_throw_pct_x': 'away_free_throw_pct', 'offensive_rebounds_x': 'away_offensive_rebounds', 'defensive_rebounds_x': 'away_defensive_rebounds',
                        'total_rebounds_x': 'away_total_rebounds', 'assists_x': 'away_assists', 'steals_x': 'away_steals', 'blocks_x': 'away_blocks', 'turnovers_x': 'away_turnovers',
                        'fouls_x': 'away_fouls', 'points_x': 'away_points', 
                        'games_y': 'home_games', 'pace_y': 'home_pace', 'field_goals_made_y': 'home_field_goals_made', 'field_goal_attempts_y': 'home_field_goal_attempts',
                        'field_goal_pct_y': 'home_field_goal_pct', '3pt_made_y': 'home_3pt_made', '3pt_attempts_y': 'home_3pt_attempts',
                        '3pt_pct_y': 'home_3pt_pct', 'free_throws_made_y': 'home_free_throws_made', 'free_throw_attempts_y': 'home_free_throw_attempts',
                        'free_throw_pct_y': 'home_free_throw_pct', 'offensive_rebounds_y': 'home_offensive_rebounds', 'defensive_rebounds_y': 'home_defensive_rebounds',
                        'total_rebounds_y': 'home_total_rebounds', 'assists_y': 'home_assists', 'steals_y': 'home_steals', 'blocks_y': 'home_blocks', 'turnovers_y': 'home_turnovers',
                        'fouls_y': 'home_fouls', 'points_y': 'home_points'}, inplace=True)
cbb_df2.to_excel('cbb_predict_raw.xlsx')

home_df = pd.read_excel('cbb_predict_raw.xlsx')
home_df.drop(columns=['Unnamed: 0'], inplace=True)
home_df.rename(columns={'home_team': 'team', 'away_team': 'opp','home_points': 'team_points', 'away_points': 'opp_points', 'home_rank': 'team_rank', 'away_rank': 'opp_rank', 'home_field_goal_attempts': 'team_field_goal_att',
                        'away_field_goal_attempts': 'opp_field_goal_att', 'home_field_goals_made': 'team_field_goal_made','away_field_goals_made': 'opp_field_goal_made', 
                        'home_field_goal_pct': 'team_field_goal_pct','away_field_goal_pct': 'opp_field_goal_pct','home_3pt_attempts': 'team_3pt_att','away_3pt_attempts': 'opp_3pt_att', 
                        'home_3pt_made': 'team_3pt_made','away_3pt_made': 'opp_3pt_made','home_3pt_pct': 'team_3pt_pct','away_3pt_pct': 'opp_3pt_pct',
                        'home_free_throw_attempts': 'team_free_throw_att','away_free_throw_attempts': 'opp_free_throw_att', 'home_free_throws_made': 'team_free_throw_made',
                        'away_free_throws_made': 'opp_free_throw_made','home_free_throw_pct': 'team_free_throw_pct','away_free_throw_pct': 'opp_free_throw_pct', 'home_total_rebounds': 'team_rebounds',
                        'away_total_rebounds': 'opp_rebounds', 'home_offensive_rebounds': 'team_off_rebounds', 'away_offensive_rebounds': 'opp_off_rebounds',
                        'home_defensive_rebounds': 'team_def_rebounds', 'away_defensive_rebounds': 'opp_def_rebounds','home_assists':'team_assists', 'away_assists': 'opp_assists', 'home_steals': 'team_steals', 'away_steals': 'opp_steals',
                        'home_blocks': 'team_blocks', 'away_blocks': 'opp_blocks', 'home_turnovers': 'team_turnovers', 'away_turnovers': 'opp_turnovers', 'home_fouls': 'team_fouls',
                        'away_fouls': 'opp_fouls', 'home_games': 'team_games', 'away_games': 'opp_games', 'home_pace': 'team_pace', 'away_pace': 'opp_pace'}, inplace=True)

away_df = pd.read_excel('cbb_predict_raw.xlsx')
away_df.drop(columns=['Unnamed: 0'], inplace=True)
away_df.rename(columns={'away_team': 'team', 'home_team': 'opp','away_points': 'team_points', 'home_points': 'opp_points', 'away_rank': 'team_rank', 'home_rank': 'opp_rank', 'away_field_goal_attempts': 'team_field_goal_att',
                        'home_field_goal_attempts': 'opp_field_goal_att', 'away_field_goals_made': 'team_field_goal_made','home_field_goals_made': 'opp_field_goal_made', 
                        'away_field_goal_pct': 'team_field_goal_pct','home_field_goal_pct': 'opp_field_goal_pct','away_3pt_attempts': 'team_3pt_att','home_3pt_attempts': 'opp_3pt_att', 
                        'away_3pt_made': 'team_3pt_made','home_3pt_made': 'opp_3pt_made','away_3pt_pct': 'team_3pt_pct','home_3pt_pct': 'opp_3pt_pct',
                        'away_free_throw_attempts': 'team_free_throw_att','home_free_throw_attempts': 'opp_free_throw_att', 'away_free_throws_made': 'team_free_throw_made',
                        'home_free_throws_made': 'opp_free_throw_made','away_free_throw_pct': 'team_free_throw_pct','home_free_throw_pct': 'opp_free_throw_pct', 'away_total_rebounds': 'team_rebounds',
                        'home_total_rebounds': 'opp_rebounds', 'away_offensive_rebounds': 'team_off_rebounds', 'home_offensive_rebounds': 'opp_off_rebounds',
                        'away_defensive_rebounds': 'team_def_rebounds', 'home_defensive_rebounds': 'opp_def_rebounds','away_assists':'team_assists', 'home_assists': 'opp_assists', 'away_steals': 'team_steals', 'home_steals': 'opp_steals',
                        'away_blocks': 'team_blocks', 'home_blocks': 'opp_blocks', 'away_turnovers': 'team_turnovers', 'home_turnovers': 'opp_turnovers', 'away_fouls': 'team_fouls',
                        'home_fouls': 'opp_fouls', 'away_games': 'team_games', 'home_games': 'opp_games', 'away_pace': 'team_pace', 'home_pace': 'opp_pace'}, inplace=True)

# Combine Home and Away DataFrames
cbb_stats_df = pd.concat([home_df, away_df])

cbb_stats_df['team_code'] = cbb_stats_df['team'].astype("category").cat.codes
cbb_stats_df['opp_code'] = cbb_stats_df['opp'].astype("category").cat.codes
cbb_stats_df['team_rank'] = cbb_stats_df['team_rank'].fillna(50)
cbb_stats_df['opp_rank'] = cbb_stats_df['opp_rank'].fillna(50)

cbb_stats_df['total_team_games'] = cbb_stats_df['team_games']
cbb_stats_df['total_opp_games'] = cbb_stats_df['opp_games']
cbb_stats_df['total_team_points'] = cbb_stats_df['team_points']/cbb_stats_df['team_points'].max()
cbb_stats_df['total_opp_points'] = cbb_stats_df['opp_points']/cbb_stats_df['opp_points'].max()
cbb_stats_df['total_team_fg_att'] = cbb_stats_df['team_field_goal_att']/cbb_stats_df['team_field_goal_att'].max()
cbb_stats_df['total_opp_fg_att'] = cbb_stats_df['opp_field_goal_att']/cbb_stats_df['opp_field_goal_att'].max()
cbb_stats_df['total_team_fg_made'] = cbb_stats_df['team_field_goal_made']/cbb_stats_df['team_field_goal_made'].max() 
cbb_stats_df['total_opp_fg_made'] = cbb_stats_df['opp_field_goal_made']/cbb_stats_df['opp_field_goal_made'].max()
cbb_stats_df['total_team_fg_pct'] = cbb_stats_df['team_field_goal_pct']/cbb_stats_df['team_field_goal_pct'].max()
cbb_stats_df['total_opp_fg_pct'] = cbb_stats_df['opp_field_goal_pct']/cbb_stats_df['opp_field_goal_pct'].max()
cbb_stats_df['total_team_3pt_att'] = cbb_stats_df['team_3pt_att']/cbb_stats_df['team_3pt_att'].max()
cbb_stats_df['total_opp_3pt_att'] = cbb_stats_df['opp_3pt_att']/cbb_stats_df['opp_3pt_att'].max()
cbb_stats_df['total_team_3pt_made'] = cbb_stats_df['team_3pt_made']/cbb_stats_df['team_3pt_made'].max()
cbb_stats_df['total_opp_3pt_made'] = cbb_stats_df['opp_3pt_made']/cbb_stats_df['opp_3pt_made'].max()
cbb_stats_df['total_team_3pt_pct'] = cbb_stats_df['team_3pt_pct']/cbb_stats_df['team_3pt_pct'].max()
cbb_stats_df['total_opp_3pt_pct'] = cbb_stats_df['opp_3pt_pct']/cbb_stats_df['opp_3pt_pct'].max()
cbb_stats_df['total_team_ft_att'] = cbb_stats_df['team_free_throw_att']/cbb_stats_df['team_free_throw_att'].max()
cbb_stats_df['total_opp_ft_att'] = cbb_stats_df['opp_free_throw_att']/cbb_stats_df['opp_free_throw_att'].max()
cbb_stats_df['total_team_ft_made'] = cbb_stats_df['team_free_throw_made']/cbb_stats_df['team_free_throw_made'].max()
cbb_stats_df['total_opp_ft_made'] = cbb_stats_df['opp_free_throw_made']/cbb_stats_df['opp_free_throw_made'].max()
cbb_stats_df['total_team_ft_pct'] = cbb_stats_df['team_free_throw_pct']/cbb_stats_df['team_free_throw_pct'].max()
cbb_stats_df['total_opp_ft_pct'] = cbb_stats_df['opp_free_throw_pct']/cbb_stats_df['opp_free_throw_pct'].max()
cbb_stats_df['total_team_rebounds'] = cbb_stats_df['team_rebounds']/cbb_stats_df['team_rebounds'].max()
cbb_stats_df['total_opp_rebounds'] = cbb_stats_df['opp_rebounds']/cbb_stats_df['opp_rebounds'].max()
cbb_stats_df['total_team_assists'] = cbb_stats_df['team_assists']/cbb_stats_df['team_assists'].max()
cbb_stats_df['total_opp_assists'] = cbb_stats_df['opp_assists']/cbb_stats_df['opp_assists'].max()
cbb_stats_df['total_team_steals'] = cbb_stats_df['team_steals']/cbb_stats_df['team_steals'].max()
cbb_stats_df['total_opp_steals'] = cbb_stats_df['opp_steals']/cbb_stats_df['opp_steals'].max()
cbb_stats_df['total_team_blocks'] = cbb_stats_df['team_blocks']/cbb_stats_df['team_blocks'].max()
cbb_stats_df['total_opp_blocks'] = cbb_stats_df['opp_blocks']/cbb_stats_df['opp_blocks'].max()
cbb_stats_df['total_team_turnovers'] = cbb_stats_df['team_turnovers'] /cbb_stats_df['team_turnovers'] .max()
cbb_stats_df['total_opp_turnovers'] = cbb_stats_df['opp_turnovers']/cbb_stats_df['opp_turnovers'].max()
cbb_stats_df['total_team_fouls'] = cbb_stats_df['team_fouls']/cbb_stats_df['team_fouls'].max()
cbb_stats_df['total_opp_fouls'] = cbb_stats_df['opp_fouls']/cbb_stats_df['opp_fouls'].max()
cbb_stats_df['total_team_code'] = cbb_stats_df['team_code']/cbb_stats_df['team_code'].max()
cbb_stats_df['total_opp_code'] = cbb_stats_df['opp_code']/cbb_stats_df['opp_code'].max()
cbb_stats_df['total_team_rank'] = cbb_stats_df['team_rank']/cbb_stats_df['team_rank'].max()
cbb_stats_df['total_opp_rank'] = cbb_stats_df['opp_rank']/cbb_stats_df['opp_rank'].max()
cbb_stats_df['total_team_pace'] = cbb_stats_df['team_pace']/cbb_stats_df['team_pace'].max()
cbb_stats_df['total_opp_pace'] = cbb_stats_df['opp_pace']/cbb_stats_df['opp_pace'].max()

# Load Models
regr = joblib.load(f"{model_path}/{model_file}")

# Define Metrics for Input Variables
X = cbb_stats_df[['total_team_games', 'total_opp_games', 'total_team_points', 'total_opp_points', 'total_team_fg_att', 'total_opp_fg_att', 'total_team_fg_made', 'total_opp_fg_made', 'total_team_fg_pct', 'total_opp_fg_pct', 
                'total_team_3pt_att', 'total_opp_3pt_att', 'total_team_3pt_made', 'total_opp_3pt_made', 'total_team_3pt_pct', 'total_opp_3pt_pct', 'total_team_ft_att', 'total_opp_ft_att',
                'total_team_ft_made', 'total_opp_ft_made', 'total_team_ft_pct', 'total_opp_ft_pct', 'total_team_rebounds', 'total_opp_rebounds', 'total_team_assists', 'total_opp_assists',
                'total_team_steals', 'total_opp_steals', 'total_team_blocks', 'total_opp_blocks', 'total_team_turnovers', 'total_opp_turnovers', 'total_team_fouls', 'total_opp_fouls',
                 'total_team_pace', 'total_opp_pace', 'team_code', 'opp_code']]
X = X.reset_index(drop=True)

# Run Model
y_pred = regr.predict(X)

# Create Predictions DataFrame
predictions = {'Home_Team' : cbb_stats_df['team'].iloc[0:int(len(cbb_stats_df)/2)], 'home_points': np.round(y_pred)[0: int(len(y_pred)/2)]}
predictions_df = pd.DataFrame(data=predictions)
predictions_df['Away_Team'] = cbb_stats_df['team'].iloc[int(len(cbb_stats_df)/2):]
predictions_df['away_points'] = np.round(y_pred)[int(len(y_pred)/2):]

# Save DFs to Excel
predictions_df.to_excel(f"{out_path}/NCAAB {date_.year}-{date_.month}-{date_.day} Score Predictions.xlsx")