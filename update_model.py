import pandas as pd
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import requests
from sportsipy.ncaab.teams import Teams, Team
from sportsipy.ncaab.schedule import Schedule
from sportsipy.ncaab.boxscore import Boxscores, Boxscore
from datetime import date, datetime 

# Read Old Dataset
cbb_norm_path = "/Users/matthew.yough/Documents/GitHub/ncaab_game_predictor"
cbb_norm_name = "pre_totaled_data.xlsx"

old_training_df = pd.read_excel(f"{cbb_norm_path}/{cbb_norm_name}", index_col=0)

date_ = date.today()
year = date_.year

if old_training_df['date'].iloc[len(old_training_df)-1] == date_:
    print("Data Up To Date")
    #return "Data Already up to Date"

teams = Teams(year=year+1)

stats_dict = {'team': [], 'pace': [], 'field_goals_made': [], 'field_goal_attempts': [], 'field_goal_pct': [], '3pt_made': [], '3pt_attempts': [],
        '3pt_pct': [], 'free_throws_made': [], 'free_throw_attempts': [], 'free_throw_pct': [], 'offensive_rebounds': [],
        'defensive_rebounds': [], 'total_rebounds': [], 'assists': [], 'steals': [], 'blocks': [], 'turnovers': [], 'fouls': [], 'points': []}
for team in teams:
    stats_dict['team'].append(team.name)
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
stats_df.to_excel('stats_df.xlsx')

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
cbb_df2.rename(columns={'pace_x': 'away_pace', 'field_goals_made_x': 'away_field_goals_made', 'field_goal_attempts_x': 'away_field_goal_attempts',
                        'field_goal_pct_x': 'away_field_goal_pct', '3pt_made_x': 'away_3pt_made', '3pt_attempts_x': 'away_3pt_attempts',
                        '3pt_pct_x': 'away_3pt_pct', 'free_throws_made_x': 'away_free_throws_made', 'free_throw_attempts_x': 'away_free_throw_attempts',
                        'free_throw_pct_x': 'away_free_throw_pct', 'offensive_rebounds_x': 'away_offensive_rebounds', 'defensive_rebounds_x': 'away_defensive_rebounds',
                        'total_rebounds_x': 'away_total_rebounds', 'assists_x': 'away_assists', 'steals_x': 'away_steals', 'blocks_x': 'away_blocks', 'turnovers_x': 'away_turnovers',
                        'fouls_x': 'away_fouls', 'points_x': 'away_points', 
                        'pace_y': 'home_pace', 'field_goals_made_y': 'home_field_goals_made', 'field_goal_attempts_y': 'home_field_goal_attempts',
                        'field_goal_pct_y': 'home_field_goal_pct', '3pt_made_y': 'home_3pt_made', '3pt_attempts_y': 'home_3pt_attempts',
                        '3pt_pct_y': 'home_3pt_pct', 'free_throws_made_y': 'home_free_throws_made', 'free_throw_attempts_y': 'home_free_throw_attempts',
                        'free_throw_pct_y': 'home_free_throw_pct', 'offensive_rebounds_y': 'home_offensive_rebounds', 'defensive_rebounds_y': 'home_defensive_rebounds',
                        'total_rebounds_y': 'home_total_rebounds', 'assists_y': 'home_assists', 'steals_y': 'home_steals', 'blocks_y': 'home_blocks', 'turnovers_y': 'home_turnovers',
                        'fouls_y': 'home_fouls', 'points_y': 'home_points'}, inplace=True)
cbb_df2.to_excel('cbb_update_raw.xlsx')

home_df = pd.read_excel('cbb_update_raw.xlsx')
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
                        'away_fouls': 'opp_fouls', 'home_pace': 'team_pace', 'away_pace': 'opp_pace'}, inplace=True)

away_df = pd.read_excel('cbb_update_raw.xlsx')
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
                        'home_fouls': 'opp_fouls', 'away_pace': 'team_pace', 'home_pace': 'opp_pace'}, inplace=True)


# Combine Home and Away DataFrames
cbb_stats_df = pd.concat([home_df, away_df])

cbb_stats_df['games'] = 1
cbb_stats_df['team_code'] = cbb_stats_df['team'].astype("category").cat.codes
cbb_stats_df['opp_code'] = cbb_stats_df['opp'].astype("category").cat.codes
cbb_stats_df['team_rank'] = cbb_stats_df['team_rank'].fillna(50)
cbb_stats_df['opp_rank'] = cbb_stats_df['opp_rank'].fillna(50)
cbb_stats_df['season'] = year
cbb_stats_df.to_excel('prejoined.xlsx')
# Combine Old Data and New Data
cbb_stats_df = pd.concat([old_training_df, cbb_stats_df])

# Summarize Data Across a Season
cbb_stats_df.sort_values(by=['season','date'], ascending=True, inplace=True)
cbb_stats_df['game_counter'] = 1

cbb_stats_df['total_team_games'] = cbb_stats_df.groupby(['season','team'])['games'].cumsum()
cbb_stats_df['total_opp_games'] = cbb_stats_df.groupby(['season','opp'])['games'].cumsum()
cbb_stats_df['total_team_points'] = cbb_stats_df.groupby(['season','team'])['team_points'].cumsum()
cbb_stats_df['total_opp_points'] = cbb_stats_df.groupby(['season','opp'])['opp_points'].cumsum()
cbb_stats_df['total_team_fg_att'] = cbb_stats_df.groupby(['season','team'])['team_field_goal_att'].cumsum()
cbb_stats_df['total_opp_fg_att'] = cbb_stats_df.groupby(['season','opp'])['opp_field_goal_att'].cumsum()
cbb_stats_df['total_team_fg_made'] = cbb_stats_df.groupby(['season','team'])['team_field_goal_made'].cumsum()
cbb_stats_df['total_opp_fg_made'] = cbb_stats_df.groupby(['season','opp'])['opp_field_goal_made'].cumsum()
cbb_stats_df['total_team_fg_pct'] = cbb_stats_df.groupby(['season','team'])['team_field_goal_pct'].cumsum()/cbb_stats_df.groupby(['season','team'])['game_counter'].cumsum() * 100
cbb_stats_df['total_opp_fg_pct'] = cbb_stats_df.groupby(['season','opp'])['opp_field_goal_pct'].cumsum()/cbb_stats_df.groupby(['season','opp'])['game_counter'].cumsum() * 100
cbb_stats_df['total_team_3pt_att'] = cbb_stats_df.groupby(['season','team'])['team_3pt_att'].cumsum()
cbb_stats_df['total_opp_3pt_att'] = cbb_stats_df.groupby(['season','opp'])['opp_3pt_att'].cumsum()
cbb_stats_df['total_team_3pt_made'] = cbb_stats_df.groupby(['season','team'])['team_3pt_made'].cumsum()
cbb_stats_df['total_opp_3pt_made'] = cbb_stats_df.groupby(['season','opp'])['opp_3pt_made'].cumsum()
cbb_stats_df['total_team_3pt_pct'] = cbb_stats_df.groupby(['season','team'])['team_3pt_pct'].cumsum()/cbb_stats_df.groupby(['season','team'])['game_counter'].cumsum() * 100
cbb_stats_df['total_opp_3pt_pct'] = cbb_stats_df.groupby(['season','opp'])['opp_3pt_pct'].cumsum()/cbb_stats_df.groupby(['season','opp'])['game_counter'].cumsum() * 100
cbb_stats_df['total_team_ft_att'] = cbb_stats_df.groupby(['season','team'])['team_free_throw_att'].cumsum()
cbb_stats_df['total_opp_ft_att'] = cbb_stats_df.groupby(['season','opp'])['opp_free_throw_att'].cumsum()
cbb_stats_df['total_team_ft_made'] = cbb_stats_df.groupby(['season','team'])['team_free_throw_made'].cumsum()
cbb_stats_df['total_opp_ft_made'] = cbb_stats_df.groupby(['season','opp'])['opp_free_throw_made'].cumsum()
cbb_stats_df['total_team_ft_pct'] = cbb_stats_df.groupby(['season','team'])['team_free_throw_pct'].cumsum()/cbb_stats_df.groupby(['season','team'])['game_counter'].cumsum() * 100
cbb_stats_df['total_opp_ft_pct'] = cbb_stats_df.groupby(['season','opp'])['opp_free_throw_pct'].cumsum()/cbb_stats_df.groupby(['season','opp'])['game_counter'].cumsum() * 100
cbb_stats_df['total_team_rebounds'] = cbb_stats_df.groupby(['season','team'])['team_rebounds'].cumsum()
cbb_stats_df['total_opp_rebounds'] = cbb_stats_df.groupby(['season','opp'])['opp_rebounds'].cumsum()
cbb_stats_df['total_team_assists'] = cbb_stats_df.groupby(['season','team'])['team_assists'].cumsum()
cbb_stats_df['total_opp_assists'] = cbb_stats_df.groupby(['season','opp'])['opp_assists'].cumsum()
cbb_stats_df['total_team_steals'] = cbb_stats_df.groupby(['season','team'])['team_steals'].cumsum()
cbb_stats_df['total_opp_steals'] = cbb_stats_df.groupby(['season','opp'])['opp_steals'].cumsum()
cbb_stats_df['total_team_blocks'] = cbb_stats_df.groupby(['season','team'])['team_blocks'].cumsum()
cbb_stats_df['total_opp_blocks'] = cbb_stats_df.groupby(['season','opp'])['opp_blocks'].cumsum()
cbb_stats_df['total_team_turnovers'] = cbb_stats_df.groupby(['season','team'])['team_turnovers'].cumsum()
cbb_stats_df['total_opp_turnovers'] = cbb_stats_df.groupby(['season','opp'])['opp_turnovers'].cumsum()
cbb_stats_df['total_team_fouls'] = cbb_stats_df.groupby(['season','team'])['team_fouls'].cumsum()
cbb_stats_df['total_opp_fouls'] = cbb_stats_df.groupby(['season','opp'])['opp_fouls'].cumsum()
cbb_stats_df['total_team_pace'] = cbb_stats_df.groupby(['season','team'])['team_pace'].cumsum()/cbb_stats_df.groupby(['season','team'])['game_counter'].cumsum() * 100
cbb_stats_df['total_opp_pace'] = cbb_stats_df.groupby(['season','opp'])['opp_pace'].cumsum()/cbb_stats_df.groupby(['season','opp'])['game_counter'].cumsum() * 100

cbb_stats_df.to_excel('cbb_raw_data.xlsx')

# Organize and Normalize Data
cbb_norm_df = pd.read_excel('cbb_raw_data.xlsx')
cbb_norm_df.drop(columns=['Unnamed: 0'], inplace=True)

cbb_norm_df['team_code'] = cbb_norm_df['team'].astype("category").cat.codes
cbb_norm_df['opp_code'] = cbb_norm_df['opp'].astype("category").cat.codes
cbb_norm_df['team_rank'] = cbb_norm_df['team_rank'].fillna(50)
cbb_norm_df['opp_rank'] = cbb_norm_df['opp_rank'].fillna(50)
cbb_norm_df.to_excel('cbb_nonnorm_data.xlsx')

cbb_norm_df['total_team_points'] = cbb_norm_df['total_team_points']/cbb_norm_df['total_team_points'].max()
cbb_norm_df['total_opp_points'] = cbb_norm_df['total_opp_points']/cbb_norm_df['total_opp_points'].max()
cbb_norm_df['total_team_fg_att'] = cbb_norm_df['total_team_fg_att']/cbb_norm_df['total_team_fg_att'].max()
cbb_norm_df['total_opp_fg_att'] = cbb_norm_df['total_opp_fg_att']/cbb_norm_df['total_opp_fg_att'].max()
cbb_norm_df['total_team_fg_made'] = cbb_norm_df['total_team_fg_made']/cbb_norm_df['total_team_fg_made'].max() 
cbb_norm_df['total_opp_fg_made'] = cbb_norm_df['total_opp_fg_made']/cbb_norm_df['total_opp_fg_made'].max()
cbb_norm_df['total_team_fg_pct'] = cbb_norm_df['total_team_fg_pct']/cbb_norm_df['total_team_fg_pct'].max()
cbb_norm_df['total_opp_fg_pct'] = cbb_norm_df['total_opp_fg_pct']/cbb_norm_df['total_opp_fg_pct'].max()
cbb_norm_df['total_team_3pt_att'] = cbb_norm_df['total_team_3pt_att']/cbb_norm_df['total_team_3pt_att'].max()
cbb_norm_df['total_opp_3pt_att'] = cbb_norm_df['total_opp_3pt_att']/cbb_norm_df['total_opp_3pt_att'].max()
cbb_norm_df['total_team_3pt_made'] = cbb_norm_df['total_team_3pt_made']/cbb_norm_df['total_team_3pt_made'].max()
cbb_norm_df['total_opp_3pt_made'] = cbb_norm_df['total_opp_3pt_made']/cbb_norm_df['total_opp_3pt_made'].max()
cbb_norm_df['total_team_3pt_pct'] = cbb_norm_df['total_team_3pt_pct']/cbb_norm_df['total_team_3pt_pct'].max()
cbb_norm_df['total_opp_3pt_pct'] = cbb_norm_df['total_opp_3pt_pct']/cbb_norm_df['total_opp_3pt_pct'].max()
cbb_norm_df['total_team_ft_att'] = cbb_norm_df['total_team_ft_att']/cbb_norm_df['total_team_ft_att'].max()
cbb_norm_df['total_opp_ft_att'] = cbb_norm_df['total_opp_ft_att']/cbb_norm_df['total_opp_ft_att'].max()
cbb_norm_df['total_team_ft_made'] = cbb_norm_df['total_team_ft_made']/cbb_norm_df['total_team_ft_made'].max()
cbb_norm_df['total_opp_ft_made'] = cbb_norm_df['total_opp_ft_made']/cbb_norm_df['total_opp_ft_made'].max()
cbb_norm_df['total_team_ft_pct'] = cbb_norm_df['total_team_ft_pct']/cbb_norm_df['total_team_ft_pct'].max()
cbb_norm_df['total_opp_ft_pct'] = cbb_norm_df['total_opp_ft_pct']/cbb_norm_df['total_opp_ft_pct'].max()
cbb_norm_df['total_team_rebounds'] = cbb_norm_df['total_team_rebounds']/cbb_norm_df['total_team_rebounds'].max()
cbb_norm_df['total_opp_rebounds'] = cbb_norm_df['total_opp_rebounds']/cbb_norm_df['total_opp_rebounds'].max()
cbb_norm_df['total_team_assists'] = cbb_norm_df['total_team_assists']/cbb_norm_df['total_team_assists'].max()
cbb_norm_df['total_opp_assists'] = cbb_norm_df['total_opp_assists']/cbb_norm_df['total_opp_assists'].max()
cbb_norm_df['total_team_steals'] = cbb_norm_df['total_team_steals']/cbb_norm_df['total_team_steals'].max()
cbb_norm_df['total_opp_steals'] = cbb_norm_df['total_opp_steals']/cbb_norm_df['total_opp_steals'].max()
cbb_norm_df['total_team_blocks'] = cbb_norm_df['total_team_blocks']/cbb_norm_df['total_team_blocks'].max()
cbb_norm_df['total_opp_blocks'] = cbb_norm_df['total_opp_blocks']/cbb_norm_df['total_opp_blocks'].max()
cbb_norm_df['total_team_turnovers'] = cbb_norm_df['total_team_turnovers'] /cbb_norm_df['total_team_turnovers'] .max()
cbb_norm_df['total_opp_turnovers'] = cbb_norm_df['total_opp_turnovers']/cbb_norm_df['total_opp_turnovers'].max()
cbb_norm_df['total_team_fouls'] = cbb_norm_df['total_team_fouls']/cbb_norm_df['total_team_fouls'].max()
cbb_norm_df['total_opp_fouls'] = cbb_norm_df['total_opp_fouls']/cbb_norm_df['total_opp_fouls'].max()
cbb_norm_df['team_code'] = cbb_norm_df['team_code']/cbb_norm_df['team_code'].max()
cbb_norm_df['opp_code'] = cbb_norm_df['opp_code']/cbb_norm_df['opp_code'].max()
cbb_norm_df['team_rank'] = cbb_norm_df['team_rank']/cbb_norm_df['team_rank'].max()
cbb_norm_df['opp_rank'] = cbb_norm_df['opp_rank']/cbb_norm_df['opp_rank'].max()
cbb_norm_df['total_team_pace'] = cbb_norm_df['total_team_pace']/cbb_norm_df['total_team_pace'].max()
cbb_norm_df['total_opp_pace'] = cbb_norm_df['total_opp_pace']/cbb_norm_df['total_opp_pace'].max()

cbb_norm_df.to_excel('cbb_norm_data2.xlsx')

