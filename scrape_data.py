import pandas as pd
import requests
import json
import http.client
import time
import numpy as np
from datetime import datetime
from sportsipy.ncaab.boxscore import Boxscores, Boxscore
from sportsipy.ncaab.teams import Teams

# Set Duration to Scrape Games

start_day = 1
end_day = 2
start_month = 12
end_month = 12
start_year = 2023
end_year = 2023

# Create Game Stats DataFrame
print("Gathering Boxscore Data")
game_datas = Boxscores(datetime(start_year, start_month, start_day), datetime(end_year, end_month, end_day))

game_dict = {'season': [], 'date': [], 'winner': [], 'winning_team': [], 'losing_team': [], 'pace': [], 
             'away_rank': [], 'away_field_goals_made': [], 'away_field_goal_attempts': [], 'away_field_goal_pct': [], 'away_3pt_made': [], 'away_3pt_attempts': [],
             'away_3pt_pct': [], 'away_free_throws_made': [], 'away_free_throw_attempts': [], 'away_free_throw_pct': [], 'away_offensive_rebounds': [],
             'away_defensive_rebounds': [], 'away_total_rebounds': [], 'away_assists': [], 'away_steals': [], 'away_blocks': [], 'away_turnovers': [], 'away_fouls': [], 'away_points': [],
             'home_rank': [], 'home_field_goals_made': [], 'home_field_goal_attempts': [], 'home_field_goal_pct': [], 'home_3pt_made': [], 'home_3pt_attempts': [],
             'home_3pt_pct': [], 'home_free_throws_made': [], 'home_free_throw_attempts': [], 'home_free_throw_pct': [], 'home_offensive_rebounds': [],
             'home_defensive_rebounds': [], 'home_total_rebounds': [], 'home_assists': [], 'home_steals': [], 'home_blocks': [], 'home_turnovers': [], 'home_fouls': [], 'home_points': []}
passed = 0
season = 0
print("Gathering Boxscores")
for day in game_datas.games:
    print(day)
    for game in game_datas.games[day]:

        if game['boxscore'][-2:] == '_w':
            continue

        try:    
            game_data = Boxscore(game['boxscore'])
        except:
            passed += 1
            continue

        print(game['boxscore'])
        if game_data.winner == "Returned None":
            print(game_data.winner)
            continue

        game_dict['season'].append(season)
        game_dict['date'].append(game_data.date)
        game_dict['winner'].append(game_data.winner)
        game_dict['winning_team'].append(game_data.winning_name)
        game_dict['losing_team'].append(game_data.losing_name)
        game_dict['pace'].append(game_data.pace)

        game_dict['away_rank'].append(game_data.away_ranking)
        game_dict['away_field_goals_made'].append(game_data.away_field_goals)
        game_dict['away_field_goal_attempts'].append(game_data.away_field_goal_attempts)
        game_dict['away_field_goal_pct'].append(game_data.away_field_goal_percentage)
        game_dict['away_3pt_made'].append(game_data.away_three_point_field_goals)
        game_dict['away_3pt_attempts'].append(game_data.away_three_point_field_goal_attempts)
        game_dict['away_3pt_pct'].append(game_data.away_three_point_field_goal_percentage)
        game_dict['away_free_throws_made'].append(game_data.away_free_throws)
        game_dict['away_free_throw_attempts'].append(game_data.away_free_throw_attempts)
        game_dict['away_free_throw_pct'].append(game_data.away_free_throw_percentage)
        game_dict['away_offensive_rebounds'].append(game_data.away_offensive_rebounds)
        game_dict['away_defensive_rebounds'].append(game_data.away_defensive_rebounds)
        game_dict['away_total_rebounds'].append(game_data.away_total_rebounds)
        game_dict['away_assists'].append(game_data.away_assists)
        game_dict['away_steals'].append(game_data.away_steals)
        game_dict['away_blocks'].append(game_data.away_blocks)
        game_dict['away_turnovers'].append(game_data.away_turnovers)
        game_dict['away_fouls'].append(game_data.away_personal_fouls)
        game_dict['away_points'].append(game_data.away_points)

        game_dict['home_rank'].append(game_data.home_ranking)
        game_dict['home_field_goals_made'].append(game_data.home_field_goals)
        game_dict['home_field_goal_attempts'].append(game_data.home_field_goal_attempts)
        game_dict['home_field_goal_pct'].append(game_data.home_field_goal_percentage)
        game_dict['home_3pt_made'].append(game_data.home_three_point_field_goals)
        game_dict['home_3pt_attempts'].append(game_data.home_three_point_field_goal_attempts)
        game_dict['home_3pt_pct'].append(game_data.home_three_point_field_goal_percentage)
        game_dict['home_free_throws_made'].append(game_data.home_free_throws)
        game_dict['home_free_throw_attempts'].append(game_data.home_free_throw_attempts)
        game_dict['home_free_throw_pct'].append(game_data.home_free_throw_percentage)
        game_dict['home_offensive_rebounds'].append(game_data.home_offensive_rebounds)
        game_dict['home_defensive_rebounds'].append(game_data.home_defensive_rebounds)
        game_dict['home_total_rebounds'].append(game_data.home_total_rebounds)
        game_dict['home_assists'].append(game_data.home_assists)
        game_dict['home_steals'].append(game_data.home_steals)
        game_dict['home_blocks'].append(game_data.home_blocks)
        game_dict['home_turnovers'].append(game_data.home_turnovers)
        game_dict['home_fouls'].append(game_data.home_personal_fouls)
        game_dict['home_points'].append(game_data.home_points)

game_df = pd.DataFrame(game_dict)
game_df['season'] = -1

season = start_year
fall = ['October', 'November', 'December']
for year in range(start_year, end_year+1):
    for ent in range(len(game_df)):
        if int(game_df['date'][ent][-4:]) == year:
            if game_df['date'][ent].split(' ')[0] in fall:
                game_df['season'][ent] = season
        elif int(game_df['date'][ent][-4:]) == year+1:
            if game_df['date'][ent].split(' ')[0] not in fall:
                game_df['season'][ent] = season
        else:
            game_df['season'][ent] = game_df['season'][ent]

    season+=1

print("Organizing Data")
game_df['games'] = 1
game_df['home_team'] = np.where(game_df['winner'] == 'Home', game_df['winning_team'], game_df['losing_team'])
game_df['away_team'] = np.where(game_df['winner'] == 'Away', game_df['winning_team'], game_df['losing_team'])

# Save un parsed date
game_df.to_excel('cbb_data.xlsx')

# Generate Home and Away DataFrames
home_df = pd.read_excel('cbb_data.xlsx')
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
                        'away_fouls': 'opp_fouls'}, inplace=True)

away_df = pd.read_excel('cbb_data.xlsx')
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
                        'home_fouls': 'opp_fouls'}, inplace=True)

# Combine Home and Away DataFrames
cbb_stats_df = pd.concat([home_df, away_df])

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
cbb_stats_df['total_team_pace'] = cbb_stats_df.groupby(['season','team'])['pace'].cumsum()/cbb_stats_df.groupby(['season','team'])['game_counter'].cumsum() * 100
cbb_stats_df['total_opp_pace'] = cbb_stats_df.groupby(['season','opp'])['pace'].cumsum()/cbb_stats_df.groupby(['season','opp'])['game_counter'].cumsum() * 100

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

cbb_norm_df.to_excel('cbb_norm_data.xlsx')