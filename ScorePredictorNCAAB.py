import pandas as pd
import requests
import json
import http.client
import time
import numpy as np
from datetime import datetime
from sportsipy.ncaab.boxscore import Boxscores, Boxscore
from sportsipy.ncaab.teams import Teams
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib


class ScorePredictorNCAAB:

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

    ###### Scrape Data Function ######
    def scrape_data(self, out_path):
        # Set Duration to Scrape Games

        start_day = 1
        end_day = 30
        start_month = 10
        end_month = 4
        start_year = 2021
        date = date.today()
        end_year = date.year

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

        season = 0
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
        game_df.to_excel(f'{out_path}/cbb_data.xlsx')

        # Generate Home and Away DataFrames
        home_df = pd.read_excel(f'{out_path}/cbb_data.xlsx')
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

        away_df = pd.read_excel(f'{out_path}/cbb_data.xlsx')
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

        cbb_stats_df.to_excel(f'{out_path}/cbb_raw_data.xlsx')

        # Organize and Normalize Data
        cbb_norm_df = pd.read_excel(f'{out_path}/cbb_raw_data.xlsx')
        cbb_norm_df.drop(columns=['Unnamed: 0'], inplace=True)

        cbb_norm_df['team_code'] = cbb_norm_df['team'].astype("category").cat.codes
        cbb_norm_df['opp_code'] = cbb_norm_df['opp'].astype("category").cat.codes
        cbb_norm_df['team_rank'] = cbb_norm_df['team_rank'].fillna(50)
        cbb_norm_df['opp_rank'] = cbb_norm_df['opp_rank'].fillna(50)
        cbb_norm_df.to_excel(f'{out_path}/cbb_nonnorm_data.xlsx')

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

        cbb_norm_df.to_excel(f'{out_path}/cbb_norm_data.xlsx')

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

    ###### Train Model ######
    def train_model(self, path, start_year, end_year):
        
        # Read Dataset for Training
        file_path = path
        file_name = "cbb_norm_data.xlsx"

        # Model Output Path and File Names
        model_path = path
        model_file = "CBB_Score_Model.pkl"

        data_df = pd.read_excel(f"{file_path}/{file_name}", index_col=0)
        data_df = data_df.dropna(axis = 0).reset_index()
        data_df = data_df[(data_df['season'] >= int(start_year)) & (data_df['season'] <= int(end_year))]

        # Define Metrics for Input Variables
        X = data_df[['total_team_points', 'total_opp_points', 'total_team_fg_att', 'total_opp_fg_att', 'total_team_fg_made', 'total_opp_fg_made', 'total_team_fg_pct', 'total_opp_fg_pct', 
                        'total_team_3pt_att', 'total_opp_3pt_att', 'total_team_3pt_made', 'total_opp_3pt_made', 'total_team_3pt_pct', 'total_opp_3pt_pct', 'total_team_ft_att', 'total_opp_ft_att',
                        'total_team_ft_made', 'total_opp_ft_made', 'total_team_ft_pct', 'total_opp_ft_pct', 'total_team_rebounds', 'total_opp_rebounds', 'total_team_assists', 'total_opp_assists',
                        'total_team_steals', 'total_opp_steals', 'total_team_blocks', 'total_opp_blocks', 'total_team_turnovers', 'total_opp_turnovers', 'total_team_fouls', 'total_opp_fouls',
                        'venue_code', 'team_code', 'opp_code', 'total_pace']]
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

        # Return Model Results
        return str(regr.score(X_test, y_test))

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

    ###### Predict Scores ######
    def predict_scores(self, path, result_path, year):

        # File Names and Paths
        data_path = path
     
        data_file = "cbb_norm_data.xlsx"

        # Model Output Path and File Names
        model_path = path
        model_file = "CBB_Score_Model.pkl"
        # Output Data Path
        out_path = result_path

        # Get Current Season Stats
        date = date.today()
        year = date.year

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

        # Get Today's Games
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

        # Merge Stats and Schedules
        cbb_df = pd.merge(game_df, stats_df, left_on='away_team', right_on='team')
        cbb_df2 = pd.merge(cbb_df, stats_df, left_on='home_team', right_on='team')

        # Rename Columns
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
        
        cbb_df2.to_excel(f'{out_path}/cbb_predict_raw.xlsx')

        # Split into Home and Away DataFrames
        home_df = pd.read_excel(f'{out_path}/cbb_predict_raw.xlsx')
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

        away_df = pd.read_excel(f'{out_path}/cbb_predict_raw.xlsx')
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

        # Organize Data for Predictinos
        cbb_stats_df['team_code'] = cbb_stats_df['team'].astype("category").cat.codes
        cbb_stats_df['opp_code'] = cbb_stats_df['opp'].astype("category").cat.codes
        cbb_stats_df['team_rank'] = cbb_stats_df['team_rank'].fillna(50)
        cbb_stats_df['opp_rank'] = cbb_stats_df['opp_rank'].fillna(50)

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

        # Make Predicions
        # Define Metrics for Input Variables
        X = cbb_stats_df[['total_team_points', 'total_opp_points', 'total_team_rank', 'total_opp_rank',  'total_team_field_goal_att',
                                'total_opp_field_goal_att',  'total_team_field_goal_made','total_opp_field_goal_made', 
                                'total_team_field_goal_pct', 'total_opp_field_goal_pct','total_team_3pt_att','total_opp_3pt_att', 
                                'total_team_3pt_made','total_opp_3pt_made', 'total_team_3pt_pct','total_opp_3pt_pct',
                                'total_team_free_throw_att','total_opp_free_throw_att',  'total_team_free_throw_made',
                                'total_opp_free_throw_made','total_team_free_throw_pct','total_opp_free_throw_pct',  'total_team_rebounds',
                                'total_opp_rebounds',  'total_team_off_rebounds', 'total_opp_off_rebounds',
                                'total_team_def_rebounds', 'total_opp_def_rebounds','total_team_assists',  'total_opp_assists','total_team_steals', 'total_opp_steals',
                                'total_team_blocks',  'total_opp_blocks',  'total_team_turnovers', 'total_opp_turnovers', 'total_team_fouls',
                                'total_opp_fouls',  'total_team_games', 'total_opp_games',  'total_team_pace',  'total_opp_pace']]
        X = X.reset_index(drop=True)

        # Load Models
        regr = joblib.load(f"{model_path}/{model_file}")
        # Run Model
        y_pred = regr.predict(X)

        # Create Predictions DataFrame
        predictions = {'Home_Team' : cbb_stats_df['team'].iloc[0:len(cbb_stats_df)/2], 'home_points': np.round(y_pred)[0: len(y_pred)/2]}
        predictions_df = pd.DataFrame(data=predictions)
        predictions_df['Away_Team'] = cbb_stats_df['team'].iloc[len(cbb_stats_df)/2:]
        predictions_df['away_points'] = np.round(y_pred)[len(y_pred)/2:]

        # Save DFs to Excel
        predictions_df.to_excel(f"{out_path}/{date} NCAAB Score Predictions.xlsx")

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

    ###### Update Model ######
    def update_model(self, path, year, current_week):
    
        # Read Old Dataset
        cbb_norm_path = path
        cbb_norm_name = "cbb_norm_data.xlsx"

        old_training_df = pd.read_excel(f"{cbb_norm_path}/{cbb_norm_name}", index_col=0)

        '''
        # Check to make sure data from same week hasn't already been added
        #old_full_df['season_week'] = old_full_df['season'].astype(str) + "," + old_full_df['week'].astype(str)
        #season_week = str(year) + "," + str(current_week)
        season_week = date.now()

        if season_week in old_full_df['season_week'].values:
            return f"Already input Season {year}, Week {current_week} Data"
        
        if bPO:
            return f"Cannot Update Model with Playoff Data"

        old_full_df.drop(columns={'season_week'}, inplace=True)
        '''

        date = date.today()
        year = date.year

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
                                'away_fouls': 'opp_fouls', 'home_games': 'team_games', 'away_games': 'opp_games', 'home_pace': 'team_pace', 'away_pace': 'opp_pace'}, inplace=True)

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
                                'home_fouls': 'opp_fouls', 'away_games': 'team_games', 'home_games': 'opp_games', 'away_pace': 'team_pace', 'home_pace': 'opp_pace'}, inplace=True)


        # Combine Home and Away DataFrames
        cbb_stats_df = pd.concat([home_df, away_df])

        cbb_stats_df['team_code'] = cbb_stats_df['team'].astype("category").cat.codes
        cbb_stats_df['opp_code'] = cbb_stats_df['opp'].astype("category").cat.codes
        cbb_stats_df['team_rank'] = cbb_stats_df['team_rank'].fillna(50)
        cbb_stats_df['opp_rank'] = cbb_stats_df['opp_rank'].fillna(50)

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

        # Combine Old Data and New Data
        cbb_stats_df = pd.concat([old_training_df, cbb_stats_df])
        cbb_stats_df.to_excel('cbb_norm_data.xlsx')

        return "Update Complete"
    
        '''
        # Retrain Model on New Data
        # Define Metrics for Input Variables
        X = updated_norm_df[['rank', 'opponent_rank', 'off_success_rate','def_success_rate', 'def_per_game_ppa',
                        'off_per_game_ppa', 'passing_yards_per_game', 'rushing_yards_per_game', 'TDs_per_game', 'time_of_possession_per_game',
                        'off_turnovers_per_game', 'penalty_yards_per_game', 'sacks_per_game', 'def_interceptions_per_game', 'team_code', 'opp_code',
                            'opp_off_success_rate','opp_def_success_rate', 'opp_def_per_game_ppa','opp_off_per_game_ppa', 'opp_passing_yards_per_game', 
                            'opp_rushing_yards_per_game', 'opp_TDs_per_game', 'opp_time_of_possession_per_game', 'opp_off_turnovers_per_game', 
                            'opp_penalty_yards_per_game', 'opp_sacks_per_game', 'opp_def_interceptions_per_game']]
        X = X.reset_index(drop=True)

        # Split Data for Home and Away Models
        X_team = X[['rank', 'opponent_rank', 'off_success_rate', 'off_per_game_ppa', 'passing_yards_per_game', 'rushing_yards_per_game', 'TDs_per_game', 
                    'time_of_possession_per_game', 'off_turnovers_per_game', 'penalty_yards_per_game', 'team_code', 'opp_code',
                    'opp_def_success_rate', 'opp_def_per_game_ppa', 'opp_time_of_possession_per_game','opp_penalty_yards_per_game', 
                    'opp_sacks_per_game', 'opp_def_interceptions_per_game']]
        X_opp = X[['rank', 'opponent_rank', 'def_success_rate', 'def_per_game_ppa','time_of_possession_per_game','penalty_yards_per_game', 'sacks_per_game',
                    'def_interceptions_per_game', 'team_code', 'opp_code','opp_off_success_rate','opp_off_per_game_ppa', 'opp_passing_yards_per_game', 
                    'opp_rushing_yards_per_game', 'opp_TDs_per_game', 'opp_time_of_possession_per_game', 'opp_off_turnovers_per_game', 
                    'opp_penalty_yards_per_game']]

        y_team = updated_norm_df['team_points']
        y_opp = updated_norm_df['opponent_points']

        # Home Linear Regression Model
        X_train, X_test, y_train, y_test = train_test_split(X_team, y_team, test_size = 0.20)

        regr_home = LinearRegression()
        regr_home.fit(X_train, y_train.values.ravel())
        print(regr_home.score(X_test, y_test))

        # Away Linear Regression model
        X_train, X_test, y_train, y_test = train_test_split(X_opp, y_opp, test_size = 0.20)

        regr_away = LinearRegression()
        regr_away.fit(X_train, y_train.values.ravel())
        print(regr_away.score(X_test, y_test))

        # Save Models
        joblib.dump(regr_home, os.path.join(model_path,home_model_file))
        joblib.dump(regr_away, os.path.join(model_path,away_model_file))

        return "Update Complete"
        '''
    
    def retrain_model(self, path, start_year, end_year):
        
        # Model Output Path and File Names
        model_path = path
        model_file = "CBB_Score_Model.pkl"

        data_df = pd.read_excel(f"{path}/{'cbb_norm_data.data.xlsx'}", index_col=0)
        data_df = data_df.dropna(axis = 0).reset_index()
        data_df = data_df[data_df['season'] >= start_year & data_df['season'] <= end_year]

        # Define Metrics for Input Variables
        X = data_df[['total_team_games', 'total_opp_games' 'total_team_points', 'total_opp_points', 'total_team_fg_att', 'total_opp_fg_att', 'total_team_fg_made', 'total_opp_fg_made', 'total_team_fg_pct', 'total_opp_fg_pct', 
                        'total_team_3pt_att', 'total_opp_3pt_att', 'total_team_3pt_made', 'total_opp_3pt_made', 'total_team_3pt_pct', 'total_opp_3pt_pct', 'total_team_ft_att', 'total_opp_ft_att',
                        'total_team_ft_made', 'total_opp_ft_made', 'total_team_ft_pct', 'total_opp_ft_pct', 'total_team_rebounds', 'total_opp_rebounds', 'total_team_assists', 'total_opp_assists',
                        'total_team_steals', 'total_opp_steals', 'total_team_blocks', 'total_opp_blocks', 'total_team_turnovers', 'total_opp_turnovers', 'total_team_fouls', 'total_opp_fouls',
                        'venue_code', 'team_code', 'opp_code']]
        X = X.reset_index(drop=True)

        y_team = data_df['team_points']

        # Linear Regression Model
        X_train, X_test, y_train, y_test = train_test_split(X, y_team, test_size = 0.20)

        regr = LinearRegression()
        regr.fit(X_train, y_train.values.ravel())
        print(regr.score(X_test, y_test))

        # Save Models
        joblib.dump(regr, f"{model_path}/{model_file}")

        # Return Model Results
        return str(round(regr.score(X, y_team),2))