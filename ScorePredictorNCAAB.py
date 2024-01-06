import pandas as pd
import requests
import json
import http.client
import time
import numpy as np
from datetime import datetime, date, timedelta
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
        start_year = 2013
        end_year = 2024

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
                game_dict['date'].append(day)
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
        fall = ['10', '11', '12']
        for year in range(start_year, end_year+1):
            for ent in range(len(game_df)):
                if int(game_df['date'][ent][-4:]) == year:
                    print(year)
                    print(game_df['date'][ent][0:2])
                    if game_df['date'][ent][0:2] in fall:
                        game_df['season'][ent] = season
                elif int(game_df['date'][ent][-4:]) == year+1:
                    if game_df['date'][ent][0:2] not in fall:
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
        cbb_stats_df['date']= pd.to_datetime(cbb_stats_df['date'])
        cbb_stats_df.sort_values(by=['season','date'], ascending=True, inplace=True)
        cbb_stats_df['game_counter'] = 1
        cbb_stats_df['team_pace'] = cbb_stats_df.groupby(['season','team'])['pace'].cumsum()/cbb_stats_df.groupby(['season','team'])['game_counter'].cumsum() * 100
        cbb_stats_df['opp_pace'] = cbb_stats_df.groupby(['season','opp'])['pace'].cumsum()/cbb_stats_df.groupby(['season','opp'])['game_counter'].cumsum() * 100
        cbb_stats_df.to_excel('pre_totaled_data.xlsx')

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
        cbb_norm_df['total_team_pace'] = cbb_norm_df['team_pace']/cbb_norm_df['team_pace'].max()
        cbb_norm_df['total_opp_pace'] = cbb_norm_df['opp_pace']/cbb_norm_df['opp_pace'].max()

        cbb_norm_df.to_excel('cbb_norm_data.xlsx')

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

        # Define Metrics for Input Variables
        X = data_df[['team_ppg', 'opp_ppg', 'team_fg_att_pg', 'opp_fg_att_pg', 'team_fg_made_pg', 'opp_fg_made_pg', 'total_team_fg_pct', 'total_opp_fg_pct', 
                        'team_3pt_att_pg', 'opp_3pt_att_pg', 'team_3pt_made_pg', 'opp_3pt_made_pg', 'total_team_3pt_pct', 'total_opp_3pt_pct', 'team_ft_att_pg', 'opp_ft_att_pg',
                        'team_ft_made_pg', 'opp_ft_made_pg', 'total_team_ft_pct', 'total_opp_ft_pct', 'team_rebounds_pg', 'opp_rebounds_pg', 'team_assists_pg', 'opp_assists_pg',
                        'team_steals_pg', 'opp_steals_pg', 'team_blocks_pg', 'opp_blocks_pg', 'team_turnovers_pg', 'opp_turnovers_pg', 'team_fouls_pg', 'opp_fouls_pg',
                         'team_code', 'opp_code', 'total_team_pace', 'total_opp_pace']]
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
        return str(regr.score(X_test, y_test))

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

    ###### Predict Scores ######
    def predict_scores(self, path, result_path):

        # File Names and Paths
        data_path = path
     
        data_file = "cbb_norm_data.xlsx"

        # Model Output Path and File Names
        model_path = path
        model_file = "CBB_Score_Model.pkl"
        # Output Data Path
        out_path = result_path

        date_ = date.today()
        year = date_.year

        try:
            teams = Teams(year=year+1)
        except:
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

        try:
            games = Boxscores(datetime.today())
        except:
            return "No Games Today"
        
        game_dict = {'date': [], 'away_team': [], 'away_rank': [], 'home_team': [], 'home_rank': []}

        cur_date = date.today()
        day = f"{str(cur_date.month)}-{str(cur_date.day)}-{str(cur_date.year)}"
        for game in games.games[day]:


            game_dict['date'].append(day)
            game_dict['away_rank'].append(game['away_rank'])
            game_dict['home_rank'].append(game['home_rank'])
            game_dict['away_team'].append(game['away_name'])
            game_dict['home_team'].append(game['home_name'])

        if len(game_dict['date']) == 0:
            return "No Games Today"
        
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
        cbb_df2.to_excel(f'{data_path}/cbb_predict_raw.xlsx')

        home_df = pd.read_excel(f'{data_path}/cbb_predict_raw.xlsx')
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

        away_df = pd.read_excel(f'{data_path}/cbb_predict_raw.xlsx')
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
        cbb_stats_df['total_team_points'] = cbb_stats_df['team_points']
        cbb_stats_df['total_opp_points'] = cbb_stats_df['opp_points']
        cbb_stats_df['total_team_fg_att'] = cbb_stats_df['team_field_goal_att']
        cbb_stats_df['total_opp_fg_att'] = cbb_stats_df['opp_field_goal_att']
        cbb_stats_df['total_team_fg_made'] = cbb_stats_df['team_field_goal_made']
        cbb_stats_df['total_opp_fg_made'] = cbb_stats_df['opp_field_goal_made']
        cbb_stats_df['total_team_fg_pct'] = cbb_stats_df['team_field_goal_pct'] * 100
        cbb_stats_df['total_opp_fg_pct'] = cbb_stats_df['opp_field_goal_pct'] * 100
        cbb_stats_df['total_team_3pt_att'] = cbb_stats_df['team_3pt_att']
        cbb_stats_df['total_opp_3pt_att'] = cbb_stats_df['opp_3pt_att']
        cbb_stats_df['total_team_3pt_made'] = cbb_stats_df['team_3pt_made']
        cbb_stats_df['total_opp_3pt_made'] = cbb_stats_df['opp_3pt_made']
        cbb_stats_df['total_team_3pt_pct'] = cbb_stats_df['team_3pt_pct'] * 100
        cbb_stats_df['total_opp_3pt_pct'] = cbb_stats_df['opp_3pt_pct'] * 100
        cbb_stats_df['total_team_ft_att'] = cbb_stats_df['team_free_throw_att']
        cbb_stats_df['total_opp_ft_att'] = cbb_stats_df['opp_free_throw_att']
        cbb_stats_df['total_team_ft_made'] = cbb_stats_df['team_free_throw_made']
        cbb_stats_df['total_opp_ft_made'] = cbb_stats_df['opp_free_throw_made']
        cbb_stats_df['total_team_ft_pct'] = cbb_stats_df['team_free_throw_pct'] * 100
        cbb_stats_df['total_opp_ft_pct'] = cbb_stats_df['opp_free_throw_pct'] * 100
        cbb_stats_df['total_team_rebounds'] = cbb_stats_df['team_rebounds']
        cbb_stats_df['total_opp_rebounds'] = cbb_stats_df['opp_rebounds']
        cbb_stats_df['total_team_assists'] = cbb_stats_df['team_assists']
        cbb_stats_df['total_opp_assists'] = cbb_stats_df['opp_assists']
        cbb_stats_df['total_team_steals'] = cbb_stats_df['team_steals']
        cbb_stats_df['total_opp_steals'] = cbb_stats_df['opp_steals']
        cbb_stats_df['total_team_blocks'] = cbb_stats_df['team_blocks']
        cbb_stats_df['total_opp_blocks'] = cbb_stats_df['opp_blocks']
        cbb_stats_df['total_team_turnovers'] = cbb_stats_df['team_turnovers']
        cbb_stats_df['total_opp_turnovers'] = cbb_stats_df['opp_turnovers']
        cbb_stats_df['total_team_fouls'] = cbb_stats_df['team_fouls']
        cbb_stats_df['total_opp_fouls'] = cbb_stats_df['opp_fouls']
        cbb_stats_df['total_team_code'] = cbb_stats_df['team_code']
        cbb_stats_df['total_opp_code'] = cbb_stats_df['opp_code']
        cbb_stats_df['total_team_rank'] = cbb_stats_df['team_rank']
        cbb_stats_df['total_opp_rank'] = cbb_stats_df['opp_rank']
        cbb_stats_df['team_code'] = cbb_stats_df['team_code']/cbb_stats_df['team_code'].max()
        cbb_stats_df['opp_code'] = cbb_stats_df['opp_code']/cbb_stats_df['opp_code'].max()
        cbb_stats_df['team_rank'] = cbb_stats_df['team_rank']/cbb_stats_df['team_rank'].max()
        cbb_stats_df['opp_rank'] = cbb_stats_df['opp_rank']/cbb_stats_df['opp_rank'].max()
        cbb_stats_df['total_team_pace'] = cbb_stats_df['team_pace']/cbb_stats_df['team_pace'].max()
        cbb_stats_df['total_opp_pace'] = cbb_stats_df['opp_pace']/cbb_stats_df['opp_pace'].max()

        cbb_stats_df['team_ppg'] = cbb_stats_df['total_team_points']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_ppg'] = cbb_stats_df['total_opp_points']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_fg_att_pg'] = cbb_stats_df['total_team_fg_att']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_fg_att_pg'] = cbb_stats_df['total_opp_fg_att']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_fg_made_pg'] = cbb_stats_df['total_team_fg_made']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_fg_made_pg'] = cbb_stats_df['total_opp_fg_made']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_3pt_att_pg'] = cbb_stats_df['total_team_3pt_att']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_3pt_att_pg'] = cbb_stats_df['total_opp_3pt_att']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_3pt_made_pg'] = cbb_stats_df['total_team_3pt_made']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_3pt_made_pg'] = cbb_stats_df['total_opp_3pt_made']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_ft_att_pg'] = cbb_stats_df['total_team_ft_att']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_ft_att_pg'] = cbb_stats_df['total_opp_ft_att']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_ft_made_pg'] = cbb_stats_df['total_team_ft_made']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_ft_made_pg'] = cbb_stats_df['total_opp_ft_made']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_rebounds_pg'] = cbb_stats_df['total_team_rebounds']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_rebounds_pg'] = cbb_stats_df['total_opp_rebounds']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_assists_pg'] = cbb_stats_df['total_team_assists']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_assists_pg'] = cbb_stats_df['total_opp_assists']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_steals_pg'] = cbb_stats_df['total_team_steals']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_steals_pg'] = cbb_stats_df['total_opp_steals']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_blocks_pg'] = cbb_stats_df['total_team_blocks']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_blocks_pg'] = cbb_stats_df['total_opp_blocks']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_turnovers_pg'] = cbb_stats_df['total_team_turnovers']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_turnovers_pg'] = cbb_stats_df['total_opp_turnovers']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_fouls_pg'] = cbb_stats_df['total_team_fouls']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_fouls_pg'] = cbb_stats_df['total_opp_fouls']/cbb_stats_df['total_opp_games']

        cbb_stats_df.reset_index(drop=True, inplace=True)
        # Load Models
        regr = joblib.load(f"{model_path}/{model_file}")

        # Define Metrics for Input Variables
        
        X = cbb_stats_df[['team_ppg', 'opp_ppg', 'team_fg_att_pg', 'opp_fg_att_pg', 'team_fg_made_pg', 'opp_fg_made_pg', 'total_team_fg_pct', 'total_opp_fg_pct', 
                        'team_3pt_att_pg', 'opp_3pt_att_pg', 'team_3pt_made_pg', 'opp_3pt_made_pg', 'total_team_3pt_pct', 'total_opp_3pt_pct', 'team_ft_att_pg', 'opp_ft_att_pg',
                        'team_ft_made_pg', 'opp_ft_made_pg', 'total_team_ft_pct', 'total_opp_ft_pct', 'team_rebounds_pg', 'opp_rebounds_pg', 'team_assists_pg', 'opp_assists_pg',
                        'team_steals_pg', 'opp_steals_pg', 'team_blocks_pg', 'opp_blocks_pg', 'team_turnovers_pg', 'opp_turnovers_pg', 'team_fouls_pg', 'opp_fouls_pg',
                        ]]
        X = X.reset_index(drop=True)
        
        # Run Model
        y_pred = regr.predict(X)

        # Create Predictions DataFrame
        predictions = {'Home_Team' : cbb_stats_df['team'].iloc[0:int(len(cbb_stats_df)/2)], 'home_points': np.round(y_pred)[0: int(len(y_pred)/2)]}
        predictions_df = pd.DataFrame(data=predictions)
        predictions_df['Away_Team'] = cbb_stats_df['team'].iloc[int(len(cbb_stats_df)/2):].values
        predictions_df['away_points'] = np.round(y_pred)[int(len(y_pred)/2):]

        # Save DFs to Excel
        predictions_df.to_excel(f"{out_path}/NCAAB {date_.year}-{date_.month}-{date_.day} Score Predictions.xlsx")

        return "Predictions Complete"
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

    ###### Update Model ######
    def update_model(self, path):

        return "Update Functionality Not Available"
    
        cbb_norm_path = path
        cbb_norm_name = "cbb_norm_data.xlsx"

        old_training_df = pd.read_excel(f"{cbb_norm_path}/{cbb_norm_name}", index_col=0)

        date_ = date.today() - timedelta(days=1)
        year = date_.year

        if old_training_df['date'].iloc[len(old_training_df)-1] == date_:
            return "Data Already up to Date"

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

        games = Boxscores(date.today() - timedelta(days=1))
        game_dict = {'date': [], 'away_team': [], 'away_rank': [], 'home_team': [], 'home_rank': []}

        cur_date = date.today() - timedelta(days=1)
        day = f"{str(cur_date.month)}-{str(cur_date.day)}-{str(cur_date.year)}"
        for game in games.games[day]:
            game_dict['date'].append(day)
            game_dict['away_rank'].append(game['away_rank'])
            game_dict['home_rank'].append(game['home_rank'])
            game_dict['away_team'].append(game['away_name'])
            game_dict['home_team'].append(game['home_name'])

        if len(game_dict['date']) == 0:
            return "No Games Today to Update Model"
        
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
        
        cbb_stats_df['team_ppg'] = cbb_stats_df['total_team_points']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_ppg'] = cbb_stats_df['total_opp_points']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_fg_att_pg'] = cbb_stats_df['total_team_fg_att']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_fg_att_pg'] = cbb_stats_df['total_opp_fg_att']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_fg_made_pg'] = cbb_stats_df['total_team_fg_made']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_fg_made_pg'] = cbb_stats_df['total_opp_fg_made']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_3pt_att_pg'] = cbb_stats_df['total_team_3pt_att']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_3pt_att_pg'] = cbb_stats_df['total_opp_3pt_att']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_3pt_made_pg'] = cbb_stats_df['total_team_3pt_made']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_3pt_made_pg'] = cbb_stats_df['total_opp_3pt_made']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_ft_att_pg'] = cbb_stats_df['total_team_ft_att']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_ft_att_pg'] = cbb_stats_df['total_opp_ft_att']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_ft_made_pg'] = cbb_stats_df['total_team_ft_made']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_ft_made_pg'] = cbb_stats_df['total_opp_ft_made']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_rebounds_pg'] = cbb_stats_df['total_team_rebounds']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_rebounds_pg'] = cbb_stats_df['total_opp_rebounds']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_assists_pg'] = cbb_stats_df['total_team_assists']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_assists_pg'] = cbb_stats_df['total_opp_assists']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_steals_pg'] = cbb_stats_df['total_team_steals']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_steals_pg'] = cbb_stats_df['total_opp_steals']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_blocks_pg'] = cbb_stats_df['total_team_blocks']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_blocks_pg'] = cbb_stats_df['total_opp_blocks']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_turnovers_pg'] = cbb_stats_df['total_team_turnovers']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_turnovers_pg'] = cbb_stats_df['total_opp_turnovers']/cbb_stats_df['total_opp_games']
        cbb_stats_df['team_fouls_pg'] = cbb_stats_df['total_team_fouls']/cbb_stats_df['total_team_games']
        cbb_stats_df['opp_fouls_pg'] = cbb_stats_df['total_opp_fouls']/cbb_stats_df['total_opp_games']
        cbb_stats_df.to_excel('cbb_raw_data.xlsx')

        # Organize and Normalize Data
        cbb_norm_df = pd.read_excel('cbb_raw_data.xlsx')
        cbb_norm_df.drop(columns=['Unnamed: 0'], inplace=True)

        cbb_norm_df['team_code'] = cbb_norm_df['team'].astype("category").cat.codes
        cbb_norm_df['opp_code'] = cbb_norm_df['opp'].astype("category").cat.codes
        cbb_norm_df['team_rank'] = cbb_norm_df['team_rank'].fillna(50)
        cbb_norm_df['opp_rank'] = cbb_norm_df['opp_rank'].fillna(50)

        cbb_stats_df['team_code'] = cbb_stats_df['team_code']/cbb_stats_df['team_code'].max()
        cbb_stats_df['opp_code'] = cbb_stats_df['opp_code']/cbb_stats_df['opp_code'].max()
        cbb_stats_df['team_rank'] = cbb_stats_df['team_rank']/cbb_stats_df['team_rank'].max()
        cbb_stats_df['opp_rank'] = cbb_stats_df['opp_rank']/cbb_stats_df['opp_rank'].max()
        cbb_stats_df['total_team_pace'] = cbb_stats_df['team_pace']/cbb_stats_df['team_pace'].max()
        cbb_stats_df['total_opp_pace'] = cbb_stats_df['opp_pace']/cbb_stats_df['opp_pace'].max()

        '''
        cbb_norm_df['total_team_points'] = cbb_norm_df['total_team_points']/cbb_norm_df['total_team_points'].max()
        cbb_norm_df['total_opp_points'] = cbb_norm_df['total_opp_points']/cbb_norm_df['total_opp_points'].max()
        cbb_norm_df['total_team_fg_att'] = cbb_norm_df['total_team_fg_att']/cbb_norm_df['total_team_fg_att'].max()
        cbb_norm_df['total_opp_fg_att'] = cbb_norm_df['total_opp_fg_att']/cbb_norm_df['total_opp_fg_att'].max()
        cbb_norm_df['total_team_fg_made'] = cbb_norm_df['total_team_fg_made']/cbb_norm_df['total_team_fg_made'].max() 
        cbb_norm_df['total_opp_fg_made'] = cbb_norm_df['total_opp_fg_made']/cbb_norm_df['total_opp_fg_made'].max()
        cbb_norm_df['total_team_3pt_att'] = cbb_norm_df['total_team_3pt_att']/cbb_norm_df['total_team_3pt_att'].max()
        cbb_norm_df['total_opp_3pt_att'] = cbb_norm_df['total_opp_3pt_att']/cbb_norm_df['total_opp_3pt_att'].max()
        cbb_norm_df['total_team_3pt_made'] = cbb_norm_df['total_team_3pt_made']/cbb_norm_df['total_team_3pt_made'].max()
        cbb_norm_df['total_opp_3pt_made'] = cbb_norm_df['total_opp_3pt_made']/cbb_norm_df['total_opp_3pt_made'].max()
        cbb_norm_df['total_team_ft_att'] = cbb_norm_df['total_team_ft_att']/cbb_norm_df['total_team_ft_att'].max()
        cbb_norm_df['total_opp_ft_att'] = cbb_norm_df['total_opp_ft_att']/cbb_norm_df['total_opp_ft_att'].max()
        cbb_norm_df['total_team_ft_made'] = cbb_norm_df['total_team_ft_made']/cbb_norm_df['total_team_ft_made'].max()
        cbb_norm_df['total_opp_ft_made'] = cbb_norm_df['total_opp_ft_made']/cbb_norm_df['total_opp_ft_made'].max()
        cbb_norm_df['total_team_rebounds'] = cbb_norm_df['total_team_rebounds']/cbb_norm_df['total_team_rebounds'].max()
        cbb_norm_df['total_opp_rebounds'] = cbb_norm_df['total_opp_rebounds']/cbb_norm_df['total_opp_rebounds'].max()
        cbb_norm_df['total_team_assists'] = cbb_norm_df['total_team_assists']/cbb_norm_df['total_team_assists'].max()
        cbb_norm_df['total_opp_assists'] = cbb_norm_df['total_opp_assists']/cbb_norm_df['total_opp_assists'].max()
        cbb_norm_df['total_team_steals'] = cbb_norm_df['total_team_steals']/cbb_norm_df['total_team_steals'].max()
        cbb_norm_df['total_opp_steals'] = cbb_norm_df['total_opp_steals']/cbb_norm_df['total_opp_steals'].max()
        cbb_norm_df['total_team_blocks'] = cbb_norm_df['total_team_blocks']/cbb_norm_df['total_team_blocks'].max()
        cbb_norm_df['total_opp_blocks'] = cbb_norm_df['total_opp_blocks']/cbb_norm_df['total_opp_blocks'].max()
        cbb_norm_df['total_team_turnovers'] = cbb_norm_df['total_team_turnovers'] /cbb_norm_df['total_team_turnovers'].max()
        cbb_norm_df['total_opp_turnovers'] = cbb_norm_df['total_opp_turnovers']/cbb_norm_df['total_opp_turnovers'].max()
        cbb_norm_df['total_team_fouls'] = cbb_norm_df['total_team_fouls']/cbb_norm_df['total_team_fouls'].max()
        cbb_norm_df['total_opp_fouls'] = cbb_norm_df['total_opp_fouls']/cbb_norm_df['total_opp_fouls'].max()
        cbb_norm_df['team_code'] = cbb_norm_df['team_code']/cbb_norm_df['team_code'].max()
        cbb_norm_df['opp_code'] = cbb_norm_df['opp_code']/cbb_norm_df['opp_code'].max()
        cbb_norm_df['team_rank'] = cbb_norm_df['team_rank']/cbb_norm_df['team_rank'].max()
        cbb_norm_df['opp_rank'] = cbb_norm_df['opp_rank']/cbb_norm_df['opp_rank'].max()
        cbb_norm_df['total_team_pace'] = cbb_norm_df['total_team_pace']/cbb_norm_df['total_team_pace'].max()
        cbb_norm_df['total_opp_pace'] = cbb_norm_df['total_opp_pace']/cbb_norm_df['total_opp_pace'].max()
        '''
        cbb_norm_df.to_excel('cbb_norm_data.xlsx')

        return "Update Complete"
    
    def retrain_model(self, path, start_year, end_year):
        
        # Model Output Path and File Names
        model_path = path
        model_file = "CBB_Score_Model.pkl"

        data_df = pd.read_excel(f"{path}/{'cbb_norm_data.xlsx'}", index_col=0)
        data_df = data_df[data_df['season'] >= int(start_year)]
        data_df = data_df[data_df['season'] <= int(end_year)]

       # Define Metrics for Input Variables
        X = data_df[['team_ppg', 'opp_ppg', 'team_fg_att_pg', 'opp_fg_att_pg', 'team_fg_made_pg', 'opp_fg_made_pg', 'total_team_fg_pct', 'total_opp_fg_pct', 
                        'team_3pt_att_pg', 'opp_3pt_att_pg', 'team_3pt_made_pg', 'opp_3pt_made_pg', 'total_team_3pt_pct', 'total_opp_3pt_pct', 'team_ft_att_pg', 'opp_ft_att_pg',
                        'team_ft_made_pg', 'opp_ft_made_pg', 'total_team_ft_pct', 'total_opp_ft_pct', 'team_rebounds_pg', 'opp_rebounds_pg', 'team_assists_pg', 'opp_assists_pg',
                        'team_steals_pg', 'opp_steals_pg', 'team_blocks_pg', 'opp_blocks_pg', 'team_turnovers_pg', 'opp_turnovers_pg', 'team_fouls_pg', 'opp_fouls_pg',
                        ]]

        Y = data_df[['team_points','opp_points', 'team_ppg', 'opp_ppg', 'team_fg_att_pg', 'opp_fg_att_pg', 'team_fg_made_pg', 'opp_fg_made_pg', 'total_team_fg_pct', 'total_opp_fg_pct', 
                        'team_3pt_att_pg', 'opp_3pt_att_pg', 'team_3pt_made_pg', 'opp_3pt_made_pg', 'total_team_3pt_pct', 'total_opp_3pt_pct', 'team_ft_att_pg', 'opp_ft_att_pg',
                        'team_ft_made_pg', 'opp_ft_made_pg', 'total_team_ft_pct', 'total_opp_ft_pct', 'team_rebounds_pg', 'opp_rebounds_pg', 'team_assists_pg', 'opp_assists_pg',
                        'team_steals_pg', 'opp_steals_pg', 'team_blocks_pg', 'opp_blocks_pg', 'team_turnovers_pg', 'opp_turnovers_pg', 'team_fouls_pg', 'opp_fouls_pg',
                         'team_code', 'opp_code', 'total_team_pace', 'total_opp_pace']]
        X = X.dropna(axis = 0)
        X = X.reset_index(drop=True)

        Y = Y.dropna(axis=0)
        Y = Y.reset_index(drop=True)
        y_team = Y['team_points']

        # Linear Regression Model
        X_train, X_test, y_train, y_test = train_test_split(X, y_team, test_size = 0.20)

        regr = LinearRegression()
        regr.fit(X_train, y_train.values.ravel())

        # Save Models
        joblib.dump(regr, f"{model_path}/{model_file}")

        # Return Model Results
        return str(round(regr.score(X_test, y_test),2))