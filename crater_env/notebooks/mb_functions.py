import pandas as pd
import numpy as np
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import random
from datetime import date
from scipy.spatial.distance import cosine
from datetime import datetime
import dateutil.relativedelta
from sklearn.cluster import AffinityPropagation
import time
import requests
from spotipy.oauth2 import SpotifyOAuth
import itertools 
from itertools import zip_longest
from sklearn.decomposition import PCA
import pymysql
import csv
import os
# import mysql.connector
import time



import requests
import pandas as pd

def workflow(search_input, access_token):
    
    # Playlist Data
    search_playlists_ids = search_playlists(search_input, access_token)
    search_playlists_audio_df = get_audio_features(*search_playlists_ids, access_token)
    search_playlists_audio_artist_df = get_artist_info(search_playlists_audio_df, access_token)
    # print(search_playlists_audio_artist_df.shape)
    # print(search_playlists_audio_artist_df.columns)
    # search_playlists_audio_artist_df.to_csv('playlist.csv', index = False)
    
    #User Data matching data
    user_top_df = get_user_top_tracks(access_token, time_range ='short_term')
    # print(user_top_df.shape)
    # print(user_top_df.columns)
    # user_top_df.to_csv('user_pref.csv', index = False)
    
    # search_playlists_audio_artist_df = pd.read_csv('playlist.csv')
    # user_top_df = pd.read_csv('user_pref.csv')
    
    
    x = get_user_pref(search_playlists_audio_artist_df,user_top_df,access_token)
    return x
    

# def search_playlists(search_input, access_token):
#     headers = {
#         'Authorization': f'Bearer {access_token}'
#     }
    
#     # Search for playlists
#     #search_url = f'https://api.spotify.com/v1/browse/featured-playlists/{search_input}/playlists?&limit=30'
#     search_url = f'https://api.spotify.com/v1/search?q={search_input}&type=playlist&limit=20'
#     response = requests.get(search_url, headers=headers)
#     playlists = response.json()['playlists']['items']
    
#     if len(playlists) < 2:
#         return pd.DataFrame()  # Return an empty data frame if less than 5 playlists found
    
#     playlist_ids = [playlist['id'] for playlist in playlists]
#     # print(playlist_ids)
#     tracks = []
    
#     # Get tracks from each playlist
#     for playlist_id in playlist_ids:
#         playlist_url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
#         response = requests.get(playlist_url, headers=headers)
#         tracks_data = response.json().get('items')
#         if tracks_data:
#             tracks.extend(tracks_data)
    
#     if tracks is None:
#         return pd.DataFrame()  # Return an empty data frame if no tracks found
    
#     track_ids = [track['track']['id'] for track in tracks if track is not None and track.get('track') is not None]

#     return track_ids, tracks


def search_playlists(search_input, access_token):
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    
    search_url = f'https://api.spotify.com/v1/search?q={search_input}&type=playlist&limit=20'
    response = requests.get(search_url, headers=headers)
    playlists = response.json()['playlists']['items']
    
    if not playlists:
        return pd.DataFrame()  # Return an empty data frame if no playlists found
    
    data = []
    for playlist in playlists:
        playlist_id = playlist['id']
        playlist_name = playlist['name']
        playlist_url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
        tracks_response = requests.get(playlist_url, headers=headers)
        tracks_data = tracks_response.json().get('items')
        
        for track in tracks_data:
            if track and track.get('track'):
                track_id = track['track']['id']
                track_name = track['track']['name']
                data.append({
                    'search_id': search_input,
                    'playlist_id': playlist_id,
                    'playlist_name': playlist_name,
                    'track_id': track_id,
                    'track_name': track_name
                })
    
    return pd.DataFrame(data)



def get_audio_features(track_ids,tracks, access_token):
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    track_data = []

     # Retrieve audio features in batches of 100
    for i in range(0, len(track_ids), 100):
        batch_track_ids = track_ids[i:i+100]
        audio_features_url = f'https://api.spotify.com/v1/audio-features/?ids={",".join(batch_track_ids)}'
        response = requests.get(audio_features_url, headers=headers)
        audio_features = response.json()['audio_features']
        
        # Add track information and audio features to the data frame
        for track, features in zip(tracks[i:i+100], audio_features):
            if features is not None:
                track_info = {
                    'Track ID': track['track']['id'],
                    'Artist ID': track['track']['artists'][0]['id'],
                    'Danceability': features['danceability'],
                    'Energy': features['energy'],
                    'Loudness': features['loudness'],
                    'Speechiness': features['speechiness'],
                    'Acousticness': features['acousticness'],
                    'Instrumentalness': features['instrumentalness'],
                    'Liveness': features['liveness'],
                    'Valence': features['valence'],
                    'Tempo': features['tempo'],
                    'Duration': features['duration_ms'],
                    'Time Signature': features['time_signature']
                    # 'Mode': features['mode'],
                    # 'Key': features['key'],
                }
                track_data.append(track_info)
                
    return pd.DataFrame(track_data)

def get_artist_info(df, access_token, for_top = False):
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    
    artist_ids = df['Artist ID'].unique().tolist()
    artist_data = []
    
    # Retrieve artist details in batches of 50
    for i in range(0, len(artist_ids), 50):
        batch_artist_ids = artist_ids[i:i+50]
        artist_ids_str = ','.join(batch_artist_ids)
        artists_url = f'https://api.spotify.com/v1/artists?ids={artist_ids_str}'
        response = requests.get(artists_url, headers=headers)
        artists = response.json()['artists']
        
        # Add artist information to the data frame
        for artist in artists:
            artist_info = {
                'Artist ID': artist['id'],
                'Artist Followers': artist['followers']['total'],
                'Artist Genres': ', '.join(artist['genres']),
                'Artist Popularity': artist['popularity']
            }
            artist_data.append(artist_info)
    
    artist_df = pd.DataFrame(artist_data)
    
    # Merge artist information with the input data frame
    if for_top:
        return artist_df
    else:
        merged_df = pd.merge(df, artist_df, on='Artist ID', how='left')
        return merged_df

def get_user_top_tracks(access_token, time_range='short_term'):
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    # Get user's top tracks
    top_tracks_url = f'https://api.spotify.com/v1/me/top/tracks?limit=50&time_range={time_range}'
    response = requests.get(top_tracks_url, headers=headers)
    top_tracks = response.json()['items']
    
    track_ids = [track['id'] for track in top_tracks]
    
    # Retrieve audio features for top tracks
    audio_features_url = f'https://api.spotify.com/v1/audio-features/?ids={",".join(track_ids)}'
    response = requests.get(audio_features_url, headers=headers)
    audio_features = response.json().get('audio_features', [])
    
    # Prepare the data for the merged data frame
    track_data = []
    for track, features in zip(tracks[i:i+100], audio_features):
        if track is not None and track.get('track') is not None and features is not None:
            track_info = {
                'Track ID': track['track']['id'],
                'Artist ID': track['track']['artists'][0]['id'],
                'Danceability': features['danceability'],
                'Energy': features['energy'],
                'Loudness': features['loudness'],
                'Speechiness': features['speechiness'],
                'Acousticness': features['acousticness'],
                'Instrumentalness': features['instrumentalness'],
                'Liveness': features['liveness'],
                'Valence': features['valence'],
                'Tempo': features['tempo'],
                'Duration': features['duration_ms'],
                'Time Signature': features['time_signature']
            }
            track_data.append(track_info)

    
    # Create the merged data frame
    merged_df = pd.DataFrame(track_data)
    
    artist_info_df = get_artist_info(merged_df, access_token, for_top = True)
    merged_df = pd.merge(merged_df, artist_info_df, on='Artist ID', how='left')
    
    return merged_df



def expand_genre(playlist, remove_dupe=True, pca_metric=5):
    # Input genre info
    if remove_dupe:
        unique_songs_playlist = playlist.drop_duplicates(subset=['Artist ID'])
    else:
        unique_songs_playlist = playlist
    all_genres = set()
    for genres in unique_songs_playlist['Artist Genres']:
        if isinstance(genres, str):
            gens = genres.split(',')
            all_genres.update(genres)
    genre_map = {genre: idx for idx, genre in enumerate(all_genres)}

    genre_df = pd.DataFrame(0, index=unique_songs_playlist.index, columns=list(all_genres))
    for idx, genres in enumerate(unique_songs_playlist['Artist Genres']):
        if isinstance(genres, list):
            genre_df.loc[idx, genres] = 1

    pca = PCA(n_components=pca_metric)
    pca_df = pd.DataFrame(pca.fit_transform(genre_df), columns=[f'pca{i}' for i in range(pca_metric)])

    if remove_dupe:
        pca_art = pd.concat([unique_songs_playlist[['Artist ID']].reset_index(drop=True), pca_df], axis=1)
    else:
        pca_art = pd.concat([unique_songs_playlist.reset_index(drop=True), pca_df], axis=1)

    return pca_art, pca, genre_map, all_genres




def get_user_pref(playlist,df,access_token):
        pca_metric = 5

        genre_info = expand_genre(playlist)
        pca_art = genre_info[0]
        pca = genre_info[1]
        genre_map = genre_info[2]
        all_genres = genre_info[3]
        
        songs_playlist = pd.merge(playlist, pca_art, on = 'Artist ID')

        print(songs_playlist.shape)
        print(songs_playlist.columns)
        #Song_count
        song_count = songs_playlist.groupby('Track ID').count().sort_values('Artist ID', ascending = False)[['Artist ID']].reset_index()
        song_count.columns = ['Track ID','song_count']
        print(song_count)
        comp_df = pd.merge(songs_playlist, song_count).drop_duplicates('Track ID')

        #top songs by song_count
        top_vibe_songs = comp_df.sort_values(['song_count','Artist Popularity'], ascending = False)[:int(len(comp_df)*.1)]

        cols = [ 'Danceability', 'Energy',
            'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness',
           'Valence', 'Tempo'] + ['pca'+str(i) for i in range(pca_metric)]
        # took out mode and key
        
        
        #Model top songs 
        model = AffinityPropagation(damping = .5)
        model_fit = model.fit(top_vibe_songs[cols])

        #give clusters
        # top_vibe_songs['clustered'] = model_fit.predict(top_vibe_songs[cols])
        comp_df['clustered'] = model_fit.predict(comp_df[cols])

#         top_vibe_songs = top_vibe_songs[top_vibe_songs.clustered.isin(comp_df.clustered.unique())]
#         cluster_df = top_vibe_songs.groupby('clustered').mean()[cols]
#         cluster_dict = dict(cluster_df.T)

#         #Get proximity from cluster
#         # Maybe remove bottom clusters from this and only include top ( or maybe just top clusters in general)
#         prox_df = pd.DataFrame()
#         for i in comp_df.clustered.unique():
#             clust_list = []
#             c_df = comp_df[comp_df.clustered == i].copy()
#             for j in np.array(c_df[cols]):
#                     clust_list.append(cosine(cluster_dict[i].values , j))
#             c_df['proximity'] = clust_list
#             c_df = c_df.sort_values('proximity')[:int(len(c_df)*self.prox_inclusion_rate)]
#             clust_len = len(c_df)
#             c_df['clust_len'] = clust_len
#             prox_df = pd.concat([prox_df, c_df])

#         top_songs = pd.concat([top_vibe_songs , prox_df]).drop_duplicates(subset = ['track_id'])

        #put genre info in users df
    #         user_genres = [i['genres'] for i in  sp.artists(artist_id)['artists']]
#         user_genres_output = get_artist_features(df.artist_id, self.spotipy_object)
        user_genres = list(df['Artist Genres'])

        ee = pd.DataFrame(user_genres).replace(genre_map)

        final = []
        for i in range(len(ee)):
            temp = [0] * len(set(all_genres))
            for j in ee.iloc[i]:
                try:
                    if j != '.':
                        temp[j] = 1
                except:
                    pass
            final.append(temp)
        pca_df = pd.DataFrame(pca.transform(final), columns = ['pca'+str(i) for i in range(pca_metric)])
        df = pd.concat([df , pca_df], axis = 1)


        df['clustered'] = model_fit.predict(df[cols])

#         #get personal preference
#         person_pref = []
#         for i in df.clustered.unique():
#             person_pref.append((len(df[df.clustered == i ])/len(df), i))

#         pref = pd.DataFrame(person_pref, columns = ['weight' , 'clustered'])

        #top clusters
        top_list  = []
        for i in df.clustered.unique():
            top_list.append((len(df[df.clustered == i]), i))

        clust_df = pd.DataFrame(top_list, columns = ['ratio','clust'])
        clust_df['ratio'] = clust_df.ratio / len(df)
        clust_df = clust_df.sort_values('ratio', ascending = False)

        #single Cluster
        top_clust = [int(clust_df.iloc[0].clust)]

        # #multiple clusters
        # top_clust = list(clust_df[:int(len(clust_df)*self.top_clust_rate)].clust.values)
            
#         if refresh:
#             clus = random.randint(0 , len(top_clust)-1)
#             while top_clust[clus] == songbird.top_clust:
#                 clus = random.randint(0 , len(top_clust)-1)
#                 print((len(top_clust), 'num_clusters'))
                
#         else:
#             clus = random.randint(0 , len(top_clust)-1)
#             print((len(top_clust), 'num_clusters'))
        
            
        # songbird.top_clust = top_clust[clus]
        clus = 0
        top_clust = [top_clust[clus]]
        final_df =  comp_df.copy()
        final_df = final_df.drop_duplicates(subset = ['Track ID'])

        playlist = final_df[final_df.clustered.isin(top_clust)]
        
#         #add extra songs
        
#         extra_songs = pd.concat([self.other_songs , self.user_top_songs])
#         ex_pca = extra_songs
#         ex_pca['song_count'] = int(round(np.mean(playlist.song_count),0))
        
#         user_genres = list(ex_pca['genres'])

#         ee = pd.DataFrame(user_genres).replace(genre_map)

#         final = []
#         for i in range(len(ee)):
#             temp = [0] * len(set(all_genres))
#             for j in ee.iloc[i]:
#                 try:
#                     if j != '.':
#                         temp[j] = 1
#                 except:
#                     pass
#             final.append(temp)
#         ex_pca = ex_pca.reset_index(drop = True)
#         pca_df = pd.DataFrame(pca.transform(final), columns = ['pca'+str(i) for i in range(self.pca_metric)])
#         trans_df = pd.concat([ex_pca , pca_df], axis = 1)
        
#         trans_df['clustered'] = model_fit.predict(trans_df[cols])
#         trans_df['proximity'] = -1
#         trans_df['clust_len'] = -1
#         trans_df['weight'] = playlist.weight.min()
        
#         trans_df = trans_df[trans_df.clustered.isin(top_clust)]
        
#         playlist = pd.concat([playlist , trans_df])
        
        top_playlist = playlist.sort_values(['song_count'], ascending = False)[:int(len(playlist)*.2)]
        print(top_playlist)
#        mid_playlist = playlist.sort_values(['song_count'], ascending = False)[int(len(playlist)*.2):int(len(playlist)*.5)]
#        low_playlist = playlist.sort_values(['song_count'], ascending = False)[int(len(playlist)*.5):int(len(playlist)*.8)]
        
        top_seeds = list(top_playlist.sample(5)['Track ID'])
#        mid_seeds = list(mid_playlist.sample(3)['Track ID'])
#        low_seeds = list(low_playlist.sample(2)['Track ID'])
        
        
        top_recs = get_recommendations(top_seeds, access_token,15, 100)
#        mid_recs = get_recommendations(mid_seeds, access_token, 3, 100)
#        low_recs = get_recommendations(low_seeds, access_token, 2, 100)
#         other_playlists = list(playlist.sample(5)['Track ID'])
        

        all_ids = top_seeds +  top_recs
        #+ mid_seeds + mid_recs + low_seeds + low_recs
       
       
        
        return all_ids
    

def get_recommendations(seed_track_ids, access_token, limit, max_popularity):
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    url = 'https://api.spotify.com/v1/recommendations'
    params = {
        'limit': limit,
        'seed_tracks': ','.join(seed_track_ids),
        'max_popularity': max_popularity
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        json_response = response.json()
        recommended_tracks = json_response['tracks']
        recommended_track_ids = [track['id'] for track in recommended_tracks]
        return recommended_track_ids
    else:
        print('Failed to retrieve recommendations. Error:', response.json())
        return []

