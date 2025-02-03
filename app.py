import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import pandas as pd

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')

songs = pd.read_pickle('songs')  
knn = pd.read_pickle('knn')      
feature_matrix = pd.read_pickle('feature_matrix')  
indices = None

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")
    
    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        return album_cover_url
    else:
        return "https://storage.googleapis.com/pr-newsroom-wp/1/2023/05/Spotify_Primary_Logo_RGB_Green.png"

def recommender(song_name, artist_name, n=5):
    global indices
    song_index = songs[(songs['track_name'] == song_name) & (songs['artist_name'] == artist_name)].index[0]
    
    distances, indices = knn.kneighbors(feature_matrix[song_index].reshape(1, -1), n_neighbors=n+1)
    
    recommended_songs = []
    recommended_song_posters = []
    
    for i in range(1, n+1):
        idx = indices[0][i]
        song = songs.iloc[idx]['track_name']
        artist = songs.iloc[idx]['artist_name']
        
        recommended_song_posters.append(get_song_album_cover_url(song, artist))
        recommended_songs.append(song)
        
    return recommended_songs, recommended_song_posters

st.header("MUSIC RECOMMENDER SYSTEM")

song_list = [f"{song} by {artist}" for song, artist in zip(songs['track_name'], songs['artist_name'])]

search_term = st.text_input("Search for a song:")
if search_term:
    filtered_songs = [song for song in song_list if search_term.lower() in song.lower()]
else:
    filtered_songs = song_list

selected_song = st.selectbox('Select a Song', filtered_songs)

selected_song_name, selected_artist_name = selected_song.split(' by ')

if st.button("Show Recommendations"):
    recommended_songs, recommended_song_posters = recommender(selected_song_name, selected_artist_name)

    cols = st.columns(len(recommended_songs)) 
    for i, (song, artist) in enumerate(zip(recommended_songs, songs.loc[indices[0][1:], 'artist_name'])):
        with cols[i]:
            st.write(f"{artist} - {song}") 
            st.image(recommended_song_posters[i], width=200)
