import requests

def fetch_movie_data(movie_id):
    url = f"http://128.2.220.241:8080/movie/{movie_id}"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        return data
    else: 
        print(f"Failed to fetch {movie_id}: {r.status_code}")

def fetch_user_data(user_id):
    url = f"http://128.2.220.241:8080/user/{user_id}"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        return data
    else: 
        print(f"Failed to fetch {user_id}: {r.status_code}")
