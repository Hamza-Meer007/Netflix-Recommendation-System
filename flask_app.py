import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests


# model loading
model = pickle.load(open(r'D:\Data Science Pojects\Netflix Recommendation System\model\nlp_model.pkl','rb'))
vectorizer = pickle.load(open(r'D:\Data Science Pojects\Netflix Recommendation System\model\tranform.pkl','rb'))


# fetching movie reviews
def fetch_tmdb_reviews(imdb_id):
    api_key = "0baaa8c7b33b5a989872a3febf289ed3"
    tmdb_url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={api_key}&external_source=imdb_id"
    response = requests.get(tmdb_url)
    data = response.json()

    # Extract movie ID from the response
    if 'movie_results' in data and len(data['movie_results']) > 0:
        movie_id = data['movie_results'][0]['id']
    else:
        return {"Error": "Movie not found or invalid IMDb ID."}

    # Fetch reviews using the movie ID
    reviews_url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={api_key}&language=en-US&page=1"
    reviews_response = requests.get(reviews_url)
    reviews_data = reviews_response.json()

    # Initialize a list to hold all the reviews
    all_reviews = []

    # Loop through the pages and collect up to 20 reviews
    page_number = 1
    while len(all_reviews) < 20 and page_number <= reviews_data['total_pages']:
        reviews_url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={api_key}&language=en-US&page={page_number}"
        reviews_response = requests.get(reviews_url)
        reviews_data = reviews_response.json()

        # Add the reviews to the list
        all_reviews.extend(reviews_data['results'])

        # If we already have 20 or more reviews, break out of the loop
        if len(all_reviews) >= 20:
            break

        page_number += 1

    if all_reviews:
        reviews = {review['author']: review['content'] for review in all_reviews[:20]}  # Limit to 20 reviews

        # Analyze sentiment of each review using the NLP model
        sentiment_reviews = {}
        for author, review in reviews.items():
            # Transform the review using the vectorizer
            review_vector = vectorizer.transform([review])

            # Predict sentiment (1: Good, 0: Bad)
            sentiment = model.predict(review_vector)
            sentiment_label = 'Good' if sentiment else 'Bad'

            # Store review and sentiment
            sentiment_reviews[review] = sentiment_label

        return sentiment_reviews
    else:
        return {"None": "No reviews found."}

# similarity between movies

def create_similarity():
    # loading the data
    data = pd.read_csv(r'Datasets\data_2022_2023\last2_data.csv')
    # creating count vectorizer
    # It converts a collection of text documents into a matrix of token counts, where each entry represents the frequency of words in the text.
    cv = CountVectorizer()
    # data['comb'] has all the things i.e; director name,actor 1,2,3 name,genre,movie title
    count_matrix = cv.fit_transform(data['comb']) 

    # Cosine Similarity measures how similar two vectors are, based on the angle between them.
    # 0 means not similar 1 means identical
    similarity = cosine_similarity(count_matrix)

    return data,similarity

# recommend closely related movies

def recommendation(title):
    title = title.lower()  # Convert title to lowercase
    try:
        # Ensure data and similarity are loaded
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()  # Load data and similarity matrix

    # Check if the title exists in the dataset
    if title not in data['movie_title'].unique():
        return "Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies"
    else:
        # Fetch the index of the requested movie
        i = data[data['movie_title'] == title].index[0]

        # Compute similarity scores
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)  # Sort by similarity
        lst = lst[1:11]  # Top 10 similar movies (excluding the input movie)

        # Generate the list of recommended movies
        recommended_movies = []
        for l in lst:
            a = l[0]
            recommended_movies.append(data['movie_title'][a])

        return recommended_movies  # Return the list of recommendations
    

# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])

def convert_to_list(value):
    if isinstance(value, str):  # Ensure it's a string before processing
        value = value.strip('[]"')  # Remove enclosing brackets/quotes
        return [item.strip().strip('"') for item in value.split(',') if item.strip()]
    return value  # Return as-is if already a list


# to get suggestions for autocomplete
def get_suggestions():
    data = pd.read_csv(r'Datasets\data_2022_2023\last2_data.csv')
    return list(data['movie_title'].str.capitalize())

#  Flask Server

app = Flask(__name__)
@app.route("/")
def home():
    suggestions = get_suggestions()
    return render_template('home.html',suggestions=suggestions)

@app.route("/similarity", methods=['POST'])
def similarity():
    movie = request.form['name']  # Get the movie name from the form
    rc = recommendation(movie)   # Call the recommendation function
    
    if isinstance(rc, str):  # Check if the function returned an error message
        return rc
    elif isinstance(rc, list):  # Check if recommendations were returned
        return '---'.join(rc)
    else:
        return "Unexpected error occurred. Please try again."



#'recommend' function to fetch and pass the reviews
@app.route("/recommend", methods=["POST"])
def recommend():

    # Getting data from AJAX request
    title = request.form.get('title')
    imdb_id = request.form.get('imdb_id')  
    cast_ids = request.form.get('cast_ids')
    cast_names = request.form.get('cast_names')
    cast_chars = request.form.get('cast_chars')
    cast_bdays = request.form.get('cast_bdays')
    cast_bios = request.form.get('cast_bios')
    cast_places = request.form.get('cast_places')
    cast_profiles = request.form.get('cast_profiles')
    poster = request.form.get('poster')
    genres = request.form.get('genres')
    overview = request.form.get('overview')
    vote_average = request.form.get('rating')
    vote_count = request.form.get('vote_count')
    release_date = request.form.get('release_date')
    runtime = request.form.get('runtime')
    status = request.form.get('status')
    rec_movies = request.form.get('rec_movies')
    rec_posters = request.form.get('rec_posters')


    # Get movie suggestions for auto-complete
    suggestions = get_suggestions()

    # Convert necessary strings to lists
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)

    # Convert cast_ids string to list
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[", "")
    cast_ids[-1] = cast_ids[-1].replace("]", "")

    # Render cast_bios strings to proper Python strings
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')

    # Ensure all lists have the same length
    min_length = min(len(cast_names), len(cast_ids), len(cast_profiles), len(cast_bdays), len(cast_places), len(cast_bios))

    # Create dictionaries for movie cards, casts, and cast details
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(min_length)}
    cast_details = {
        cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] 
        for i in range(min_length)
    }

    # Fetch reviews from TMDB using the IMDb ID
    movie_reviews = fetch_tmdb_reviews(imdb_id)

    # Pass all the data to the HTML file
    return render_template(
        'recommend.html',
        title=title,
        poster=poster,
        overview=overview,
        vote_average=vote_average,
        vote_count=vote_count,
        release_date=release_date,
        runtime=runtime,
        status=status,
        genres=genres,
        movie_cards=movie_cards,
        reviews=movie_reviews,
        casts=casts,
        cast_details=cast_details,
    )

if __name__ == "__main__":
    app.run(debug=True)
