from flask import Flask, render_template,request,jsonify
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import DuckDuckGoSearchRun
from scipy.sparse import csr_matrix 
import pandas as pd 
import numpy as np
from sklearn.neighbors import NearestNeighbors 
import re
import os 

app = Flask(__name__)
api_key = open('secret.txt', 'r').read().replace('\n','')
print(api_key)
os.environ['OPENAI_API_KEY'] = str(api_key)
llm = OpenAI(temperature=0.5)

# Create a list of movie streaming channels
movie_channels = [
    "Netflix",
    "Amazon Prime Video",
    "HBO Max",
    "Disney+",
    "Disney Plus",
    "Apple TV+",
    "Hulu",
    "Peacock",
    "Paramount+",
    "Paramount Plus",
    "YouTube Movies",
    "Vudu",
    "Tubi",
    "Crave",
    "Sony Crackle",
    "FandangoNOW",
    "Roku Channel",
    "Starz",
    "Showtime"
    # Add more channels here
]

def load_reviews_data():
    df = pd.read_csv("Rotten_Tomatoes_Dataset.csv")
    df = df[['rotten_tomatoes_link', 'critic_name','movie_title','review_score']]

    critic_to_id = {critic: idx for idx, critic in enumerate(df["critic_name"].unique())}
    movie_to_id = {movie: idx for idx, movie in enumerate(df["movie_title"].unique())}

    df["Critic_ID"] = df["critic_name"].map(critic_to_id)
    df["Movie_ID"] = df["movie_title"].map(movie_to_id)

    df = df[['movie_title','Movie_ID','Critic_ID','review_score']]

    return df



  
def create_matrix(df): 
      
    N = len(df['Critic_ID'].unique()) 
    M = len(df['Movie_ID'].unique()) 
      
    # Map Ids to indices 
    user_mapper = dict(zip(np.unique(df["Critic_ID"]), list(range(N)))) 
    movie_mapper = dict(zip(np.unique(df["Movie_ID"]), list(range(M)))) 
      
    # Map indices to IDs 
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["Critic_ID"]))) 
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["Movie_ID"]))) 
      
    user_index = [user_mapper[i] for i in df['Critic_ID']] 
    movie_index = [movie_mapper[i] for i in df['Movie_ID']] 
  
    X = csr_matrix((df["review_score"], (movie_index, user_index)), shape=(M, N)) 
      
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper 

def find_similar_movies(movie_id,movie_mapper,movie_inv_mapper,X, k, metric='cosine', show_distance=False): 
      
    neighbour_ids = [] 
      
    movie_ind = movie_mapper[movie_id] 
    movie_vec = X[movie_ind] 
    k+=1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric) 
    kNN.fit(X) 
    movie_vec = movie_vec.reshape(1,-1) 
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance) 
    for i in range(0,k): 
        n = neighbour.item(i) 
        neighbour_ids.append(movie_inv_mapper[n]) 
    neighbour_ids.pop(0) 
    return neighbour_ids 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inputs', methods=['POST'])
def get_movie_inputs():
    generated_review = None
    streaming_channels = None
    if request.method == 'POST':
        user_input = request.form.get('userInput')
        sentiment = request.form.get('sentiment')
        review_length = request.form.get('reviewLength')
        num_recommendations = int(request.form.get('numRecommendations'))
         
        
        # Call your functions to generate review and movie channels
        generated_review = generate_review(user_input, sentiment, review_length)
        streaming_channels = generate_movie_channels(user_input)

        df = load_reviews_data()
        X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(df) 

       

        movie_titles = dict(zip(df['Movie_ID'], df['movie_title'])) 

        search_value = user_input
        movie_id = [key for key, value in movie_titles.items() if value == search_value][0]

        

        similar_ids = find_similar_movies(movie_id,movie_mapper,movie_inv_mapper,X, num_recommendations, metric='cosine', show_distance=False)
        movie_title = movie_titles[movie_id] 

        similar_movies = [movie_titles[i] for i in similar_ids] 
        print(similar_movies) 
      

    # Render the template and pass the data as template variables
    return render_template('index.html', generated_review=generated_review,similar_movies = similar_movies,streaming_channels=streaming_channels)




def generate_review(user_input, sentiment, review_length):
    prompt_template = PromptTemplate.from_template(
    "Can you write me a {adjective} review for the following movie, {title}, in {number} words?"        
    )
    prompt_template.format(adjective="positive",title ='Die Hard',number=10)

    chain = LLMChain(llm=llm,prompt=prompt_template)
    movie_review = chain.run({'adjective': sentiment,'title':user_input,'number': review_length})

    return movie_review

def generate_movie_channels(user_input):

    search = DuckDuckGoSearchRun()
    result = search.run("Where is {} streaming on?".format(user_input))
    
    search_results = []
    for channel in movie_channels:
        match = re.search(channel, result)

        if match:
            print("Match found:", match.group())
            search_results.append(match.group())
        else:
            print("No match found.")
        
    return search_results



if __name__ == '__main__':
    app.run(debug=True)