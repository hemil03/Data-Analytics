import warnings
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pandas as pd
from random import randint
from requests import get
from time import sleep, time


# Lists to store the scraped data in
names = []        #names of movies will be stored in this list
years = []        #year of movies will be stored here
years_url = list(range(2010,2021,1))      #list of years for searching
pages = list(range(1,250,50))         #list of nubmber of pages for searching
imdb_ratings = []       #IMDB rating of movies will be stored here
metascores = []        #metascore of movies will be stored here
votes = []          #votes of movies will be stored here

# Preparing the monitoring of the loop
start_time = time()
requests = 0

#loop through years 2010 - 2020     
for year in years_url:
    sleep(3)        #sleep for 3 seconds
    requests = 0           #counter for requests
    url_yr = ('https://www.imdb.com/search/title/?release_date='+str(year))     #url of IMDB for searching movies by year
  
    #loop through pages 1 - 5      
    for page in pages:    
        # Modified the request URL for year and page      
        url = (url+'&sort=num_votes,desc&start='+str(page))
        response = get(url)

        # Throw a warning for non-200 status codes     
        if(response.status_code != 200):
            warnings.warn('Response code is "not" 200!!!!!')

        #initiating beautifulsoup for extracting data
        bs = BeautifulSoup(response.text, 'html.parser')

        #find movie containers in perticuler webpage
        movie_containers = bs.find_all('div', class_ = 'lister-item mode-advanced')

        
        # Extract data from individual movie container
        for container in movie_containers:
            # If the movie has Metascore, then extract:
            if container.find('div', class_ = 'inline-block ratings-metascore') is not None:
                requests += 1

                name = container.h3.a.text      #extracting name of movie 
                names.append(name)
            
                year = container.h3.find('span', class_ = 'lister-item-year text-muted unbold').text        #extracting year of movie
                years.append(year)
            
                imdb = float(container.strong.text)     #extracting ratings of movie
                imdb_ratings.append(imdb)
        
                m_score = container.find('span', class_ = 'inline-block ratings-metascore').text        #extracting metascore of movie
                metascores.append(int(m_score))
        
                vote = container.find('span', attrs = {'name':'nv'})['data-value']      #extracting votes of movie
                votes.append(int(vote))

            # Break the loop if the number of requests is greater than 70 per year
            if(requests == 70):
                break
        if(requests == 70):
                print('breaking container')
                break

# Create the Movies Pandas Dataframe
df_imdb = list(zip(names, years, imdb_ratings, metascores, votes))
movies_df = pd.DataFrame(df_imdb, columns = ['Name', 'Year', 'IMDB_ratings', 'metascore', 'votes'])

# Clean up the year column - have it just be the year nothing else
movies_df.loc[:, 'year'] = movies_df['year'].str[-5:-1].astype(int)
print(movies_df['year'].head(3))

#Need to make the IMDB and metascore ratings the same scale - so multiply the IMDB column by 10
movies_df['n_imdb'] = movies_df['imdb'] * 10

#write the results to a CSV
movies_df.to_csv('movies.csv')

#Create two histograms to visually compare the average IMDB rating to the average metascore rating
movies_df['metascore'] = movies_df.metascore.astype('float')

plt.hist(movies_df['n_imdb'], label='IMDB Ratings', alpha=.7, edgecolor='red')
plt.hist(movies_df['metascore'], label='Metascore', alpha=0.7, edgecolor='yellow')
plt.legend()
plt.show()