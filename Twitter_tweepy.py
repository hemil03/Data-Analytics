from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
import pandas as pd
import itertools
from collections import Counter
import re
import matplotlib.pyplot as plt

#Twitter developer credentials
consumer_key="888888888"
consumer_secret="888888888"
access_token="88888-88888"
access_token_secret="888888"

#initialising tweepy API using abow credentials
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
auth_api = API(auth)
user_name = '@billgates'

#collecting data of tweeter account using given user name
item = auth_api.get_user(user_name)
print("name: " + item.name)
print("screen_name: " + item.screen_name)
print("description: " + item.description)
print("statuses_count: " + str(item.statuses_count))
print("friends_count: " + str(item.friends_count))
print("followers_count: " + str(item.followers_count))

#collecting data of all tweets
tweets = auth_api.user_timeline(screen_name = user_name, count = 200, include_rts = False, tweet_mode = 'extended')

#downloading data of all tweets of given user
all_tweets = []
all_tweets.extend(tweets)
oldest_id = tweets[-1].id
while True:
    tweets = auth_api.user_timeline(screen_name = user_name, count = 200, include_rts = False, max_id = oldest_id - 1, tweet_mode = 'extended')
    if len(tweets) == 0:
        break
    oldest_id = tweets[-1].id
    all_tweets.extend(tweets)

#saving downloaded data into .csv file
outtweets = [[tweet.id_str, 
              tweet.created_at, 
              tweet.favorite_count, 
              tweet.retweet_count,
              tweet.source,
              tweet.full_text.encode('utf-8').decode('utf-8')] 
             for idx,tweet in enumerate(all_tweets)]
df = pd.DataFrame(outtweets, columns = ['id', 'created_at', 'favorite_count', 'retweet_count', 'source', 'text'])
df.to_csv('tweets.csv',index = False)

#remove urls from collected tweets for word count
def remove_url(txt):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())
tweet_no_url = [remove_url(tweet) for tweet in df['text']] 

#count words used in tweets and convert it into lower string
words = [string.lower().split() for string in tweet_no_url]
filtered_words = list(itertools.chain(words))

m = []
for i in range(len(filtered_words)):
    for j in range(len(filtered_words[i])):
        m.append(filtered_words[i][j])


#finding most common words used in every collected tweets
final_common = []
for i in m:
    if(len(i) >= 4):
        final_common.append(i)

Counter = Counter(final_common)
Counter.most_common(20)

print('Most liked tweet')
print(df.loc[(df['favorite_count'] == df['favorite_count'].max())])

print('Most retweeted tweet')
print(df.loc[(df['retweet_count'] == df['retweet_count'].max())])

#plotting number of tweets uploded using different sources
source = df.source.unique()
s_count = []
for i in source:
    s_count.append(len(df.loc[(df['source'] == i)]))
    s_csv = df.loc[(df['source'] == i)]
    s_csv.to_csv('source_' + i + '_tweets.csv',index = False)

cs = pd.DataFrame(list(zip(source, s_count)), columns = ['Source', 'Count'])
cs['Count'].plot(kind="bar")
plt.xticks(range(len(source)), list(source))
plt.xlabel('Sources')
plt.ylabel('No. of tweets')
plt.title('Tweets by Sources')
for i, v in enumerate(list(s_count)):
    plt.text(i - 0.25, v, str(v))
plt.show()

#plotting number of tweets by year
y_count = []
year = list(range(2011, 2022, 1))

for i in year:
    y_count.append(len(df[df['created_at'].dt.year == i]))
    y_csv = df[df['created_at'].dt.year == i]
    y_csv.to_csv('year_' + str(i) + '_tweets.csv',index = False)

cy = pd.DataFrame(list(zip(year, y_count)), columns = ['Year', 'Count'])
cy['Count'].plot(kind="bar")
plt.xticks(range(len(year)), list(year))
plt.xlabel('Year')
plt.ylabel('No. of tweets')
plt.title('Tweets by Year')
for i, v in enumerate(list(y_count)):
    plt.text(i - 0.25, v, str(v))
plt.show()

#plotting number of tweets by month 
m_count = []
month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

for i in range(1, 13):
    m_count.append(len(df[df['created_at'].dt.month == i]))
    m_csv = df[df['created_at'].dt.month == i]
    m_csv.to_csv('month_' + month[i-1] + '_tweets.csv',index = False)
    
cm = pd.DataFrame(list(zip(month, m_count)), columns = ['Month', 'Count'])
cm['Count'].plot(kind="bar")
plt.xticks(range(len(month)), list(month))
plt.xlabel('Month')
plt.ylabel('No. of tweets')
plt.title('Tweets by Month')
for i, v in enumerate(list(m_count)):
    plt.text(i - 0.25, v, str(v))
plt.show()

#plotting number of tweets by day
d_count = []
day = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

for i in range(7):
    d_count.append(len(df[df['created_at'].dt.weekday == i]))
    d_csv = df[df['created_at'].dt.weekday == i]
    d_csv.to_csv('day_' + day[i] + '_tweets.csv',index = False)
    
cd = pd.DataFrame(list(zip(day, d_count)), columns = ['Day', 'Count'])
cd['Count'].plot(kind="bar")
plt.xticks(range(len(day)), list(day))
plt.xlabel('Weekday')
plt.ylabel('No. of tweets')
plt.title('Tweets by Day')
for i, v in enumerate(list(d_count)):
    plt.text(i - 0.25, v, str(v))
plt.show()