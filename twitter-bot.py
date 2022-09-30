import tweepy
import keys
import pandas as pd
import numpy as np
from transformers import pipeline


def api():
    auth = tweepy.OAuth1UserHandler(keys.api_key, keys.api_secret)
    auth.set_access_token(keys.access_token, keys.access_secret)

    return tweepy.API(auth)

api = api()

request = api.mentions_timeline(count = 1)
hashtag = request[0].entities['hashtags'][0]['text']
id = request[0].id

print(hashtag)

text = []
searched_tweets = api.search_tweets(q=hashtag, lang="en", count = 100)
for tweet in searched_tweets:
    text.append(tweet.text)


classifier = pipeline("text-classification")

outputs = classifier(text)

df = pd.DataFrame(outputs) 

print(df)

counter = 0
df['multiplier'] = 0
for i in df['label']:
    
    if i == "NEGATIVE":
        df['multiplier'][counter] = -1
    else:
        df['multiplier'][counter] = 1
    counter += 1
    
df['weighted'] = df['score'] * df['multiplier']

sentiment_score = np.mean((df['weighted']))

print(sentiment_score)
