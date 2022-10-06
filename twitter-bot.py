import os
import tweepy
import keys
import pandas as pd
import numpy as np
from transformers import pipeline
from datetime import datetime
import time



now = datetime.now()

dt_string = now.strftime("%d/%m/%Y %H:%M:%S")


auth = tweepy.OAuthHandler(keys.api_key, keys.api_secret)
auth.set_access_token(keys.access_token, keys.access_secret)
api = tweepy.API(auth)

FILE_NAME = 'last_seen.txt'

def read_last_seen(FILE_NAME):
    file_read = open(FILE_NAME, "r")
    last_seen_id = file_read.read().strip()
    file_read.close()
    return last_seen_id

def store_last_seen(FILE_NAME, last_seen_id):
    file_write = open(FILE_NAME, 'w')
    file_write.write(str(last_seen_id))
    file_write.close()
    return

classifier = pipeline("text-classification", model = "jakegehri/twitter-emotion-classifier-BERT")

tweets = api.mentions_timeline()

tweet_id = str(tweets[0].id)

last_seen = read_last_seen(FILE_NAME)

store_last_seen(FILE_NAME, tweet_id)


def get_hashtag(tweets):
    try:
        return tweets[0].entities['hashtags'][0]['text']
    except IndexError:
        return None

tweet_hashtag = get_hashtag(tweets)

"""
while True:
    time.sleep(5)
    tweets = api.mentions_timeline()
    tweet_id = str(tweets[0].id)
    tweet_text = tweets[0].text
    tweet_hashtag = get_hashtag(tweets)
    last_seen = read_last_seen(FILE_NAME)
    if tweet_id != last_seen:
        text = []
        searched_tweets = api.search_tweets(q=hashtag, lang="en", count = 100)
        for tweet in searched_tweets:
            text.append(tweet.text)
        store_last_seen(FILE_NAME, tweet_id)
    else:
        continue
"""


text = []
searched_tweets = api.search_tweets(q=tweet_hashtag, lang="en", count = 100)
for tweet in searched_tweets:
    text.append(tweet.text)

outputs = classifier(text, top_k=6)

df = pd.DataFrame(outputs)

print(df[0][0])

"""


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

if sentiment_score > 0:
    sentiment = "POSITIVE"
else:
    sentiment = "NEGATIVE"

reply = 'On ' + dt_string + ', #' + hashtag + ' has a ' + sentiment + ' sentiment.' + f' (score: {sentiment_score})'

# api.update_status(status=reply, in_reply_to_status_id = id)

"""

    
