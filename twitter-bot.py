import os
import tweepy
import keys
import pandas as pd
import numpy as np
from transformers import pipeline
from datetime import datetime



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

last = read_last_seen(FILE_NAME)
print(last)


def store_last_seen(FILE_NAME, last_seen_id):
    file_write = open(FILE_NAME, 'w')
    file_write.write(str(last_seen_id))
    file_write.close()
    return

# tweets = api.mentions_timeline(read_last_seen(FILE_NAME), tweet_mode='extended')
"""
for tweet in reversed(tweets):
    print(str(tweet.id) + " - " + tweet.full_text)
    api.update_status("@" + tweet.user.screen_name + " Auto reply works.", tweet.id)
    store_last_seen(FILE_NAME, tweet.id)
    """

def sentiment_hashtag():
    request = api.mentions_timeline(count = 1)
    hashtag = request[0].entities['hashtags'][0]['text']
    id = request[0].id

    text = []
    searched_tweets = api.search_tweets(q=hashtag, lang="en", count = 100)
    for tweet in searched_tweets:
        text.append(tweet.full_text)


    classifier = pipeline("text-classification")

    outputs = classifier(text)

    df = pd.DataFrame(outputs) 

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

    api.update_status(status=reply, in_reply_to_status_id = id)

    
