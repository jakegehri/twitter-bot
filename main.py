import os
import secrets
from ssl import ALERT_DESCRIPTION_UNKNOWN_PSK_IDENTITY
import tweepy
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from transformers import pipeline
from datetime import datetime
import time

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("ACCESS_SECRET")

now = datetime.now()

dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
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

labels = {"LABEL_0" : "sadness", "LABEL_1" : "joy", "LABEL_2" : "love", 
        "LABEL_3" : "anger", "LABEL_4" : "fear", "LABEL_5" : "surprise"}

while True:
    time.sleep(5)
    tweets = api.mentions_timeline()
    tweet_id = str(tweets[0].id)
    tweet_text = tweets[0].text
    tweet_hashtag = get_hashtag(tweets)
    last_seen = read_last_seen(FILE_NAME)
    if tweet_id != last_seen:
        text = []
        searched_tweets = api.search_tweets(q=tweet_hashtag, include_entities = False, lang="en", count = 100)
        for tweet in searched_tweets:
            text.append(tweet.text)
        df = pd.DataFrame({'label': [], 'score' : []})
        for i in range(len(text)):
            outputs = classifier(text[i], top_k=6)
            df = pd.concat([df, pd.DataFrame(outputs)], ignore_index=True, axis=0)

        df = df.replace({"label": labels})

        api.update_status(status = df.groupby('label')['score'].mean().sort_values(ascending=False), in_reply_to_status_id = tweet_id , auto_populate_reply_metadata=True)
        store_last_seen(FILE_NAME, tweet_id)
    else:
        continue

