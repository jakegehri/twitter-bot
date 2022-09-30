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

hashtag = "#tesla"

text = []
searched_tweets = api.search_tweets(q=hashtag, lang="en", count = 100)
for tweet in searched_tweets:
    text.append(tweet.text)

print(len(searched_tweets))

print(searched_tweets[1].text)

classifier = pipeline("text-classification")

outputs = classifier(text)
print(pd.DataFrame(outputs))

