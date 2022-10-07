# Emotion analysis hashtag twitter bot

This exercise is to build a twitter bot that will return the presiding emotional analysis of a given hashtag. model.py fine tunes a Hugging Hace BERT encoder on a dataset of tweets classified into 1 of 6 emotions and send the fine tuned model to the Hugging Face model hub. main.py utlizes this fine tuned encoder to pull tweets given a specific hashtag and aggregate the emotional predictions of these tweets. The bot replies to the user a breakdown of the aggregated emotional feelings the twitter commuinty has about a given hashtag.
