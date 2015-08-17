from twython import Twython
import psycopg2
from datetime import datetime
import time
import os
from apscheduler.schedulers.blocking import BlockingScheduler
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import NMF

def clean_tokenized_text(doc):
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(['this','a', 'lol', 'haha', 'say','best','want','see','meet','via','week','time','one','two','three'])
    stop_words = set(stop_words)
    doc = doc.replace('#','')
    rx = re.compile('\W+')
    text = rx.sub(" ", doc).strip().lower()
    wnl = WordNetLemmatizer()
    split_cleaned_text = []
    for word in text.split():
        temp = wnl.lemmatize(word, pos='v')
        if len(temp) > 2 and temp.isalpha() and temp not in stop_words:
            split_cleaned_text.append(temp)
    return split_cleaned_text

def cluster_score_store():
    print "start new run clustering, scoring and storing result"
    conn = psycopg2.connect(database='hedda', user='hedda')
    c = conn.cursor()
    #get last days tweets from db
    c.execute('''SELECT id, text, url, (favcount + retweetcount), date_time FROM tweets
                 WHERE date_time::date > current_date - interval '1' day;''')
    texts = pd.DataFrame(c.fetchall())
    texts.columns = ['id','text','url','sumfavrtcount', 'date_time']
    texts['date_time'] = texts['date_time'].apply(pd.to_datetime)
    texts['hrs'] = texts['date_time'].apply(lambda x: (datetime.now()-x).total_seconds()/60/60)
    texts['favrt_hour'] = texts['sumfavrtcount']/texts['hrs']

    #prepare vectorized text
    tfidfvect = TfidfVectorizer(max_features=100, max_df=0.7, min_df=.01, tokenizer=clean_tokenized_text)
    tfidfvect.fit(texts['text'].values)
    X = tfidfvect.transform(texts['text'].values)
    feature_names = tfidfvect.get_feature_names()

    nmf = NMF(n_components = 25, max_iter = 5000).fit(X)
    topic_labels = []

    for topic_idx, topic in enumerate(nmf.components_):
        #print "Topic %s: %s" % (topic_idx,  ' '.join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))
        topic_labels.append(" ".join([feature_names[i]
                        for i in topic.argsort()[:-3 - 1:-1]]))

    y_hat = nmf.transform(X)
    y_norm = [y/ y.sum() for y in y_hat]

    df_y_hat = pd.DataFrame(y_norm, 
                            columns=topic_labels,  
                            index=texts.index.values) 
    df_y_hat = df_y_hat.fillna(0)
    df_nmf = texts.join(df_y_hat) 

    #pick winning cluster for every tweet
    df_nmf['HighScore'] = df_nmf[topic_labels].max(axis=1)
    df_nmf['Cluster'] = df_nmf[topic_labels].idxmax(axis=1)

    #keep only high scores above 0.85
    df_nmf = df_nmf[df_nmf['HighScore']>=0.85]

    #create cluster dataset with highest scoring tweets per cluster 
    #based on the amount of retweets and favorites per hour
    idx = df_nmf.groupby(['Cluster'])['favrt_hour'].transform(max) == df_nmf['favrt_hour']
    df_clus_favrt = pd.DataFrame(df_nmf[idx][['Cluster','id','date_time','text','url','HighScore','sumfavrtcount','favrt_hour']])
    df_temp = df_nmf.groupby('Cluster').sum()['sumfavrtcount'].reset_index()
    df_temp.columns=['Cluster','total-rt-fav']
    df_clus_favrt = pd.merge(df_clus_favrt, df_temp, on ='Cluster')
    df_clus_favrt.columns=['descr','id','date_time','tweettext','url','score', 'sum-rt-fav-toptweet', 'favrt_hour','total-rt-fav']
     
    #export tweet URL, tweet text, tweet date and any additional information you found useful
    #choice: best cluster is the one with highest total retweets and favorites
    export = df_clus_favrt.sort('total-rt-fav', ascending=False).head(5)[['descr','url','tweettext','date_time']]
    export.to_csv('resultnewstweets-%s.csv' % datetime.now(), index=False)
    print "new file created at: %s" % datetime.now()

def downloadnewtweets():
    print "start new run downloading new tweets"
    conn = psycopg2.connect(database='hedda', user='hedda')
    c = conn.cursor() 
    c.execute('''SELECT accountname, lastid FROM accounts''')
    count = 0
    for user, lastid in c.fetchall():
        if len(user) > 0:
            try:
                user_timeline = twitter.get_user_timeline(screen_name=user, since_id=lastid)
            except Exception, e:
                pass
            else:
                maxid = 0
                tweetrows=[]
                for tweet in user_timeline:
                    if tweet['lang'] == 'en' and tweet['id'] > lastid:
                        count+=1
                        if tweet['id'] > maxid: maxid = tweet['id']
                        if len(tweet['entities']['hashtags'])>0:
                            ht = ' '.join(text['text'] for text in tweet['entities']['hashtags'])
                        else: ht =''
                        #username, text, id, datetime, url, favoritecount, retweetcount, hashtags
                        tweetrows.append([user, 
                                          tweet['text'], 
                                          tweet['id'], 
                                          tweet['created_at'], 
                                          'https://twitter.com/%s/status/%s' %(user,tweet['id']),
                                          tweet['favorite_count'], 
                                          tweet['retweet_count'], 
                                          ht])  
                c.executemany('INSERT INTO tweets VALUES (%s,%s,%s,%s,%s,%s,%s,%s)', tweetrows )
                conn.commit()
                maxid = max(lastid,maxid)  
                if maxid > 0:
                    c.execute('''UPDATE accounts SET lastid='%s' WHERE accountname='%s';''' % (maxid, user))
                    conn.commit()
    print('At %s : %d new updates' % (datetime.now(), count))
    conn.close()


twitter = Twython(app_key='...',     
        app_secret='...',
        oauth_token='...',
        oauth_token_secret='...')

if __name__ == "__main__":
    print "start tweetnews.py"
    # Start the scheduler
    sched = BlockingScheduler()

    # download new tweets every 10 minutes
    sched.add_job(downloadnewtweets, 'interval', id='downloadnewtweets',  minutes=10)

    # find clusters and save file every hour
    sched.add_job(cluster_score_store, 'interval', id='clusterscorestore',  hours=1)
    sched.start()
