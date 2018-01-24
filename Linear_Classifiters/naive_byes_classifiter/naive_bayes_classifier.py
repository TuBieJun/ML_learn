#!/usr/bin/python

from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.externals import joblib



if __name__ == "__main__":

    # get data
    news = fetch_20newsgroups(subset='all')
    print len(news.data)
    print news.data[0]

    #split the data
    x_train, x_test, y_train, y_test = train_test_split(
                                       news.data,
                                       news.target,
                                       test_size=0.25,
                                       random_state=33
                                        )

    #extract feature and normal
    vec = CountVectorizer()
    x_train = vec.fit_transform(x_train)
    x_test = vec.transform(x_test)

    #use Naive Bayes model
    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)
    y_predict = mnb.predict(x_test)

    #show the result
    print "The accuary of Naive Bayes Classifier is", mnb.score(x_test, y_test)
    print classification_report(y_test, y_predict, target_names=news.target_names)

    #save model
    joblib.dump(mnb, "mnb_model.m")











     


    


