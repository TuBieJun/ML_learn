#!/usr/bin/python

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.externals import joblib

if __name__ == "__main__":

    # load data
    digits = load_digits()
    print digits.data.shape

    # split data to train and test set
    x_train, x_test, y_train, y_test = train_test_split(
                                            digits.data,
                                            digits.target,
                                            test_size=0.25,
                                            random_state=33
                                            )

    print y_test, len(y_test.shape)
    
    # normal data
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    # use Linear SVC
    lsvc = LinearSVC()
    lsvc.fit(x_train, y_train)
    y_predict = lsvc.predict(x_test)

    # estimate model
    print "The Accuracy of Linear SVC is", lsvc.score(x_test, y_test)
    print classification_report(
                                y_test,
                                y_predict,
                                target_names=digits.target_names.astype(str)
                                  )

    #save model
    joblib.dump(lsvc, "svm_model.m")










    

