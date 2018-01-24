import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report
from sklearn.externals import joblib

if __name__ == "__main__":

    column_names = [ 
                    "Sample code number",
                     "Clump Thickness", 
                     "Uniformity of Cell Size",
                     "Uniformity of Cell",
                     "Mariginal Adhesion",
                     "Single Epithelial Cell Size",
                     "Bare Nuclei",
                     "Bland Chromatin",
                     "Normal Nucleoli",
                     "Mitoses",
                     "Class"
                    ]

    # get the data from internet
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning" \
                      "-databases/breast-cancer-wisconsin/breast-cancer-" \
                      "wisconsin.data", names=column_names)

    # repalce the ? value to nan
    data = data.replace(to_replace='?', value=np.nan)
    
    # drop the nan data
    data = data.dropna(how='any')

    # split the data set to train_set and test_set
    x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]],
                                                        data[column_names[10]],
                                                        test_size = 0.25,
                                                        random_state = 33)

    # normalization data   std=1  mean=0
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    #use LogisticRession
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    lr_y_predict = lr.predict(x_test)

    #estimate model
    print "Accuracy of LR classifiter%s"%(lr.score(x_test, y_test))
    print classification_report(y_test, lr_y_predict, target_names=["positive", "negative"])

    #save the model 
    joblib.dump(lr, "logistic_model.m")
    








