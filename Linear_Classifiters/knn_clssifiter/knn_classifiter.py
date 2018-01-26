from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib

if __name__ == "__main__":

    #get data
    iris = load_iris()
    print iris.data.shape

    #split data to train set and test set
    x_train, x_test, y_train, y_test = train_test_split(
                                        iris.data,
                                        iris.target,
                                        test_size=0.25,
                                        random_state=33
                                        )

    #normal data
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    #use knn
    knnc = KNeighborsClassifier()
    knnc.fit(x_train, y_train)
    y_predict = knnc.predict(x_test)

    #stat reuslt
    print 'The accuracy of K-Nearest Neighbor Classifier is', knnc.score(x_test, y_test)
    print classification_report(y_test, y_predict, target_names=iris.target_names)

    #save model
    joblib.dump(knnc, "knnc_model.m")
    
    
    






