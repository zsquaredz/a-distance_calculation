import pre
from sklearn.metrics import hinge_loss
from sklearn import svm

def train_classifier(feature_matrix, labels):
    clf = svm.LinearSVC(random_state=42, loss="hinge")
    clf.fit(feature_matrix, labels)
    return clf


def calc_distance(src_path, trg_path, ngram, min_freq):
    train_feature_matrix, y_train, test_feature_matrix, y_test = pre.get_features(src_path, trg_path, ngram, min_freq)
    clf = train_classifier(train_feature_matrix, y_train)
    pred_decision = clf.decision_function(test_feature_matrix)
    a_distance = hinge_loss(y_test, pred_decision)
    return 1-a_distance

