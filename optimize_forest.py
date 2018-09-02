# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.metrics.pairwise import pairwise_distances

from modlamp.descriptors import PeptideDescriptor


def get_tree_pred(model, X):
    preds = np.empty((X.shape[0], len(model.estimators_)))
    for i, tree in enumerate(model.estimators_):
        preds[:, i] = tree.predict_proba(X.astype('float32'), check_input=False)[:, 1]  # don't always check input dim
    return preds


Pos = PeptideDescriptor('/Users/modlab/y/pycharm/activelearning/retrospective/input/B/Pos.fasta', 'PPCALI')
Pos.keep_natural_aa()
Neg = PeptideDescriptor('/Users/modlab/y/pycharm/activelearning/retrospective/input/B/Neg.fasta', 'PPCALI')
Neg.keep_natural_aa()
y = np.array(len(Pos.sequences) * [1] + len(Neg.sequences) * [0])  # target vector

Data = PeptideDescriptor(Pos.sequences + Neg.sequences, 'PPCALI')
Data.calculate_autocorr(7)

# Scaler
scaler = StandardScaler()
X = scaler.fit_transform(Data.descriptor)

# Classifier
clf = RandomForestClassifier(bootstrap=True, class_weight=None,
                             criterion='gini', max_depth=None, max_features='auto',
                             max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
                             min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
                             oob_score=True, random_state=3, verbose=0,
                             warm_start=False)

# 5-fold cross validation
mcc_scorer = make_scorer(matthews_corrcoef)
scores = cross_val_score(clf, X, y, cv=5, scoring=mcc_scorer)
print("5-fold cross validation:\n\nMCC: %.3f\n" % np.mean(scores))

# plot internal distances (cosine-similarity)
clf.fit(X, y)
leafs = clf.apply(X)
dists = list()
for j in range(len(X)):
    dists.extend(np.mean(pairwise_distances(X, X[j].reshape(1, -1), metric='cosine'), axis=1).tolist())

plt.hist(dists, 100)
plt.title('Internal Pairwise Cosine Similarity')
plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# move decision boundary and check MCC in 5-fold cross validation
for b in np.arange(0., 0.3, 0.02):  # range for moving probas
    mccs = list()
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)
        for p in range(len(y_pred)):  # move predicted probability 0.1 --> cutoff level at 0.4 for classification
            y_pred[p, 0] -= b
            y_pred[p, 1] += b
        mccs.append(matthews_corrcoef(y_test, y_pred.round(0)[:, 1]))
    mcc = np.mean(mccs)[0]
    print("moved: %.2f\nMCC:   %.3f" % (b, mcc))
