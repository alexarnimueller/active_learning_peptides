# -*- coding:utf-8 -*-

import cPickle as pickle
import os

import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score
from progressbar import ProgressBar

from modlamp.core import read_fasta
from modlamp.descriptors import PeptideDescriptor

seed = np.random.RandomState(seed=42)

for d in os.listdir('./output'):
    if os.path.isdir('./output/' + d):
        print("\nRunning %s..." % d)
        sclr = pickle.load(open('./output/' + d + '/scaler.p', 'r'))
        pos = read_fasta('./input/' + d + '/Pos.fasta')[0]
        neg = read_fasta('./input/' + d + '/Neg.fasta')[0]
        desc = PeptideDescriptor(pos + neg, 'PPCALI')
        desc.calculate_autocorr(7)
        X = sclr.transform(desc.descriptor)
        y = np.array(len(pos) * [1] + len(neg) * [0])
        skf = StratifiedKFold(y, n_folds=10)
        
        synth = pd.read_csv('./output/' + d + '/synthesis_selection.csv')
        
        print("\tPerforming 10-fold cross-validation")
        mcc = list()
        acc = list()
        pbar = ProgressBar()
        for train, test in pbar(skf):
            clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                         max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                                         min_samples_leaf=1, min_samples_split=2,
                                         min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
                                         oob_score=True, random_state=42, verbose=0, warm_start=False)
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            mcc.append(matthews_corrcoef(y[test], y_pred))
            acc.append(accuracy_score(y[test], y_pred))
        print("\t10-fold cross-validation MCC:      %.3f±%.3f" % (np.mean(mcc), np.std(mcc)))
        print("\t10-fold cross-validation accuracy: %.3f±%.3f" % (np.mean(acc), np.std(acc)))
        
        print("\tPredicting all picked sequences...")
        df = pd.read_csv('./picked_sequences.csv', sep=',')
        picked = PeptideDescriptor(df['Sequence'].values, 'PPCALI')
        picked.calculate_autocorr(7)
        pckd = sclr.transform(picked.descriptor)
        clf.fit(X, y)
        pred_pckd = clf.predict(pckd)
        mcc = matthews_corrcoef(df['AMP'], pred_pckd)
        acc = accuracy_score(df['AMP'], pred_pckd)
        print("\tMCC on all picked sequences:      %.3f±%.3f" % (np.mean(mcc), np.std(mcc)))
        print("\tAccuracy on all picked sequences: %.3f±%.3f" % (np.mean(acc), np.std(acc)))
