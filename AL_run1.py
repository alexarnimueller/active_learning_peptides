# -*- coding: utf-8 -*-
"""
ACTIVE LEARNING WITH AMP - RUN 1
================================
2016 modlab ETH Zurich - Alex Müller - alex.mueller@pharma.ethz.ch

This script can be used to fit a Random Forest (RF) classifier with 500 trees on a peptide class (file "Pos.fasta")
vs. other peptides ("Neg.fasta"), followed by presenting a mixed library of peptides to the classifier and determining
predictions as well as RF similarity to all known training samples. As a result, the
script gives the an ordered list of peptides with the ordering score being the probability of being class "Pos".

The script needs three input files with peptide sequences:
- Positive training examples (e.g. AMPs) named "Pos.fasta"
- Negative training examples (e.g. TM sequences) named "Neg.fasta"
- The screening library to pick peptides from named "Lib.fasta"

:Usage:
python AL_run.py <input_folder> <output_folder>

:param input_folder:  folder containing the above mentioned three files.
:param output_folder: folder where generated output files will be saved to.

:Example:
python AL_run.py input/ output/

:Important:
This project requires **scikit-learn** version **0.17.1** to function properly!
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
import cPickle as pickle
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from modlamp.descriptors import PeptideDescriptor

seed = np.random.RandomState(seed=3)
globstart = time.time()


def get_tree_pred(model, x):
    """ Get the predictions from all single ensemble members.
    
    :param model: ensemble model
    :param x: data to be predicted as a numpy.array
    :return: predictions from all members in a numpy.array
    """
    preds = np.empty((x.shape[0], len(model.estimators_)))
    for i, tree in enumerate(model.estimators_):
        preds[:, i] = tree.predict_proba(x.astype('float32'), check_input=False)[:, 1]  # don't always check input dim
    return preds


def main(infolder, outfolder):

    descriptor = 'PPCALI'
    
    print "RF Peptide Learning Info\n========================\n"
    print datetime.now().strftime("%Y-%m-%d_%H-%M") + "\n"
    print("INPUT:\nInputfolder is\t%s\nOutputfolder is\t%s\nDescriptor is\t%s , auto-correlated (window 7)\n" %
            (infolder, outfolder, descriptor))

    # -------------------------------- TRAINING --------------------------------
    print "LOG:\nLoading data..."
    Pos = PeptideDescriptor(infolder + '/Pos.fasta', descriptor)
    Pos.filter_duplicates()
    Neg = PeptideDescriptor(infolder + '/Neg.fasta', descriptor)
    Neg.filter_duplicates()
    targets = np.array(len(Pos.sequences) * [1] + len(Neg.sequences) * [0])  # target vector

    # Descriptor calculation
    print "Calculating %s descriptor..." % descriptor
    Data = PeptideDescriptor(Pos.sequences + Neg.sequences, descriptor)
    Data.calculate_autocorr(7)
    
    # Standard Scaling
    print "Standard scaling %s descriptor..." % descriptor
    scaler = StandardScaler()
    Data = scaler.fit_transform(Data.descriptor)

    # Classifier
    clf = RandomForestClassifier(bootstrap=True, class_weight=None,
                criterion='gini', max_depth=None, max_features='sqrt',
                max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
                oob_score=True, random_state=seed, verbose=0,
                warm_start=False)

    # fitting classifier
    print "Fitting Random Forest classifier..."
    clf.fit(Data, targets)
    fit_leafs = clf.apply(Data)
    print "\tRF out-of-bag score: %.2f" % clf.oob_score_

    # -------------------------------- LIBRARY --------------------------------
    # Loading library
    print "Loading sequence library..."
    Lib = PeptideDescriptor(infolder + '/Lib.fasta', descriptor)
    class_labels = [l[:3] for l in Lib.names]  # extract class labels from sequence names
    
    print "\tLibrary size: %i" % len(Lib.sequences)
    print "\tLibrary composition is:\n\t\thel: %i\n\t\tasy: %i\n\t\tnCM: %i" % (class_labels.count('hel'),
                                                                                class_labels.count('asy'),
                                                                                class_labels.count('nCM'))

    # Calculating descriptors for library members
    print "Calculating %s descriptor for library..." % descriptor
    D = PeptideDescriptor(Lib.sequences, descriptor)
    D.calculate_autocorr(7)
   
    # combining both libraries and scaling descriptor
    print "Standard scaling %s descriptor for library..." % descriptor
    X = scaler.transform(D.descriptor)

    # -------------------------------- PREDICTING --------------------------------
    # get single tree predictions and calculate stdev
    print "Predicting single tree results, standard deviation and entropy for library..."
    start = time.time()
    preds = get_tree_pred(clf, X)

    print "Predicting class probabilities for library..."
    probas = clf.predict_proba(X)
    probas = probas[:, 1].tolist()
    variance = np.var(preds, axis=1)
    print ("\tPredictions took %.1f s" % (time.time() - start))

    # calculate similarity of library members to training data
    print "Calculating Random Forest similarity (cosine)..."
    start = time.time()
    lib_leafs = clf.apply(X)  # leaf indices where library samples end up in -> RF intrinsic similarity measure
    D_RF = pairwise_distances(lib_leafs, fit_leafs, metric='cosine')
    RF_dist = D_RF.mean(axis=1).tolist()
    print ("\tDistance calculation took %.1f s" % (time.time() - start))

    # scaling all output features
    print "Min-Max scaling outputs..."
    sclr = MinMaxScaler()
    # some transformations from lists to numpy matrices to arrays back to min-max scaled list:
    variance = np.squeeze(sclr.fit_transform(variance.reshape(-1, 1))).tolist()
    RF_dist = np.squeeze(sclr.fit_transform(np.array(RF_dist).reshape(-1, 1))).tolist()

    # construct final list with all values (prediction, RF_dist, var, sum)
    print "Creating result dictionaries..."
    sums = [x + 0.5 * y + 0.5 * z for x, y, z in zip(probas, RF_dist, variance)]  # weighed [1,0.5,0.5] sum of all values

    # create data frame with all values
    d = pd.DataFrame({'Class': class_labels, 'Prediction': probas, 'RFDistance': RF_dist, 'TreeVariance': variance,
                    'WeighedSum': sums}, index=Lib.sequences)
    d.index.name = 'Sequence'
    d = d[['Class', 'Prediction', 'RFDistance', 'TreeVariance', 'WeighedSum']].sort_values('Prediction', ascending=False)
    
    # get top and bottom two predictions for every class (total 12 sequences = one synthesis)
    d_hel_top = d.loc[d['Class'] == 'hel'].sort_values('Prediction', ascending=False)[:2]
    d_hel_bot = d.loc[d['Class'] == 'hel'].sort_values('Prediction', ascending=True)[:2]
    d_asy_top = d.loc[d['Class'] == 'asy'].sort_values('Prediction', ascending=False)[:2]
    d_asy_bot = d.loc[d['Class'] == 'asy'].sort_values('Prediction', ascending=True)[:2]
    d_nCM_top = d.loc[d['Class'] == 'nCM'].sort_values('Prediction', ascending=False)[:2]
    d_nCM_bot = d.loc[d['Class'] == 'nCM'].sort_values('Prediction', ascending=True)[:2]
    synth_sele = pd.concat([d_hel_top, d_hel_bot, d_asy_top, d_asy_bot, d_nCM_top, d_nCM_bot])

    # writing output
    print "Saving files to output directory..."
    synth_sele.to_csv(outfolder + '/' + datetime.now().strftime("%Y-%m-%d_%H-%M") + 'synthesis_selection.csv')
    d.to_csv(outfolder + '/library_pred.csv')
    
    # saving scaler and classifier to pickle file for later usage
    pickle.dump(sclr, open(outfolder + datetime.now().strftime("%Y-%m-%d_%H-%M") + '-scaler.p', 'w'))
    pickle.dump(clf, open(outfolder + datetime.now().strftime("%Y-%m-%d_%H-%M") + '-classifier.p', 'w'))

    print("Total runtime: %.1f s\n" % (time.time() - globstart))
    print "\nALL DONE SUCCESSFULLY"
    print "Look for your results file in %s\nAnd maybe save this terminal output to a logfile ;-)" % outfolder


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)

    print "Running...\n"
    infolder = os.path.abspath(sys.argv[1])
    outfolder = os.path.abspath(sys.argv[2])

    # run the main function
    main(infolder, outfolder)
