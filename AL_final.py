# -*- coding: utf-8 -*-
"""
ACTIVE LEARNING WITH AMP - FINAL RUN
====================================
2016 modlab ETH Zurich - Alex Müller - alex.mueller@pharma.ethz.ch

This script can be used to fit a Random Forest (RF) classifier with 500 trees on a peptide class (file "Pos.fasta")
vs. other peptides ("Neg.fasta"), followed by presenting a mixed library of peptides to the classifier and determining
predictions as well as RF similarity to all known training samples. As a result, the
script gives the an ordered list of peptides with the ordering score being the probability of being class "Pos".

The script needs three input files with peptide sequences:
- Positive training examples (e.g. AMPs) named "Pos.fasta"
- Negative training examples (e.g. TM sequences) named "Neg.fasta"
- The screening library to pick peptides from named "Lib.fasta"

In addition, this script needs a pre-trained classifier and scaler as pickle files in the input folder:
- "scaler.p"
- "classifier.p"
These are obtained from the first screening script AL_run1.py

:Usage:
python AL_run1.py <input_folder> <output_folder>

:param input_folder:  folder containing the above mentioned three files.
:param output_folder: folder where generated output files will be saved to.

:Example:
python AL_run.py input/ output/

:Important:
This project requires **scikit-learn** version **0.17.1** to function properly!
"""

import cPickle as pickle
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

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
    print "Loading prefitted scaler and standard scaling %s descriptor..." % descriptor
    scaler = pickle.load(open(infolder + '/scaler.p', 'r'))
    Data = scaler.transform(Data.descriptor)

    # Classifier
    print "Loading pretrained classifier..."
    clf = pickle.load(open(infolder + '/classifier.p', 'r'))
    
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
    sums = [0.5 * (x * (1 - y) + z) for x, y, z in zip(variance, RF_dist, probas)]  # dens-weight + proba

    # create data frame with all values
    d = pd.DataFrame({'Class': class_labels, 'Prediction': probas, 'RFSimilarity': RF_dist, 'TreeVariance': variance,
                    'WeighedSum': sums}, index=Lib.sequences)
    d.index.name = 'Sequence'
    d = d[['Class', 'Prediction', 'RFSimilarity', 'TreeVariance', 'WeighedSum']].sort_values('Prediction',
                                                                                             ascending=False)
    
    # get top 5 and bottom 5 predictions according to the AMP prediction
    synth_sele_top = d[:5]
    synth_sele_bottom = d[-5:]
    synth_sele = pd.concat([synth_sele_top, synth_sele_bottom])

    # writing output
    print "Saving output files to output directory..."
    synth_sele.to_csv(outfolder + '/' + datetime.now().strftime("%Y-%m-%d_%H-%M") + 'synthesis_selection.csv')
    d.to_csv(outfolder + '/library_pred.csv')

    # saving scaler and classifier to pickle file for later usage
    pickle.dump(sclr, open(outfolder + '/' + datetime.now().strftime("%Y-%m-%d_%H-%M") + '-scaler.p', 'w'))
    pickle.dump(clf, open(outfolder + '/' + datetime.now().strftime("%Y-%m-%d_%H-%M") + '-classifier.p', 'w'))

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
