# -*- coding: utf-8 -*-
"""Functions to help the tracking process.

The idea is to have some production ready functions that are optimized to the best fitting machine
learning algorithms, hyperparameters and features.
"""
# pylint:disable=no-member
from collections import OrderedDict
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from ..data.constants import DETKEY
from .scoring import distance_positions_v, score_id_sim_orientation_v
from .training import train_bin_clf


def make_detection_score_fun(dw_truth, frame_diff=1, radius=110, clf=None, **kwargs):
    """Function to generate a scoring function that scores tracks and matching detections.

    Note:
        The default classifier, parameters and features should be set to the best default
        for beesbook tracking.

    This score function uses a :class:`LinearSVC` SVM Classifier from scikit-learn and the features
    distance via :func:`distances_positions_v` and id similarity via
    :func:`.score_id_sim_orientation_v`.

    Arguments:
        dw_truth (:class:`.DataWrapperTruth`): :class:`.DataWrapperTruth` with truth data

    Keyword Arguments:
        clf (scikit-learn classifier): a scikit-learn classifier that will be trained
        frame_diff (int): after n frames close track if no matching object is found
        radius (int): radius in image coordinates to restrict neighborhood search
        **kwargs (:obj:`dict`): keyword arguments for :func:`train_bin_clf`

    Returns:
        tuple: tuple containing:

             - **score_fun** (:obj:`func`): scoring function to score tracks and matching detections
             - **clf** (scikit-learn classifier): the trained scikit-learn classifier
    """
    if clf is None:
        clf = make_pipeline(StandardScaler(), LinearSVC(dual=False))
    features = OrderedDict()
    features['score_distances'] = lambda tracks, detections:\
        distance_positions_v([track.meta[DETKEY][-1] for track in tracks], detections)
    features['score_id_sim_orientation'] = lambda tracks, detections:\
        score_id_sim_orientation_v([track.meta[DETKEY][-1] for track in tracks], detections)
    train_bin_clf(clf, dw_truth, features, frame_diff, radius, **kwargs)

    def score_fun(tracks, detections_test):
        """A scoring function to score tracks and matching detections.

        Arguments:
            tracks (:obj:`list` of :obj:`.Track`): iterable with :obj:`.Track` objects that might
                be extended
            detections_test (:obj:`list` of :obj:`.Detection`): Iterable with :obj:`.Detection`
                objects that might match tracks

        Returns:
            :obj:`np.array`: iterable with negative scores (the smaller the better) or infinity
        """
        detections_path = [track.meta[DETKEY][-1] for track in tracks]
        score_orientations = score_id_sim_orientation_v(detections_path, detections_test)
        score_distances = distance_positions_v(detections_path, detections_test)
        clf_data = np.array((score_distances, score_orientations)).T
        if hasattr(clf, "predict_proba"):
            # we have do adapt the return of predict_proba to be compatible with decision_function
            class_scores = clf.predict_proba(clf_data)
            clf_score = class_scores[:, 1]
            clf_score[class_scores[:, 0] >= clf_score] = -2
        else:
            clf_score = clf.decision_function(clf_data)
        # we use linear sum assignment also known as minimum weight matching in bipartite graphs
        clf_score = -clf_score
        clf_score[clf_score > 0] = np.inf
        return clf_score
    return score_fun, clf


def make_track_score_fun(dw_truth, frame_diff=15, radius=np.inf, clf=None, **kwargs):
    """Function to generate a scoring function that scores tracks and matching tracks.

    Note:
        The default classifier, parameters and features should be set to the best default
        for beesbook tracking.

    Arguments:
        dw_truth (:class:`.DataWrapperTruthTracks`): DataWrapperTruthTracks with truth data
        frame_diff (int): after n frames close track if no matching object is found
        radius (int): radius in image coordinates to restrict neighborhood search

    Keyword Arguments:
        clf (scikit-learn classifier): a scikit-learn classifier that will be trained
        **kwargs (:obj:`dict`): keyword arguments for :func:`.train_bin_clf`

    Returns:
        tuple: tuple containing:

             - **score_fun** (:obj:`func`): scoring function to score matching tracks
             - **clf** (scikit-learn classifier): the trained scikit-learn classifier

    Todo:

         - set default classifier, parameters and features!
    """
    if clf is None:
        clf = make_pipeline(StandardScaler(), LinearSVC(dual=False))
    features = OrderedDict()
    features['score_distances'] = lambda tracks, tracks_test:\
        distance_positions_v([track.meta[DETKEY][-1] for track in tracks],
                             [track.meta[DETKEY][0] for track in tracks_test])
    features['score_id_sim_orientation'] = lambda tracks, tracks_test:\
        score_id_sim_orientation_v([track.meta[DETKEY][-1] for track in tracks],
                                   [track.meta[DETKEY][0] for track in tracks_test])
    train_bin_clf(clf, dw_truth, features, frame_diff, radius, **kwargs)

    def score_fun(tracks, tracks_test):
        """A scoring function to score tracks and matching tracks.

        Arguments:
            tracks (:obj:`list` of :obj:`.Track`): iterable with :obj:`.Track` objects that
                might be extended
            tracks_test (:obj:`list` of :obj:`.Track`): iterable with :obj:`.Track` objects
                that might match tracks

        Returns:
            :obj:`np.array`: iterable with negative scores (the smaller the better) or infinity
        """
        detections_path = [track.meta[DETKEY][-1] for track in tracks]
        detections_test = [track.meta[DETKEY][0] for track in tracks_test]
        score_orientations = score_id_sim_orientation_v(detections_path, detections_test)
        score_distances = distance_positions_v(detections_path, detections_test)
        clf_data = np.array((score_distances, score_orientations)).T
        if hasattr(clf, "predict_proba"):
            # we have do adapt the return of predict_proba to be compatible with decision_function
            class_scores = clf.predict_proba(clf_data)
            clf_score = class_scores[:, 1]
            clf_score[class_scores[:, 0] >= clf_score] = -2
        else:
            clf_score = clf.decision_function(clf_data)
        # we use linear sum assignment also known as minimum weight matching in bipartite graphs
        clf_score = -clf_score
        clf_score[clf_score > 0] = np.inf
        return clf_score
    return score_fun, clf
