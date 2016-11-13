"""Code to help train and evaluate models, generate scoring functions and training data.

To help generating learning data we provide some helpers. The provided functions will **combine**
the generation of learning data with training and evaluation of classifiers. You may also use the
generated learning data and scoring functions for your own training and evaluation process.
"""
from __future__ import print_function
import copy
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedShuffleSplit
from sklearn.metrics import classification_report, roc_curve, auc
from ..data import DataWrapperTracks
from ..validation import Validator, convert_validated_to_pandas
from .walker import SimpleWalker


def generate_learning_data(dw_truth, features, frame_diff, radius):
    """Function to generate learning data using truth data.

    The features are are the same that could be used to train a binary classifier.

    Warning:
        It is recommended to use an :obj:`collections.OrderedDict` instead of a regular :obj:`dict`
        for `features` because the order in :obj:`dict` is not guaranteed.

    Arguments:
        dw_truth (:class:`.DataWrapperTruth`): :class:`.DataWrapperTruth` with truth data
        features (:obj:`dict`): ``{:attr:`feature`: score_fun(tracks, frame_objects_test)}`` mapping
        frame_diff (int): after n frames a close track if no matching object is found
        radius (int): radius in image coordinates to restrict neighborhood search

    Returns:
        tuple: tuple containing:

            - **x_data** (:obj:`np.array`): The learning data
            - **y_data** (:obj:`np.array`): The correct classes for the learning data
    """
    def score_fun_learning(tracks_path, frame_objects_test):
        """Scoring function that fills ``learning_data`` and ``learning_target`` for classifier.

        Arguments:
            tracks_path (:obj:`list` of :obj:`.Track`): A list with the :obj:`.Track` objects
            frame_objects_test (:obj:`list` of :obj:`.Track` or :obj:`.Detection`): A list with
                either :obj:`.Track` or :obj:`.Detection` that is scored with `tracks_path`.

        Returns:
            :obj:`list`: distintive weights to allow the walker to continue on the truth path
        """
        for key, fun in features.items():
            learning_data[key].extend(fun(tracks_path, frame_objects_test))
        weights = []
        for track_path, frame_object_test in zip(tracks_path, frame_objects_test):
            truth_path = dw_truth.get_truthid(track_path)
            truth_test = dw_truth.get_truthid(frame_object_test)
            if truth_path is None and truth_test is None:  # pragma: no cover
                # avoid accidentally learning from False Positive Tracks
                weights.append(np.inf)
                learning_target.append(-1)
            elif truth_path == truth_test:
                weights.append(0)
                learning_target.append(1)
            else:
                weights.append(np.inf)
                learning_target.append(0)
        return weights

    learning_data = {key: list() for key in features.keys()}
    learning_target = list()
    walker = SimpleWalker(dw_truth, score_fun_learning, frame_diff, radius)
    tracks = walker.calc_tracks()

    # just make sure that we correctly "tracked" the truth data
    scores = convert_validated_to_pandas(
        Validator(dw_truth).validate(
            tracks, frame_diff - 1, gap_l=False, gap_r=False, cam_gap=True))
    # remove scores with deletes for track training: we have missing fragments in the test data
    if isinstance(dw_truth, DataWrapperTracks):
        scores = scores[scores.deletes == 0]
    assert np.all(scores.inserts == 0) and np.all(scores.deletes == 0) and np.all(scores.value >= 1)

    x_data = np.array([learning_data[key] for key in features.keys()]).T
    y_data = np.array(learning_target)
    y_data[y_data == 0] = False
    y_data[y_data == 1] = True
    mask = np.ones(len(y_data), dtype=bool)
    mask[y_data == -1] = False
    y_data = y_data[mask]
    x_data = x_data[mask, ]
    return x_data, y_data


def train_and_evaluate(clf, x_data, y_data, verbose=False, **kwargs):
    """Function to train and evaluate a Classifier.

    When :attr:`verbose` is True then several metrics are calculated and printed.
    For training only 90 percent of the dataset is used!

    Arguments:
        clf (scikit-learn classifier): a scikit-learn classifier that will be trained
        x_data (:obj:`np.array`): learning data
        y_data (:obj:`np.array`): the classes for the learning data

    Keyword Arguments:
        verbose (Optional bool): if true calculates accuracy, 10-fold cross validation...
        **kwargs (:obj:`dict`): Keyword arguments for ``clf.fit()``.
    """
    train_indices, test_indices = list(StratifiedShuffleSplit(n_splits=1).split(x_data, y_data))[0]
    clf.fit(x_data[train_indices], y_data[train_indices], **kwargs)
    if verbose:
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(x_data[test_indices])[:, 1]
        else:
            y_score = clf.decision_function(x_data[test_indices])
        fpr, tpr, _ = roc_curve(y_data[test_indices], y_score)
        print("ROC AUC: {:.4f}".format(auc(fpr, tpr)))
        print("Accuracy on training set: {:.4f}".format(clf.score(x_data[train_indices],
                                                                  y_data[train_indices])))
        print("Accuracy on testing set: {:.4f}.".format(clf.score(x_data[test_indices],
                                                                  y_data[test_indices])))
        y_pred = cross_val_predict(clf, x_data, y_data, cv=10, fit_params=kwargs)
        print("Classification Report (10 fold cross validation):")
        print(classification_report(y_data, y_pred, digits=4))
        clf.fit(x_data[train_indices], y_data[train_indices], **kwargs)


def train_bin_clf(clf, dw_truth, features, frame_diff, radius, **kwargs):
    """Function to train a binary classifier using truth data.

    The features are used to train a binary classifier to corresponding frame objects.

    Warning:
        It is recommended to use an :obj:`collections.OrderedDict` instead of a regular :obj:`dict`
        for `features` because the order in :obj:`dict` is not guaranteed.

    You will also get a generic scoring function that is compatible with :class:`.SimpleWalker` and
    the generated learning data for training the classifier. Use this data if you want to use some
    *custom* training methods.

    Arguments:
        clf (scikit-learn classifier): a scikit-learn classifier that will be trained
        dw_truth (:class:`.DataWrapperTruth`): :class:`.DataWrapperTruth` with truth data
        features (:obj:`dict`): ``{:attr:`feature`: score_fun(tracks, frame_objects_test)}`` mapping
        frame_diff (int): after n frames a close track if no matching object is found
        radius (int): radius in image coordinates to restrict neighborhood search

    Keyword Arguments:
        verbose (Optional bool): if true prints some information about training success
        **kwargs (:obj:`dict`): Keyword arguments for :func:`train_and_evaluate()` that are also
            passed to ``clf.fit()``.

    Returns:
        tuple: tuple containing:

            - **x_data** (:obj:`np.array`): The learning data
            - **y_data** (:obj:`np.array`): The correct classes for the learning data
            - **score_fun** (:obj:`func`): A generic scoring function for the Classifier
    """

    def score_fun_generic(tracks_path, frame_objects_test):
        """Generic scoring function that might be used together with the trained classifier.

        Note:
            For performance it is recommended to use a custom scoring function in production!

        Arguments:
            tracks_path (:obj:`list` of :obj:`.Track`): A list with the :obj:`.Track` objects
            frame_objects_test (:obj:`list` of :obj:`.Track` or :obj:`.Detection`): A list with
                either :obj:`.Track` or :obj:`.Detection` that is scored with `tracks_path`.

        Returns:
            :obj:`np.array`: Scores for`tracks_path` and `frame_objects_test` pairs.
        """
        clf_data = np.array([fun(tracks_path, frame_objects_test) for fun in features.values()]).T
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

    # make a copy of the feature functions because the training and scoring functions depend on them
    features = copy.deepcopy(features)
    x_data, y_data = generate_learning_data(dw_truth, features, frame_diff, radius)
    train_and_evaluate(clf, x_data, y_data, **kwargs)
    return x_data, y_data, score_fun_generic
