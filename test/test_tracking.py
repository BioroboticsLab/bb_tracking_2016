# -*- coding: utf-8 -*-
"""Adding tests to training and tracking functions.

These tests are more like integration end-to-end tests.

Note:
    The tests in this file are marked as **slow**.
    You will have to run them separately with ``pytest -m slow``.
    They also have a separate job on the continuous integration server.
"""
from collections import OrderedDict
import itertools
import pytest
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from bb_binary import binary_id_to_int
from bb_tracking.data import DataWrapperTruthTracks, Track
from bb_tracking.data.constants import DETKEY
from bb_tracking.tracking import distance_positions_v, make_detection_score_fun, \
    train_bin_clf, make_track_score_fun
from test.conftest import cmp_tracks, generate_random_walker


@pytest.mark.slow
@pytest.mark.parametrize("score_fun_type", ["simple", "detection_clf", "other_clf",
                                            "svm_clf", "bayes_clf", "track_clf", "track_other_clf"])
def test_calc_tracks(score_fun_type):
    """Test the calculation of tracks for a single set of random detection data."""
    walker, detections = next(generate_random_walker())
    bayes = GaussianNB()
    svm = make_pipeline(StandardScaler(), LinearSVC(dual=False))
    features = OrderedDict()
    features['score_distances'] = lambda tracks, detections:\
        distance_positions_v([track.meta[DETKEY][-1] for track in tracks], detections)
    frame_diff = 1
    radius = 10
    if "track" in score_fun_type:
        frame_diff = 10
        radius = 500
        features['score_distances'] = lambda tracks, tracks_test:\
            distance_positions_v([track.meta[DETKEY][-1] for track in tracks],
                                 [track.meta[DETKEY][0] for track in tracks_test])
        cam_timestamps = {cam_id: walker.data.get_timestamps(cam_id=cam_id)
                          for cam_id in walker.data.get_camids()}
        tracks = [Track(id=det.id, ids=[det.id], timestamps=[det.timestamp], meta={DETKEY: [det]})
                  for det in walker.data.get_detections(walker.data.get_all_detection_ids()[0])]
        walker.data = DataWrapperTruthTracks(tracks, cam_timestamps, walker.data)

    if score_fun_type == "detection_clf":
        walker.score_fun, _ = make_detection_score_fun(walker.data, frame_diff=frame_diff,
                                                       radius=radius, verbose=True)
    elif score_fun_type == "track_clf":
        walker.score_fun, _ = make_track_score_fun(walker.data, frame_diff=frame_diff,
                                                   radius=radius, verbose=True)
    elif score_fun_type == "other_clf":
        walker.score_fun, _ = make_detection_score_fun(walker.data, frame_diff=frame_diff,
                                                       radius=radius, clf=bayes, verbose=True)
    elif score_fun_type == "track_other_clf":
        walker.score_fun, _ = make_track_score_fun(walker.data, frame_diff=frame_diff,
                                                   radius=radius, clf=bayes, verbose=True)
    elif "svm" in score_fun_type:
        _, _, walker.score_fun = train_bin_clf(svm, walker.data, features, frame_diff, radius,
                                               verbose=True)
    elif "bayes" in score_fun_type:
        _, _, walker.score_fun = train_bin_clf(bayes, walker.data, features, frame_diff, radius,
                                               verbose=True)

    truth_tracks = detections[1]
    walker.frame_diff = 11
    walker.prune_weight = 4

    assert walker.prune_weight < 5
    assert walker.frame_diff > 10

    # test start parameter
    assert walker.calc_tracks(start=1000) == []

    # test stop parameter
    assert walker.calc_tracks(stop=0) == []

    # test normal parameter
    test_tracks = walker.calc_tracks()
    cmp_tracks_helper(truth_tracks, test_tracks)


@pytest.mark.slow
@pytest.mark.parametrize("score_fun_type", ["simple", "classifier"])
def test_calc_tracks_random(score_fun_type):
    """Test the calculation of tracks for multiple random detection data."""
    frame_diff = 1
    radius = 10
    for (walker, detections), _ in zip(generate_random_walker(), itertools.repeat(None, 10)):
        if score_fun_type == "classifier":
            walker.score_fun, _ = make_detection_score_fun(walker.data, frame_diff, radius)
        truth_tracks = detections[1]
        walker.frame_diff = 11
        walker.prune_weight = 4

        test_tracks = walker.calc_tracks()
        cmp_tracks_helper(truth_tracks, test_tracks)


def cmp_tracks_helper(truth_tracks, test_tracks):
    """Helper to compare truth tracks with test tracks.

    Arguments:
        truth_tracks (iterable): iterable with expected ``Track``s.
        test_tracks (iterable): iterable with ``Track``s to test.
    """
    for test_track in test_tracks:
        truth_id = binary_id_to_int(test_track.meta[DETKEY][0].beeId, endian='little')
        cmp_tracks(test_track, truth_tracks[truth_id], cmp_trackid=False)
    # test length after content to have more detailed information on Problem
    assert len(truth_tracks) == len(test_tracks)
