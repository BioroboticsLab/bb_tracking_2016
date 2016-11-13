# -*- coding: utf-8 -*-
"""Adding tests to scoring functions."""
# pylint:disable=protected-access,redefined-outer-name,too-many-arguments,too-many-locals
from __future__ import division, print_function
from itertools import chain, combinations
import math
import random
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist
from bb_binary import int_id_to_binary
from bb_tracking.data import Detection, Track
from bb_tracking.data.constants import DETKEY
from bb_tracking.tracking import score_id_sim, score_id_sim_v, \
    score_id_sim_orientation, score_id_sim_orientation_v,\
    score_id_sim_rotating, score_id_sim_rotating_v, score_id_sim_tracks_median_v,\
    distance_orientations, distance_orientations_v, distance_positions_v,\
    bit_array_to_int_v
# load deprecated scoring functions separately
from bb_tracking.tracking.scoring import score_ids_best_fit, score_ids_best_fit_rotating, \
    score_ids_and_orientation


def test_id_best_fit():
    """Tests the scoring of two lists of ids."""
    # test setup
    with pytest.raises(AssertionError):
        score_ids_best_fit([], [1, 2, 3])

    with pytest.raises(AssertionError):
        score_ids_best_fit([1, 2, 3], [])

    assert score_ids_best_fit([1, 2, 33, 4], [33, 5], length=10) == 0
    assert score_ids_best_fit(range(5), range(5, 11), length=10) == 0.1
    assert score_ids_best_fit([2], [3], length=10) == 0.1
    assert score_ids_best_fit([2], [8], length=10) == 0.2
    assert score_ids_best_fit([0], [3], length=2) == 1


def test_id_best_fit_rotating():
    """Tests the scoring of two lists of ids using rotation."""
    # test setup
    with pytest.raises(AssertionError):
        score_ids_best_fit_rotating([], [1, 2, 3])

    with pytest.raises(AssertionError):
        score_ids_best_fit_rotating([1, 2, 3], [])

    assert score_ids_best_fit_rotating([1, 2, 33, 4], [33, 5], length=10) == 0
    assert score_ids_best_fit_rotating([2], [3], length=10) == 0.1

    # just rotate left or right and match
    assert score_ids_best_fit_rotating([1], [4], length=3) == 1. / 3
    assert score_ids_best_fit_rotating([4], [1], length=3) == 1. / 3
    assert score_ids_best_fit_rotating([2], [4], length=4) == 1. / 4
    assert score_ids_best_fit_rotating([4], [2], length=4) == 1. / 4

    # increase score to make rotating more unlikely
    assert score_ids_best_fit_rotating([1], [4], rotation_penalty=2, length=3) == 2. / 3
    assert score_ids_best_fit_rotating([4], [1], rotation_penalty=2, length=3) == 2. / 3
    assert score_ids_best_fit_rotating([2], [4], rotation_penalty=2, length=4) == 2. / 4
    assert score_ids_best_fit_rotating([4], [2], rotation_penalty=2, length=4) == 2. / 4

    # zero penalty for rotation
    assert score_ids_best_fit_rotating([1], [8], rotation_penalty=0, length=4) == 0
    assert score_ids_best_fit_rotating([8], [1], rotation_penalty=0, length=4) == 0


def test_score_ids_and_orientation():
    """Tests the scoring of id similarities using orientation as bonus option."""
    # test setup
    with pytest.raises(AssertionError):
        score_ids_and_orientation(([1, 2, 3], [0, math.pi, math.pi]),
                                  ([], []))

    with pytest.raises(AssertionError):
        score_ids_and_orientation(([], []),
                                  ([1, 2, 3], [0, math.pi, math.pi]))

    with pytest.raises(AssertionError):
        score_ids_and_orientation(([1, 2, 3], [0, math.pi]),
                                  ([1, 2, 3], [0, math.pi, math.pi]))

    with pytest.raises(AssertionError):
        score_ids_and_orientation(([1, 2, 3], [0, math.pi, math.pi]),
                                  ([1, 2, 3], [0, math.pi]))

    # simple case, ids are identical
    assert score_ids_and_orientation(([1], [0]), ([2, 3, 1], [0, 0, 0])) == 0

    # no orientation match => same as score_ids_best_fit
    assert score_ids_and_orientation(([1, 2, 33, 4], [math.pi] * 4), ([33, 5], [0] * 2),
                                     length=10) == 0
    assert score_ids_and_orientation((range(5), [math.pi] * 5),
                                     (range(5, 11), [0] * 6), length=10) == 0.1
    assert score_ids_and_orientation(([2], [math.pi]), ([3], [0]), length=10) == 0.1
    assert score_ids_and_orientation(([2], [math.pi]), ([8], [0]), length=10) == 0.2
    assert score_ids_and_orientation(([0], [math.pi]), ([3], [0]), length=2) == 1

    # orientation match, so check bonus
    assert score_ids_and_orientation(([1], [0]), ([3], [math.pi]), length=10) == 0.1  # no bonus!
    assert score_ids_and_orientation(([1], [0]), ([3, 3], [math.pi, math.pi / 8])) == 0


@pytest.fixture
def id_detections():
    """Fixture for scoring of id frequencies - Detections."""
    detections = dict()

    # simple cases
    detections['det0'] = make_detection(beeid=[0], orientation=0)
    detections['det05'] = make_detection(beeid=[0.5], orientation=0)
    detections['det1'] = make_detection(beeid=[1], orientation=0)

    # array like
    n = 5
    factor = 2
    detections['detv1'] = make_detection(beeid=range(n))
    detections['detv2'] = make_detection(beeid=range(1, 1 + n))

    # arrays with floats
    detections['detv12'] = make_detection(beeid=np.array([float(x) / factor
                                                          for x in detections['detv1'].beeId]))
    detections['detv22'] = make_detection(beeid=np.array([float(x) / factor
                                                          for x in detections['detv2'].beeId]))

    return detections


@pytest.fixture
def id_sim_ids(id_detections):
    """Fixture for scoring of id frequencies."""
    detections = pd.DataFrame(columns=['det1', 'det2', 'expected'])

    # some detections that are going to be compared
    det0 = id_detections['det0']
    det05 = id_detections['det05']
    det1 = id_detections['det1']
    det0pi = make_detection(beeid=det0.beeId, orientation=math.pi)

    detv1 = id_detections['detv1']
    detv2 = id_detections['detv2']

    detv12 = id_detections['detv12']
    detv22 = id_detections['detv22']

    # some simple cases
    detections.loc[len(detections)] = (det0, det0pi, 0)
    detections.loc[len(detections)] = (det1, det0pi, 1)
    detections.loc[len(detections)] = (det05, det0pi, 0.5)

    # use array like structures
    n = len(detv1.beeId)
    detv1pi = make_detection(beeid=detv1.beeId, orientation=math.pi)
    detv2pi = make_detection(beeid=detv2.beeId, orientation=math.pi)

    detections.loc[len(detections)] = (detv1, detv1pi, 0)
    detections.loc[len(detections)] = (detv1, detv2pi, n)
    detections.loc[len(detections)] = (detv2, detv1pi, n)

    # use floats in array like structures
    factor = 2
    detv12pi = make_detection(beeid=detv12.beeId, orientation=math.pi)
    detv22pi = make_detection(beeid=detv22.beeId, orientation=math.pi)

    detections.loc[len(detections)] = (detv12, detv12pi, 0)
    detections.loc[len(detections)] = (detv12, detv22pi, float(n) / factor)
    detections.loc[len(detections)] = (detv22, detv12pi, float(n) / factor)

    detections['id_length'] = [len(det.beeId) for det in detections.det1]
    return detections


@pytest.fixture
def id_sim_orientations(id_detections):
    """Fixture for scoring of id frequencies with bonus for orientation."""
    detections = pd.DataFrame(columns=['det1', 'det2', 'bonus_r', 'bonus', 'bits', 'expected'])

    # parameter settings
    bonus_r = math.pi / 6
    bonus = 1.
    n = 1

    # some detections that are going to be compared
    det0 = id_detections['det0']
    det05 = id_detections['det05']
    det1 = id_detections['det1']

    detv12 = id_detections['detv12']
    detv22 = id_detections['detv22']

    # some simple cases with orientation bonus
    detections.loc[len(detections)] = (det0, det0, bonus_r, bonus, n, 0)
    detections.loc[len(detections)] = (det1, det0, bonus_r, bonus, n, 0)
    detections.loc[len(detections)] = (det05, det0, bonus_r, 0.1, n, 0.4)

    # arrays and orientation bonus
    factor = 2
    n = len(detv12.beeId)
    expected = float(n) / factor - 1. / n
    detections.loc[len(detections)] = (detv12, detv12, bonus_r, bonus, n, 0)
    detections.loc[len(detections)] = (detv12, detv22, bonus_r, bonus, n, expected)
    detections.loc[len(detections)] = (detv22, detv12, bonus_r, bonus, n, expected)

    return detections


@pytest.fixture
def id_sim_rotating(id_detections):
    """Fixture for rotating scoring of id frequency."""
    detections = pd.DataFrame(columns=['det1', 'det2', 'penalty', 'bits', 'expected'])

    # some detections that are going to be compared
    detv1 = id_detections['detv1']

    # parameter settings
    n = len(detv1.beeId)
    penalty = 0.5

    # array is 1 off (left + right)
    detv3 = make_detection(beeid=np.array([4, 0, 1, 2, 3]))
    detv4 = make_detection(beeid=np.array([1, 2, 3, 4, 0]))
    detections.loc[len(detections)] = (detv1, detv3, penalty, n, penalty)
    detections.loc[len(detections)] = (detv1, detv4, penalty, n, penalty)

    # array is 2 off (left + right)
    detv5 = make_detection(beeid=np.array([3, 4, 0, 1, 2]))
    detv6 = make_detection(beeid=np.array([2, 3, 4, 0, 1]))
    detections.loc[len(detections)] = (detv1, detv5, penalty, n, 2. * penalty)
    detections.loc[len(detections)] = (detv1, detv6, penalty, n, 2. * penalty)

    # array is 1 off but penalty is too high
    penalty2 = 10
    detections.loc[len(detections)] = (detv1, detv3, penalty2, n, 8)
    detections.loc[len(detections)] = (detv1, detv4, penalty2, n, 8)

    return detections


def test_id_sim(id_sim_ids):
    """Tests the scoring of id similarities."""
    # test setup
    with pytest.raises(AssertionError):
        score_id_sim([], [1, 2])

    for det1, det2, expected, _ in id_sim_ids.itertuples(index=False):
        assert score_id_sim(det1.beeId, det2.beeId) == expected


def test_id_sim_v(id_sim_ids):
    """Tests the scoring of id similarities."""
    # test setup
    empty_detection = make_detection(beeid=[])
    some_detection = make_detection(beeid=[1, 2])
    with pytest.raises(AssertionError):
        score_id_sim_v([], [some_detection])
    with pytest.raises(AssertionError):
        score_id_sim_v([empty_detection], [some_detection])

    for _, group in id_sim_ids.groupby('id_length'):
        assert np.all(score_id_sim_v(group.det1, group.det2) == group.expected)


def test_id_sim_rotating(id_sim_ids, id_sim_rotating):
    """Tests the scoring of id similarities using rotation."""
    # test setup
    with pytest.raises(AssertionError):
        score_id_sim_rotating([], [1, 2])

    with pytest.raises(AssertionError):
        score_id_sim_rotating([1, 2], [])

    for det1, det2, expected, _ in id_sim_ids.itertuples(index=False):
        assert score_id_sim_rotating(det1.beeId, det2.beeId) == expected

    for det1, det2, penalty, _, expected in id_sim_rotating.itertuples(index=False):
        assert score_id_sim_rotating(det1.beeId, det2.beeId, rotation_penalty=penalty) == expected


def test_id_sim_rotating_v(id_sim_ids, id_sim_rotating):
    """Tests the scoring of id similarities using rotation (vectorized)."""
    # test setup
    empty_detection = make_detection(beeid=[])
    some_detection = make_detection(beeid=[1, 2])
    with pytest.raises(AssertionError):
        score_id_sim_rotating_v([], [some_detection])
    with pytest.raises(AssertionError):
        score_id_sim_rotating_v([empty_detection], [some_detection])

    for _, group in id_sim_ids.groupby('id_length'):
        assert np.all(score_id_sim_rotating_v(group.det1, group.det2) == group.expected)

    for (penalty, _), group in id_sim_rotating.groupby(['penalty', 'bits']):
        assert np.all(score_id_sim_rotating_v(group.det1, group.det2, rotation_penalty=penalty) ==
                      group.expected)


def test_id_sim_orientation(id_sim_ids, id_sim_orientations):
    """Tests the scoring of similarities using orientation as bonus option."""
    # test setup
    with pytest.raises(AssertionError):
        score_id_sim_orientation([], None, [1, 2], None)

    for det1, det2, expected, _ in id_sim_ids.itertuples(index=False):
        assert score_id_sim_orientation(det1.beeId, det1.orientation,
                                        det2.beeId, det2.orientation) == expected

    for det1, det2, bonus_r, bonus, _, expected in id_sim_orientations.itertuples(index=False):
        assert expected == score_id_sim_orientation(det1.beeId, det1.orientation,
                                                    det2.beeId, det2.orientation,
                                                    range_bonus_orientation=bonus_r,
                                                    value_bonus_orientation=bonus)


def test_id_sim_orientation_v(id_sim_ids, id_sim_orientations):
    """Tests the scoring of similarities using orientation as bonus option (vectorized)."""
    # test setup
    empty_detection = make_detection(beeid=[])
    some_detection = make_detection(beeid=[1, 2])
    with pytest.raises(AssertionError):
        score_id_sim_orientation_v([], [some_detection])
    with pytest.raises(AssertionError):
        score_id_sim_orientation_v([empty_detection], [some_detection])

    for _, group in id_sim_ids.groupby('id_length'):
        assert np.all(score_id_sim_orientation_v(group.det1, group.det2) == group.expected)

    for (bonus_r, bonus, _), group in id_sim_orientations.groupby(['bonus_r', 'bonus', 'bits']):
        test_values = score_id_sim_orientation_v(group.det1, group.det2,
                                                 range_bonus_orientation=bonus_r,
                                                 value_bonus_orientation=bonus)
        assert np.all(group.expected.values == test_values)


def test_id_sim_tracks_median_v():
    """Tests the scoring of tracks via id similarities."""
    # test setup
    some_detection = make_detection(beeid=[1, 2])
    empty_track = Track(id=0, ids=[], timestamps=[], meta={DETKEY: []})
    some_track = Track(id=1, ids=[some_detection.id], timestamps=[some_detection.timestamp],
                       meta={DETKEY: [some_detection]})
    with pytest.raises(AssertionError):
        score_id_sim_tracks_median_v([], [some_track])
    with pytest.raises(AssertionError):
        score_id_sim_tracks_median_v([empty_track], [some_track])

    n_bits = 12
    # test with tracks with only one detection
    detections = [make_detection(det_id=i, timestamp=i, beeid=[i] * n_bits) for i in range(3)]
    tracks = [Track(id=det.id, ids=[det.id], timestamps=[det.timestamp],
                    meta={DETKEY: [det]}) for det in detections]
    assert list(score_id_sim_tracks_median_v(tracks, tracks)) == [0] * len(tracks)

    assert list(score_id_sim_tracks_median_v([tracks[0]], [tracks[1]])) == [n_bits]

    # tracks with multiple detections
    track1 = Track(id=1, ids=[1, 2, 3], timestamps=[1, 2, 3], meta={DETKEY: detections})
    track2 = Track(id=2, ids=[4, 5, 6], timestamps=[1, 2, 3],
                   meta={DETKEY: [detections[0], detections[1],
                                  make_detection(beeid=[0.5] * n_bits)]})
    results = score_id_sim_tracks_median_v([track1, track1, track2], [track1, track2, track2])

    assert list(results) == [0, n_bits / 2, 0]


def test_distance_orientations():
    """Tests the calculation of the distance between two orientations."""
    assert distance_orientations(0, 0) == 0
    assert distance_orientations(math.pi, math.pi) == 0
    assert distance_orientations(-math.pi, math.pi) == 0
    assert distance_orientations(0, math.pi) == 180
    assert distance_orientations(math.pi / 2, math.pi) == 90
    assert distance_orientations(-math.pi / 2, math.pi) == 90


def test_distance_orientations_v():
    """Tests the calculation of the distance between two lists of orientations (vectorized)."""
    det_0 = make_detection(orientation=0, meta={'meta_orientation': 0})
    det_pi = make_detection(orientation=math.pi, meta={'meta_orientation': math.pi})
    det_neg_pi = make_detection(orientation=-math.pi, meta={'meta_orientation': -math.pi})
    det_half_pi = make_detection(orientation=math.pi / 2, meta={'meta_orientation': math.pi / 2})
    det_neg_half_pi = make_detection(orientation=-math.pi / 2,
                                     meta={'meta_orientation': -math.pi / 2})
    detections1, detections2, expected_results = [], [], []

    # 0, 0
    detections1.append(det_0)
    detections2.append(det_0)
    expected_results.append(0)

    # pi, pi
    detections1.append(det_pi)
    detections2.append(det_pi)
    expected_results.append(0)

    # -pi, pi
    detections1.append(det_neg_pi)
    detections2.append(det_pi)
    expected_results.append(0)

    # 0, pi
    detections1.append(det_0)
    detections2.append(det_pi)
    expected_results.append(math.pi)

    # pi / 2, pi
    detections1.append(det_half_pi)
    detections2.append(det_pi)
    expected_results.append(math.pi / 2)

    # -pi / 2, pi
    detections1.append(det_neg_half_pi)
    detections2.append(det_pi)
    expected_results.append(math.pi / 2)

    results = distance_orientations_v(detections1, detections2)
    assert np.all(results == expected_results)

    meta_results = distance_orientations_v(detections1, detections2, meta_key='meta_orientation')
    assert np.all(meta_results == expected_results)


def test_distance_positions_v():
    """Tests the calculate of the distance between positions of orientations (vectorized)."""
    n = 100
    must_have_seeds = [123]
    for test_seed in chain([random.randint(0, 2**10)], must_have_seeds):
        print("Last used seed: {}".format(test_seed))
        random.seed(test_seed)
        xy_test = np.random.randint(0, high=n * 10, size=(n, 2))
        expected_dists = pdist(xy_test)
        list1, list2 = [], []
        for idx1, idx2 in combinations(range(n), 2):
            xpos1, ypos1 = xy_test[idx1]
            xpos2, ypos2 = xy_test[idx2]
            list1.append(make_detection(xpos=xpos1, ypos=ypos1))
            list2.append(make_detection(xpos=xpos2, ypos=ypos2))
        test_dists = distance_positions_v(list1, list2)
        assert np.all(expected_dists == test_dists)


def test_bit_array_to_int_v():
    """Tests the conversion of bit arrays to integers (vectorized)."""
    expected_ids, detections = [], []
    for i in range(2**12):
        expected_ids.append(i)
        detections.append(make_detection(beeid=int_id_to_binary(i)[::-1] / 2))

    calculated_ids = bit_array_to_int_v(detections)
    assert len(expected_ids) == len(calculated_ids)
    assert set(expected_ids) == set(calculated_ids)

    with pytest.raises(AssertionError):
        bit_array_to_int_v([make_detection(beeid=[1] * 13)])


def make_detection(det_id=0, timestamp=0, xpos=0, ypos=0, orientation=0, beeid=None, meta=None):
    """Helper to generate a Detection with default values."""
    if beeid is None:
        beeid = []
    meta = meta or {}
    return Detection(id=det_id, timestamp=timestamp, x=xpos, y=ypos, orientation=orientation,
                     beeId=beeid, meta=meta)
