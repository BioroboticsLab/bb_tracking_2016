# -*- coding: utf-8 -*-
"""Adding tests to validation methods."""
# pylint:disable=protected-access,too-many-branches,too-many-statements
import copy
import numpy as np
import pandas as pd
import pytest
import six
from bb_tracking.data import Score, ScoreMetrics, Track
from bb_tracking.data.constants import DETKEY, FPKEY
from bb_tracking.validation import Validator, validation_score_fun_all, calc_fragments,\
    convert_validated_to_pandas, track_statistics


def test_init(data_truth, timestamps):
    """Tests initialization of :class:`Validator` Class."""
    validator = Validator(data_truth)
    assert validator.truth == data_truth
    assert np.all(pd.to_datetime(validator.timestamps) == pd.to_datetime(timestamps))
    assert set(validator.cam_timestamps.keys()) == set([0, 2])


def test_remove_false_positives(validator, timestamps, id_translator):
    """Tests the removal of False positive Tracks."""
    get_ids = id_translator(validator.truth)
    tracks = (
        Track(id=1, ids=get_ids(*range(1, 4)), timestamps=timestamps[0:3], meta={}),
        Track(id=2, ids=get_ids(14, 15), timestamps=[timestamps[4]], meta={}),
        Track(id=3, ids=get_ids(1), timestamps=[timestamps[0]], meta={FPKEY: True}),
        Track(id=4, ids=get_ids(1, 14), timestamps=[timestamps[0]], meta={}),
    )
    tracks_clean = validator.remove_false_positives(tracks)

    assert len(tracks_clean) == 2
    assert tracks_clean[0].id == 1
    assert tracks_clean[1].id == 4


def test_sanity_check(validator, id_translator):
    """Tests the sanity check for Tracks."""
    get_ids = id_translator(validator.truth)
    # duplicate ids
    with pytest.raises(AssertionError) as excinfo:
        validator.sanity_check((Track(id=1, ids=get_ids(1, 1), timestamps=[1, 2], meta={}), ))
    assert str(excinfo.value) == "Duplicate ids in track 1."
    # duplicate timestamps
    with pytest.raises(AssertionError) as excinfo:
        validator.sanity_check((Track(id=1, ids=get_ids(1, 2), timestamps=[1, 1], meta={}), ))
    assert str(excinfo.value) == "Duplicate tstamps in track 1."
    # duplicate track ids
    with pytest.raises(AssertionError) as excinfo:
        tracks = (Track(id=1, ids=get_ids(1, 3), timestamps=[1, 2], meta={}),
                  Track(id=1, ids=get_ids(4, 5), timestamps=[1, 2], meta={}),)
        validator.sanity_check(tracks)
    assert str(excinfo.value) == "Duplicate track id 1."
    # multiple id assignment
    with pytest.raises(AssertionError) as excinfo:
        tracks = (Track(id=1, ids=get_ids(1, 2), timestamps=[1, 2], meta={}),
                  Track(id=2, ids=get_ids(1, 2), timestamps=[1, 2], meta={}),)
        validator.sanity_check(tracks, cam_gap=False)
    # test a little more complicated because of unordered sets
    partial_message = "are assigned multiple times."
    assert partial_message in str(excinfo.value)
    for test_id in get_ids(1, 2):
        assert str(test_id) in str(excinfo.value)
    assert str(excinfo.value).count(',') == 1
    # unknown ids
    with pytest.raises(AssertionError) as excinfo:
        ids = get_ids(1)
        ids.append(99)
        tracks = (Track(id=1, ids=ids, timestamps=[1, 2], meta={}), )
        validator.sanity_check(tracks, cam_gap=False)
    if six.PY2:
        assert str(excinfo.value) == "There are unknown ids in tracks: set([99])."
    else:
        assert str(excinfo.value) == "There are unknown ids in tracks: {99}."
    # false positive track
    tracks = (Track(id=1, ids=get_ids(14, 15), timestamps=[1, 2], meta={}), )
    validator.sanity_check(tracks, cam_gap=False)
    assert tracks[0].meta[FPKEY]
    # return values
    tracks = (Track(id=1, ids=get_ids(1, 2, 14), timestamps=[1, 2, 3], meta={}),
              Track(id=2, ids=get_ids(3, 4, 5), timestamps=[1, 2, 3], meta={}), )
    positives, negatives, f_positives, f_negatives = validator.sanity_check(tracks, cam_gap=False)
    assert positives == set(get_ids(*range(1, 6)))
    assert negatives == set(get_ids(15))
    assert f_positives == set(get_ids(14))
    assert f_negatives == set(get_ids(*range(6, 14)))


def test_validate(validator, timestamps, id_translator):
    """Tests validation of a set of tracks with truth data."""
    # pylint:disable=unused-argument
    def score_fun_1(*args):
        """Scoring function that always returns 1 to inject into validation."""
        return 1

    get_ids = id_translator(validator.truth)
    tracks = (
        Track(id=1, ids=get_ids(*range(1, 4)), timestamps=timestamps[0:3], meta={}),
        Track(id=2, ids=get_ids(*range(6, 8)), timestamps=timestamps[2:4], meta={}),
        Track(id=3, ids=get_ids(*range(12, 14)), timestamps=timestamps[5:7], meta={}),
        Track(id=4, ids=get_ids(15, 12, 13), timestamps=timestamps[4:7], meta={}),
        Track(id=5, ids=get_ids(1, 2, 6, 7, 9), timestamps=timestamps[0:5], meta={}),
        Track(id=6, ids=get_ids(1, 2, 3, 10, 11), timestamps=timestamps[0:5], meta={}),
    )

    expected_tracks = {1: 1, 2: 4, 3: 5, 4: 5, 5: 1, 6: 1}
    validator.timestamps = timestamps

    # test setup
    with pytest.raises(AssertionError):
        validator.validate(None, -1, check=False)

    gap = 0
    # test correct assigning of truth id paths
    scores = validator.validate(tracks, gap, cam_gap=False, val_score_fun=score_fun_1,
                                val_calc_id_fun=score_fun_1, check=False)
    for track_id, score in scores.items():
        if track_id is 5:
            assert score.alternatives == [4, 5]
        else:
            assert score.alternatives == []
        assert score.track_id == track_id
        assert score.truth_id == expected_tracks[track_id]
        assert score.value == 1
        assert score.calc_id == 1

    # test with standard score function resulting in choosing the best truth id path
    track = tracks[4]
    scores = validator.validate([track], gap, cam_gap=False,
                                val_calc_id_fun=score_fun_1, check=False)
    assert len(scores) == 1
    score = scores[track.id]
    assert score.track_id == track.id
    assert score.truth_id == expected_tracks[track.id]
    assert score.alternatives == []


def test_score_track_ids_equal(validator):
    """Tests scoring of tracks considered (locally) equal."""
    n = 20
    validator.timestamps = np.arange(n)
    track1 = Track(0, ids=range(n, 2 * n), timestamps=range(n), meta={})

    # equal
    score_expected = make_score_metric({"track_length": n, "truth_track_length": n,
                                        "adjusted_length": n, "id_matches": n})
    score, _ = validator.score_track(track1, track1)
    assert score_expected == score

    # equal prefix
    for i in range(1, n):
        track2 = Track(i, ids=range(n, n + i), timestamps=range(i), meta={})
        score_expected = make_score_metric({"track_length": i, "truth_track_length": n,
                                            "adjusted_length": i + 1,
                                            "id_matches": i, "deletes": 1, "gap_right": False})
        score, _ = validator.score_track(track1, track2)
        assert score_expected == score

    # equal suffix
    for i in range(1, n):
        track2 = Track(i, ids=range(n + i, 2 * n), timestamps=range(i, n), meta={})
        score_expected = make_score_metric({"track_length": n - i, "truth_track_length": n,
                                            "adjusted_length": n - i + 1,
                                            "id_matches": n - i, "deletes": 1, "gap_left": False})
        score, _ = validator.score_track(track1, track2)
        assert score_expected == score

    # equal center
    track2 = Track(2, ids=range(n + 5, n + 15), timestamps=range(5, 15), meta={})
    score_expected = make_score_metric({"track_length": 10, "truth_track_length": 20,
                                        "adjusted_length": 12, "id_matches": 10,
                                        "deletes": 2, "gap_left": False, "gap_right": False})
    score, _ = validator.score_track(track1, track2)
    assert score_expected == score


def test_score_track_ids_not_equal(validator):
    """Tests scoring of tracks considered different."""
    n = 20
    validator.timestamps = np.arange(n)

    track1 = Track(1, ids=(5, 6, 9), timestamps=(5, 6, 9), meta={})
    track2 = Track(2, ids=range(5, 10), timestamps=range(5, 10), meta={})
    track3 = Track(3, ids=(5, 8, 6, 7, 9), timestamps=range(5, 10), meta={})

    # some mismatches
    score_expected = make_score_metric({"track_length": 5, "truth_track_length": 5,
                                        "adjusted_length": 7, "id_matches": 2,
                                        "id_mismatches": 3, "gap_matches": 2})
    score, _ = validator.score_track(track2, track3)
    assert score_expected == score

    # some inserts
    score_expected = make_score_metric({"track_length": 5, "truth_track_length": 5,
                                        "adjusted_length": 7, "id_matches": 3,
                                        "inserts": 2, "gap_matches": 2})
    score, _ = validator.score_track(track1, track2)
    assert score_expected == score

    # some deletes
    score_expected = make_score_metric({"track_length": 5, "truth_track_length": 5,
                                        "adjusted_length": 7, "id_matches": 3,
                                        "deletes": 2, "gap_matches": 2})
    score, _ = validator.score_track(track2, track1)
    assert score_expected == score

    # empty matches
    score_expected = make_score_metric({"track_length": 5, "truth_track_length": 5,
                                        "adjusted_length": 7, "id_matches": 3,
                                        "gap_matches": 4})
    score, _ = validator.score_track(track1, track1)
    assert score_expected == score


def test_score_track_gaps(validator):
    """Tests scoring of tracks with missed gaps."""
    n = 20
    validator.timestamps = np.arange(n)

    # gap before not found
    track1 = Track(1, ids=(0, 2, 3, 4), timestamps=(0, 2, 3, 4), meta={})
    track2 = Track(2, ids=(2, 3, 4), timestamps=(2, 3, 4), meta={})
    score_expected = make_score_metric({"track_length": 3, "truth_track_length": 5,
                                        "adjusted_length": 7, "id_matches": 3,
                                        "gap_matches": 3, "deletes": 1, "gap_left": False})
    score, _ = validator.score_track(track1, track2, gap=1)
    assert score_expected == score

    # gap after not found
    track1 = Track(1, ids=(0, 1, 2, 4), timestamps=(0, 1, 2, 4), meta={})
    track2 = Track(2, ids=range(3), timestamps=range(3), meta={})
    score_expected = make_score_metric({"track_length": 3, "truth_track_length": 5,
                                        "adjusted_length": 5, "id_matches": 3,
                                        "gap_matches": 1, "deletes": 1, "gap_right": False})
    score, _ = validator.score_track(track1, track2, gap=1)
    assert score_expected == score

    # both gaps not found
    track1 = Track(1, ids=(0, 2, 4), timestamps=(0, 2, 4), meta={})
    track2 = Track(2, ids=(2, ), timestamps=(2, ), meta={})
    score_expected = make_score_metric({"track_length": 1, "truth_track_length": 5,
                                        "adjusted_length": 5, "id_matches": 1,
                                        "gap_matches": 2, "deletes": 2,
                                        "gap_left": False, "gap_right": False})
    score, _ = validator.score_track(track1, track2, gap=1)
    assert score_expected == score

    # id switch before
    track1 = Track(1, ids=range(6, 9), timestamps=range(6, 9), meta={})
    track2 = Track(2, ids=range(3, 9), timestamps=range(3, 9), meta={})
    score_expected = make_score_metric({"track_length": 6, "truth_track_length": 3,
                                        "adjusted_length": 8, "id_matches": 3,
                                        "gap_matches": 2, "inserts": 3, "gap_left": False})
    score, _ = validator.score_track(track1, track2, gap=1)
    assert score_expected == score

    # id switch after
    track1 = Track(1, ids=range(3, 6), timestamps=range(3, 6), meta={})
    track2 = Track(2, ids=range(3, 9), timestamps=range(3, 9), meta={})
    score_expected = make_score_metric({"track_length": 6, "truth_track_length": 3,
                                        "adjusted_length": 8, "id_matches": 3,
                                        "gap_matches": 2, "inserts": 3, "gap_right": False})
    score, _ = validator.score_track(track1, track2, gap=1)
    assert score_expected == score

    # edge case: last element of gap is correct
    track1 = Track(1, ids=(1, 3, 5), timestamps=(1, 3, 5), meta={})
    track2 = Track(2, ids=(2, 3, 4), timestamps=(2, 3, 4), meta={})
    score_expected = make_score_metric({"track_length": 3, "truth_track_length": 5,
                                        "adjusted_length": 7,
                                        "id_matches": 1, "gap_matches": 2,
                                        "deletes": 2, "inserts": 2,
                                        "gap_left": False, "gap_right": False})
    score, _ = validator.score_track(track1, track2, gap=1)
    assert score_expected == score

    # edge case: gap is one bigger then start
    track1 = Track(1, ids=(0, 2, 4), timestamps=(0, 2, 4), meta={})
    track2 = Track(2, ids=(1, 2, 3), timestamps=(1, 2, 3), meta={})
    score_expected = make_score_metric({"track_length": 3, "truth_track_length": 5,
                                        "adjusted_length": 6,
                                        "id_matches": 1, "gap_matches": 1,
                                        "deletes": 2, "inserts": 2,
                                        "gap_left": False, "gap_right": False})
    score, _ = validator.score_track(track1, track2, gap=1)
    assert score_expected == score


def test_score_track_gaps_cam(validator, timestamps, id_translator):
    """Tests scoring of tracks with camera gaps."""
    get_ids = id_translator(validator.truth)
    n = 5
    validator.timestamps = [pd.to_datetime(t) for t in validator.timestamps]
    validator.cam_timestamps = {cam_id: [pd.to_datetime(t) for t in cam_timestamps]
                                for cam_id, cam_timestamps in validator.cam_timestamps.items()}
    # test the case with camera gaps off
    local_ids = get_ids(1, 2, 3, 10, 11)
    local_timestamps = [pd.to_datetime(timestamps[i]) for i in range(n)]

    track_test = Track(1, ids=local_ids, timestamps=local_timestamps, meta={})
    track_truth = validator.truth.get_truth_track(1)

    score, _ = validator.score_track(track_truth, track_test, gap=1)
    score_expected = make_score_metric({"track_length": n, "truth_track_length": n,
                                        "adjusted_length": 7, "id_matches": n, "gap_matches": 2})
    assert score_expected == score

    # test the case with camera gaps on
    n = n - 1
    track_test.ids.pop(1)
    track_test.timestamps.pop(1)
    track_truth = validator.truth.get_truth_track(1, cam_id=0)
    score, _ = validator.score_track(track_truth, track_test, gap=1, cam_id=0)
    score_expected = make_score_metric({"track_length": n, "truth_track_length": n,
                                        "adjusted_length": n, "id_matches": n})
    assert score_expected == score


def test_validation_score_fun_all():
    """Tests calculating score value from metrics."""
    # test setup
    with pytest.raises(AssertionError):
        validation_score_fun_all(None, gap=-1)

    # empty score
    score_test = make_score_metric(dict())
    assert validation_score_fun_all(score_test) == 0

    # all id matches
    score_test = make_score_metric({"adjusted_length": 5, "id_matches": 5})
    assert validation_score_fun_all(score_test) == 1

    # all id mismatches
    score_test = make_score_metric({"adjusted_length": 5, "id_mismatches": 5})
    assert validation_score_fun_all(score_test) == 0

    # all gap matches
    score_test = make_score_metric({"adjusted_length": 5, "gap_matches": 5})
    assert validation_score_fun_all(score_test) == 1

    # all gap mismatches
    score_test = make_score_metric({"adjusted_length": 5, "deletes": 3, "inserts": 2})
    assert validation_score_fun_all(score_test) == 0

    # mixed id and gap matches
    score_test = make_score_metric({"adjusted_length": 5, "id_matches": 3, "gap_matches": 2})
    assert validation_score_fun_all(score_test) == 1

    # mixed id and gap mismatches
    score_test = make_score_metric({"adjusted_length": 5, "id_mismatches": 3,
                                    "deletes": 2, "inserts": 1})
    assert validation_score_fun_all(score_test) == 0

    # everything mixed
    score_test = make_score_metric({"adjusted_length": 6, "id_matches": 2, "id_mismatches": 3,
                                    "gap_matches": 1, "deletes": 1, "inserts": 1})
    assert validation_score_fun_all(score_test) == 0.5


def test_calc_fragments(data_truth):
    """Tests calculating some information about fragments in the truth data."""
    # fragments are divided via camera and gaps
    gap = 0
    fragments, tracks, ids, lengths = calc_fragments(data_truth, gap)
    assert fragments, tracks == (8, 4)
    #   truth_ids: 1  1  2  3  4  1  5  5
    assert ids == [1, 3, 1, 1, 2, 1, 2, 2]
    assert ids == lengths

    # fragments are divided via gaps
    fragments, tracks, ids, lengths = calc_fragments(data_truth, gap, cam_gap=False)
    assert fragments, tracks == (6, 4)
    #   truth_ids: 1  2  3  4  5  5
    assert ids == [5, 1, 1, 2, 2, 2]
    assert ids == lengths

    gap = 10
    # fragments are divided via camera
    fragments, tracks, ids, lengths = calc_fragments(data_truth, gap)
    assert fragments, tracks == (6, 6)  # 1 cam_1 and 1 cam_2 are considered different tracks!
    #   truth_ids: 1  1  2  4  3  5
    assert ids == [4, 1, 1, 2, 1, 4]
    assert lengths == [5, 1, 1, 2, 1, 5]

    # fragments are basically not divided
    fragments, tracks, ids, lengths = calc_fragments(data_truth, gap, cam_gap=False)
    assert fragments, tracks == (5, 5)
    #   truth_ids: 1  2  3  4  5
    assert ids == [5, 1, 1, 2, 4]
    assert lengths == [5, 1, 1, 2, 5]


def test_convert_validated():
    """Tests converting the scores dictionary from Validator to a Pandas DataFrame."""
    metric = make_score_metric({"track_length": 5})
    score = Score(value=1.5, track_id=1, truth_id=2, calc_id=2,
                  metrics=metric, alternatives=[1, 2, 3])
    scores = {0: score}
    scores_df = convert_validated_to_pandas(scores)

    assert scores_df.shape[0] == 1
    row = scores_df.iloc[0]
    assert row.value == score.value
    assert row.track_id == score.track_id
    assert row.truth_id == score.truth_id
    assert row.track_length == metric.track_length
    assert row.alternatives == score.alternatives


def test_track_statistics(validator, tracks_test, id_translator):
    """Tests the calculation of track statistics."""
    get_ids = id_translator(validator.truth)
    tracks_clean = [Track(id=track.id, ids=get_ids(*track.ids), timestamps=track.timestamps,
                          meta=track.meta) for track in tracks_test]
    for track in tracks_clean:
        track.meta[DETKEY] = validator.truth.get_detections(track.ids)
    gap = 10

    # no errors
    tracks = copy.deepcopy(tracks_clean)
    scores = validator.validate(tracks, gap, cam_gap=False)
    metrics = track_statistics(tracks, scores, validator, gap, cam_gap=False)
    for key, (wrong, n_fragments) in metrics['fragments'].items():
        assert wrong == 0, "{} is wrong".format(key)
        assert n_fragments == 5, "{} is wrong".format(key)

    for key, (correct, n_items) in metrics['tracks'].items():
        assert correct > 0, "{} is wrong".format(key)
        assert correct == n_items, "{} is wrong".format(key)

    for key, (correct, n_items) in metrics['truth_ids'].items():
        if "default" in key:
            correct += 1
        assert correct > 0, "{} is wrong".format(key)
        assert correct == n_items, "{} is wrong".format(key)

    # one camera gap
    tracks = copy.deepcopy(tracks_clean)
    del tracks[0].ids[1]
    del tracks[0].timestamps[1]
    del tracks[0].meta[DETKEY][1]
    scores = validator.validate(tracks, gap)
    metrics = track_statistics(tracks, scores, validator, gap)
    for key, (wrong, n_fragments) in metrics['fragments'].items():
        assert wrong == 0, "{} is wrong".format(key)
        assert n_fragments == 5, "{} is wrong".format(key)

    for key, (correct, n_items) in metrics['tracks'].items():
        if key != "tracks_in_scope":
            correct += 1  # we are missing one detection (separate fragment + track)
        assert correct > 0, "{} is wrong".format(key)
        assert correct == n_items, "{} is wrong".format(key)

    for key, (correct, n_items) in metrics['truth_ids'].items():
        if "default" in key or "truth" in key:
            correct += 1
        assert correct > 0, "{} is wrong".format(key)
        assert correct == n_items, "{} is wrong".format(key)

    # test no scoring on gaps
    tracks = copy.deepcopy(tracks_clean)
    tracks.append(Track(id=9, ids=[tracks[0].ids[0]], timestamps=[tracks[0].timestamps[0]],
                        meta={DETKEY: [validator.truth.get_detection(tracks[0].ids[0])]}))
    # remove the first two items in track 1
    del tracks[0].ids[0]
    del tracks[0].ids[0]
    del tracks[0].timestamps[0]
    del tracks[0].timestamps[0]
    del tracks[0].meta[DETKEY][0]
    del tracks[0].meta[DETKEY][0]
    scores = validator.validate(tracks, gap, gap_l=False, gap_r=False, cam_gap=False)
    metrics = track_statistics(tracks, scores, validator, gap, cam_gap=False)
    for key, (wrong, n_fragments) in metrics['fragments'].items():
        assert wrong == 0, "{} is wrong".format(key)
        assert n_fragments == 6, "{} is wrong".format(key)

    for key, (correct, n_items) in metrics['tracks'].items():
        if key == "track_detections_correct":
            correct += 1  # we removed one detections from track data
        elif key == "tracks_complete":
            correct += 1  # track 1 is not complete
        assert correct > 0, "{} is wrong".format(key)
        assert correct == n_items, "{} is wrong".format(key)

    for key, (correct, n_items) in metrics['truth_ids'].items():
        if "default" in key:
            correct += 1
        assert correct > 0, "{} is wrong".format(key)
        assert correct == n_items, "{} is wrong".format(key)


def make_score_metric(score_dict):
    """Helper to create :obj:`ScoreMetrics` namedtuple."""
    default_dict = {"track_length": 0, "truth_track_length": 0, "adjusted_length": 0,
                    "id_matches": 0, "id_mismatches": 0,
                    "inserts": 0, "deletes": 0, "gap_matches": 0,
                    "gap_left": True, "gap_right": True}
    for key, value in default_dict.items():
        if key not in score_dict:
            score_dict[key] = value
    return ScoreMetrics(**score_dict)
