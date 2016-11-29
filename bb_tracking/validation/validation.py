# -*- coding: utf-8 -*-
"""Provides classes and functions to validate tracks (:obj:`.Track`).

The scores that are calculated and evaluated are defined in :obj:`.Score` and :obj:`.ScoreMetrics`.
"""
# pylint:disable=too-many-arguments,too-many-locals
import math
import numpy as np
import pandas as pd
from ..data import DataWrapperPandas, DataWrapperTracks, Score, ScoreMetrics
from ..data.constants import CAMKEY, FPKEY, FRAMEIDXKEY
from ..tracking.scoring import calc_track_ids, bit_array_to_int_v


class Validator(object):
    """Class to validate :obj:`.Track` objects with truth data.

    It also has some helpers e.g. to check tracks for sanity and remove false positives.
    """
    truth = None
    """:class:`.DataWrapperTruth`: the :class:`.DataWrapperTruth` with truth data"""
    timestamps = None
    """:obj:`list` of timestamps: sorted list with all timestamps in truth"""
    cam_timestamps = None
    """:obj:`dict` of :obj:`list`: sorted lists of timestamps in truth for a cam"""

    def __init__(self, truth_dw):
        """Initialization of class attributes

        Arguments:
            truth_dw (:class:`.DataWrapperTruth`): data wrapper with truth data
        """
        self.truth = truth_dw
        self.timestamps = truth_dw.get_timestamps()
        self.cam_timestamps = {cam: truth_dw.get_timestamps(cam_id=cam)
                               for cam in truth_dw.get_camids()}

    def remove_false_positives(self, tracks):
        """Removes tracks with only false positives.

        Arguments:
            tracks (iterable of :obj:`.Track`): iterable structure with :obj:`.Track`

        Returns:
            iterable of :obj:`.Track`: iterable structure with cleaned :obj:`.Track`
        """
        tracks_clean = []
        for track in tracks:
            if FPKEY in track.meta.keys() and track.meta[FPKEY]:
                continue
            for detection_id in track.ids:
                if self.truth.get_truthid(detection_id) != self.truth.fp_id:
                    tracks_clean.append(track)
                    break
        return tracks_clean

    def sanity_check(self, tracks, cam_gap=True):
        """Performs some sanity checks on tracks.

        The following checks are performed:

            - No duplicate ids in tracks.
            - Only one detection per timestamp
            - Every detection belongs to one track only
            - No unknown detections
            - If `cam_gap` is True a track only has detections from one camera
            - Each track id is unique

        Arguments:
            tracks (:obj:`list` of :obj:`.Track`): iterable structure with :obj:`.Track`.

        Keyword Arguments:
            cam_gap (bool): flag indicating that a camera switch is a insurmountable gap

        Returns:
            tuple: tuple containing:

                - **true_positives** (:obj:`int`): assigned positives
                - **true_negatives** (:obj:`int`): not assigned false positives
                - **false_positives** (:obj:`int`): assigned false positives
                - **false_negatives** (:obj:`int`): not assigned positives
        """
        assigned_ids = set()
        track_ids = set()
        positives, false_positives = self.truth.get_all_detection_ids()
        for track in tracks:
            assert track.id not in track_ids, "Duplicate track id {}.".format(track.id)
            track_ids.add(track.id)
            track_detection_ids = set(track.ids)
            assert len(track_detection_ids) == len(track.ids), \
                "Duplicate ids in track {}.".format(track.id)
            tstamps = set(track.timestamps)
            assert len(tstamps) == len(track_detection_ids), \
                "Duplicate timestamps in track {}.".format(track.id)
            assert all(i < j for i, j in zip(track.timestamps, track.timestamps[1:])),\
                "Timestamps in track {} not in order.".format(track.id)
            duplicates = assigned_ids & track_detection_ids
            assert len(duplicates) == 0, "IDs {} are assigned multiple times.".format(duplicates)
            assigned_ids |= track_detection_ids
            if track_detection_ids <= false_positives:
                track.meta[FPKEY] = True
            if cam_gap:
                assert len(self.truth.get_camids(frame_object=track)) == 1, \
                    "Track {} has detection ids from multiple cameras.".format(track.id)
        unknown_ids = assigned_ids - (positives | false_positives)
        assert len(unknown_ids) == 0, "There are unknown ids in tracks: {}.".format(unknown_ids)
        return (assigned_ids & positives, false_positives - assigned_ids,
                assigned_ids & false_positives, positives - assigned_ids)

    def validate(self, tracks, gap, gap_l=True, gap_r=True, cam_gap=True, val_score_fun=None,
                 val_calc_id_fun=None, check=True):
        """Validates the given :obj:`.Track` with truth data.

        The default `score_fun` is :func:`validation_score_fun_all()`.
        Use it as primer to implement your own.

        Arguments:
            tracks (:obj:`list` of :obj:`.Track`): list of :obj:`.Track` objects to validate
            gap (int): the gap the algorithm should be capable to overcome

        Keyword Arguments:
            gap_l (bool): flag indicating to consider **left** gap in scoring
            gap_r (bool): flag indicating consider **right** gap in scoring
            cam_gap (bool): flag indicating that a camera switch is a insurmountable gap
            val_score_fun (:obj:`func`): function that takes a :obj:`.ScoreMetrics` and calculates
                :obj:`.Score`
            check (bool): flag indicating that sanity check should be performed

        Returns:
            :obj:`dict`: the scores for each :obj:`.Track` as dictionary with :obj:`.Score` objects
            and mapping `{id => score}`
        """
        assert gap >= 0

        def default_calc_id_fun(track):
            """Default implementation to calculate the id of a :obj:`.Track` object."""
            return calc_track_ids([track])[0]
        if val_score_fun is None:
            val_score_fun = validation_score_fun_all
        if val_calc_id_fun is None:
            val_calc_id_fun = default_calc_id_fun

        # just make sure everything is all right
        if check:
            self.sanity_check(tracks, cam_gap=cam_gap)
        # we do not score false positives
        tracks = self.remove_false_positives(tracks)
        truth = self.truth
        scores = dict()
        for track in tracks:
            truth_ids = truth.get_truthids(frame_object=track)
            cam_id = list(truth.get_camids(frame_object=track))[0] if cam_gap else None
            for truth_id in truth_ids:
                # no scoring for false positive tracks
                if truth_id == self.truth.fp_id:
                    continue
                truth_track = self.truth.get_truth_track(truth_id, cam_id=cam_id)
                metrics, _ = self.score_track(truth_track, track,
                                              gap=gap, gap_l=gap_l, gap_r=gap_r, cam_id=cam_id)
                score_value = val_score_fun(metrics, gap)
                if track.id not in scores.keys() or\
                   score_value > scores[track.id].value:
                    scores[track.id] = Score(
                        value=score_value,
                        track_id=track.id,
                        truth_id=truth_id,
                        calc_id=val_calc_id_fun(track),
                        metrics=metrics,
                        alternatives=[])
                elif (track.id in scores.keys() and
                      score_value == scores[track.id].value):
                    scores[track.id].alternatives.append(truth_id)
        return scores

    def score_track(self, track_truth, track_test, gap=0, gap_l=True, gap_r=True, cam_id=None):
        """Scores the equality of two :obj:`.Track` using local alignment methods.

        Arguments:
            track_truth (:obj:`.Track`): the first :obj:`.Track` that is considered the source
            track_test (:obj:`.Track`): the second :obj:`.Track` that is considered the copy

        Keyword Arguments:
            gap (int): the gap the algorithm should be capable to overcome
            gap_l (bool): flag indicating to consider **left** gap in scoring
            gap_r (bool): flag indicating consider **right** gap in scoring
            cam_id (int): limit the scores on one camera

        Returns:
            :obj:`.ScoreMetrics`: :obj:`.ScoreMetrics` tuple with alignment information
        """
        def to_timeformat(time_something):
            """Helper to convert lists to a common time format."""
            return pd.to_datetime(time_something, utc=True)
        assert len(set(track_test.timestamps)) == len(set(track_test.ids)), \
            "You might have duplicate timestamps in the test track."
        assert len(set(track_truth.timestamps)) == len(set(track_truth.ids)), \
            "You might have duplicate timestamps in the truth track."

        timestamps = self.cam_timestamps[cam_id] if cam_id is not None else self.timestamps
        # we use only one timestamps format for comparison
        timestamps = to_timeformat(timestamps)
        timestamps_test = to_timeformat(track_test.timestamps)
        timestamps_truth = to_timeformat(track_truth.timestamps)
        assert timestamps_test[0] >= timestamps[0], "Track is out of scope for ground truth data."
        assert timestamps_test[-1] <= timestamps[-1], "Track is out of scope for ground truth data."

        # calculate start and end positions for truth track with gaps
        gap_l_offset = gap + 1 if gap_l else 0
        gap_r_offset = gap + 1 if gap_r else 0
        truth_track_length = math.fabs(list(timestamps).index(timestamps_truth[-1]) -
                                       list(timestamps).index(timestamps_truth[0])) + 1
        start_idx = list(timestamps).index(timestamps_test[0])
        end_idx = list(timestamps).index(timestamps_test[-1]) + 1
        tstamps = timestamps[
            max(0, start_idx - gap_l_offset):min(len(timestamps), end_idx + gap_r_offset)]

        truth_local = pd.Series(track_truth.ids, index=timestamps_truth, name='truth')

        if tstamps[0] > timestamps_truth[0]:
            truth_local = truth_local[truth_local.index >= tstamps[0]]

        if tstamps[-1] < timestamps_truth[-1]:
            truth_local = truth_local[truth_local.index <= tstamps[-1]]

        # reset gap offset
        start_idx_t = list(timestamps).index(truth_local.index[0])
        end_idx_t = list(timestamps).index(truth_local.index[-1]) + 1
        tstamps = timestamps[
            max(0, start_idx - gap_l_offset, min(start_idx, start_idx_t - gap_l_offset)):
            min(len(timestamps), end_idx + gap_r_offset, max(end_idx, end_idx_t + gap_r_offset))]

        # combine truth and test data in one dataframe
        data = pd.concat([pd.Series(tstamps, index=tstamps, name='timestamps'),
                          truth_local,
                          pd.Series(track_test.ids, index=timestamps_test, name='test')],
                         axis=1)

        data['gap_truth'] = data.truth.isnull()
        data['gap_test'] = data.test.isnull()
        data['gap_matches'] = data.gap_truth & data.gap_test
        data['id_matches'] = data.truth == data.test
        data['inserts'] = (np.logical_not(data.id_matches) & data.gap_truth &
                           np.logical_not(data.gap_test))
        data['deletes'] = (np.logical_not(data.id_matches) & data.gap_test &
                           np.logical_not(data.gap_truth))
        data['id_mismatches'] = np.logical_not(data.id_matches | data.gap_test | data.gap_truth)
        gap_left_found = np.all(data.gap_matches.iloc[:gap_l_offset] |
                                data.id_matches.iloc[:gap_l_offset]) if gap_l else True
        gap_right_found = np.all(data.gap_matches.iloc[-gap_r_offset:] |
                                 data.id_matches.iloc[-gap_r_offset:]) if gap_r else True

        return ScoreMetrics(track_length=end_idx - start_idx,
                            truth_track_length=truth_track_length,
                            adjusted_length=data.timestamps.shape[0],
                            id_matches=data.id_matches.sum(),
                            id_mismatches=data.id_mismatches.sum(),
                            inserts=data.inserts.sum(),
                            deletes=data.deletes.sum(),
                            gap_matches=data.gap_matches.sum(),
                            gap_left=gap_left_found,
                            gap_right=gap_right_found), data


def validation_score_fun_all(metrics, gap=0):
    """Scoring function that considers all ids and gaps and calculates the percentage of matches.

    Note:
        Used as default implementation to calculate :attr:`.Score.value`.

    Arguments:
        metrics (:obj:`.ScoreMetrics`): :obj:`.ScoreMetrics` to calculate accumulated score

    Keyword Arguments:
        gap (int): the gap the algorithm should be capable to overcome

    Returns:
        float: value between 0 (bad) and 1 (good) indicating how good the matching is considered
    """
    assert gap >= 0
    if metrics.adjusted_length is 0:
        return 0
    return float(metrics.id_matches + metrics.gap_matches) / metrics.adjusted_length


def convert_validated_to_pandas(validated_results):
    """Converter to convert the result of a :class:`.Validator` to a Pandas DataFrame.

    Arguments:
        validated_results (:obj:`dict` of :obj:`.Score`): a dictionary with Score tuples as values

    Returns:
        :obj:`pd.DataFrame`: a Pandas Dataframe with the score information as columns
    """
    metric_key = "metrics"
    scores = []
    for score in validated_results.values():
        row = {key: getattr(score, key) for key in Score._fields if key != metric_key}
        row.update({key: getattr(getattr(score, metric_key), key) for key in ScoreMetrics._fields})
        scores.append(row)
    return pd.DataFrame(scores)


def calc_fragments(dw_truth, gap, cam_gap=True):
    """Calculates some information about the fragments in the truth data.

    Arguments:
        dw_truth (:obj:`.DataWrapperTruth`): :obj:`.DataWrapperTruth` object
        gap (int): the gap the algorithm should be capable to overcome

    Keyword Arguments:
        cam_gap (bool): flag indicating that a camera switch is a insurmountable gap

    Returns:
        tuple: tuple containing

            - **fragment_counter** (:obj:`int`): number of fragments
            - **track_no_gaps_counter** (:obj:`int`): number of tracks without gaps
            - **fragment_ids** (:obj:`int`): list with number of ids per fragments
            - **fragment_lenghts** (:obj:`int`): list with length of fragments
    """
    fragment_counter = 0
    track_no_gaps_counter = 0
    fragment_ids = list()
    fragment_lengths = list()
    if isinstance(dw_truth, DataWrapperTracks):
        dw_truth = dw_truth.data

    if isinstance(dw_truth, DataWrapperPandas):
        tcol = dw_truth.cols['truthId']
        detections = dw_truth.detections
        cam_groups = detections.groupby(dw_truth.cols[CAMKEY]) if cam_gap else ((-1, detections), )
    else:
        cam_groups = dw_truth.cam_tracks.items() if cam_gap else ((-1, dw_truth.tracks), )
    for _, camg in cam_groups:
        tracks = camg.groupby(tcol) if isinstance(dw_truth, DataWrapperPandas) else camg.items()
        for truth_id, track in tracks:
            if truth_id == dw_truth.fp_id:
                continue
            fragment_counter += 1
            if isinstance(dw_truth, DataWrapperPandas):
                track = track.sort_values(dw_truth.cols['timestamp'], axis=0)
                frame_idx = track[dw_truth.cols['frameIdx']].values
            else:
                frame_idx = np.array(track.meta[FRAMEIDXKEY])

            if len(frame_idx) < 2:
                track_no_gaps_counter += 1
                fragment_ids.append(len(frame_idx))
                fragment_lengths.append(len(frame_idx))
                continue

            fragment_counter_before = fragment_counter

            frame_diffs = frame_idx[1:] - frame_idx[:-1]
            frame_diffs = np.insert(frame_diffs, 0, 0)  # re-add first element
            frame_diffs_bool = frame_diffs > gap + 1
            fragment_counter += np.sum(frame_diffs_bool)
            if (fragment_counter - fragment_counter_before) == 0:
                track_no_gaps_counter += 1

            for fragment in np.split(frame_diffs, frame_diffs_bool.nonzero()[0]):
                fragment_ids.append(fragment.shape[0])
                fragment_lengths.append(np.sum(fragment[1:]) + 1)

    assert fragment_counter == len(fragment_ids)
    assert fragment_counter == len(fragment_lengths)
    return fragment_counter, track_no_gaps_counter, fragment_ids, fragment_lengths


def track_statistics(tracks, scores, validator, gap, cam_gap=True):
    """Calculates some statistics about tracks for validation.

    Arguments:
        tracks (:obj:`list` of :obj:`.Track`): iterable with tracks to analyze
        scores (:obj:`dict` or :obj:`pd.Dataframe`): result of :func:`Validator.validate()`
        validator (:class:`.Validator`): Validator class to calculate track statistics
        gap (int): the gap the algorithm should be capable to overcome

    Keyword Arguments:
        cam_gap (bool): flag indicating that a camera switch is a insurmountable gap

    Returns:
        :obj:`dict`: some metrics about tracks
    """
    assert len(tracks) > 0
    truth = validator.truth
    if isinstance(scores, dict):
        scores = convert_validated_to_pandas(scores)
    fragment_c, track_no_gaps_c, fragment_ids, _ = calc_fragments(truth, cam_gap=cam_gap, gap=gap)
    positives, negatives, f_positives, f_negatives = validator.sanity_check(tracks, cam_gap=cam_gap)
    if cam_gap:
        # we are considering tracks with the same truth id on different cameras as separate tracks
        n_tracks = np.sum([len(truth.get_truthids(cam_id=cid) - set([truth.fp_id]))
                           for cid in truth.get_camids()])
    else:
        n_tracks = len(truth.get_truthids() - set([truth.fp_id]))
    n_track_detections = np.sum(fragment_ids)
    assert n_track_detections == len(positives) + len(f_negatives)
    assert len(tracks) == scores.shape[0]
    n_tracks_test = len(tracks)

    # Handles the case that gaps before and after a fragment are not analyzed. It is not possible
    # to compare to real fragments, so we use the number of calculated fragments as reference.
    if scores.shape[0] > track_no_gaps_c and np.all(scores.gap_left) and np.all(scores.gap_right):
        fragment_c = scores.shape[0]

    # calculate how many truth fragments do have converging ids
    truth_calc_ids, truth_ids, truth_lengths = [], [], []
    if cam_gap:
        truth_tracks = []
        for cam_id in truth.get_camids():
            truth_tracks.extend([track for track in truth.get_truth_tracks(cam_id=cam_id)
                                 if track.id != truth.fp_id])
    else:
        truth_tracks = [track for track in truth.get_truth_tracks() if track.id != truth.fp_id]

    truth_calc_ids = calc_track_ids(truth_tracks)
    truth_ids = np.array([track.id for track in truth_tracks])
    truth_lengths = np.array([len(track.ids) for track in truth_tracks])

    matching_ids = scores.truth_id == scores.calc_id
    matching_ids_truth = truth_ids == truth_calc_ids

    # calculate how many detections have the correct id from start
    truth_positives, _ = truth.get_all_detection_ids()
    detections = truth.get_detections(truth_positives)
    detection_truth_ids = np.array([truth.get_truthid(det) for det in detections])
    detection_ids = bit_array_to_int_v(detections)

    metrics_dict = {
        "detections": {
            "detections_id_mismatches": (np.sum(scores.id_mismatches), n_track_detections),
            "detections_deletes": (np.sum(scores.deletes), n_track_detections),
            "detections_inserts": (np.sum(scores.inserts), n_track_detections),
        },
        "fragments": {
            "fragments_missed_gap_left": (np.logical_not(scores.gap_left).sum(), n_tracks_test),
            "fragments_missed_gap_right": (np.logical_not(scores.gap_right).sum(), n_tracks_test),
            "fragments_deletes": (np.sum(scores.deletes > 0), n_tracks_test),
            "fragments_inserts": (np.sum(scores.inserts > 0), n_tracks_test),
            "fragments_id_mismatches": (np.sum(scores.id_mismatches > 0), n_tracks_test),
        },
        "tracks": {
            "track_detections_correct": (scores.id_matches.sum(), n_track_detections),
            "track_fragments_complete": (scores[(scores.value >= 1) &
                                                ((scores.id_matches + scores.gap_matches) ==
                                                 scores.adjusted_length)].shape[0], fragment_c),
            "tracks_in_scope": (track_no_gaps_c, n_tracks),
            "tracks_complete": (scores[(scores.value >= 1) &
                                       ((scores.id_matches + scores.gap_matches) ==
                                        scores.adjusted_length) &
                                       (scores.track_length ==
                                        scores.truth_track_length)].shape[0], n_tracks)
        },
        "confusion_matrix": {
            "true_positives": len(positives),
            "true_negatives": len(negatives),
            "false_positives": len(f_positives),
            "false_negatives": len(f_negatives),
        },
        "truth_ids": {
            "detection_ids_correct_default": (np.sum(detection_ids == detection_truth_ids),
                                              len(detection_ids)),
            "detection_ids_correct_truth": (np.sum(truth_lengths[matching_ids_truth]),
                                            np.sum(truth_lengths)),
            "detection_ids_correct_tracking": (np.sum(scores[matching_ids].id_matches +
                                                      scores[matching_ids].id_mismatches),
                                               np.sum(scores.id_matches + scores.id_mismatches)),
            "fragment_ids_correct_tracking": (np.sum(matching_ids), scores.shape[0]),
            "track_ids_correct_truth": (np.sum(matching_ids_truth), len(truth_ids)),
        },
    }
    return metrics_dict
