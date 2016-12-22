# -*- coding: utf-8 -*-
"""Defines some functions that compare features and score them.

Deprecated scoring functions are not imported to the parent module!

Note:
    Functions with suffix **_v** are vectorized with numpy.
"""
from itertools import chain
from scipy.spatial import distance
import math
import numpy as np
from scipy.spatial.distance import cityblock, hamming
from bb_binary import int_id_to_binary
from bb_tracking.data import Detection
from bb_tracking.data.constants import DETKEY


def score_ids_best_fit(ids1, ids2, length=12):
    """Compares two lists of ids by choosing the pair with the best score.

    .. deprecated:: September 2016
        This scoring is used with data from the *old* Computer Vision Pipeline.

    Arguments:
        ids1 (:obj:`list` of int): Iterable with ids as integers as base ids
        ids2 (:obj:`list` of int): Iterable with ids as integers to compare base to

    Keyword Arguments:
        length (Optional int): number of bits in the bit array used to compare ids

    Returns:
        float: Uses Hamming distance of best matching pair
    """
    assert len(ids1) > 0
    assert len(ids2) > 0
    ids1, ids2 = set(ids1), set(ids2)

    # best case: we already have the same ids in both sets
    if ids1 & ids2:
        return 0

    best_score = float("inf")
    for id1 in ids1:
        id1 = int_id_to_binary(id1, nb_bits=length)
        for id2 in ids2:
            id2 = int_id_to_binary(id2, nb_bits=length)
            test_score = hamming(id1, id2)
            if test_score < best_score:
                best_score = test_score
    return best_score


def score_ids_best_fit_rotating(ids1, ids2, rotation_penalty=1, length=12):
    """Compares two lists of ids by choosing the pair best score by rotating the bits.

    .. deprecated:: September 2016
        This scoring is used with data from the *old* Computer Vision Pipeline.

    Arguments:
        ids1 (:obj:`list` of int): Iterable with ids as integers as base ids
        ids2 (:obj:`list` of int): Iterable with ids as integers to compare base to

    Keyword Arguments:
        rotation_penalty (Optional float): the penalty that is added for a rotation
            of 1 to the left or right
        length (Optional int): number of bits in the bit array used to compare ids

    Returns:
        float: Uses Hamming distance of best matching (rotated) pair
    """
    assert len(ids1) > 0
    assert len(ids2) > 0

    # best case: we already have the same ids in both sets
    if set(ids1) & set(ids2):
        return 0

    rotation_penalty = float(rotation_penalty) / length
    best_score = float("inf")
    for id1 in ids1:
        id1 = int_id_to_binary(id1, nb_bits=length)
        for id2 in ids2:
            id2 = int_id_to_binary(id2, nb_bits=length)
            # rotate left and right
            for direction in [-1, 1]:
                for i in range(int(math.ceil(float(length / 2))) + 1):
                    if i * rotation_penalty >= best_score:
                        break
                    test_score = hamming(id1, np.roll(id2, direction * i)) + i * rotation_penalty
                    if test_score < best_score:
                        best_score = test_score
                        if best_score <= 0:
                            return best_score

    return best_score


def score_ids_and_orientation(id_orientation_tuple1, id_orientation_tuple2, length=12,
                              range_bonus_orientation=30):
    """Compares lists of ids by choosing the pair with the best score and considering orientation.

    The bonus is equal to the negative score of one non matching bit.

    .. deprecated:: September 2016
        This scoring is used with data from the *old* Computer Vision Pipeline.

    Arguments:
        id_orientation_tuple1 (tuple): (Iterable with ids, Iterable with orientations)
        id_orientation_tuple2 (tuple): (Iterable with ids, Iterable with orientations)

    Keyword Arguments:
        length (int): number of bits in the bit array used to compare ids
        range_bonus_orientation (float): range in degrees, so that two orientations get a bonus

    Returns:
        float: Uses Hamming distance of best matching pair with bonus for same orientation
    """
    ids1, orientations1 = id_orientation_tuple1
    ids2, orientations2 = id_orientation_tuple2
    assert len(ids1) > 0
    assert len(ids2) > 0
    assert len(ids1) == len(orientations1)
    assert len(ids2) == len(orientations2)

    # best case: we already have the same ids in both sets
    if set(ids1) & set(ids2):
        return 0

    best_score = float("inf")
    for id1, or1 in zip(ids1, orientations1):
        id1 = int_id_to_binary(id1, nb_bits=length)
        for id2, or2 in zip(ids2, orientations2):
            id2 = int_id_to_binary(id2, nb_bits=length)
            # bonus for orientation
            orientation_score = 0.0
            if distance_orientations(or1, or2) <= range_bonus_orientation:
                orientation_score = -1. / length
            test_score = hamming(id1, id2) + orientation_score
            if test_score < best_score:
                best_score = test_score
                if best_score <= 0:
                    return best_score
    return best_score


def score_id_sim(id1, id2):
    """Compares two id frequency distributions for similarity

    Arguments:
        id1 (:obj:`list` of float): id bit frequency distribution of base id
        id2 (:obj:`list` of float): id bit frequency distribution of id to compare base to

    Returns:
        float: Use Manhattan distance :math:`\\sum_i |id1_i - id2_i|`
    """
    assert len(id1) == len(id2)
    return cityblock(id1, id2)


def score_id_sim_v(detections1, detections2):
    """Compares two id frequency distributions for similarity (vectorized)

    Arguments:
        detections1 (:obj:`list` of :obj:`.Detection`): Iterable with Detections
        detections2 (:obj:`list` of :obj:`.Detection`): Iterable with Detections

    Returns:
        :obj:`np.array`: Use Manhattan distance :math:`\\sum_i |id1_i - id2_i|`
    """
    assert len(detections1) == len(detections2), "Detection lists do not have the same length."
    arr1 = np.array([det.beeId for det in detections1])
    arr2 = np.array([det.beeId for det in detections2])
    assert np.all(arr1.shape == arr2.shape), "Detections do not have the same length of id bits."
    return np.sum(np.fabs(arr1 - arr2), axis=1)


def score_id_sim_orientation(id1, or1, id2, or2, range_bonus_orientation=30,
                             value_bonus_orientation=1.):
    """Compares two id frequency distributions for similarity

    Arguments:
        id1 (:obj:`list` of float): id bit frequency distribution of base id
        or1 (float): orientation belonging to id1
        id2 (:obj:`list` of float): id bit frequency distribution of id to compare base to
        or2 (float): orientation belonging to id2

    Keyword Arguments:
        range_bonus_orientation (Optional float): range in degrees, so that two orientations
            get a bonus
        value_bonus_orientation (Optional float): value to add if orientations are within
            `range_bonus_orientation`

    Returns:
        float: Use Manhattan distance :math:`\\sum_i |id1_i - id2_i|`
    """
    # pylint:disable=too-many-arguments
    assert len(id1) == len(id2)
    # bonus for orientation
    orientation_score = 0.0
    if distance_orientations(or1, or2) <= range_bonus_orientation:
        orientation_score = float(value_bonus_orientation) / len(id1)
    return max(0, cityblock(id1, id2) - orientation_score)


def score_id_sim_orientation_v(detections1, detections2, range_bonus_orientation=(math.pi / 6),
                               value_bonus_orientation=1.):
    """Compares id frequency distributions for similarity (vectorized)

    Arguments:
        detections1 (:obj:`list` of :obj:`.Detection`): Iterable with `.Detection`
        detections2 (:obj:`list` of :obj:`.Detection`): Iterable with `.Detection`

    Keyword Arguments:
        range_bonus_orientation (Optional float): range in degrees, so that two orientations
            get a bonus
        value_bonus_orientation (Optional float): value to add if orientations are within
            `range_bonus_orientation`

    Returns:
        :obj:`np.array`: Use Manhattan distance :math:`\\sum_i |id1_i - id2_i|`
    """
    assert len(detections1) == len(detections2), "Detection lists do not have the same length."
    # Note: use *-operator instead of chaining when supporting python3 only!
    arr1 = np.array([tuple(chain((det.orientation, ), det.beeId)) for det in detections1])
    arr2 = np.array([tuple(chain((det.orientation, ), det.beeId)) for det in detections2])
    assert np.all(arr1.shape == arr2.shape), "Detections do not have the same length of id bits."
    score_orientations = (np.sum(np.fabs(arr1[:, 1:] - arr2[:, 1:]), axis=1) -
                          ((np.fabs(arr1[:, 0] - arr2[:, 0]) <= range_bonus_orientation) *
                           float(value_bonus_orientation) / (arr1.shape[1] - 1)))
    score_orientations[score_orientations < 0] = 0
    return score_orientations


def score_id_sim_rotating(id1, id2, rotation_penalty=0.5):
    """Compares two id frequency distributions for similarity by rotating them

    Instead of only using the distance metric, one id is rotated to check for better results.

    Arguments:
        id1 (:obj:`list`): id bit frequency distribution of base id
        id2 (:obj:`list`): id bit frequency distribution of id to compare base to

    Keyword Arguments:
        rotation_penalty (float): the penalty that is added for a rotation of 1 to the left or right

    Returns:
        float: Manhattan distance but also rotated with added rotation_penalty.
    """
    n = float(len(id2))
    assert len(id1) == n
    best_score = float("inf")

    # rotate left and right
    for direction in [-1, 1]:
        for i in range(int(math.ceil(n / 2)) + 1):
            if i * rotation_penalty >= best_score:
                break
            test_score = cityblock(id1, np.roll(id2, direction * i)) + i * rotation_penalty
            if test_score < best_score:
                best_score = test_score
                if best_score <= 0:
                    return best_score

    return best_score


def score_id_sim_rotating_v(detections1, detections2, rotation_penalty=0.5):
    """Compares id frequency distributions for similarity by rotating them (vectorized)

    Instead of only using the distance metric, one id is rotated to check for better results.

    Note:
        The calculation of this feature is quite expensiv compared to :func:`score_id_sim_v`!

    Arguments:
        detections1 (:obj:`list` of :obj:`.Detection`): Iterable with `.Detection`
        detections2 (:obj:`list` of :obj:`.Detection`): Iterable with `.Detection`

    Keyword Arguments:
        rotation_penalty (Optional float): the penalty that is added for a rotation of 1
            to the left or right

    Returns:
        :obj:`np.array`: Manhattan distance but also rotated with added rotation_penalty.
    """
    assert len(detections1) == len(detections2), "Detection lists do not have the same length."
    arr1 = np.array([det.beeId for det in detections1])
    arr2 = np.array([det.beeId for det in detections2])
    assert np.all(arr1.shape == arr2.shape), "Detections do not have the same length of id bits."

    n = float(arr1.shape[0])
    score = np.full(arr1.shape[0], np.inf)
    # rotate left and right
    for direction in [-1, 1]:
        for i in range(int(math.ceil(n / 2)) + 1):
            test_score = np.sum(np.fabs(arr1 - np.roll(arr2, direction * i, axis=1)), axis=1)
            score = np.minimum(score, test_score + i * rotation_penalty)
    return score


def score_id_sim_tracks_median_v(tracks1, tracks2):
    """Compares id frequency distributions of tracks by comparing the median (vectorized)

    Arguments:
        tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
        tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

    Returns:
        :obj:`np.array`: Use Manhattan distance :math:`\\sum_i |Median(ids1)_i - Median(ids2)_i|`
    """
    assert len(tracks1) == len(tracks2), "Track lists do not have the same length."

    arr1 = calc_median_ids(tracks1)
    arr2 = calc_median_ids(tracks2)
    assert np.all(arr1.shape == arr2.shape), "Detections do not have the same length of id bits."
    return np.sum(np.fabs(arr1 - arr2), axis=1)


def distance_orientations(rad1, rad2):
    """Calculates the distance between two orientations.

    Orientations in rad are converted to degrees then the difference is calculated.

    Arguments:
        rad1 (float): orientation of first detection in rad
        rad2 (float): orientation of second detection in rad

    Returns:
        float: distance between two orientation in degrees, e.g. 90 for -pi/2 and pi
    """
    distance = math.fabs(math.degrees(rad1) - math.degrees(rad2))
    if distance > 180:
        distance = 360 - distance
    return distance


def distance_orientations_v(detections1, detections2, meta_key=None):
    """Calculates the distances between orientations (vectorized)

    Orientations are expected to be in rad.

    Arguments:
        detections1 (:obj:`list` of :obj:`.Detection`): Iterable with `.Detection`
        detections2 (:obj:`list` of :obj:`.Detection`): Iterable with `.Detection`

    Keyword Arguments:
        meta_key (Optional str): Instead of :attr:`.Detection.orientation` use value from
            :attr:`.Detection.meta`

    Returns:
        :obj:`np.array`: distance between orientations of detections line by line in rad
    """
    if meta_key is None:
        arr1 = np.array([det.orientation for det in detections1])
        arr2 = np.array([det.orientation for det in detections2])
    else:
        arr1 = np.array([det.meta[meta_key] for det in detections1])
        arr2 = np.array([det.meta[meta_key] for det in detections2])

    distance = np.fabs(arr1 - arr2)
    mask = distance > math.pi
    distance[mask] = 2 * math.pi - distance[mask]
    return distance


def distance_positions_v(detections1, detections2):
    """Calculates the euclidean distances between the x and y positions (vectorized)

    Arguments:
        detections1 (:obj:`list` of :obj:`.Detection`): Iterable with `.Detection`
        detections2 (:obj:`list` of :obj:`.Detection`): Iterable with `.Detection`

    Returns:
        :obj:`np.array`: Euclidean distance between detections line by line
    """
    arr1 = np.array([(det.x, det.y) for det in detections1])
    arr2 = np.array([(det.x, det.y) for det in detections2])
    return np.linalg.norm(arr1 - arr2, axis=1)


def bit_array_to_int_v(bit_arrays, threshold=0.5, endian='little'):
    """Converts the bit frequency distribution of the id to an integer representation.

    Note:
        Instead of bit_arrays you could also pass lists with :obj:`.Detection` objects.
        In this case the :attr:`.Detection.beeId` is used and interpreted as bit array.

    Arguments:
        bit_arrays (:obj:`list` of arrays or :obj:`.Detection`): Iterable with ids to decode.

    Keyword Arguments:
        threshold (Optional float): ``values >= threshold`` are interpreted as 1
        endian (Optional str): Either `little` for little endianess or `big` for big endianess.

    Returns:
        :obj:`list` of int: the decoded ids represented as integer
    """
    if isinstance(bit_arrays[0], Detection):
        bit_arrays = [det.beeId for det in bit_arrays]
    if endian == 'little':
        bit_arrays = [arr[::-1] for arr in bit_arrays]
    assert len(bit_arrays[0]) == 12, "Only implemented for 12 bit representation."
    arr = np.packbits(np.array(bit_arrays) >= threshold, axis=1)
    arr = arr.astype(np.int16, copy=False)
    return np.left_shift(arr[:, 0], 4) | np.right_shift(arr[:, 1], 4)


def calc_median_ids(tracks):
    """Helper to calculate the median bit for all the ids in the given track.

    Note:
        For performance reasons the median id is saved as meta key and only is recalculated
        if the length of the track changes.

    Arguments:
        tracks(:obj:`list` of :obj:`.Track`): Iterable with Tracks

    Returns:
        :obj:`np.array`: median for all the bits in the given track
    """
    meta_key = 'median_id'
    ids_median = []
    for track in tracks:
        if meta_key in track.meta.keys() and track.meta[meta_key][0] == len(track.ids):
            ids_median.append(track.meta[meta_key][1])
        else:
            track_median = np.median([det.beeId for det in track.meta[DETKEY]], axis=0)
            ids_median.append(track_median)
            track.meta[meta_key] = (len(track.ids), track_median)
    return np.array(ids_median)


def calc_track_ids(tracks):
    """Function to calculate an id for a :obj:`.Track`.

    This functions calculates the median for each bit of the detections in the track and then uses a
    threshold to decide whether a bit is set or not.

    Note:
        Used as default implementation to calculate :attr:`.Score.calc_id`.

    Arguments:
        tracks (:obj:`list` of :obj:`.Track`): A list of :obj:`.Track` object to calculate
            an id based on it's list of :obj:`Detection` objects.

    Returns:
        int: the calculated id for the :obj:`.Track`
    """
    return bit_array_to_int_v(calc_median_ids(tracks))


def confidence_id_sim_v(tracks1, tracks2):
    """Compares id confidence of tracks (version 1)

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :obj:`np.array`: difference between mean confidence in the tracks
        """
    arr1conf, arr2conf = [], []
    for track1, track2 in zip(tracks1, tracks2):
        arr1 = np.array([det.beeId for det in track1.meta[DETKEY]])
        arr2 = np.array([det.beeId for det in track2.meta[DETKEY]])

        arr1conf.append(np.mean(np.min(np.abs(0.5 - arr1) * 2, axis=1)))
        arr2conf.append(np.mean(np.min(np.abs(0.5 - arr2) * 2, axis=1)))

    return np.fabs(np.array(arr1conf) - np.array(arr2conf))


def confidence_id_sim_v2(tracks1, tracks2):
    """Compares id confidency of tracks (version 2)

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :obj:`np.array`: difference between mean confidence in the tracks
        """
    scores = []
    for track1, track2 in zip(tracks1, tracks2):
        arr1 = np.array([det.beeId for det in track1.meta[DETKEY]])
        arr2 = np.array([det.beeId for det in track2.meta[DETKEY]])

        scores.append(math.fabs((np.mean(np.min(np.abs(0.5 - arr1) * 2, axis=1))) -
                                (np.mean(np.min(np.abs(0.5 - arr2) * 2, axis=1)))))

    return scores


def short_confidence_id_sim_v(tracks1, tracks2):
    """Compares id confidence of last detections in tracks1 and first detections in tracks2

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :obj:`np.array`: difference between confidence values of the detections
        """
    scores = []
    for track1, track2 in zip(tracks1, tracks2):
        length = min(2, len(track1.timestamps), len(track2.timestamps))
        arr1 = np.array([det.beeId for det in track1.meta[DETKEY][-length:]])
        arr2 = np.array([det.beeId for det in track2.meta[DETKEY][:length]])
        scores.append(math.fabs((np.mean(np.min(np.abs(0.5 - arr1) * 2, axis=1))) -
                                (np.mean(np.min(np.abs(0.5 - arr2) * 2, axis=1)))))
    return scores


def calculate_speed(track):
    """Helper to calculate the average speed of a given track.

        Arguments:
            track(:obj:`.Track`): A given Tracks

        Returns:
            float: total distance between all detections per total time of the track
    """
    start = track.meta[DETKEY][0].timestamp
    end = track.meta[DETKEY][-1].timestamp

    length = (end - start)
    assert length > 0, "Start and end time are the same."

    # we start with the first detection
    startpoint = (track.meta[DETKEY][0].x, track.meta[DETKEY][0].y)

    # sum up all the distances
    eucl = 0

    for det in track.meta[DETKEY][1:]:
        currentpoint = (det.x, det.y)
        eucl += distance.euclidean(startpoint, currentpoint)
        startpoint = currentpoint

    return eucl / length


def speed_diff(tracks1, tracks2):
    """Compares the average speed of the tracks

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :(:obj:`list` of float: difference between speed
    """
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        # If one of the tracks consists only of one detection, we have a point
        # A point does not have a speed, so we append a non-existing value
        if len(track1.timestamps) < 2 or len(track2.timestamps) < 2:
            scores.append(-1.0)
        else:
            speed1 = calculate_speed(track1)
            assert np.all(np.isfinite(speed1)), "In speed of track 1, a division by zero has " \
                                                "occured"

            speed2 = calculate_speed(track2)
            assert np.all(np.isfinite(speed2)), "In speed of track 2, a division by zero has " \
                                                "occured"

            scores.append(math.fabs(speed1 - speed2))
    return scores


def x_vectors(tracks1, tracks2):
    """Compares the last move from track 1 with the first one from track 2 on the x-axis

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :obj:`list` of float: absolute difference between x-movements
    """
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        # If one of the tracks consists only of one detection, we have a point
        # Comparing a point and a vector is useless, so we append a non-existing value
        if len(track1.timestamps) < 2 or len(track2.timestamps) < 2:
            scores.append(-1.0)
        else:
            det1_front = track1.meta[DETKEY][-2]
            det1_back = track1.meta[DETKEY][-1]
            det2_front = track2.meta[DETKEY][0]
            det2_back = track2.meta[DETKEY][1]

            x_vector1 = det1_back.x - det1_front.x
            x_vector2 = det2_back.x - det2_front.x

            # runterbrechen auf die Zeitänderung
            time_diff1 = (det1_back.timestamp - det1_front.timestamp)
            time_diff2 = (det2_back.timestamp - det2_front.timestamp)

            assert time_diff1 > 0, "Start and end time are the same."
            assert time_diff2 > 0, "Start and end time are the same."

            vector1 = x_vector1 / time_diff1
            vector2 = x_vector2 / time_diff2

            assert np.all(np.isfinite(vector1))
            assert np.all(np.isfinite(vector2))

            diff = math.fabs(vector1 - vector2)
            scores.append(diff)

    return scores


def y_vectors(tracks1, tracks2):
    """Compares the last move from track 1 with the first one from track 2 on the y-axis

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :obj:`list` of float: absolute difference between y-movements
    """
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        # If one of the tracks consists only of one detection, we have a point
        # Comparing a point and a vector is useless, so we append a non-existing value
        if (len(track1.timestamps) < 2 or len(track2.timestamps) < 2):
            scores.append(-1.0)
        else:

            det1_front = track1.meta[DETKEY][-2]
            det1_back = track1.meta[DETKEY][-1]
            det2_front = track2.meta[DETKEY][0]
            det2_back = track2.meta[DETKEY][1]

            y_vector1 = det1_back.y - det1_front.y
            y_vector2 = det2_back.y - det2_front.y

            time_diff1 = (det1_back.timestamp - det1_front.timestamp)
            time_diff2 = (det2_back.timestamp - det2_front.timestamp)

            # make sure we won't divide by zero
            assert time_diff1 > 0
            assert time_diff2 > 0

            vector1 = y_vector1 / time_diff1
            vector2 = y_vector2 / time_diff2

            assert np.all(np.isfinite(vector1))
            assert np.all(np.isfinite(vector2))

            diff = math.fabs(vector1 - vector2)
            scores.append(diff)

    return scores


def xy_vectors(tracks1, tracks2):
    """Compares the last move from track 1 with the first one from track 2 on the both axis

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :obj:`list` of float: absolute difference between both vector length
    """
    scores = []

    for track1, track2 in zip(tracks1, tracks2):

        # If one of the tracks consists only of one detection, we have a point
        # Comparing a point and a vector is useless, so we append a non-existing value
        if (len(track1.timestamps) < 2 or len(track2.timestamps) < 2):
            scores.append(-1.0)
        else:

            det1_front = track1.meta[DETKEY][-2]
            det1_back = track1.meta[DETKEY][-1]
            det2_front = track2.meta[DETKEY][0]
            det2_back = track2.meta[DETKEY][1]

            time_diff1 = (det1_back.timestamp - det1_front.timestamp)
            time_diff2 = (det2_back.timestamp - det2_front.timestamp)

            # make sure we won't divide by zero
            assert time_diff1 > 0
            assert time_diff2 > 0

            # Berechne die Vektoren zwischen den Detektionen
            vector1 = np.asarray([det1_back.x - det1_front.x, det1_back.y - det1_front.y])
            vector2 = np.asarray([det2_back.x - det2_front.x, det2_back.y - det2_front.y])

            normalized_vector1 = vector1 / time_diff1
            normalized_vector2 = vector2 / time_diff2

            # Betrachte den Unterschied der Vektorlängen
            length_diff = np.linalg.norm(normalized_vector1 - normalized_vector2)
            scores.append(length_diff)

    return scores


def gap_speed(tracks1, tracks2):
    """Combines the features time_diff and distance by normalizing the distance between the last
    detection of the first track and the first detection of the second track with the time that
    collapsed between these detections

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :obj:`list` of float: distance per time
    """
    scores = []
    for track1, track2 in zip(tracks1, tracks2):
        det1 = track1.meta[DETKEY][-1]
        det2 = track2.meta[DETKEY][0]

        time_diff = (det2.timestamp - det1.timestamp)

        # make sure we won't divide by zero
        assert time_diff > 0, "The detections come from the same frame."

        distance = math.sqrt((det2.x - det1.x) ** 2 + (det2.y - det1.y) ** 2)
        normalized_distance = distance / time_diff

        assert np.all(np.isfinite(normalized_distance))

        scores.append(normalized_distance)

    return scores


def movement_area(tracks1, tracks2):
    """Calculates the movement area for both tracks (movement area = the rectangle where a track
    takes place, calculated by the difference of the highest and lowest x and y position of any
    Detection in the Track). Then the area is normalized by the time that collapses during the
    track and finally the values for both tracks are subtracted.

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :obj:`list` of float: difference between movement areas of the tracks
    """
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        # It is not useful to compare the areas if one of the track only consists of one detection
        if len(track1.timestamps) < 2 or len(track2.timestamps) < 2:
            scores.append(-1.0)
        else:

            points1 = np.asarray([[det.x, det.y] for det in track1.meta[DETKEY]])
            points2 = np.asarray([[det.x, det.y] for det in track2.meta[DETKEY]])

            minx = (np.min(points1, axis=0))[0]
            maxx = (np.max(points1, axis=0))[0]
            miny = (np.min(points1, axis=0))[1]
            maxy = (np.max(points1, axis=0))[1]

            minx2 = (np.min(points2, axis=0))[0]
            maxx2 = (np.max(points2, axis=0))[0]
            miny2 = (np.min(points2, axis=0))[1]
            maxy2 = (np.max(points2, axis=0))[1]

            square1 = (maxx - minx) * (maxy - miny)
            square2 = (maxx2 - minx2) * (maxy2 - miny2)

            # Time collapsed from beginning of the track until the end
            time_diff1 = (track1.meta[DETKEY][-1].timestamp - track1.meta[DETKEY][0].timestamp)
            time_diff2 = (track2.meta[DETKEY][-1].timestamp - track2.meta[DETKEY][0].timestamp)

            # make sure we won't divide by zero
            assert time_diff1 > 0
            assert time_diff2 > 0

            result1 = square1 / time_diff1
            result2 = square2 / time_diff2

            # Append the difference between the areas
            scores.append(math.fabs(result1 - result2))

    return scores


def len_dif(tracks1, tracks2):
    """Compares the length of track1 and track2

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :int: absolute length difference between both tracks
    """
    scores = []

    for track1, track2 in zip(tracks1, tracks2):
        len_diff = math.fabs(len(track1.timestamps) - len(track2.timestamps))
        scores.append(len_diff)
    return scores


def len_quot(tracks1, tracks2):
    """Compares the length of track1 and track2

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :float: length of track1 divided by length of track2
    """
    scores = []

    for track1, track2 in zip(tracks1, tracks2):
        len_quot = math.fabs(len(track2.timestamps) / len(track1.timestamps))
        assert np.all(np.isfinite(len_quot))
        scores.append(len_quot)
    return scores


def distance_orientations_per_time(detections1, detections2, meta_key=None):
    """Combines features distance_orientations and time_diff

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :float: distance orientation normalized by time
    """
    assert len(detections1) == len(detections2), \
        "Both detection lists should have the same size"

    if meta_key is None:
        arr1 = np.array([det.orientation for det in detections1])
        arr2 = np.array([det.orientation for det in detections2])
    else:
        arr1 = np.array([det.meta[meta_key] for det in detections1])
        arr2 = np.array([det.meta[meta_key] for det in detections2])

    distance = np.fabs(arr1 - arr2)
    mask = distance > math.pi
    distance[mask] = 2 * math.pi - distance[mask]

    time_difs = [(det2.timestamp - det1.timestamp)
                 for det1, det2 in zip(detections1, detections2)]

    dist_per_time = distance / time_difs

    assert np.all(np.isfinite(dist_per_time))
    return dist_per_time


def forward_error(tracks1, tracks2):
    """Calculates the bee's movement in the last two detections of track1 and forecasts where
    the bee is expected to be by the time track2 begins

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :float: difference between expected and real location of first detection in track2
    """
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        # If there is only one detection, we cannot predict a movement
        if (len(track1.timestamps) < 2):
            scores.append(-1.0)
        else:
            det1_front = track1.meta[DETKEY][-2]
            det1_back = track1.meta[DETKEY][-1]
            det2 = track2.meta[DETKEY][0]

            vector = np.asarray([det1_back.x - det1_front.x, det1_back.y - det1_front.y])
            time_diff = (det1_back.timestamp - det1_front.timestamp)
            # make sure we won't divide by zero
            assert time_diff > 0, "The detections come from the same frame."
            move_per_time = vector / time_diff

            time_between_tracks = (det2.timestamp - det1_back.timestamp)
            predicted_vector = move_per_time * time_between_tracks

            predicted_point = np.asarray([det1_back.x, det1_back.y]) + predicted_vector

            error = np.linalg.norm(predicted_point - np.asarray([det2.x, det2.y]))
            scores.append(error)
    return scores


def backward_error(tracks1, tracks2):
    """Calculates the bee's movement in the first two detections of track2 and calculates where
    the bee should have been by the end of track1

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :float: difference between expected and real location of last detection in track1
    """
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        # If there is only one detection, we cannot predict a movement
        if (len(track2.timestamps) < 2):
            scores.append(-1.0)
        else:
            det2_front = track2.meta[DETKEY][0]
            det2_back = track2.meta[DETKEY][1]
            det1 = track1.meta[DETKEY][-1]

            vector = np.asarray([det2_back.x - det2_front.x, det2_back.y - det2_front.y])
            time_diff = (det2_back.timestamp - det2_front.timestamp)
            # make sure we won't divide by zero
            assert time_diff > 0, "The detections come from the same frame."
            move_per_time = vector / time_diff

            time_between_tracks = (det2_front.timestamp - det1.timestamp)
            predicted_vector = move_per_time * time_between_tracks

            predicted_point = np.asarray([det2_front.x, det2_front.y]) - predicted_vector

            error = np.linalg.norm(predicted_point - np.asarray([det1.x, det1.y]))
            scores.append(error)
    return scores


def z_orientation_forward_error(tracks1, tracks2):
    """Calculates the change of orientation (=the angle in which the bee looks on the hive) within
    the last two detections of track1

            Arguments:
                tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
                tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

            Returns:
                :float: difference between expected and real orientation of the first detection
                in track2
    """
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        # If there is only one detection, we cannot predict a change
        # A value that cannot appear in the real data is appended
        if (len(track1.timestamps) < 2):
            scores.append(2 * math.pi)
        else:
            det1_front = track1.meta[DETKEY][-2]
            det1_back = track1.meta[DETKEY][-1]
            det2 = track2.meta[DETKEY][0]

            # The rotations must be given in the range [-pi, pi]
            assert -math.pi <= det1_front.orientation <= math.pi
            assert -math.pi <= det1_back.orientation <= math.pi
            assert -math.pi <= det2.orientation <= math.pi

            orientation_change = (det1_back.orientation - det1_front.orientation) % (2 * math.pi)
            if (orientation_change > math.pi):
                orientation_change = orientation_change - (2 * math.pi)

            time_diff = (det1_back.timestamp - det1_front.timestamp)
            z_change_per_time = orientation_change / time_diff

            time_between_tracks = (det2.timestamp - det1_back.timestamp)

            predicted_change = z_change_per_time * time_between_tracks

            predicted_z = (det1_back.orientation + predicted_change) % (2 * math.pi)

            error = (det2.orientation - predicted_z) % (2 * math.pi)
            if (error > math.pi):
                error = error - (2 * math.pi)

            scores.append(error)
    return scores


def z_orientation_backward_error(tracks1, tracks2):
    """Calculates the change of orientation (=the angle in which the bee looks on the hive) within
    the first two detections of track2

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :float: difference between expected and real orientation of the last detection
            in track1
    """
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        # If there is only one detection, we cannot predict a change
        # A value that cannot appear in the real data is appended
        if (len(track2.timestamps) < 2):
            scores.append(2 * math.pi)
        else:
            det2_front = track2.meta[DETKEY][0]
            det2_back = track2.meta[DETKEY][1]
            det1 = track1.meta[DETKEY][-1]

            # The rotations must be given in the range [-pi, pi]
            assert -math.pi <= det2_front.orientation <= math.pi
            assert -math.pi <= det2_back.orientation <= math.pi
            assert -math.pi <= det1.orientation <= math.pi

            # How does orientation change within the two detections
            orientation_change = det2_front.orientation - det2_back.orientation
            if (orientation_change > math.pi):
                orientation_change = orientation_change - (2 * math.pi)

            # Break it down to time
            time_diff = (det2_back.timestamp - det2_front.timestamp)
            z_change_per_time = orientation_change / time_diff

            time_between_tracks = (det2_front.timestamp - det1.timestamp)

            predicted_change = z_change_per_time * time_between_tracks

            predicted_z = (det2_front.orientation + predicted_change) % (2 * math.pi)

            error = (det1.orientation - predicted_z) % (2 * math.pi)
            if (error > math.pi):
                error = error - (2 * math.pi)

            scores.append(error)
    return scores


def angle_difference(tracks1, tracks2):
    """Function that calculates the angle between the movement-vector of the last two detections
    in track1 and the first two in track2


        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :float: absolute angle difference
    """
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        # It's not possible to calculate an angle between points
        if len(track1.timestamps) < 2 or len(track2.timestamps) < 2:
            scores.append(-1.0)

        else:

            det1_front = track1.meta[DETKEY][-2]
            det1_back = track1.meta[DETKEY][-1]
            vector1 = np.asarray([det1_back.x - det1_front.x, det1_back.y - det1_front.y])

            det2_front = track2.meta[DETKEY][0]
            det2_back = track2.meta[DETKEY][1]
            vector2 = np.asarray([det2_back.x - det2_front.x, det2_back.y - det2_front.y])

            # If the bee didn't move, we still have a point
            # And cannot calculate an angle difference
            if (np.sum(np.fabs(vector1)) == 0 or np.sum(
                    np.fabs(vector2)) == 0):
                scores.append(-1.0)

            else:
                # calculate the dot-product
                dot = np.dot(vector1, vector2)
                x_modulus = np.sqrt((vector1 * vector1).sum())
                y_modulus = np.sqrt((vector2 * vector2).sum())

                cos_angle = np.divide(np.divide(dot, x_modulus), y_modulus)
                # force numpy to not make cos_angle 1 or -1
                # as those are not defined for the arccos!
                cos_angle = np.around(cos_angle, decimals=10)
                angle = np.arccos(cos_angle)

                scores.append(angle)
    return scores


def calculate_weighted_window_vector(points, track1):

    """Helper to calculate a vector from multiple vectors which are weighted based on how
    close they are to the end (start) of a track.

    Arguments:
        points (:obj:`list` of [int,int,float]): [x position of a detection, y porition of a
            detection, timestamp of the detection]
        track1 (bool): is it track1 or  not in this case it is track2), this influences if we
            weight from the front to the end or vice versa

    Returns:
        :obj:`list` of float: weighted vector normalized on time
    """

    # a default vector for tracks of length1
    vector = np.asarray([0.0, 0.0])
    start_point = points[0]

    if len(points) == 1:
        return vector

    # A list to append all the differences between the vectors calculated from the given points
    vector_diffs = []

    for point in points[1:]:
        vector_diff = point - start_point
        normalized_diff = vector_diff / (point[2] - start_point[2])
        vector_diffs.append(normalized_diff)
        start_point = point

    # The weights are powers of 2
    to_be_weighted_len = len(vector_diffs)
    weights = np.asarray([2 ** i for i in range(to_be_weighted_len)])

    # In track1 the last vector is weighted highest because it is close to the gap
    if track1:
        for i in range(to_be_weighted_len):
            vector_diffs[i] = (vector_diffs[i] * weights[i])[:2]  # get rid of the timestamp
    # In track2 the first vector is weighted highest because it is close to the gap
    else:
        for i in range(to_be_weighted_len):
            vector_diffs[i] = (vector_diffs[i] * weights[-1 - i])[:2]

    # Sum of all the vectors
    vector_sum = np.sum(vector_diffs, axis=0)
    # Divided by the weights
    vector = vector_sum / (sum(weights))
    return vector


def weighted_window_xy_length_difference(tracks1, tracks2):
    """Function that calculates length difference of the weighted vector for the last movements of
    track1 and the first movements of track2

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :float: absolute length difference of the two weighted vectors
    """
    window = 4  # consider the last three vectors (if existent)
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        if len(track1.timestamps) < 2 or len(track2.timestamps) < 2:
            scores.append(-1.0)

        else:
            short1points = [np.asarray([det.x, det.y, det.timestamp]) for det in
                            track1.meta[DETKEY][-window:]]
            short2points = [np.asarray([det.x, det.y, det.timestamp]) for det in
                            track2.meta[DETKEY][:window]]

            vector1 = calculate_weighted_window_vector(short1points, True)
            vector2 = calculate_weighted_window_vector(short2points, False)

            eucl1 = np.linalg.norm(vector1)
            eucl2 = np.linalg.norm(vector2)
            scores.append(np.fabs(np.subtract(eucl1, eucl2)))

    return scores


def weighted_window_x_length_difference(tracks1, tracks2):
    """Function that calculates length difference of the weighted vector for the last movements of
    track1 and the first movements of track2 on the x axis

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :float: absolute length difference of the two weighted vectors
    """
    window = 4
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        if (len(track1.timestamps) < 2 or len(track2.timestamps) < 2):
            scores.append(-1.0)

        else:
            short1points = [np.asarray([det.x, det.y, det.timestamp]) for det in
                            track1.meta[DETKEY][-window:]]
            short2points = [np.asarray([det.x, det.y, det.timestamp]) for det in
                            track2.meta[DETKEY][:window]]

            vector1 = calculate_weighted_window_vector(short1points, True)
            vector2 = calculate_weighted_window_vector(short2points, False)

            len1 = vector1[0]
            len2 = vector2[0]
            scores.append(np.subtract(len1, len2))

    return scores


def weighted_window_y_length_difference(tracks1, tracks2):
    """Function that calculates length difference of the weighted vector for the last movements of
    track1 and the first movements of track2 on the y axis

        Arguments:
            tracks1 (:obj:`list` of :obj:`.Track`): Iterable with Tracks
            tracks2 (:obj:`list` of :obj:`.Track`): Iterable with Tracks

        Returns:
            :float: absolute length difference of the two weighted vectors
    """
    window = 4
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        if len(track1.timestamps) < 2 or len(track2.timestamps) < 2:
            scores.append(-1.0)  # es ergibt keinen sinn, winkel bei punkten zu betrachten

        else:
            short1points = [np.asarray([det.x, det.y, det.timestamp]) for det in
                            track1.meta[DETKEY][-window:]]
            short2points = [np.asarray([det.x, det.y, det.timestamp]) for det in
                            track2.meta[DETKEY][:window]]

            vector1 = calculate_weighted_window_vector(short1points, True)
            vector2 = calculate_weighted_window_vector(short2points, False)

            # vergleiche die vektorlänge
            len1 = vector1[1]
            len2 = vector2[1]
            scores.append(np.subtract(len1, len2))

    return scores
