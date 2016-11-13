# -*- coding: utf-8 -*-
"""Defines some functions that compare features and score them.

Deprecated scoring functions are not imported to the parent module!

Note:
    Functions with suffix **_v** are vectorized with numpy.
"""
from itertools import chain
import math
import numpy as np
from scipy.spatial.distance import cityblock, hamming
from bb_binary import int_id_to_binary
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


def bit_array_to_int_v(detections, threshold=0.5):
    """Converts the bit frequency distribution of the id to an integer representation.


    Note:
        This method assumes little endianess and a 12 bit representation!

    Arguments:
        detections (:obj:`list` of :obj:`.Detection`): Iterable with `.Detection`

    Keyword Arguments:
        threshold (Optional float): `beeId` values >= threshold are interpreted as 1

    Returns:
        :obj:`list` of int: the decoded ids represented as integer
    """
    assert len(detections[0].beeId) == 12, "Only implemented for 12 bit representation."
    arr = np.packbits(np.array([det.beeId[::-1] for det in detections]) >= threshold, axis=1)
    arr = arr.astype(np.int16, copy=False)
    return np.left_shift(arr[:, 0], 4) | np.right_shift(arr[:, 1], 4)


def calc_median_ids(tracks):
    """Helper to calculate the median bit for all the ids in the given track.

    Note:
        For performance reasons the median id is saved as meta key and only is recalculated
        if the length of the track changes.

    Arguments:
        tracks(:obj:`list` of :obj`.Track`): Iterable with Tracks

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
