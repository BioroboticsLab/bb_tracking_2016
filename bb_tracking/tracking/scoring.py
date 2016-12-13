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


# Features for the tracks
# Feature, um die id-simularity zu scoren
def score_track_id_sim(tracks1, tracks2):
    scores = []
    for track1, track2 in zip(tracks1, tracks2):
        arr1 = np.array([det.beeId for det in track1.meta[DETKEY]])
        arr2 = np.array([det.beeId for det in track2.meta[DETKEY]])
        scores.append(score_id_sim(np.median(arr1, axis=0), np.median(arr2, axis=0)))
    return scores


# Feature, um die id-simularity um die Lücke herum zu scoren
def score_track_id_sim_window(tracks1, tracks2):
    window = 2
    scores = []
    for track1, track2 in zip(tracks1, tracks2):
        arr1 = np.array([det.beeId for det in track1.meta[DETKEY][-window:]])
        arr2 = np.array([det.beeId for det in track2.meta[DETKEY][:window]])
        scores.append(score_id_sim(np.median(arr1, axis=0), np.median(arr2, axis=0)))
    return scores


def confidency_id_sim_v(tracks1, tracks2):
    arr1conf, arr2conf = [], []
    for track1, track2 in zip(tracks1, tracks2):
        arr1 = np.array([det.beeId for det in track1.meta[DETKEY]])
        arr2 = np.array([det.beeId for det in track2.meta[DETKEY]])

        arr1conf.append(np.mean(np.min(np.abs(0.5 - arr1) * 2, axis=1)))
        arr2conf.append(np.mean(np.min(np.abs(0.5 - arr2) * 2, axis=1)))

    return np.fabs(np.array(arr1conf) - np.array(arr2conf))


def confidency_id_sim_v2(tracks1, tracks2):
    idconf = []
    for track1, track2 in zip(tracks1, tracks2):
        arr1 = np.array([det.beeId for det in track1.meta[DETKEY]])
        arr2 = np.array([det.beeId for det in track2.meta[DETKEY]])

        idconf.append(math.fabs((np.mean(np.min(np.abs(0.5 - arr1) * 2, axis=1))) -
                                (np.mean(np.min(np.abs(0.5 - arr2) * 2, axis=1)))))

    return idconf


def short_confidency_id_sim_v2(tracks1, tracks2):
    """Only take the last and first elements of tracks"""
    window = 2
    idconf = []
    for track1, track2 in zip(tracks1, tracks2):
        length = min(window, len(track1.timestamps), len(track2.timestamps))
        arr1 = np.array([det.beeId for det in track1.meta[DETKEY][-length:]])
        arr2 = np.array([det.beeId for det in track2.meta[DETKEY][:length]])
        idconf.append(math.fabs((np.mean(np.min(np.abs(0.5 - arr1) * 2, axis=1))) -
                                (np.mean(np.min(np.abs(0.5 - arr2) * 2, axis=1)))))
    return idconf


# UNPERFORMANT THINGS

# Geschwindigkeiten der Biene
def tempo(tracks1, tracks2):
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        if (len(track1.timestamps) < 2 or len(track2.timestamps) < 2):
            scores.append(-1.0)  # ein punkt hat keine geschwindigkeit!
        else:
            speed1 = calculate_speed(track1)
            assert np.all(np.isfinite(speed1))

            speed2 = calculate_speed(track2)
            assert np.all(np.isfinite(speed2))

            scores.append(math.fabs(speed1 - speed2))
    return scores


# hilfsfunktion
def calculate_speed(track):
    # track length to divide the total distance by
    start = track.meta[DETKEY][0].timestamp
    end = track.meta[DETKEY][-1].timestamp

    length = (end - start)

    assert length > 0  # cause we don't want to divide by zero

    # we start with the first detection
    startpoint = (track.meta[DETKEY][0].x, track.meta[DETKEY][0].y)

    # sum up all the distances
    eucl = 0

    for det in track.meta[DETKEY][1:]:
        currentpoint = (det.x, det.y)
        eucl += distance.euclidean(startpoint, currentpoint)
        startpoint = currentpoint

    return eucl / length


# Bei den Vektorlängen macht es schon Sinn, dass einer der Vektoren Null sein kann.
# Dann hat sich die Biene halt nicht bewegt
# Betrachtet die Differenz in den X-Vektoren der letzten 2 und der ersten 2 Detektionen
# der beiden Tracks
def x_vectors(tracks1, tracks2):
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        if (len(track1.timestamps) < 2 or len(track2.timestamps) < 2):
            scores.append(
                -1.0)  # wir können keine Vektoren und Punkte vergleichen. hänge unmöglichen wert an
        else:
            detvorend = track1.meta[DETKEY][-2]
            detend = track1.meta[DETKEY][-1]
            detvorend2 = track2.meta[DETKEY][0]
            detend2 = track2.meta[DETKEY][1]

            end = detend.x - detvorend.x
            start = detend2.x - detvorend2.x

            # runterbrechen auf die Zeitänderung
            timedif1 = (detend.timestamp - detvorend.timestamp)
            timedif2 = (detend2.timestamp - detvorend2.timestamp)

            # make sure we won't divide by zero
            assert timedif1 > 0
            assert timedif2 > 0

            vektor1 = end / timedif1
            vektor2 = start / timedif2

            assert np.all(np.isfinite(vektor1))
            assert np.all(np.isfinite(vektor2))

            dif = math.fabs(
                vektor1 - vektor2)  # kein, räumliche änderung egal in welche richtung
            scores.append(dif)

    return scores


# Betrachtet die Differenz in den Y-Vektoren der letzten 2 und der ersten
# 2 Detektionen der beiden Tracks
def y_vectors(tracks1, tracks2):
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        if (len(track1.timestamps) < 2 or len(track2.timestamps) < 2):
            scores.append(
                -1.0)  # wir können keine Vektoren und Punkte vergleichen. hänge unmöglichen wert an
        else:

            detvorend = track1.meta[DETKEY][-2]
            detend = track1.meta[DETKEY][-1]
            detvorend2 = track2.meta[DETKEY][0]
            detend2 = track2.meta[DETKEY][1]

            end = detend.y - detvorend.y
            start = detend2.y - detvorend2.y

            # runterbrechen auf die Zeitänderung
            timedif1 = (detend.timestamp - detvorend.timestamp)
            timedif2 = (detend2.timestamp - detvorend2.timestamp)

            # make sure we won't divide by zero
            assert timedif1 > 0
            assert timedif2 > 0

            vektor1 = end / timedif1
            vektor2 = start / timedif2

            assert np.all(np.isfinite(vektor1))
            assert np.all(np.isfinite(vektor2))

            dif = math.fabs(vektor1 - vektor2)
            scores.append(dif)

    return scores


# Betrachtet die Differenz in den Vektoren der letzten 2 und der
# ersten 2 Detektionen der beiden Tracks
def xy_vector(tracks1, tracks2):
    scores = []

    for track1, track2 in zip(tracks1, tracks2):

        if (len(track1.timestamps) < 2 or len(track2.timestamps) < 2):
            scores.append(
                -1.0)  # wir können keine Vektoren und Punkte vergleichen. hänge unmöglichen wert an
        else:

            detvorend = track1.meta[DETKEY][-2]
            detend = track1.meta[DETKEY][-1]

            timedif1 = (detend.timestamp - detvorend.timestamp)

            detvorend2 = track2.meta[DETKEY][0]
            detend2 = track2.meta[DETKEY][1]
            timedif2 = (detend2.timestamp - detvorend2.timestamp)

            # make sure we won't divide by zero
            assert timedif1 > 0
            assert timedif2 > 0

            # Berechne die Vektoren zwischen den Detektionen
            vector1 = np.asarray([detend.x - detvorend.x, detend.y - detvorend.y])
            vector2 = np.asarray([detend2.x - detvorend2.x, detend2.y - detvorend2.y])

            normalized_vector1 = vector1 / timedif1
            normalized_vector2 = vector2 / timedif2

            # Betrachte den Unterschied der Vektorlängen
            eucl1 = np.linalg.norm(normalized_vector1 - normalized_vector2)
            scores.append(eucl1)

    return scores


# Setzt Zeit- und Ortsdistanz zwischen letzter Detektion des ersten Tracks
# und erster des zweiten Tracks in Relation
def gap_speed(tracks1, tracks2):
    scores = []
    for track1, track2 in zip(tracks1, tracks2):
        det1 = track1.meta[DETKEY][-1]
        det2 = track2.meta[DETKEY][0]

        timedif = (det2.timestamp - det1.timestamp)
        # make sure we won't divide by zero
        assert timedif > 0

        placedif = math.sqrt((det2.x - det1.x) ** 2 + (det2.y - det1.y) ** 2)
        dif = placedif / timedif

        assert np.all(np.isfinite(dif))

        scores.append(dif)

    return scores


# Berechne die Differenz der Größe der Quadrate, auf denen sich die Biene
# während des Tracks bewegt (pro Zeiteinheit)
def movement_radius(tracks1, tracks2):
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        if (len(track1.timestamps) < 2 or len(track2.timestamps) < 2):
            scores.append(
                -1.0)  # wir können keine Vektoren und Punkte vergleichen. hänge unmöglichen wert an
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

            # falls die Biene sich nicht bewegt, ist ihr Radius eben Null, kein Problem

            # Bestimme die kompletten Zeitfenster, auf denen sich der Track bewegt
            timedif1 = (track1.meta[DETKEY][-1].timestamp - track1.meta[DETKEY][0].timestamp)
            timedif2 = (track2.meta[DETKEY][-1].timestamp - track2.meta[DETKEY][0].timestamp)

            # make sure we won't divide by zero
            assert timedif1 > 0
            assert timedif2 > 0

            result1 = (square1 / (timedif1))
            result2 = square2 / (timedif2)

            # nimm die Differenz der Bewegungsradien
            scores.append(math.fabs(result1 - result2))

            # oder nimm die verhältnismäßigen Größen
            # if result2<=result1:
            # scores.append(math.fabs(result2/result1))
            # else:
            # scores.append(math.fabs(result1/result2))

    return scores


def len_dif(tracks1, tracks2):
    scores = []
    for track1, track2 in zip(tracks1, tracks2):
        lendif = math.fabs(len(track1.timestamps) - len(track2.timestamps))
        scores.append(lendif)
    return scores


def len_quot(tracks1, tracks2):
    scores = []
    for track1, track2 in zip(tracks1, tracks2):
        lenquot = math.fabs(len(track2.timestamps) / len(track1.timestamps))

        assert np.all(np.isfinite(lenquot))

        scores.append(lenquot)
    return scores


# meine implementierung, in der die verstrichene Zeit eine Rolle spielt
def distance_orientations_per_time(detections1, detections2, meta_key=None):
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

    # assert np.all(np.isfinite(dist_per_time))
    return dist_per_time


def foreward_error(tracks1, tracks2):
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        if (len(track1.timestamps) < 2):
            scores.append(-1.0)  # wert, der niemals vorkommen könnte
        else:
            det1v = track1.meta[DETKEY][-2]
            det1h = track1.meta[DETKEY][-1]
            det2 = track2.meta[DETKEY][0]

            vector = np.asarray([det1h.x - det1v.x, det1h.y - det1v.y])
            timedif = (det1h.timestamp - det1v.timestamp)
            move_per_time = vector / timedif

            time_between_tracks = (det2.timestamp - det1h.timestamp)
            predicted_vector = move_per_time * time_between_tracks

            predicted_point = np.asarray([det1h.x, det1h.y]) + predicted_vector

            error = np.linalg.norm(predicted_point - np.asarray([det2.x, det2.y]))
            scores.append(error)
    return scores


def backward_error(tracks1, tracks2):
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        if (len(track2.timestamps) < 2):
            scores.append(-1.0)  # wert, der niemals vorkommen kann
        else:
            det2v = track2.meta[DETKEY][0]
            det2h = track2.meta[DETKEY][1]
            det1 = track1.meta[DETKEY][-1]

            vector = np.asarray([det2h.x - det2v.x, det2h.y - det2v.y])
            timedif = (det2h.timestamp - det2v.timestamp)
            move_per_time = vector / timedif

            time_between_tracks = (det2v.timestamp - det1.timestamp)
            predicted_vector = move_per_time * time_between_tracks

            predicted_point = np.asarray([det2v.x, det2v.y]) - predicted_vector

            error = np.linalg.norm(predicted_point - np.asarray([det1.x, det1.y]))
            scores.append(error)
    return scores


def z_orientation_foreward_error(tracks1, tracks2):
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        if (len(track1.timestamps) < 2):
            scores.append(2 * math.pi)  # wert, der nicht vorkommen kann
        else:
            det1v = track1.meta[DETKEY][-2]
            det1h = track1.meta[DETKEY][-1]
            det2 = track2.meta[DETKEY][0]

            # The rotations must be given in the range [-pi, pi]
            assert -math.pi <= det1v.orientation <= math.pi
            assert -math.pi <= det1h.orientation <= math.pi
            assert -math.pi <= det2.orientation <= math.pi

            wert = (det1h.orientation - det1v.orientation) % (2 * math.pi)
            if (wert > math.pi):
                wert = wert - (2 * math.pi)

            timedif = (det1h.timestamp - det1v.timestamp)
            z_change_per_time = wert / timedif

            time_between_tracks = (det2.timestamp - det1h.timestamp)

            predicted_change = z_change_per_time * time_between_tracks

            predicted_z = (det1h.orientation + predicted_change) % (2 * math.pi)

            error = (det2.orientation - predicted_z) % (2 * math.pi)
            if (error > math.pi):
                error = error - (2 * math.pi)

            scores.append(error)
    return scores


# wie groß ist der abstand zwischen dem erwarteten und dem realen
def z_orientation_backward_error(tracks1, tracks2):
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        if (len(track2.timestamps) < 2):
            scores.append(2 * math.pi)  # wert, der nicht vorkommen kann
        else:
            det2v = track2.meta[DETKEY][0]
            det2h = track2.meta[DETKEY][1]
            det1 = track1.meta[DETKEY][-1]

            # The rotations must be given in the range [-pi, pi]
            assert -math.pi <= det2v.orientation <= math.pi
            assert -math.pi <= det2h.orientation <= math.pi
            assert -math.pi <= det1.orientation <= math.pi

            # Wie verändert sich die Rotation auf dem letzten Vektor des Tracks?
            rotation = det2v.orientation - det2h.orientation
            if (rotation > math.pi):
                rotation = rotation - (2 * math.pi)

            # Wie verändert sich die Rotation also pro Millisekunde?
            timedif = (det2h.timestamp - det2v.timestamp)
            z_change_per_time = rotation / timedif

            # Wie viele Millisekunden liegen zwischen den Tracks?
            time_between_tracks = (det2v.timestamp - det1.timestamp)

            predicted_change = z_change_per_time * time_between_tracks

            predicted_z = (det2v.orientation + predicted_change) % (2 * math.pi)

            error = (det1.orientation - predicted_z) % (2 * math.pi)
            if (error > math.pi):
                error = error - (2 * math.pi)

            scores.append(error)
    return scores


# Hilfsfunktion, gibt den winkelabstand der letzten vektoren an
def winkelabstand(tracks1, tracks2):
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        if (len(track1.timestamps) < 2 or len(track2.timestamps) < 2):
            scores.append(
                -1.0)  # es ergibt keinen sinn, den winkelabstand zwischen punkten zu bestimmen

        else:

            detvorend = track1.meta[DETKEY][-2]
            detend = track1.meta[DETKEY][-1]
            arr1 = np.asarray([detend.x - detvorend.x, detend.y - detvorend.y])

            detvorend2 = track2.meta[DETKEY][0]
            detend2 = track2.meta[DETKEY][1]
            arr2 = np.asarray([detend2.x - detvorend2.x, detend2.y - detvorend2.y])

            if (np.sum(np.fabs(arr1)) == 0 or np.sum(
                    np.fabs(arr2)) == 0):  # falls sich die Biene einfach nicht bewegt hat
                scores.append(
                    -1.0)  # es ergibt keinen sinn, den winkelabstand zwischen punkten zu bestimmen

            else:
                # berechne das skalarprodukt
                dot = np.dot(arr1, arr2)
                x_modulus = np.sqrt((arr1 * arr1).sum())
                y_modulus = np.sqrt((arr2 * arr2).sum())

                cos_angle = np.divide(np.divide(dot, x_modulus), y_modulus)
                cos_angle = np.around(cos_angle,
                                      decimals=10)  # force numpy to not make cos_angle 1 or -1
                # as those are not defined for the arccos!
                angle = np.arccos(cos_angle)

                scores.append(angle)

    return scores


# Hilfsfunktion, welche für einen Track auf den letzten Detektionen ein gewichtetes
# Fenster-verfahren anwendet
def calculate_weighted_window_vector(points, track1):
    # points is an array of np.arrays of length 3,
    # representing a point and its timestamp [[x,y,timestampt], , ]
    # track1 is a boolean, stating if we are in track1 or 2, thats important for the weighting

    # a defaultvector for tracks of length1
    vector = np.asarray([0.0, 0.0])
    startpoint = points[0]

    if (len(points) == 1):
        return vector

    vektordifs = []

    for point in points[1:]:
        vektordif = point - startpoint
        normalized_dif = vektordif / (point[2] - startpoint[2])
        vektordifs.append(normalized_dif)
        startpoint = point

    to_be_weighted_len = len(vektordifs)
    weights = np.asarray([2 ** i for i in range(to_be_weighted_len)])

    if track1:
        for i in range(to_be_weighted_len):
            vektordifs[i] = (vektordifs[i] * weights[i])[:2]  # get rid of the timestamp
    else:  # im hinteren track werden die ersten detektionen mehr gewichtet
        for i in range(to_be_weighted_len):
            vektordifs[i] = (vektordifs[i] * weights[-1 - i])[:2]

    vektorsum = np.sum(vektordifs, axis=0)
    vector = vektorsum / (sum(weights))
    return vector


def weighted_window_xy_length_difference(tracks1, tracks2):
    window = 4  # 4 bedeutet die letzten 3 vektoren zu betrachten
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        if (len(track1.timestamps) < 2 or len(track2.timestamps) < 2):
            scores.append(-1.0)  # es ergibt keinen sinn, winkel bei punkten zu betrachten

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
    window = 4
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        if (len(track1.timestamps) < 2 or len(track2.timestamps) < 2):
            scores.append(-1.0)  # es ergibt keinen sinn, winkel bei punkten zu betrachten

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
    window = 4
    scores = []
    for track1, track2 in zip(tracks1, tracks2):

        if (len(track1.timestamps) < 2 or len(track2.timestamps) < 2):
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
