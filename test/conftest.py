# -*- coding: utf-8 -*-
"""Setup for tests and fixtures.

In this file you will find almost all the used fixtures for tests and some helper.

Note:
    Some data for fixtures are read from CSV files in the `fixtures/` directory.

    There is also a bash script to convert the ODS file to CSV files.

    The CSV files are checked in so that they can be used for testing e.g. on continuous
    integration servers without having to install Gnumeric as dependency.

Todo:
    If this module gets bigger it might be necessary to refactor this file and put fixtures and
    helper functions into separate files.
"""
# pylint:disable=no-member,redefined-outer-name,too-many-arguments,too-many-locals
from __future__ import division, print_function
from fractions import Fraction
import inspect
import math
import os
import random
import shutil
import warnings
import numpy as np
import pandas as pd
import pytest
from bb_binary import build_frame_container, build_frame_container_from_df, Repository,\
    int_id_to_binary, binary_id_to_int, Frame
from bb_tracking.data import Detection, Track, DataWrapperPandas, DataWrapperTruthPandas, \
    DataWrapperBinary, DataWrapperTruthBinary, DataWrapperTracks, DataWrapperTruthTracks
from bb_tracking.data.constants import CAMKEY, DETKEY, TRUTHKEY
from bb_tracking.tracking import SimpleWalker
from bb_tracking.validation import Validator

# enable stack trace for annoying Warnings
warnings.simplefilter("error", pd.core.common.SettingWithCopyWarning)
PATH = os.path.join(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
    'fixtures',
    '')


def parse_float_list(string):
    """Helper to parse a string to float list.

    Arguments:
        string (str): string  with integers and floats

    Returns:
        list: list with floats

    Example:
        >>> parse_float_list("[1,23; 2,34, 1/3, 5]")
        [1.23, 2.34, 0.33, 5]
    """
    new_list = []
    convert_fun = int
    for num in string[1:-1].split(';'):
        if '/' in num:
            num = float(Fraction(num))
            convert_fun = float
        elif ',' in num or '.' in num:
            num = float(num.replace(',', '.'))
            convert_fun = float
        elif num == "inf":
            convert_fun = float
        new_list.append(num)
    return [convert_fun(x) for x in new_list]


def cmp_tracks(track1, track2, cmp_trackid=True):
    """Compare two tracks.

    Arguments:
        track1 (:obj:`Track`): expected track
        track2 (:obj:`Track`): test track

    Keyword Arguments:
        cmp_trackid (bool): True if track ids should be compared. False otherwise.
    """
    if cmp_trackid:
        assert track1.id == track2.id
    assert len(track1.ids) == len(track2.ids)
    track_ids_equal = list(track1.ids) == list(track2.ids)
    if isinstance(track_ids_equal, bool):
        assert track_ids_equal
    else:
        assert all(track_ids_equal)
    assert len(track1.timestamps) == len(track2.timestamps)

    track_timestamps_equal = list(track1.timestamps) == list(track2.timestamps)
    if isinstance(track_timestamps_equal, bool):
        assert track_timestamps_equal
    else:
        assert all(track_timestamps_equal)


@pytest.fixture()
def detections():
    """Fixture for detections (not cleaned).

    The data for this fixture is partly arbitrary so
    do **not** use it for verifying Algorithms!
    """
    frame = pd.read_csv(PATH + 'detections.csv', decimal=',')
    frame.beeID = frame.beeID.apply(parse_float_list)
    frame.descriptor = frame.descriptor.apply(parse_float_list)
    return frame


@pytest.fixture
def detections_clean():
    """Fixture for detections (cleaned).

    The data for this fixture is partly arbitrary so do **not** use it for verifying Algorithms!
    """
    frame = pd.read_csv(PATH + 'detections_clean.csv', decimal=',')
    frame.set_index('id', drop=False, inplace=True, verify_integrity=True)
    frame.beeID = frame.beeID.apply(parse_float_list)
    frame.descriptor = frame.descriptor.apply(parse_float_list)
    return frame


@pytest.fixture
def detections_binary(detections_clean):
    """Fixture for bb_binary detections (not cleaned).

    The data for this fixture is partly arbitrary so do **not** use it for verifying Algorithms!
    """
    detections_clean[CAMKEY] = detections_clean['camID']
    detections_clean['decodedId'] = detections_clean['beeID']
    detections_clean['xRotation'] = detections_clean['zrotation']
    detections_clean['yRotation'] = detections_clean['zrotation']
    detections_clean['zRotation'] = detections_clean['zrotation']
    detections_clean['radius'] = [0] * detections_clean.shape[0]

    union_type = 'detectionsDP'
    fc_cam0, offset = build_frame_container_from_df(detections_clean, union_type, 0)
    fc_cam2, _ = build_frame_container_from_df(detections_clean, union_type, 2, frame_offset=offset)
    test_repo = PATH + 'test_repo'
    if os.path.exists(test_repo):
        shutil.rmtree(test_repo)
    os.makedirs(test_repo)
    repo = Repository(test_repo)
    repo.add(fc_cam0)
    repo.add(fc_cam2)
    return repo


@pytest.fixture
def detections_binary_empty():
    """Fixture for bb_binary repository with empty frames."""
    frame0 = Frame.new_message()
    frame0.id = 0
    frame0.timestamp = 0
    frame0.detectionsUnion.init('detectionsDP', 0)

    frame1 = Frame.new_message()
    frame1.id = 1
    frame1.timestamp = 1
    frame1.detectionsUnion.init('detectionsDP', 0)
    fc0 = build_frame_container(0, 1, 0)
    fc0.init('frames', 2)
    fc0.frames[0] = frame0
    fc0.frames[1] = frame1
    test_repo = PATH + 'test_repo'
    if os.path.exists(test_repo):
        shutil.rmtree(test_repo)
    os.makedirs(test_repo)
    repo = Repository(test_repo)
    repo.add(fc0)
    return repo


@pytest.fixture
def truth():
    """Fixture for truth (not cleaned).

    The data for this fixture is partly arbitrary so do **not** use it for verifying Algorithms!
    """
    frame = pd.read_csv(PATH + 'truth.csv', decimal=',')
    return frame


@pytest.fixture
def truth_clean():
    """Fixture for truth (cleaned).

    The data for this fixture is partly arbitrary so do **not** use it for verifying Algorithms!
    """
    frame = pd.read_csv(PATH + 'truth_clean.csv', decimal=',')
    frame.set_index('id', drop=False, inplace=True, verify_integrity=True)
    frame.beeID = frame.beeID.apply(parse_float_list)
    frame.descriptor = frame.descriptor.apply(parse_float_list)
    return frame


@pytest.fixture
def truth_binary(truth):
    """Fixture for bb_binary truth (not cleaned).

    The data for this fixture is partly arbitrary so do **not** use it for verifying Algorithms!
    """
    truth[CAMKEY] = truth['camID']

    union_type = 'detectionsTruth'
    fc_cam0, offset = build_frame_container_from_df(truth, union_type, 0)
    fc_cam2, _ = build_frame_container_from_df(truth, union_type, 2, frame_offset=offset)
    test_repo = PATH + 'test_repo_truth'
    if os.path.exists(test_repo):
        shutil.rmtree(test_repo)
    os.makedirs(test_repo)
    repo = Repository(test_repo)
    repo.add(fc_cam0)
    repo.add(fc_cam2)
    return repo


@pytest.fixture
def data_pandas_truth(detections, truth):
    """Fixture for DataWrapperTruthPandas with cleaned data.

    The data for this fixture is partly arbitrary so do **not** use it for verifying Algorithms!
    """
    return DataWrapperTruthPandas(detections, truth, 1)


@pytest.fixture
def data_pandas(detections):
    """Fixture for DataWrapperPandas with cleaned data.

    The data for this fixture is partly arbitrary so do **not** use it for verifying Algorithms!
    """
    return DataWrapperPandas(detections, duplicates_radius=1)


@pytest.fixture
def data_binary(detections_binary):
    """Fixture for DataWrapperBinary with cleaned data.

    The data for this fixture is partly arbitrary so do **not** use it for verifying Algorithms!
    """
    return DataWrapperBinary(detections_binary)


@pytest.fixture
def data_binary_truth(detections_binary, truth_binary):
    """Fixture for DataWrapperTruthBinary with cleaned data.

    The data for this fixture is partly arbitrary so do **not** use it for verifying Algorithms!
    """
    return DataWrapperTruthBinary(detections_binary, truth_binary, 1)


@pytest.fixture
def data_tracks(data_binary, tracks_test, id_translator):
    """Fixture for DataWrapperTracks with cleaned data.

    The data for this fixture is partly arbitrary so do **not** use it for verifying Algorithms!
    """
    get_ids = id_translator(data_binary)
    tracks = [Track(id=track.id, ids=get_ids(*track.ids), timestamps=track.timestamps,
                    meta={DETKEY: data_binary.get_detections(get_ids(*track.ids))})
              for track in tracks_test]
    return DataWrapperTracks(tracks, data_binary.cam_timestamps, data=data_binary)


@pytest.fixture
def data_tracks_no_detections(data_binary, tracks_test, id_translator):
    """Fixture for DataWrapperTracks with cleaned data but without detections.

    The data for this fixture is partly arbitrary so do **not** use it for verifying Algorithms!
    """
    get_ids = id_translator(data_binary)
    tracks = [Track(id=track.id, ids=get_ids(*track.ids), timestamps=track.timestamps,
                    meta={DETKEY: data_binary.get_detections(get_ids(*track.ids))})
              for track in tracks_test]
    return DataWrapperTracks(tracks, data_binary.cam_timestamps)


@pytest.fixture
def data_tracks_truth(data_binary_truth, tracks_test, id_translator):
    """Fixture for DataWrapperTruthTracks with cleaned data.

    The data for this fixture is partly arbitrary so do **not** use it for verifying Algorithms!
    """
    get_ids = id_translator(data_binary)
    tracks = [Track(id=track.id, ids=get_ids(*track.ids), timestamps=track.timestamps,
                    meta={DETKEY: data_binary_truth.get_detections(get_ids(*track.ids))})
              for track in tracks_test]
    return DataWrapperTruthTracks(tracks, data_binary_truth.cam_timestamps, data=data_binary_truth)


@pytest.fixture(params=["pandas", "pandas_truth", "binary", "binary_truth", "tracks", "tracks_nd",
                        "tracks_truth"])
def data(request, data_pandas, data_pandas_truth, data_binary, data_binary_truth, data_tracks,
         data_tracks_no_detections, data_tracks_truth):
    """Fixture to run all implementations of DataWrapper"""
    return {"pandas": data_pandas,
            "pandas_truth": data_pandas_truth,
            "binary": data_binary,
            "binary_truth": data_binary_truth,
            "tracks": data_tracks,
            "tracks_nd": data_tracks_no_detections,
            "tracks_truth": data_tracks_truth}[request.param]


@pytest.fixture(params=["pandas_truth", "binary_truth", "tracks_truth"])
def data_truth(request, data_pandas_truth, data_binary_truth, data_tracks_truth):
    """Fixture to run all implementations of DataWrapperTruth"""
    return {"pandas_truth": data_pandas_truth,
            "binary_truth": data_binary_truth,
            "tracks_truth": data_tracks_truth}[request.param]


@pytest.fixture
def detections_test(timestamps):
    """Fixture for detections in {key: namedtuple} format."""
    detections_test = {
        'f1d0c0': Detection(1, timestamps[0], 1, 1, 2.69913,
                            [2 / 3, 1 / 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            {TRUTHKEY: 1, CAMKEY: 0}),
        'f2d0c2': Detection(2, timestamps[1], 2, 2, -0.314444,
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], {TRUTHKEY: 1, CAMKEY: 2}),
        'f3d0c0': Detection(3, timestamps[2], 3, 3, 0.516632,
                            [2 / 3, 1 / 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            {TRUTHKEY: 1, CAMKEY: 0}),
        'f1d1c0': Detection(4, timestamps[0], 4, 4, 2.27196,
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], {TRUTHKEY: 2, CAMKEY: 0}),
        'f1d2c0': Detection(5, timestamps[0], 5, 5, 0,
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], {TRUTHKEY: 3, CAMKEY: 0}),
        'f3d1c0': Detection(6, timestamps[2], 6, 6, 0.56614,
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], {TRUTHKEY: 4, CAMKEY: 0})
    }
    return detections_test


@pytest.fixture
def tracks_test(timestamps):
    """Fixture for tracks in namedtuple format."""
    tracks_test = (
        Track(1, [1, 2, 3, 10, 11], list(timestamps[0:5]), meta={}),
        Track(2, [4, ], (timestamps[0],), meta={}),
        Track(3, [5, ], (timestamps[0],), meta={}),
        Track(4, [6, 7], timestamps[2:4], meta={}),
        Track(5, [8, 9, 12, 13], timestamps[3:7], meta={}),
    )
    return tracks_test


@pytest.fixture
def timestamps():
    """Fixture for timestamps as floats with microseconds."""
    timestamps = (     # Index
        1459516622.1,  # 0
        1459516622.2,  # 1
        1459516622.3,  # 2
        1459516623.0,  # 3
        1459516623.1,  # 4
        1459516623.3,  # 5
        1459516624.0,  # 6
    )
    return timestamps


@pytest.fixture()
def id_translator(detections_clean):
    """Fixture to translate ids to id format of :class:`DataWrapper`."""
    def translate_ids(dwi):
        """Generates a function that translates the ids to the format the instance expects.

        Arguments:
            dwi (:class:`DataWrapper`): A :class:`DataWrapper` instance to determine format

        Returns:
            A function that expects ids and returns a list of ids in the expected format.
        """
        def get_ids(*args):
            """Translates the ids to the format the :class:`DataWrapper` instance expects.

            Arguments:
                *args (int): integers with :obj:`Detection` ids

            Returns:
                list of int or str: A list with the ids in the expected format.
            """
            return [ids[i] for i in args]
        if isinstance(dwi, DataWrapperTracks):
            dwi = dwi.data
        if isinstance(dwi, DataWrapperPandas):
            values = detections_clean.id
        elif isinstance(dwi, DataWrapperBinary):
            values = detections_clean.generatedID
        else:
            values = detections_clean.generatedID
        ids = {key: val for key, val in zip(detections_clean.id.tolist(), values)}
        return get_ids
    return translate_ids


@pytest.fixture
def tracking_generation_seed():
    """Fixture for default seed for generation of random tracking data."""
    return 112


@pytest.fixture
def detections_simple_tracking(tracking_generation_seed):
    """Fixture for detections with basic tracking testing."""
    detections_dict = {
        "id": [],
        "timestamp": [],
        "camID": [],
        "xpos": [],
        "ypos": [],
        "zrotation": [],
        "beeID": [],
        "truthID": []
    }

    random.seed(tracking_generation_seed)
    truth_tracks = dict()
    detection_index, cam_id, detection_id, xpos, ypos, zrotation, bee_id, truth_id = [0] * 8
    while detection_index < 500:
        track_length = random.randint(1, 10)
        detection_index += track_length
        timestamps = random.sample(range(0, 11), track_length)
        timestamps.sort()
        truth_track = Track(id=truth_id, ids=[], timestamps=timestamps,
                            meta={DETKEY: []})
        for j in range(track_length):
            detection = Detection(id=detection_id, timestamp=timestamps[j],
                                  orientation=zrotation, beeId=list(int_id_to_binary(bee_id))[::-1],
                                  x=xpos, y=ypos, meta={})
            detections_dict["id"].append(detection.id)
            detections_dict["timestamp"].append(detection.timestamp)
            detections_dict["camID"].append(cam_id)
            detections_dict["xpos"].append(detection.x)
            detections_dict["ypos"].append(detection.y)
            detections_dict["zrotation"].append(detection.orientation)
            detections_dict["beeID"].append(detection.beeId)
            detections_dict["truthID"].append(truth_id)

            truth_track.ids.append(detection_id)
            truth_track.meta[DETKEY].append(detection)
            detection_id += 1
        truth_tracks[truth_track.id] = truth_track
        truth_id += 1
        bee_id += 1
        cam_id = random.randint(0, 3)
        xpos += 5
        ypos += 5
        zrotation += math.pi / 3
    return pd.DataFrame(detections_dict), truth_tracks


@pytest.fixture
def data_simple_tracking(detections_simple_tracking):
    """Fixture for DataWrapperTruth with basic tracking testing."""
    detections = detections_simple_tracking[0]
    detections_truth = detections.copy()
    detections_truth["decodedId"] = [binary_id_to_int(bee_id, endian='little')
                                     for bee_id in detections_truth["beeID"]]
    detections_truth["readability"] = DataWrapperTruthPandas.code_unknown
    return DataWrapperTruthPandas(detections, detections_truth, 1, meta_keys={'camID': CAMKEY})


@pytest.fixture
def simple_walker(data_simple_tracking):
    """Fixture for Simple Walker."""
    def dist_fun(tracks, detections_test):
        """Function to calculate distance between track and detection."""
        np_track = np.array([(track.meta[DETKEY][-1].x, track.meta[DETKEY][-1].y)
                             for track in tracks])
        np_test = np.array([(detection.x, detection.y) for detection in detections_test])
        return np.linalg.norm(np_track - np_test, axis=1)
    return SimpleWalker(data_simple_tracking, dist_fun, 1, 10)


def generate_random_walker():
    """Generate a new walker with random detection and track data."""
    # must have seeds that generate known problems
    must_have_seeds = [112, 308, 393]
    for seed in must_have_seeds:
        print("Last used seed: {}".format(seed))
        detections = detections_simple_tracking(seed)
        yield simple_walker(data_simple_tracking(detections)), detections
    while True:
        seed = random.randint(0, 2**10)
        print("Last used seed: {}".format(seed))
        detections = detections_simple_tracking(seed)
        yield simple_walker(data_simple_tracking(detections)), detections


@pytest.fixture
def validator(data_truth):
    """Fixture for Validator."""
    return Validator(data_truth)
