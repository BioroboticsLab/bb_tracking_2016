# -*- coding: utf-8 -*-
"""Adding tests to DataWrapper implementations."""
# pylint:disable=protected-access,redefined-outer-name
import math
from datetime import datetime
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.util.testing import assert_frame_equal
import pytest
import pytz
from bb_tracking.data import DataWrapper, DataWrapperTruth, DataWrapperPandas, \
    DataWrapperTruthPandas, DataWrapperBinary, DataWrapperTruthBinary, DataWrapperTracks, \
    Detection, Track
from bb_tracking.data.constants import CAMKEY, DETKEY, TRUTHKEY
from test.conftest import cmp_tracks


@pytest.fixture
def detections_track_duplicates(detections):
    """Fixture for detections with added track duplicates."""
    # add problematic detection
    new_row = detections[detections.id == 17].copy()
    new_row.reset_index(drop=True, inplace=True)
    new_row.loc[0, 'id'] = 18
    new_row.loc[0, ['xpos', 'ypos']] = 4.6
    new_detections = detections.append(new_row, ignore_index=True)
    new_detections.loc[[did in [13, 16, 17] for did in new_detections.id], ['xpos', 'ypos']] = 3.5
    return new_detections


def test_datawrapper_abstract():
    """Tests the Abstract DataWrapper class."""
    data = DataWrapper()
    with pytest.raises(NotImplementedError):
        data.get_camids()

    with pytest.raises(NotImplementedError):
        data.get_detection("detection_id")

    with pytest.raises(NotImplementedError):
        data.get_detections("detection_ids")

    with pytest.raises(NotImplementedError):
        data.get_frame_objects()

    with pytest.raises(NotImplementedError):
        data.get_neighbors("frame_object", "cam_id")

    with pytest.raises(NotImplementedError):
        data.get_timestamps()


def test_datawrappertruth_abstract():
    """Tests the Abstract DataWrapperTruth Class."""
    data = DataWrapperTruth()
    with pytest.raises(NotImplementedError):
        data.get_all_detection_ids()

    with pytest.raises(NotImplementedError):
        data.get_truth_track("truth_id")

    with pytest.raises(NotImplementedError):
        data.get_truth_tracks()

    with pytest.raises(NotImplementedError):
        data.get_truthid("frame_object")

    with pytest.raises(NotImplementedError):
        data.get_truthids()


def test_init_datawrapper(detections, detections_clean):
    """Test initializing the DataWrapper Class."""
    data = DataWrapperPandas(detections, duplicates_radius=1)
    detections_data = make_unique_order(data, data.detections.copy())
    detections_data.drop('meta', 1, inplace=True)

    detections_clean = make_unique_order(data, detections_clean)
    assert_frame_equal(detections_data, detections_clean)

    # test replacing column names
    test_replace = 'test_col_replacement'
    data = DataWrapperPandas(detections, cols={'readability': test_replace})
    assert data.cols['readability'] == test_replace

    # test meta keys column
    meta_keys = {'xpos': 'xpos', 'ypos': 'ypos'}
    data = DataWrapperPandas(detections, meta_keys=meta_keys)
    detection = data.get_detection(1)
    assert set(detection.meta.keys()) == set(meta_keys)
    assert detection.x == detection.meta['xpos']
    assert detection.y == detection.meta['ypos']


def test_init_binary(detections_binary):
    """Test initializing the DataWrapperBinary Class."""
    data = DataWrapperBinary(detections_binary)
    detection = data.get_detection('f3d0c0')
    assert detection.meta == {CAMKEY: 0}

    meta_keys = {'xpos': 'x', 'ypos': 'y'}
    data = DataWrapperBinary(detections_binary, meta_keys=meta_keys)
    detection = data.get_detection('f3d0c0')
    assert set(detection.meta.keys()) == set(meta_keys.values()) | set([CAMKEY])
    assert detection.x == detection.meta['x']
    assert detection.y == detection.meta['y']


def test_init_truth(detections, truth, truth_clean):
    """Test initializing the DataWrapperTruthPandas Class."""
    data = DataWrapperTruthPandas(detections, truth, 1)
    detections_data = make_unique_order(data, data.detections.copy())
    detections_data.drop('meta', 1, inplace=True)

    truth_clean = make_unique_order(data, truth_clean)
    assert_frame_equal(detections_data, truth_clean)

    # increase radius, expect warning
    with pytest.raises(UserWarning):
        DataWrapperTruthPandas(detections, truth, 10)


def test_init_truth_binary(detections_binary, truth_binary, truth_clean):
    """Test initializing the DataWrapperTruthBinary Class."""
    data = DataWrapperTruthBinary(detections_binary, truth_binary, 1)
    truth_ids = set(truth_clean.truthID.unique()) - set([data.fp_id])
    assert truth_ids == set(data.tracks.keys())

    cam_ids = set(truth_clean.camID.unique())
    assert cam_ids == set(data.cam_tracks.keys())

    # increase radius, expect warning
    with pytest.raises(UserWarning):
        DataWrapperTruthBinary(detections_binary, truth_binary, 10)


def test_init_tracks(data_binary, tracks_test, id_translator):
    """Test initializing the DataWrapperTracks Class."""
    get_ids = id_translator(data_binary)
    tracks = [Track(id=track.id, ids=get_ids(*track.ids),
                    timestamps=[time.to_pydatetime().replace(tzinfo=pytz.utc)
                                for time in track.timestamps],
                    meta={DETKEY: data_binary.get_detections(get_ids(*track.ids))})
              for track in tracks_test]
    data = DataWrapperTracks(tracks, data_binary.cam_timestamps, data=data_binary)
    assert data.cam_ids == data_binary.cam_ids
    assert data.timestamps == data_binary.timestamps
    assert len(data.tracks) == np.sum([len(ftracks) for ftracks in data.frame_track_end.values()])
    assert len(data.tracks) == np.sum([len(ftracks) for ftracks in data.frame_track_start.values()])


def test_merge_truth_radius_problem(detections_track_duplicates, truth):
    r"""Tests that duplicates identified by truth data are considered.

    Example: a and b are detections with distance 2 (> duplicates_radius=1)
    and c is the truth data that is between the two detections and within distance 1
    (<= duplicates_radius=1). So a and b will be added to the same track but are essentially
    duplicates that need to be merged or handled otherwise.
       a - 2 - b
        \1   1/
           c
    """
    with pytest.raises(UserWarning):
        DataWrapperTruthPandas(detections_track_duplicates, truth, 1)


def test_get_timestamps(data, timestamps):
    """Test the extraction of unique timestamps in order."""
    timestamps_data = data.get_timestamps()
    assert all(pd.to_datetime(timestamps_data) == pd.to_datetime(timestamps))

    timestamps_cam0 = [timestamps[i] for i in [0, 2, 3, 4]]
    timestamps_data = data.get_timestamps(cam_id=0)
    assert all(pd.to_datetime(timestamps_data) == pd.to_datetime(timestamps_cam0))

    timestamps_cam2 = [timestamps[i] for i in [1, 3, 4, 5, 6]]
    timestamps_data = data.get_timestamps(cam_id=2)
    assert all(pd.to_datetime(timestamps_data) == pd.to_datetime(timestamps_cam2))


def test_get_camids(data, data_binary, id_translator):
    """Test the extraction of unique camera ids."""
    assert data.get_camids() == set([0, 2])

    if not isinstance(data, DataWrapperTracks) or data.data is not None:
        data_provider = data
    else:
        data_provider = data_binary

    get_ids = id_translator(data_provider)
    ids = list(get_ids(1, 2, 3))

    detection = data_provider.get_detection(ids[0])
    # test with detection
    assert data.get_camids(frame_object=detection) == set([0])

    # test with track
    if not isinstance(data, DataWrapperTracks) or data.data is not None:
        track = Track(id=0, ids=ids, timestamps=[0, 1, 2], meta={})
        assert data.get_camids(frame_object=track) == set([0, 2])

    # test with track with detections meta data
    track = Track(id=0, ids=ids, timestamps=[0, 1, 2],
                  meta={DETKEY: [data_provider.get_detection(det_id) for det_id in ids]})
    assert data.get_camids(frame_object=track) == set([0, 2])

    # test other types
    with pytest.raises(TypeError) as excinfo:
        data.get_camids(frame_object=0)
    assert str(excinfo.value) == "Type {} not supported.".format(type(0))


def test_get_detection(data, detections_test):
    """Test the return of a specific detection."""
    detection_test = detections_test['f3d0c0']
    if isinstance(data, (DataWrapperBinary, DataWrapperTracks)):
        key = 'f3d0c0'
    else:
        key = detection_test.id
    meta = True if isinstance(data, DataWrapperTruthBinary) else False

    if isinstance(data, DataWrapperTracks) and data.data is None:
        with pytest.raises(NotImplementedError):
            data.get_detection(key)
        return
    detection_data = data.get_detection(key)
    cmp_detections([detection_data], [detection_test], meta=meta)


def test_get_detections(data, detections_test):
    """Test the return of a list of detections."""
    detection_ids = detections_test.keys()
    test_detections = [detections_test[test_id] for test_id in detection_ids]
    test_ids = detection_ids if isinstance(data, (DataWrapperBinary, DataWrapperTracks))\
        else [detections_test[test_id].id for test_id in detection_ids]
    meta = True if isinstance(data, DataWrapperTruthBinary) else False

    if isinstance(data, DataWrapperTracks) and data.data is None:
        with pytest.raises(NotImplementedError):
            data.get_detections(test_ids)
        return
    detections_data = data.get_detections(test_ids)
    cmp_detections(detections_data, test_detections, meta=meta)


def test_get_track(data_tracks, tracks_test, id_translator):
    """Test the return of a specific track."""
    get_ids = id_translator(data_tracks)
    expected_track = tracks_test[0]
    track = data_tracks.get_track(1)
    assert get_ids(*expected_track.ids) == track.ids


def test_get_tracks(data_tracks, tracks_test, id_translator):
    """Test the return of a list of tracks."""
    get_ids = id_translator(data_tracks)
    tracks = data_tracks.get_tracks([track.id for track in tracks_test])
    for expected_track, track in zip(tracks_test, tracks):
        assert get_ids(*expected_track.ids) == track.ids


def test_get_frame_objects(data, detections_test, timestamps):
    """Test the return of a list of detections from a frame."""
    if isinstance(data, DataWrapperTracks):
        return
    cam_id = 0
    detections_test = [detections_test[test_id] for test_id in ['f1d0c0', 'f1d1c0', 'f1d2c0']]
    timestamp = timestamps[0]
    if isinstance(data, DataWrapperBinary):
        timestamp = data.cam_timestamps[cam_id][0]
    detections_data = data.get_frame_objects(cam_id=cam_id, timestamp=timestamp)
    cmp_detections(detections_data, detections_test)


def test_get_frame_objects_tracks(data_tracks):
    """Test the return of a list of tracks from a frame."""
    cam_id = 0
    timestamp = data_tracks.cam_timestamps[cam_id][0]
    tracks = data_tracks.get_frame_objects(cam_id=cam_id, timestamp=timestamp)
    assert set([2, 3]) == set([track.id for track in tracks])

    timestamp = data_tracks.cam_timestamps[cam_id][2]
    tracks = data_tracks.get_frame_objects(cam_id=cam_id, timestamp=timestamp)
    assert set([4]) == set([track.id for track in tracks])


def test_get_duplicate_ids(detections):
    """Test the calculation of duplicate entries via their distance."""
    data = DataWrapperPandas(detections)
    left, right = data._get_duplicate_ids(detections, 1)
    left.sort()
    right.sort()
    assert len(left) == 3
    assert len(right) == 3
    assert left == [13, 13, 16]
    assert right == [16, 17, 17]


def test_merge_entries(data_pandas):
    """Test the merging of entries in a dataframe given their indexes."""
    df_dict = {data_pandas.cols[CAMKEY]: range(5),
               data_pandas.cols['timestamp']: range(5),
               data_pandas.cols['x']: [1] * 5,
               data_pandas.cols['y']: [2] * 5,
               data_pandas.cols['beeId']: [[i] * data_pandas.beeId_digits for i in range(5)],
               data_pandas.cols['localizer']: [1] * 5}

    df_merge = pd.DataFrame(df_dict, index=range(5))
    left = [1, 2]
    right = [3, 4]
    data_pandas._merge_entries(df_merge, left, right)
    assert all([0, 1, 2] == df_merge.index.values)


def test_merge_ids(data_pandas):
    """Test the merging of the bit frequency distribution of two ids."""
    id_dist = [1., 0., 0.3]
    with pytest.raises(AssertionError):
        data_pandas._merge_ids(id_dist, id_dist)
    data_pandas.beeId_digits = 3

    assert_allclose(id_dist, data_pandas._merge_ids(id_dist, id_dist))
    assert_allclose([0.75, 0.25, 0.45], data_pandas._merge_ids(id_dist, [0.5, 0.5, 0.6]))
    assert_allclose([0.5, 0.5, 0.5], data_pandas._merge_ids([1., 0., 1.], [0., 1., 0.]))


def test_get_neighbors(data, detections, timestamps):
    """Test the calculation of nearest neighbors."""
    if isinstance(data, DataWrapperTracks):
        return
    ids = detections['generatedID'] if isinstance(data, DataWrapperBinary) else detections['id']
    # ignore false positives
    ids = ids[0:-2]
    detections = data.get_detections(ids)
    timestamps = data.timestamps if isinstance(data, DataWrapperBinary) else timestamps
    # expect empty result
    assert len(data.get_neighbors(detections[0], 0, radius=0.5)) == 0

    # timestamp without matching frame
    assert len(data.get_neighbors(detections[0], 0, timestamp=timestamps[5])) == 0

    # expect all detections from first frame except the one we are checking
    neighbors = data.get_neighbors(detections[0], 0)
    assert set((ids[3], ids[4])) == set([det.id for det in neighbors])

    distance = math.sqrt(2 * 3**2)
    # expect no detection (radius is exclusive)
    assert len(data.get_neighbors(detections[0], 0, radius=distance)) == 0
    # expect exactly one detection
    neighbors = data.get_neighbors(detections[0], 0, radius=distance + 0.1)
    assert set((ids[3], )) == set([det.id for det in neighbors])

    # different timestamp
    neighbors = data.get_neighbors(detections[0], 0, timestamp=timestamps[3])
    assert set((ids[6], ids[9])) == set([det.id for det in neighbors])

    # test with Track instead of detection id
    track = Track(id=0, ids=[ids[0]], timestamps=[timestamps[3]], meta={})
    neighbors = data.get_neighbors(track, 0, timestamp=timestamps[3])
    assert set((ids[6], ids[9])) == set([det.id for det in neighbors])

    # test other types
    with pytest.raises(TypeError) as excinfo:
        data.get_neighbors(ids[0], 0)
    assert str(excinfo.value) == "Type {} not supported.".format(type(ids[0]))


def test_get_neighbors_tracks(data_tracks):
    """Test the calculation of nearest neighbors for tracks."""
    cam_id = 0
    timestamp = data_tracks.timestamps[0]
    test_track = Track(id=2, ids=[4, ], timestamps=[timestamp, ], meta={})
    tracks = data_tracks.get_neighbors(test_track, cam_id)
    assert set([1, 3]) == set([track.id for track in tracks])

    timestamp = data_tracks.timestamps[2]
    tracks = data_tracks.get_neighbors(test_track, cam_id, timestamp=timestamp)
    assert set([4]) == set([track.id for track in tracks])

    with pytest.raises(TypeError) as excinfo:
        data_tracks.get_neighbors(0, cam_id)
    assert str(excinfo.value) == "Type {} not supported.".format(type(0))


def test_get_all_detection_ids(data_truth, id_translator):
    """Test the extraction of all detection ids."""
    get_ids = id_translator(data_truth)
    positives, false_positives = data_truth.get_all_detection_ids()
    assert positives == set(get_ids(*range(1, 14)))
    assert false_positives == set(get_ids(14, 15))


def test_get_truth_track_single(data_truth, tracks_test, detections_clean):
    """Test the extraction of a track via its truthId."""
    # Track is not available
    assert data_truth.get_truth_track(9999) is None

    if isinstance(data_truth, (DataWrapperBinary, DataWrapperTracks)):
        tracks_test = [Track(id=track.id, timestamps=track.timestamps, meta=track.meta,
                             ids=[detections_clean.loc[detections_clean.id == did,
                                                       'generatedID'].values[0]
                                  for did in track.ids])
                       for track in tracks_test]
    # available tracks
    for track in tracks_test:
        cmp_tracks(data_truth.get_truth_track(track.id), track)

    # restrict on camera id
    for track in tracks_test:
        if track.id == 1:
            track.ids.pop(1)
            track.timestamps.pop(1)
        cam_id = 2 if track.id == 5 else 0
        cmp_tracks(data_truth.get_truth_track(track.id, cam_id=cam_id), track)


def test_get_truth_tracks(data_truth, tracks_test, detections_clean):
    """Test the extraction of all tracks."""
    if isinstance(data_truth, (DataWrapperBinary, DataWrapperTracks)):
        tracks_test = [Track(id=track.id, timestamps=track.timestamps, meta=track.meta,
                             ids=[detections_clean.loc[detections_clean.id == did,
                                                       'generatedID'].values[0]
                                  for did in track.ids])
                       for track in tracks_test]
    tracks = list(data_truth.get_truth_tracks())
    assert len(tracks) == len(tracks_test)
    for track1, track2 in zip(tracks, tracks_test):
        cmp_tracks(track1, track2)

    cam_2_track_ids = [[2], [8, 9, 12, 13]]
    tracks_cam_2 = list(data_truth.get_truth_tracks(cam_id=2))
    assert len(tracks_cam_2) == 2
    if isinstance(data_truth, (DataWrapperBinary, DataWrapperTracks)):
        cam_2_track_ids = [[detections_clean.loc[detections_clean.id == did,
                                                 'generatedID'].values[0]
                            for did in ids] for ids in cam_2_track_ids]
    for track, ids in zip(tracks_cam_2, cam_2_track_ids):
        assert set(track.ids) == set(ids)


def test_get_truthid(data_truth, truth_clean, id_translator):
    """Test the access to truthIds via function."""
    get_ids = id_translator(data_truth)
    false_positives = get_ids(14, 15)
    false_positives.append(99999)
    for detection in truth_clean.itertuples(index=False):
        det = Detection(id=get_ids(detection.id)[0], x=detection.xpos, y=detection.ypos,
                        timestamp=detection.timestamp, orientation=detection.zrotation,
                        beeId=detection.beeID, meta={})
        track = Track(id=0, ids=[det.id], timestamps=[det.timestamp], meta={DETKEY: [det]})
        track_meta = Track(id=0, ids=[det.id], timestamps=[det.timestamp],
                           meta={DETKEY: [det], TRUTHKEY: detection.truthID})
        if det.id in false_positives:
            assert data_truth.get_truthid(det.id) == data_truth.fp_id
            assert data_truth.get_truthid(det) == data_truth.fp_id
            assert data_truth.get_truthid(track) == data_truth.fp_id
            assert data_truth.get_truthid(track_meta) == data_truth.fp_id
        else:
            assert detection.truthID == data_truth.get_truthid(det.id)
            assert detection.truthID == data_truth.get_truthid(det)
            assert detection.truthID == data_truth.get_truthid(track)
            assert detection.truthID == data_truth.get_truthid(track_meta)


def test_get_truthids(data_truth, id_translator):
    """Test the access to multiple truthIds via function."""

    get_ids = id_translator(data_truth)
    detection = data_truth.get_detection(get_ids(1)[0])
    track = Track(id=0, ids=get_ids(1, 2, 3), timestamps=[0, 1, 2], meta={})
    track2 = Track(id=0, ids=get_ids(1, 2, 3, 4, 14), timestamps=[0, 1, 2, 3, 4], meta={})

    # test all
    truth_ids = data_truth.get_truthids()
    assert truth_ids == set(range(1, 6)) | set([-1])

    # test cam 0
    truth_ids = data_truth.get_truthids(cam_id=0)
    assert truth_ids == set(range(1, 5)) | set([-1])

    # test cam 2
    truth_ids = data_truth.get_truthids(cam_id=2)
    assert truth_ids == set([1, 5]) | set([-1])

    # test detection
    truth_ids = data_truth.get_truthids(frame_object=detection)
    assert truth_ids == set([1])

    # test track
    truth_ids = data_truth.get_truthids(frame_object=track)
    assert truth_ids == set([1])

    # test track with multiple truthIds
    truth_ids = data_truth.get_truthids(frame_object=track2)
    assert truth_ids == set([-1, 1, 2])

    # test frame object AND cam
    with pytest.raises(ValueError) as excinfo:
        data_truth.get_truthids(frame_object=track, cam_id=0)
    assert str(excinfo.value) == "You can not use frame_object and cam_id together."

    # test other types
    with pytest.raises(TypeError) as excinfo:
        data_truth.get_truthids(frame_object=0)
    assert str(excinfo.value) == "Type {} not supported.".format(type(0))


def cmp_detections(dets1, dets2, meta=False):
    """Compares two lists of detections.

    Arguments:
        dets1 (iterable): iterable with expected :obj:`Detection`s
        dets2 (iterable): iterable with :obj:`Detection`s to test

    Keyword Arguments
        meta (bool): flag to indicate whether meta data should be compared or not
    """
    assert len(dets1) == len(dets2)
    for det1, det2 in zip(dets1, dets2):
        for key in Detection._fields:
            val1 = getattr(det1, key)
            val2 = getattr(det2, key)
            # do not compare ids because different DataWrappers have their own system
            if key == 'id':
                continue
            elif isinstance(val1, datetime):
                # we do not care about timezone offsets in tests
                val1 = val1.replace(tzinfo=val2.tzinfo)
                assert val1 == val2, "{}: {}, {}".format(key, val1, val2)
            elif isinstance(val1, float) or isinstance(val2, float):
                assert np.allclose(val1, val2), "{}: {}, {}".format(key, val1, val2)
            elif key != 'meta' or meta:
                assert val1 == val2, "{}: {}, {}".format(key, val1, val2)


def make_unique_order(data, frame):
    """Sorts and orders DataFrame to be able to compare the contents.

    Arguments:
        data (DataWrapper): :obj:`DataWrapper` with information on columns.
        frame (pd.DataFrame): Pandas DataFrame with data that should be sorted.

    Returns:
        pd.DataFrame: sorted Pandas DataFrame.
    """
    frame = frame.sort_values(data.cols['id'], axis=0)
    frame = frame.sort_index(axis=1)
    return frame
