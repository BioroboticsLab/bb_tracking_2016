# -*- coding: utf-8 -*-
"""Adding tests to walker functions."""
# pylint:disable=protected-access,redefined-outer-name,too-many-locals
import copy
import numpy as np
import pytest
import six
from scipy.spatial.distance import euclidean
from bb_binary import binary_id_to_int
from bb_tracking.data import Track
from bb_tracking.data.constants import CAMKEY, DETKEY


def test_init_simple_walker(simple_walker, data_simple_tracking):
    """Test initializing the :class:`SimpleWalker` Class."""
    assert simple_walker.data == data_simple_tracking
    assert simple_walker.frame_diff == 1
    assert simple_walker.radius == 10


@pytest.fixture(params=["detections", "tracks"])
def frame_objects_data(simple_walker, request):
    """Fixture to test with different frame objects like detections and tracks."""
    time_index = 0
    cam_id = list(simple_walker.data.get_camids())[0]
    timestamps = simple_walker.data.get_timestamps(cam_id=cam_id)
    frame_objects = simple_walker.data.get_frame_objects(cam_id=cam_id,
                                                         timestamp=timestamps[time_index])
    if request.param == 'tracks':
        frame_objects = [Track(id=det.id, ids=[det.id], timestamps=[det.timestamp],
                               meta={DETKEY: [det]}) for det in frame_objects]
    return frame_objects, time_index, timestamps, cam_id


def test_calc_tracks():
    """This test was moved to ``test_tracking.py``."""
    assert True


def test_calc_initialize(simple_walker, frame_objects_data):
    """Test the initialization of the waiting list."""
    frame_objects, time_index, timestamps, _ = frame_objects_data
    new_waiting = simple_walker._calc_initialize(time_index, frame_objects, [])
    ids = set()
    # test basic setup like last update and track properties
    for i, (last_update, track) in enumerate(new_waiting):
        assert last_update == time_index
        assert len(track.ids) == 1
        assert len(track.timestamps) == 1
        assert track.timestamps[0] == timestamps[time_index]
        assert track.ids[0] not in ids
        ids.add(track.ids[0])
        assert track.ids[0] == frame_objects[i].id
        assert DETKEY in track.meta.keys()
        if isinstance(frame_objects[i], Track):
            assert track.meta[DETKEY] == frame_objects[i].meta[DETKEY]
        else:
            assert track.meta[DETKEY] == [frame_objects[i], ]

    # test that all frame_objects have started a new track
    assert set([det.id for det in frame_objects]) == ids

    # test with track prefix
    simple_walker.track_prefix = "test_"
    new_waiting = simple_walker._calc_initialize(time_index, frame_objects, [])
    for _, track in new_waiting:
        assert isinstance(track.id, six.string_types)
        assert "test_" in track.id

    # test other types
    with pytest.raises(TypeError) as excinfo:
        simple_walker._calc_initialize(time_index, [0, 1, 2], [])
    assert str(excinfo.value) == "Type {} not supported.".format(type(1))


def test_calc_close_tracks(simple_walker):
    """Test the closing of tracks in the waiting list."""
    diff = 2
    simple_walker.frame_diff = diff
    tracks = list(range(5))
    waiting_pattern = [[0, i] for i in tracks]

    # no change, no track closed
    waiting = copy.deepcopy(waiting_pattern)
    closed_tracks = []
    new_waiting = simple_walker._calc_close_tracks(diff, waiting, closed_tracks)
    assert waiting == waiting_pattern
    assert new_waiting == waiting_pattern
    assert not closed_tracks

    # empty list, all tracks closed
    waiting = copy.deepcopy(waiting_pattern)
    closed_tracks = []
    new_waiting = simple_walker._calc_close_tracks(diff + 1, waiting, closed_tracks)
    assert waiting == waiting_pattern
    assert not new_waiting
    assert closed_tracks == tracks


def test_calc_weight(simple_walker, detections_simple_tracking):
    """Test the calculation of weights."""
    def gen_new_track(track, index):
        """Generates a new Track with a single detection."""
        detection = track.meta[DETKEY][index]
        return Track(id=track.id, ids=[detection.id], timestamps=[detection.timestamp],
                     meta={DETKEY: [detection]})
    truth_tracks = detections_simple_tracking[1]
    track_before = None
    tracks_path, detections_test, expected_mask = [], [], []
    for track in truth_tracks.values():
        if track_before:
            tracks_path.append(gen_new_track(track_before, -1))
            detections_test.append(track.meta[DETKEY][0])
            expected_mask.append(False)
        for i in range(1, len(track.ids)):
            tracks_path.append(gen_new_track(track, -1))
            detections_test.append(track.meta[DETKEY][i])
            expected_mask.append(True)
        track_before = track
    expected_mask = np.array(expected_mask)
    calculated_weights = simple_walker.score_fun(tracks_path, detections_test)
    assert np.all(calculated_weights[expected_mask] == 0)
    assert np.all(calculated_weights[np.logical_not(expected_mask)] > 0)


def setup_for_assignment(simple_walker, object_type=DETKEY, time_idx=1):
    """Helper to generate setup for assignment tests."""
    cam_id = list(simple_walker.data.get_camids())[0]
    timestamps = simple_walker.data.get_timestamps(cam_id=cam_id)
    detections = simple_walker.data.get_frame_objects(cam_id=cam_id,
                                                      timestamp=timestamps[time_idx - 1])

    waiting = simple_walker._calc_initialize(0, detections, [])
    frame_objects = simple_walker.data.get_frame_objects(cam_id=cam_id,
                                                         timestamp=timestamps[time_idx])
    if object_type == 'tracks':
        frame_objects = [Track(id=det.id, ids=[det.id], timestamps=[det.timestamp],
                               meta={DETKEY: [det]}) for det in frame_objects]
    return time_idx, timestamps, frame_objects, waiting


@pytest.mark.parametrize("object_type", ["detections", "tracks"])
def test_calc_make_claims(simple_walker, detections_simple_tracking, object_type):
    """Test the setting of claims by tracks."""
    truth_tracks = detections_simple_tracking[1]
    # setup waiting list...
    tidx, timestamps, frame_objects, waiting = setup_for_assignment(simple_walker,
                                                                    object_type=object_type,
                                                                    time_idx=2)
    # add a Track that is not due in this round
    waiting.append([tidx + 1, Track(id=-1, ids=[], timestamps=[], meta={})])

    cam_id = 0
    # calculate cost matrix
    expected_cost_matrix = np.full((len(waiting), len(frame_objects)), simple_walker.max_weight)
    # get list of frame object Indices
    detection_indices = np.array([frame_object.id for frame_object in frame_objects])
    # expected distance for neighbors
    neighbor_distance = euclidean([0, 0], [5, 5])

    # verify cost matrix with truth data
    for row, (_, track) in enumerate(waiting):
        if track.id == -1:
            neighbors = None
        else:
            truth_id = binary_id_to_int(track.meta[DETKEY][0].beeId, endian='little')
            neighbors = simple_walker.data.get_neighbors(track.meta[DETKEY][-1], cam_id,
                                                         radius=simple_walker.radius,
                                                         timestamp=timestamps[tidx])
        if not neighbors:
            continue
        for neighbor in neighbors:
            col = np.where(detection_indices == neighbor.id)[0]
            if neighbor.id in truth_tracks[truth_id].ids:
                expected_cost_matrix[row, col] = 0
            else:
                expected_cost_matrix[row, col] = neighbor_distance

    cost_matrix = simple_walker._calc_make_claims(cam_id, tidx, timestamps[tidx],
                                                  frame_objects, waiting)
    assert set([0, neighbor_distance, simple_walker.max_weight]) == set(np.unique(cost_matrix))
    assert np.allclose(expected_cost_matrix, cost_matrix)

    if object_type == "detections":
        # no matching frame objects for ALL tracks in waiting list
        data = simple_walker.data
        data.detections.loc[data.detections[data.cols['timestamp']] >= timestamps[tidx],
                            data.cols['x']] += 1000
        tidx, timestamps, frame_objects, waiting = setup_for_assignment(simple_walker,
                                                                        object_type=object_type,
                                                                        time_idx=2)
        # idee: füge eine weitere detektion mit unnereeicbaren abstand hinzu.
        cost_matrix = simple_walker._calc_make_claims(cam_id, tidx, timestamps[tidx],
                                                      frame_objects, waiting)
        assert np.all(cost_matrix == simple_walker.max_weight)


def test_resolve_claims(simple_walker):
    """Test the resolving of claims with conflicts."""
    # having one best fit and a second best fit with best fit in next frame
    with pytest.raises(AssertionError):  # not resolved with this method!
        cost_matrix = np.array([[0, 5], [5, simple_walker.max_weight]])
        rows, cols = simple_walker._resolve_claims(cost_matrix)
        assert all(rows == [0, 1]) and all(cols == [0, 1])

    # set prune weight to be able to resolve this problem
    simple_walker.prune_weight = 4
    cost_matrix = np.array([[0, 5], [5, simple_walker.max_weight]])
    rows, cols = simple_walker._resolve_claims(cost_matrix)
    assert all(rows == [0, 1]) and all(cols == [0, 1])

    # no assignments
    cost_matrix = np.full((5, 4), simple_walker.max_weight)
    rows, cols = simple_walker._resolve_claims(cost_matrix)
    assert len(rows) == 4
    assert len(cols) == 4


@pytest.mark.parametrize("object_type", ["detections", "tracks"])
def test_calc_assign(simple_walker, detections_simple_tracking, object_type):
    """Test the assignment of frame objects to tracks."""
    truth_tracks = detections_simple_tracking[1]

    # test with empty frame objects
    assert ([], set()) == simple_walker._calc_assign(0, 0, None, [], [])

    # setup waiting list...
    time_idx, timestamps, frame_objects, waiting = setup_for_assignment(simple_walker, object_type)
    cam_id = 0

    if object_type == "tracks":
        for _, track in waiting:
            track.meta['DO_NOT_OVERWRITE'] = True

    expected_assigned = set()
    waiting_expected = copy.deepcopy(waiting)

    for waiting_idx, (_, track) in enumerate(waiting_expected):
        truth_id = binary_id_to_int(track.meta[DETKEY][0].beeId, endian='little')
        truth_track = truth_tracks[truth_id]
        timestamp = truth_track.timestamps[time_idx]
        detection = truth_track.meta[DETKEY][time_idx]
        detection.meta[CAMKEY] = cam_id
        if len(truth_track.ids) > 1 and timestamp == timestamps[time_idx]:
            expected_assigned.add(detection.id)
            waiting_expected[waiting_idx][0] = time_idx
            if object_type == "tracks":
                waiting_expected[waiting_idx][0] += 1
            track.ids.append(detection.id)
            track.timestamps.append(timestamp)
            track.meta[DETKEY].append(detection)

    # verify new waiting and assigned list with truth data
    waiting_new, assigned_new = simple_walker._calc_assign(cam_id, time_idx, timestamps[time_idx],
                                                           frame_objects, waiting)
    assert expected_assigned == assigned_new
    assert waiting_expected == waiting_new

    # test other types
    with pytest.raises(TypeError) as excinfo:
        simple_walker._calc_assign(cam_id, time_idx, timestamps[time_idx], [1, 2, 3], waiting)
    assert str(excinfo.value) == "Type {} not supported.".format(type(1))
