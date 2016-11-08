# -*- coding: utf-8 -*-
"""
This implementations of :class:`.DataWrapper` are using dictionaries with shared data objects as
backend. Therefore they are tuned for fast performance and **not** for data analysis on the
detection data.

The :class:`DataWrapperBinary` implementations are only compatible with *bb_binary* data.
If have other data formats either try to `convert
<http://bb-binary.readthedocs.io/en/latest/api/converting.html>`_ them
or use :class:`.DataWrapperPandas`.
"""
import numpy as np
from scipy.spatial import cKDTree
from bb_binary import to_datetime
from .constants import CAMKEY, DETKEY, FRAMEIDXKEY, TRUTHKEY
from .datastructures import Detection, Track
from .datawrapper import DataWrapper, DataWrapperTruth


class DataWrapperBinary(DataWrapper):
    """Class for fast access to detections via *bb_binary* :class:`Repository`.

    This class is recommended for performance reasons and only compatible to the *bb_binary* format.
    It uses dictionaries for fast lookup and cheap memory because the :obj:`.Detection` objects are
    shared between the different dictionaries.
    """
    cam_ids = None
    """:obj:`list` of int: sorted list with all available camera ids"""
    cam_timestamps = None
    """:obj:`dict`: ``{cam_id: sorted list of timestamps}`` mapping"""
    detections_dict = None
    """:obj:`dict`: ``{detection_id: detection}`` mapping for :obj:`.Detection`"""
    frame_detections = None
    """:obj:`dict`: ``{(cam_id, timestamp): detection}`` mapping for :obj:`.Detection`"""
    frame_trees = None
    """:obj:`dict`: ``{(cam_id, timestamp): KDTree}`` mapping"""
    timestamps = None
    """:obj:`list` of timestamp: sorted list with all available timestamps"""

    def __init__(self, repository, meta_keys=None, **kwargs):
        """Necessary initialization to organize detection data.

        Arguments:
            repository (:class:`bb_binary.Repository`): *bb_binary* Repository with detections

        Keyword Arguments:
            meta_keys (Optional :obj:`dict`): ``{detecion_key: meta_key}`` mapping that is added
                as meta field in detections
            **kwargs (:obj:`dict`): keyword arguments for :func:`Repository.iter_frames()`
        """
        # convert detections to python objects and create dictionaries for fast lookup
        self.detections_dict = dict()
        self.frame_detections = dict()
        self.frame_trees = dict()
        # use local variables because we need to sort the data later
        cam_ids = set()
        timestamps = set()
        cam_timestamps = dict()
        last_fc_id = None
        meta_keys = meta_keys or dict()
        for frame, frame_container in repository.iter_frames(**kwargs):
            cam_id = frame_container.camId
            # a different frame containers might have data from a different camera
            if frame_container.id != last_fc_id:
                last_fc_id = frame_container.id
                if cam_id not in cam_ids:
                    cam_ids.add(cam_id)
                    cam_timestamps[cam_id] = set()
                assert frame.detectionsUnion.which() == 'detectionsDP',\
                    "Only implemented for union type 'detectionsDP'!"

            # each frame has a different timestamp
            timestamp = to_datetime(frame.timestamp)
            cam_timestamps[cam_id].add(timestamp)
            timestamps.add(timestamp)
            # now iterate through detections
            self._iterate_detections(frame, cam_id, meta_keys)

        self.cam_ids = list(sorted(cam_ids))
        self.timestamps = list(sorted(timestamps))
        self.cam_timestamps = {cam_id: list(sorted(tstamps))
                               for cam_id, tstamps in cam_timestamps.items()}
        assert len(self.detections_dict) > 0, "Repository is empty."

    def _iterate_detections(self, frame, cam_id, meta_keys):
        """Helper to iterate through detections and extract information.

        This helper will iterate through all detections of a frame, create the :obj:`.Detection`
        :obj:`namedtuple` and :obj:`cKDTree` for each frame.

        Arguments:
            frame (Frame): bb_binary Frame object
            cam_id (int): the id of the camera this frame belongs to
            meta_keys (Optional :obj:`dict`): ``{detecion_key: meta_key}``mapping that is added
                as meta field in detections
        """
        xy_cols = list()
        timestamp = to_datetime(frame.timestamp)
        self.frame_detections[(cam_id, timestamp)] = list()
        # iterate through detections and make some data cleaning
        for detection in frame.detectionsUnion.detectionsDP:
            detection_id = 'f{}d{}c{}'.format(frame.id, detection.idx, cam_id)
            detection_tuple = Detection(
                id=detection_id, timestamp=timestamp, x=detection.xpos, y=detection.ypos,
                orientation=0 if np.isinf(detection.zRotation) else detection.zRotation,
                beeId=[x / 255. for x in detection.decodedId],
                meta={mkey: getattr(detection, dkey) for dkey, mkey in meta_keys.items()})
            detection_tuple.meta[CAMKEY] = cam_id
            self.detections_dict[detection_id] = detection_tuple
            self.frame_detections[(cam_id, timestamp)].append(detection_tuple)
            xy_cols.append((detection.xpos, detection.ypos))
        self.frame_trees[(cam_id, timestamp)] = cKDTree(xy_cols)

    def get_camids(self, frame_object=None):
        if frame_object is None:
            cam_ids = self.cam_ids
        elif isinstance(frame_object, Track):
            if DETKEY in frame_object.meta.keys():
                detections = frame_object.meta[DETKEY]
            else:
                detections = self.get_detections(frame_object.ids)
            cam_ids = [detection.meta[CAMKEY] for detection in detections]
        elif isinstance(frame_object, Detection):
            cam_ids = [frame_object.meta[CAMKEY], ]
        else:
            raise TypeError("Type {0} not supported.".format(type(frame_object)))
        return frozenset(cam_ids)

    def get_detection(self, detection_id):
        return self.detections_dict[detection_id]

    def get_detections(self, detection_ids):
        return [self.detections_dict[d_id] for d_id in detection_ids]

    def get_frame_objects(self, cam_id=None, timestamp=None):
        cam_id = cam_id or self.cam_ids[0]
        timestamp = timestamp or self.cam_timestamps[cam_id][0]
        return self.frame_detections[(cam_id, timestamp)]

    def get_neighbors(self, frame_object, cam_id, radius=10, timestamp=None):
        if isinstance(frame_object, Track):
            detection = self.detections_dict[frame_object.ids[-1]]
        elif isinstance(frame_object, Detection):
            detection = frame_object
        else:
            raise TypeError("Type {0} not supported.".format(type(frame_object)))
        # determine search parameters
        timestamp = timestamp or detection.timestamp
        frame_key = (cam_id, timestamp)

        # use spatial tree for efficient neighborhood search
        if frame_key not in self.frame_trees:
            return []
        tree = self.frame_trees[frame_key]
        indices = tree.query_ball_point((detection.x, detection.y), radius)

        # translate tree Indices in detection ids and remove search item
        detections = self.frame_detections[frame_key]
        found = [detections[did] for did in indices]
        if timestamp == detection.timestamp:
            found = [det for det in found if det.id != detection.id]
        return found

    def get_timestamps(self, cam_id=None):
        if cam_id is not None:
            return self.cam_timestamps[cam_id]
        return self.timestamps


class DataWrapperTruthBinary(DataWrapperBinary, DataWrapperTruth):
    """Class for fast access to truth data via *bb_binary* :class:`Repository`.

    This class is recommended for performance reasons and only compatible to the *bb_binary* format.
    It uses dictionaries for fast lookup and cheap memory because the :obj:`.Detection` objects are
    shared between the different dictionaries.

    Note:
        The generation and access to the objects is not as fast as in :class:`DataWrapperBinary`.
        This is no problem, because the truth data is most likely not as big as real datasets.
    """

    cam_false_positives = None
    """:obj:`dict`: ``{cam_id: set of detection ids}`` mapping"""
    cam_tracks = None
    """:obj:`dict`: ``{cam_id: {truth_id: Track}}`` mapping for :obj:`.Track`"""
    false_positives = None
    """:obj:`set` of ids: Ids of all the detections **without** associated truth id"""
    positives = None
    """:obj:`set` of ids: Ids of all the detections **with** associated truth id"""
    tracks = None
    """:obj:`dict`: ``{truth_id: Track}`` mapping for :obj:`.Track`"""

    def __init__(self, repo_detections, repo_truth, radius, meta_keys=None, **kwargs):
        """Necessary initialization to organize detection data.

        Arguments:
            repo_detections (:class:`bb_binary.Repository`): Repository with detections
            repo_truth (:class:`bb_binary.Repository`): Repository with truth data
            radius (int): merge detections and truth via distance

        Keyword Arguments:
            meta_keys (Optional :obj:`dict`): ``{detecion_key: meta_key}`` mapping that is added
                as meta field in :obj:`.Detection` objects
            **kwargs (:obj:`dict`): keyword arguments for :func:`Repository.iter_frames()`
        """

        super(DataWrapperTruthBinary, self).__init__(repo_detections, meta_keys=meta_keys, **kwargs)
        # generate truth tracks
        self.tracks = dict()
        self.cam_tracks = {cam_id: dict() for cam_id in self.cam_ids}
        for frame, frame_container in repo_truth.iter_frames(**kwargs):
            assert frame.detectionsUnion.which() == 'detectionsTruth',\
                "Only implemented for union type 'detectionsTruth'!"
            cam_id = frame_container.camId
            frame_truth_ids, xy_cols = self._init_truth_tracks(cam_id, frame)
            self._match_truth_with_pipeline(cam_id, frame, xy_cols, frame_truth_ids, radius)
        self._sort_track_values()
        self._calc_tp_fp()
        assert len(self.positives) > 0, "No matched detections!"

    def _init_truth_tracks(self, cam_id, frame):
        """Initialize truth tracks and extract x- and y-positions from detections."""
        xy_cols = list()
        frame_truth_ids = list()
        for truth_detection in frame.detectionsUnion.detectionsTruth:
            # remove truth detections where tag was not visible
            if truth_detection.readability == 'none':
                continue
            truth_id = truth_detection.decodedId
            frame_truth_ids.append(truth_id)
            xy_cols.append([truth_detection.xpos, truth_detection.ypos])
            # initialize track if it does not exist
            if truth_id not in self.tracks.keys():
                self.tracks[truth_id] = Track(id=truth_id, timestamps=[], ids=[],
                                              meta={DETKEY: [], FRAMEIDXKEY: []})
            if truth_id not in self.cam_tracks[cam_id]:
                self.cam_tracks[cam_id][truth_id] = Track(id=truth_id, timestamps=[], ids=[],
                                                          meta={DETKEY: [], FRAMEIDXKEY: []})
        return frame_truth_ids, xy_cols

    def _match_truth_with_pipeline(self, cam_id, frame, xy_cols, frame_truth_ids, radius):
        """Determine matching pairs of truth data and repository output."""
        # pylint:disable=too-many-arguments
        frame_key = (cam_id, to_datetime(frame.timestamp))
        tree = self.frame_trees[frame_key]
        indices = tree.query_ball_tree(cKDTree(xy_cols), radius)
        for i, idx in enumerate(indices):
            if len(idx) == 0:
                continue
            elif len(idx) == 1:
                truth_id = frame_truth_ids[idx[0]]
                detection = self.frame_detections[frame_key][i]
                assert TRUTHKEY not in detection.meta.keys(), \
                    "Do not assign {} twice.".format(TRUTHKEY)
                detection.meta[TRUTHKEY] = truth_id
                self.tracks[truth_id].ids.append(detection.id)
                self.tracks[truth_id].timestamps.append(detection.timestamp)
                self.tracks[truth_id].meta[DETKEY].append(detection)
                self.tracks[truth_id].meta[FRAMEIDXKEY].append(frame.frameIdx)

                self.cam_tracks[cam_id][truth_id].ids.append(detection.id)
                self.cam_tracks[cam_id][truth_id].timestamps.append(detection.timestamp)
                self.cam_tracks[cam_id][truth_id].meta[DETKEY].append(detection)
                self.cam_tracks[cam_id][truth_id].meta[FRAMEIDXKEY].append(frame.frameIdx)
            elif len(idx) > 1:
                raise UserWarning('Truth Data has detections in each others radius.')
            else:  # pragma: no cover
                Exception()

    def _sort_track_values(self):
        """Makes sure that ids, timestamps and meta data in tracks are all in the correct order."""
        del_ids = []
        for tid, track in self.tracks.items():
            if len(track.ids) == 0:
                del_ids.append(tid)
                continue
            tstamps, ids, dets, frame_idx = zip(*sorted(zip(track.timestamps, track.ids,
                                                            track.meta[DETKEY],
                                                            track.meta[FRAMEIDXKEY])))
            self.tracks[tid] = Track(id=tid, timestamps=tstamps, ids=ids,
                                     meta={DETKEY: dets, FRAMEIDXKEY: frame_idx})
        # delete empty tracks after iterating over them!
        for did in del_ids:
            del self.tracks[did]

        del_ids = []
        for cam_id in self.cam_ids:
            for tid, track in self.cam_tracks[cam_id].items():
                if len(track.ids) == 0:
                    del_ids.append((cam_id, tid))
                    continue
                tstamps, ids, dets, frame_idx = zip(*sorted(zip(track.timestamps, track.ids,
                                                                track.meta[DETKEY],
                                                                track.meta[FRAMEIDXKEY])))
                self.cam_tracks[cam_id][tid] = Track(id=tid, timestamps=tstamps, ids=ids,
                                                     meta={DETKEY: dets, FRAMEIDXKEY: frame_idx})
        # delete empty tracks after iterating over them!
        for cam_id, did in del_ids:
            del self.cam_tracks[cam_id][did]

    def _calc_tp_fp(self):
        """Calculate some information about positives and false positives in truth data."""
        self.positives = set()
        self.false_positives = set()
        self.cam_false_positives = {key: set() for key in self.get_camids()}
        for detection in self.detections_dict.values():
            if TRUTHKEY in detection.meta and detection.meta[TRUTHKEY] != self.fp_id:
                self.positives.add(detection.id)
            else:
                self.cam_false_positives[detection.meta[CAMKEY]].add(detection.id)
                detection.meta[TRUTHKEY] = self.fp_id
                self.false_positives.add(detection.id)

    def get_all_detection_ids(self):
        return self.positives, self.false_positives

    def get_truth_track(self, truth_id, cam_id=None):
        if cam_id is not None and cam_id in self.cam_tracks.keys() \
           and truth_id in self.cam_tracks[cam_id].keys():
            return self.cam_tracks[cam_id][truth_id]
        elif cam_id is None and truth_id in self.tracks.keys():
            return self.tracks[truth_id]
        return None

    def get_truth_tracks(self, cam_id=None):
        if cam_id is None:
            tracks = self.tracks.values()
        else:
            tracks = self.cam_tracks[cam_id].values()
        for track in tracks:
            yield track

    def get_truthid(self, frame_object):
        if isinstance(frame_object, Detection):
            detection_id = frame_object.id
        elif isinstance(frame_object, Track):
            if TRUTHKEY in frame_object.meta:
                return frame_object.meta[TRUTHKEY]
            detection_id = frame_object.meta[DETKEY][-1].id
        else:
            detection_id = frame_object
        detection = self.get_detection(detection_id)
        return detection.meta[TRUTHKEY]

    def get_truthids(self, cam_id=None, frame_object=None):
        if frame_object is not None and cam_id is not None:
            raise ValueError("You can not use frame_object and cam_id together.")
        elif frame_object is None and cam_id is None:
            truth_ids = list(self.tracks.keys())
            if len(self.false_positives) > 0:
                truth_ids.append(self.fp_id)
        elif frame_object is None and cam_id is not None:
            truth_ids = list(self.cam_tracks[cam_id].keys())
            if len(self.cam_false_positives[cam_id]) > 0:
                truth_ids.append(self.fp_id)
        elif isinstance(frame_object, Track):
            truth_ids = [self.get_truthid(det_id) for det_id in frame_object.ids]
        elif isinstance(frame_object, Detection):
            truth_ids = [self.get_truthid(frame_object), ]
        else:
            raise TypeError("Type {0} not supported.".format(type(frame_object)))
        return set(truth_ids)
