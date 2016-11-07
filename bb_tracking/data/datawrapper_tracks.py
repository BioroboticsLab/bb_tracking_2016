# -*- coding: utf-8 -*-
"""
This implementations of :class:`.DataWrapper` are also using dictionaries with shared data
objects as backend. The difference to :class:`.DataWrapperBinary` is that they are managing
:obj:`.Track` objects instead of :obj:`.Detection` objects. Therefore the :class:`DataWrapperTracks`
implementations have to know where a track starts and where it ends.

Sometimes it is necessary to have additional information about detections for training and
validation purposes. So it is posssible to inject another instance of :class:`.DataWrapper` and
:class:`.DataWrapperTracks` will delegate tasks it can not fullfill to this instance.
"""
from .constants import CAMKEY, DETKEY
from .datastructures import Detection, Track
from .datawrapper import DataWrapper, DataWrapperTruth


class DataWrapperTracks(DataWrapper):
    """Class for access to tracks.

    This class is implemented with a focus on performance. It is possible to inject a
    :class:`.DataWrapper` for other, non performant related tasks.

    Todo:
        :func:`get_neighbors()` does not support the radius argument!
    """
    cam_ids = None
    """:obj:`list` of int: sorted list with all available camera ids"""
    cam_timestamps = None
    """:obj:`dict`: ``{cam_id: sorted list of timestamps}`` mapping"""
    data = None
    """:class:`.DataWrapper`: Optional DataWrapper instance to access detection data"""
    frame_track_end = None
    """:obj:`dict`: ``{(cam_id, timestamp): list of track}`` :obj:`.Track` ends in frame"""
    frame_track_start = None
    """:obj:`dict`: ``{(cam_id, timestamp): list of track}`` :obj:`.Track` starts in frame"""
    timestamps = None
    """:obj:`list` of timestamp: sorted list with all available timestamps"""
    tracks = None
    """:obj:`dict`: ``{track_id: track}`` mapping for :obj:`.Track`"""

    def __init__(self, tracks, cam_timestamps, data=None, end_offset=0):
        """Initialization for DataWrapperTracks

        Note:
            The `data` keyword argument is optional, but sometimes it is convenient to have one
            :class:`.DataWrapper` that provides access to :obj:`.Track` and :obj:`.Detection`.

        Arguments:
            tracks (:obj:`list` of :obj:`.Track`): Iterable with :obj:`.Track`
            cam_timestamps (:obj:`dict`): ``{cam_id: sorted list of timestamps}`` mapping.
                Same as :attr:`.DataWrapperBinary.cam_timestamps`.

        Keyword Arguments:
            data (Optional :class:`.DataWrapper`): A DataWrapper provides access to detections.
            end_offset (Optional int): The track will end on ``len(track) - end_offset``
        """
        self.cam_ids = list(cam_timestamps.keys())
        self.cam_timestamps = cam_timestamps
        timestamps = list()
        for timestamps_cam in cam_timestamps.values():
            timestamps.extend(timestamps_cam)
        timestamps = list(set(timestamps))
        timestamps.sort()
        self.timestamps = timestamps
        self.data = data

        # initialize track dictionaries
        self.frame_track_start = dict()
        self.frame_track_end = dict()
        for cam_id, timestamps_cam in self.cam_timestamps.items():
            for time in timestamps:
                self.frame_track_start[(cam_id, time)] = list()
                self.frame_track_end[(cam_id, time)] = list()

        # fill track dictionaries
        self.tracks = dict()
        for track in tracks:
            assert track.id not in self.tracks.keys(), "Duplicate track ids."
            self.tracks[track.id] = track
            cam_id_start = list(self.get_camids(frame_object=track.meta[DETKEY][0]))[0]
            time_start = track.timestamps[0]
            self.frame_track_start[(cam_id_start, time_start)].append(track)

            cam_id_end = list(self.get_camids(frame_object=track.meta[DETKEY][-1 - end_offset]))[0]
            time_end = track.timestamps[-1 - end_offset]
            self.frame_track_end[(cam_id_end, time_end)].append(track)

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

    def get_detection(self, *args):
        if self.data is None:
            raise NotImplementedError()
        return self.data.get_detection(*args)

    def get_detections(self, *args):
        if self.data is None:
            raise NotImplementedError()
        return self.data.get_detections(*args)

    def get_track(self, track_id):
        """Returns all information concerning the track with `track_id`.

        Arguments:
            track_id (int or str): the id of the track as used in the data scheme

        Returns:
            :obj:`.Track`: data formated as :obj:`.Track`
        """
        return self.tracks[track_id]

    def get_tracks(self, track_ids):
        """Same as :func:`get_track()` but with multiple ids.

        Arguments:
            track_ids (:obj:`list` of int or str): Iterable structure with track ids

        Returns:
            :obj:`list` of :obj:`.Track`: iterable structure with :obj:`.Track`
        """
        return [self.tracks[tid] for tid in track_ids]

    def get_frame_objects(self, cam_id=None, timestamp=None):
        cam_id = cam_id or self.cam_ids[0]
        timestamp = timestamp or self.cam_timestamps[cam_id][0]
        return self.frame_track_end[(cam_id, timestamp)]

    def get_frame_objects_starting(self, cam_id=None, timestamp=None):
        """Gets all tracks starting on `cam_id` in frame with `timestamp`.

        Keyword Arguments:
            cam_id (Optional int): the cam to consider, starts with smallest cam if None
            timestamp (Optional timestamp): frame with timestamp, starts with first frame if None

        Returns:
            :obj:`list` of :obj:`.Track`: iterable structure with :obj:`.Track`
        """
        cam_id = cam_id or self.cam_ids[0]
        timestamp = timestamp or self.cam_timestamps[cam_id][0]
        return self.frame_track_start[(cam_id, timestamp)]

    def get_neighbors(self, frame_object, cam_id, radius=10, timestamp=None):
        if not isinstance(frame_object, Track):
            raise TypeError("Type {0} not supported.".format(type(frame_object)))

        timestamp = timestamp or frame_object.timestamps[-1]

        return [track for track in
                self.get_frame_objects_starting(cam_id=cam_id, timestamp=timestamp)
                if track.id != frame_object.id]

    def get_timestamps(self, cam_id=None):
        if cam_id is not None:
            return self.cam_timestamps[cam_id]
        return self.timestamps


class DataWrapperTruthTracks(DataWrapperTracks, DataWrapperTruth):
    """ Class for access to tracks and truth data.

    As :class:`DataWrapperTruthTracks` are only used in the context of training and validation it is
    necessary to inject a :class:`.DataWrapperTruth` instance. This implementation will delegate all
    truth data related tasks to the implementation in :attr:`DataWrapperTracks.data`.
    """

    def __init__(self, tracks, cam_timestamps, data, **kwargs):
        """Initialization for DataWrapperTruthTracks

        Arguments:
            tracks (:obj:`list` of :obj:`.Track`): Iterable with :obj:`.Track`
            cam_timestamps (:obj:`dict`): ``{cam_id: sorted list of timestamps}`` mapping
            data (:class:`.DataWrapperTruth`): A DataWrapper that provides access to truth data.

        Keyword Arguments:
            **kwargs (:obj:`dict`): Keyword arguments for :class:`DataWrapperTracks` without `data`.
        """
        super(DataWrapperTruthTracks, self).__init__(tracks, cam_timestamps, data=data, **kwargs)

    def get_all_detection_ids(self, *args, **kwargs):
        return self.data.get_all_detection_ids(*args, **kwargs)

    def get_truth_track(self, *args, **kwargs):
        return self.data.get_truth_track(*args, **kwargs)

    def get_truth_tracks(self, *args, **kwargs):
        return self.data.get_truth_tracks(*args, **kwargs)

    def get_truthid(self, *args, **kwargs):
        return self.data.get_truthid(*args, **kwargs)

    def get_truthids(self, *args, **kwargs):
        return self.data.get_truthids(*args, **kwargs)
