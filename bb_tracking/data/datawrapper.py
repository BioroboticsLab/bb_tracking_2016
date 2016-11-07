# -*- coding: utf-8 -*-
"""
Provides classes for abstract access to :obj:`.Detection` and :obj:`.Track` objects.

There are two abstract classes describing the interface of a :obj:`DataWrapper`.
A :obj:`DataWrapper` is only providing access to data, whereas the :obj:`DataWrapperTruth`
also considers ground truth data. This distinction has the following reasons:

    1. No *accidental* tracking on truth data (e.g. truth ids)
    2. Better performance in generation and access of data
    3. Smaller interface for production usage

There are currently three implementations:

:class:`.DataWrapperBinary`:
    Directly integrates with `bb_binary
    <http://bb-binary.readthedocs.io/en/latest/api/repository.html#module-bb_binary.repository>`_
    and is implemented with a focus on performance.

:class:`.DataWrapperPandas`:
    Uses `Pandas <http://pandas.pydata.org/>`_ as backend. Therefore it is more complicated to
    generate an instance of this DataWrapper. But it offers more possibilities in using other data
    sources or performing experimens with data, that is not available in *bb_binary*.

:class:`.DataWrapperTracks`:
    This is a DataWrapper to organize the access to :obj:`.Track` objects to be able to perform
    tracking and validation on track level (assigning tracks to other tracks instead of detections
    to tracks).
"""


class DataWrapper(object):
    """Abstract class that describes the access to detections."""

    def get_camids(self, frame_object=None):
        """Returns an iterable with camera ids.

        If `frame_object` is :obj:`None` returns **all** available camera ids.
        Otherwise it returns the camera ids the :obj:`.Detection` objects of
        `frame_object` are associated with. So possible multiple ids if `frame_object` is a
        :obj:`.Track` and a single id if `frame_object` is a :obj:`.Detection`.

        Keyword Arguments:
            frame_object (Optional :obj:`.Detection` or :obj:`.Track`): frame object to restrict
            camera ids to.

        Returns:
            :obj:`list` of int: iterable structure with camera ids
        """
        raise NotImplementedError()

    def get_detection(self, detection_id):
        """Returns all information concerning the :obj:`.Detection` with `detection_id`.

        Arguments:
            detection_id (int or str): the id of the detection as used in the data schema

        Returns:
            :obj:`.Detection`: data formated as :obj:`.Detection`
        """
        raise NotImplementedError()

    def get_detections(self, detection_ids):
        """Returns all information concerning the :obj:`.Detection` objects with `detection_ids`.

        Arguments:
            detection_ids (:obj:`list` of int or str): iterable structure with detection ids

        Returns:
            :obj:`list` of :obj:`.Detection`: iterable structure with :obj:`.Detection` objects
        """
        raise NotImplementedError()

    def get_frame_objects(self, cam_id=None, timestamp=None):
        """Gets all detections or tracks on `cam_id` in frame with `timestamp`.

        Keyword Arguments:
            cam_id (Optional int): the cam to consider, starts with **smallest** cam id if
                :obj:`None`
            timestamp (Optional timestamp): frame with timestamp, starts with **first** frame if
                :obj:`None`

        Returns:
            :obj:`list` of :obj:`.Detection` or :obj:`.Track` : iterable structure with
            :obj:`.Detection` or :obj:`.Track` depending on :class:`.DataWrapper` implementation.
        """
        raise NotImplementedError()

    def get_neighbors(self, frame_object, cam_id, radius=10, timestamp=None):
        """Gets all detections or tracks in the neighborhood of the given frame_object.

        Arguments:
            frame_object (:obj:`.Detection` or :obj:`.Track`): frame object to search neighborhood
            cam_id (int): usually cam_id is already available so use this parameter for speedup

        Keyword arguments:
            radius (Optional int): the radius to search in image coordinates
            timestamp (Optional timestamp): consider frame objects of frame with different timestamp

        Returns:
            :obj:`list` of :obj:`.Detection` or :obj:`.Track` : iterable structure with
            :obj:`.Detection` or :obj:`.Track` depending on :class:`DataWrapper` implementation.
        """
        raise NotImplementedError()

    def get_timestamps(self, cam_id=None):
        """Extracts all timestamps as unique ordered Iterable.

        Keyword Arguments:
            cam_id (Optional int): select only timestamps that are available for this camera id

        Returns:
            :obj:`list` of time representations: list with timestamps as ordered iterable
        """
        raise NotImplementedError()


class DataWrapperTruth(DataWrapper):
    """Special wrapper for truth data.

    This is separated from :class:`.DataWrapper` for performance and convenience reasons.

    The main difference is, that this class will merge ground truth data with pipeline output and
    generate truth tracks that can be used for training and validation.
     """

    fp_id = -1
    """int: id for possibly false positive detections (no matching truth_id)"""
    code_unknown = "unknown"
    """str: default value for readability"""

    def get_all_detection_ids(self):
        """Returns the ids of all positives and false positives.

        Returns:
            tuple: tuple containing:

                - **positives** (:obj:`set`): Ids of detections **with** associated truth id
                - **false positives** (:obj:`set`): Ids of detections **without** truth id

        """
        raise NotImplementedError()

    def get_truth_track(self, truth_id, cam_id=None):
        """Returns the :obj:`.Track` to the given `truth_id`.

        Arguments:
            truth_id (int or str): we are using the `truth_id` to identify a certain track

        Keyword Arguments:
            cam_id (Optional int): the cam to consider

        Returns:
            :obj:`.Track`: the track to the given `truth_id`
        """
        raise NotImplementedError()

    def get_truth_tracks(self, cam_id=None):
        """Returns an iterator over all the truth tracks.

        Keyword Arguments:
            cam_id (Optional int): restrict on tracks on this camera

        Returns:
            iterator: iterator over all the truth :obj:`.Tracks`
        """
        raise NotImplementedError()

    def get_truthid(self, frame_object):
        """Returns the truth id for the corresponding `frame_object` id or :obj:`None`.

        Keyword Arguments:
            frame_object (:obj:`.Detection` or :obj:`.Track` or id): frame object or the id
                of the :obj:`.Detection` so extract the truth id from

        Returns:
            int or str: the truth id for the `frame_object` or :obj:`None`
            if `frame_object` is a false positive
        """
        raise NotImplementedError()

    def get_truthids(self, cam_id=None, frame_object=None):
        """Returns all truth ids of either the whole truth data or restricted on some parameters.

        Note:
            `cam_id` and `frame_object` can not be used together!

        Keyword Arguments:
            cam_id (Optional int): get truth ids that are associated with this camera.
            frame_object (:obj:`.Detection` or :obj:`.Track`): get truth ids that are associated
                with this frame object.

        Returns:
            :obj:`set`: all the truth ids that might be associated with a frame object or camera.
        """
        raise NotImplementedError()
