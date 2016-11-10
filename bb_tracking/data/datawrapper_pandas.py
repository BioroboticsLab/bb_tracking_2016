# -*- coding: utf-8 -*-
"""Provides classes for easy access to detections beyond data formats.

This DataWrapppers are using Pandas as backend. This helps when you are using data from other data
sources, but the instantiation is a bit more complicated. You basically create a Pandas DataFrame
and provide a mapping from required default names to the column names in your Pandas DataFrame.

Row lookups in Pandas are also quite expensive so this this classes are quite slow and **not**
recommended for production.

Note:
    If not used for experiments in the future they might be removed.

Todo:
    Refactor detection cache and add cache for frame trees.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .constants import CAMKEY, DETKEY
from .datastructures import Detection, Track
from .datawrapper import DataWrapper, DataWrapperTruth


class DataWrapperPandas(DataWrapper):
    """Class for flexible access to detections.

    This class is recommended for testing or data exploration. The data input is configurable and
    possible additional attributes are not discarded. But it is not as efficient as the
    :class:`.DataWrapperBinary` class in terms of memory and lookup speed.
    """

    detections = None
    """:obj:`pd.DataFrame`: pandas dataframe with all detections"""
    detections_dict = None
    """:obj`dict`: dictionary with ``{detection_id: detection}`` mapping for :obj:`.Detection`"""
    beeId_digits = 12
    """int: number of digits use to encode a ``beeID`` on a tag"""
    cols = None
    """:obj`dict`: strings to identify certain columns"""
    duplicates_radius = None
    """int: detections within this radius are considered to be duplicates"""
    mean_duplicates_merge_columns = None
    """:obj`dict`: use mean value for this columns when merge duplicates"""

    def __init__(self, detections, cols=None, duplicates_radius=None, meta_keys=None):
        """Necessary initialization to reformat detections dataframe.

        Arguments:
            detections (:obj:`pd.DataFrame`): pandas dataframe with detections

        Keyword Arguments:
            cols (Optional :obj:`dict`): dictionary with ``{key: key in dataframe}`` mapping.
            duplicates_radius (Optional int): distance to determine if detections are duplicates
            meta_keys (Optional :obj:`dict`): ``{detecion_key: meta_key}`` mapping that is added
                as meta field in detections
        """
        self.cols = {
            'id': 'id',
            'x': 'xpos',
            'y': 'ypos',
            'beeId': 'beeID',
            'timestamp': 'timestamp',
            CAMKEY: 'camID',
            'orientation': 'zrotation',
            'localizer': 'localizerSaliency',
            'decoder': 'decoder',
            'frameIdx': 'frameIdx',
            'truthId': 'truthID',
            'readability': 'readability',
            'mergeId': 'merge_id_col'
        }

        cols = cols or dict()
        for (key, val) in cols.items():
            if key in self.cols:
                self.cols[key] = val
        mkeys = list(meta_keys.keys()) if meta_keys is not None else []
        assert set(mkeys) <= set(detections.columns), "Keys not available in detections."

        self.duplicates_radius = duplicates_radius

        self.mean_duplicates_merge_columns = [col for col in ['x', 'y', 'localizer']
                                              if self.cols[col] in detections.columns]

        self.detections = self.clean_data(detections.copy())
        # note that initial meta data for detections will slow the setup time
        if meta_keys:
            assert "meta" not in detections.columns, "Keyword `meta` not allowed in columns!"
            self.detections["meta"] = [{mkey: getattr(row, key) for key, mkey in meta_keys.items()}
                                       for row in self.detections[mkeys].itertuples()]
        else:
            self.detections["meta"] = [{}] * self.detections.shape[0]
        # order is important!
        detection_cols = [self.cols['id'], self.cols['timestamp'], self.cols['x'], self.cols['y'],
                          self.cols['orientation'], self.cols['beeId'], 'meta']
        # using a dictionary gives a great lookup speedup with a small generation overhead!
        self.detections_dict = {row.id: Detection(*row)
                                for row in self.detections[detection_cols].itertuples(index=False)}

    def clean_data(self, data):
        """Clean the dataframes to use a common format to access all the data.

        Arguments:
            data (:obj:`pd.DataFrame`): pandas dataframe

        Returns:
            :obj:`pd.DataFrame`: clean dataframe
        """
        # drop truthId column if given
        if self.cols['truthId'] in data.columns:
            data.drop(self.cols['truthId'], axis=1, inplace=True)

        # first finish cleaning before merging duplicates
        data = self._clean_data(data)

        # merge duplicate entries
        if self.duplicates_radius is not None:
            self.detections = data
            left, right = self._get_duplicate_ids(data, self.duplicates_radius)
            data = self._merge_entries(data, left, right)
        return data

    def _clean_data(self, data):
        """Common cleaning operations for class and subclasses

        Arguments:
            data (:obj:`pd.DataFrame`): pandas dataframe

        Returns:
            :obj:`pd.DataFrame`: clean dataframe
        """
        # sometimes we have inf values in orientation so we replace them with 0
        if self.cols['orientation'] in data.columns:
            data.loc[np.isinf(data[self.cols['orientation']]), self.cols['orientation']] = 0
        # use id col as index
        data.set_index([self.cols['id']], drop=False, inplace=True, verify_integrity=True)
        return data

    def _get_duplicate_ids(self, data, radius):
        """Gets detection duplicates using radius to determine duplicates.

        Arguments:
            data (:obj:`pd.DataFrame`): pandas dataframe with detections
            radius (int): distance used to determine if two detections are duplicates

        Returns:
            tuple: tuple containing:

                - **left** (:obj:`list` of :obj:`int`): id mapping to merge
                - **right** (:obj:`list` of :obj:`int`): id mapping to merge
        """
        left, right = [], []
        for (cam_id, timestamp), _ in data.groupby([self.cols[CAMKEY],
                                                    self.cols['timestamp']]):
            left_idx, right_idx = [], []
            tree, index = self._get_tree(cam_id, timestamp)
            neighbors = tree.query_pairs(radius)
            for i, j in neighbors:
                left_idx.append(i)
                right_idx.append(j)
            left.extend(index[left_idx])
            right.extend(index[right_idx])
        return (list(left), list(right))

    def _merge_entries(self, data, left, right):
        """Merges two entries by merging their ids.

        Arguments:
            data (:obj:`pd.DataFrame`): pandas dataframe with detections
            left (list of int): id or ids that are used as base
            right (list of int): id or ids that are merged to their partner left

        Returns:
            :obj:`pd.DataFrame`: pandas dataframe with merged detections
        """
        for i, j in zip(left, right):
            data.at[i, self.cols['beeId']] =\
                self._merge_ids(
                    data.loc[i, self.cols['beeId']],
                    data.loc[j, self.cols['beeId']])

            for column in ['x', 'y', 'localizer']:
                data.at[i, self.cols[column]] = np.mean(data.loc[[i, j], self.cols[column]])
        data.drop(right, inplace=True)
        return data

    def _merge_ids(self, id1, id2):
        """Merge two id distributions.

        Arguments:
            id1 (list of float): bit array with frequency
            id2 (list of float): bit array with frequency

        Returns:
            list of float: merged id distribution
        """
        assert len(id1) == self.beeId_digits
        assert len(id1) == len(id2)
        return [(float(x) + float(y)) / 2 for x, y in zip(id1, id2)]

    def get_camids(self, frame_object=None):
        if frame_object is None:
            cam_ids = self.detections[self.cols[CAMKEY]].unique().tolist()
        elif isinstance(frame_object, Track):
            cam_ids = self.detections.loc[frame_object.ids, self.cols[CAMKEY]].unique().tolist()
        elif isinstance(frame_object, Detection):
            cam_ids = [self.detections.loc[frame_object.id, self.cols[CAMKEY]], ]
        else:
            raise TypeError("Type {0} not supported.".format(type(frame_object)))
        return set(cam_ids)

    def get_detection(self, detection_id):
        return self.detections_dict[detection_id]

    def get_detections(self, detection_ids):
        return [self.detections_dict[d_id] for d_id in detection_ids]

    def get_frame_objects(self, cam_id=None, timestamp=None):
        frame = self._get_frame(cam_id, timestamp)
        return self.get_detections(frame[self.cols['id']].values)

    def get_neighbors(self, frame_object, cam_id, radius=10, timestamp=None):
        if isinstance(frame_object, Track):
            detection = self.get_detection(frame_object.ids[-1])
        elif isinstance(frame_object, Detection):
            detection = frame_object
        else:
            raise TypeError("Type {0} not supported.".format(type(frame_object)))
        # determine search parameters
        timestamp = timestamp or detection.timestamp

        # use spatial tree for efficient neighborhood search
        tree, index = self._get_tree(cam_id, timestamp)
        if index is None:
            return []
        indices = tree.query_ball_point((detection.x, detection.y), radius)

        # translate tree Indices in detection ids and remove search item
        ids = index[indices]
        return self.get_detections(ids[ids != detection.id].tolist())

    def get_timestamps(self, cam_id=None):
        if cam_id is None:
            timestamps = self.detections[self.cols['timestamp']].unique()
        else:
            timestamps = self.detections.loc[self.detections[self.cols[CAMKEY]] == cam_id,
                                             self.cols['timestamp']].unique()
        timestamps.sort()
        # keep numpy array for bulk access on DataFrame
        return timestamps

    def _get_frame(self, cam_id=None, timestamp=None):
        """Helper to extract all detections of a frame via `cam_id` and `timestamp`.

        Keyword Arguments:
            cam_id (Optional int): the id of the camera or the first if None
            timestamp (Optional timestamp): the timestamp of the frame or the first if None

        Returns:
            :obj:`pd.DataFrame`: subset of detections restricted on cam_id and timestamp.
        """
        data = self.detections
        cam_id = cam_id or data[self.cols[CAMKEY]].min()
        timestamp = timestamp or data[self.cols['timestamp']].min()
        frame = data[(data[self.cols[CAMKEY]] == cam_id) &
                     (data[self.cols['timestamp']] == timestamp)]
        return frame

    def _get_tree(self, cam_id, timestamp):
        """Helper to generate a spatial tree from frame defined via cam_id and timestamp.

        Basically there is one tree for each frame.

        Arguments:
            cam_id (Optional int): the id of the camera
            timestamp (Optional timestamp): the timestamp of the frame

        Returns:
            tuple: tuple containing:

                - **tree** (:obj:`scipy.spatial.cKDTree`): spatial tree for neighborhood search.
                - **index** (:obj:`list` of ids): the frame index to map tree ids to detection ids
        """
        frame = self._get_frame(cam_id, timestamp)
        if frame.empty:
            return cKDTree([[], []]), None
        xy_cols = frame[[self.cols['x'], self.cols['y']]]
        return cKDTree(xy_cols), frame.index.values


class DataWrapperTruthPandas(DataWrapperPandas, DataWrapperTruth):
    """Special wrapper for truth data with a Pandas Backend.
    """
    merge_cols = None
    """:obj:`dict`: strings to identify how columns in truth should be named in detections"""
    duplicate_cols = None
    """tuple of str: columns to identify track duplicates"""

    def __init__(self, detections, truth, radius, merge_cols=None, **kwargs):
        """Necessary initialization to reformat detections dataframe.

        Also performs merge of `detections` and `truth` dataframes.

        Arguments:
            detections (:obj:`pd.DataFrame`): pandas dataframe with detections
            truth (:obj:`pd.DataFrame`): pandas dataframe with truth
            radius (int): merge detections and truth via distance

        Keyword Arguments:
            merge_cols (:obj:`dict`): ``{key in truth: new key in detections}`` mapping.
            **kwargs (:obj:`dict`): keyword arguments for :class:`DataWrapperPandas`
        """
        if 'duplicates_radius' not in kwargs.keys():
            kwargs['duplicates_radius'] = radius
        super(DataWrapperTruthPandas, self).__init__(detections, **kwargs)

        self.duplicate_cols = [self.cols["camId"], self.cols["timestamp"], self.cols["truthId"]]
        self.merge_cols = merge_cols or {
            'decodedId': self.cols['truthId'],
            'readability': self.cols['readability']  # we need this to determine which truth to use
        }
        truth = self.clean_data(truth)
        self.detections = super(DataWrapperTruthPandas, self).clean_data(self.detections)

        mck = set(self.merge_cols.keys())
        tck = set(truth.columns)
        assert mck <= tck, "Truth data is missing columns: {}".format(mck - tck)
        assert set(self.merge_cols.values()).isdisjoint(set(self.detections.columns)),\
            "Detections data already contains merge column name(s)."
        self.detections = self._merge_truth(self.detections, truth, radius, self.merge_cols)

        track_duplicates = self._get_track_duplicates(self.detections)
        if track_duplicates:
            raise UserWarning("Truth Data has duplicates in tracks: {}.".format(track_duplicates))

    def clean_data(self, data):
        # set negative id for false positives
        if self.cols['truthId'] in data.columns:
            data[self.cols['truthId']].fillna(self.fp_id, inplace=True)
            data[self.cols['truthId']] = data[self.cols['truthId']].astype(int)

        # remove truth detections where tag was not visible
        if self.cols['readability'] in data.columns:
            data = data[data[self.cols['readability']] != 'none']

        return self._clean_data(data)

    def _merge_truth(self, detections, truth, radius, merge_cols):
        """Helper to merge `truth` data with `detections`.

        The detections are merged using their proximity (x-, y-coordinates) via `radius`.

        Arguments:
            detections (:obj:`pd.DataFrame`): pandas dataframe with detections
            truth (:obj:`pd.DataFrame`): pandas dataframe with truth
            radius (int): merge detections and truth via distance
            merge_cols (dict): ``{key in truth: new key in detections}`` mapping.

        Returns:
            :obj:`pd.DataFrame`: merged detections and truth
        """
        detections[self.cols['mergeId']] = self._get_merge_ids(detections, truth, radius)
        to_merge = pd.DataFrame(truth[list(merge_cols.keys())].values,
                                columns=[list(merge_cols.values())],
                                index=truth.id.values)
        df_merged = pd.merge(detections, to_merge, how='left',
                             left_on=self.cols['mergeId'],
                             right_index=True)

        df_merged.drop(self.cols['mergeId'], axis=1, inplace=True)
        df_merged[self.cols['truthId']].fillna(self.fp_id, inplace=True)
        df_merged[self.cols['readability']].fillna(self.code_unknown, inplace=True)

        return df_merged

    def _get_track_duplicates(self, data):
        """Helper to search for duplicates in tracks.

        Arguments:
            data (:obj:`pd.DataFrame`): dataframe like `detections`

        Returns:
            list of ids: list of duplicate ids
        """
        data_test = data[data[self.cols['truthId']] != self.fp_id]
        data_test_no_duplicates = data[self.duplicate_cols].drop_duplicates(keep=False)
        duplicate_ids = data_test.index.difference(data_test_no_duplicates.index)
        return duplicate_ids.tolist()

    def _get_merge_ids(self, detections, truth, radius):
        """Helper to calculate the mapping of index in detections dataframe and truth dataframe.

        The detections are merged using their proximity (x-, y-coordinates) via `radius`.

        Arguments:
            detections (:obj:`pd.DataFrame`): pandas dataframe with detections
            truth (:obj:`pd.DataFrame`): pandas dataframe with truth
            radius (int): merge detections and truth via distance

        Returns:
            obj:`pd.Series`: mapping of detection ids to ids in truth dataframe
        """
        merge_ids = pd.Series(data=np.zeros_like(detections.index), index=detections.index.tolist())
        for (cam_id, timestamp), group in\
                truth.groupby([self.cols[CAMKEY], self.cols['timestamp']]):
            g_mask, f_mask = [], []
            frame_tree, frame_ids = self._get_tree(cam_id, timestamp)
            idx_mapping = frame_tree.query_ball_tree(
                cKDTree(group[[self.cols['x'], self.cols['y']]]),
                radius)
            for i, idx in enumerate(idx_mapping):
                if len(idx) == 0:
                    continue
                elif len(idx) == 1:
                    f_mask.append(i)
                    g_mask.append(idx[0])
                elif len(idx) > 1:
                    raise UserWarning('Truth Data has detections in each others radius.')
                else:  # pragma: no cover
                    Exception()
            merge_ids.ix[frame_ids[f_mask]] = group.index.values[g_mask]
        return merge_ids

    def get_all_detection_ids(self):
        pmask = self.detections[self.cols['truthId']] != self.fp_id
        positives = set(self.detections.loc[pmask, self.cols['id']])
        false_positives = set(self.detections.loc[np.logical_not(pmask), self.cols['id']])
        return positives, false_positives

    def get_truth_track(self, truth_id, cam_id=None):
        restrictions = self.detections[self.cols['truthId']] == truth_id
        if cam_id is not None:
            restrictions = restrictions & (self.detections[self.cols[CAMKEY]] == cam_id)
        detections = self.detections[restrictions].sort_values('timestamp')
        if detections.shape[0] == 0:
            return None
        return Track(truth_id,
                     detections[self.cols['id']].tolist(),
                     pd.to_datetime(detections[self.cols['timestamp']]).tolist(),
                     meta={})

    def get_truth_tracks(self, cam_id=None):
        if cam_id is None:
            truth_ids = self.detections[self.cols['truthId']].unique()
        else:
            truth_ids = self.detections.loc[self.detections[self.cols[CAMKEY]] == cam_id,
                                            self.cols['truthId']].unique()
        for truth_id in truth_ids:
            if truth_id == self.fp_id:
                continue
            yield self.get_truth_track(truth_id, cam_id=cam_id)

    def get_truthid(self, frame_object):
        if isinstance(frame_object, Detection):
            detection_id = frame_object.id
        elif isinstance(frame_object, Track):
            detection_id = frame_object.meta[DETKEY][-1].id
        else:
            detection_id = frame_object
        return self.detections.loc[detection_id, self.cols['truthId']]

    def get_truthids(self, cam_id=None, frame_object=None):
        if frame_object is not None and cam_id is not None:
            raise ValueError("You can not use frame_object and cam_id together.")
        elif frame_object is None and cam_id is None:
            truthids = self.detections[self.cols['truthId']].unique()
        elif frame_object is None and cam_id is not None:
            truthids = self.detections.loc[self.detections[self.cols[CAMKEY]] == cam_id,
                                           self.cols['truthId']].unique()
        elif isinstance(frame_object, Track):
            truthids = self.detections.ix[list(frame_object.ids)][self.cols['truthId']].unique()
        elif isinstance(frame_object, Detection):
            truthids = [self.detections.loc[frame_object.id, self.cols['truthId']], ]
        else:
            raise TypeError("Type {0} not supported.".format(type(frame_object)))
        return set(truthids)
