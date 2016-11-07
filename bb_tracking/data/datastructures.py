# -*- coding: utf-8 -*-
"""Defines commonly used data structures for a convenient exchange format.

Note that :obj:`collections.namedtuple` are used for fast data exchange objects but they do
not support docstring in Python 2. Setting the docstring is only possible in Python 3.
So depending on the Python version that was used to compile the documentation there will be some
missing information.
"""
from collections import namedtuple
from six import PY3

Detection = namedtuple('Detection', ['id', 'timestamp', 'x', 'y', 'orientation', 'beeId', 'meta'])
if PY3:
    Detection.__doc__ = """
A :obj:`Detection` is one data point produced where (possibly) a bee was observed.

Keyword Arguments:
    id (int or str): identifier for detection
    timestamp (time representation): timestamp of detection
    x (float): x position in image coordinates
    y (float): y position in image coordinates
    orientation (float): the orientation of the tag/bee (zrotation in schema)
    beeId (int or iterable): interpretation of id
    meta (:obj:`dict`): dictionary with further information about a detection
        like localizer scores, truth ids, camera ids...
"""

Track = namedtuple('Track', ['id', 'ids', 'timestamps', 'meta'])
if PY3:
    Track.__doc__ = """
A :obj:`Track` is a group of :obj:`Detection` objects produced by (possibly) one bee.

To avoid handing a lot of data that we won't need only the identifiers and timestamps are
stored/provided in a :obj:`Track`. Specific data for a detection could be retrieved via
the :obj:`.DataWrapper`.

Depending on the :obj:`.DataWrapper` implementation the detections are also stored as objects in the
meta field. The key identifier for the list of detections in the meta field is :obj:`.DETKEY`.

Note:
    It might be necessary to clear the meta dictionary after some processing steps to free memory.

Keyword Arguments:
    id (int or str): identifier for :obj:`Track`
    ids (list of :obj:`Detection`): iterable with ids of :obj:`Detection`
    timestamps (iterable): iterable with timestamps of detections
    meta (:obj:`dict`): dictionary with uncommon information about a :obj:`Track`
        like scoring metrics
"""

Score = namedtuple('Score', ['value', 'track_id', 'truth_id', 'metrics', 'alternatives'])
if PY3:
    Score.__doc__ = """
A :obj:`Score` defines how similar two :obj:`Track` objects are.

Keyword Arguments:
    value (float): The scoring value calculated via a scoring function using the :obj:`ScoreMetrics`
    track_id (int or str): :obj:`Track` id we are scoring
    truth_id (int or str): truth id of the :obj:`Track` we are matching against
    metrics (:obj:`ScoreMetrics`): metrics describing the differences between :obj:`Track` objects
    alternatives (:obj:`list`): alternative truth ids that have the same score
"""

ScoreMetrics = namedtuple(
    'ScoreMetrics', ['track_length', 'truth_track_length', 'adjusted_length',
                     'id_matches', 'id_mismatches',
                     'inserts', 'deletes', 'gap_matches', 'gap_left', 'gap_right'])
if PY3:
    ScoreMetrics.__doc__ = """
A :obj:`ScoreMetrics` provides metrics to determine how similar two :obj:`Track` objects are.

Keyword Arguments:
    track_length (int): length of :obj:`Track` **including** gaps
    truth_track_length (int): length of truth track we are matching against
    adjusted_length (int): length of :obj:`Track` with added gaps left and right
    id_matches (int): number of matched ids
    id_mismatches (int): number of mismatched ids
    inserts (int): basically number of False Positives in the :obj:`Track`
    deletes (int): basically number of False Negatives in the :obj:`Track`
    gap_matches (int): correctly matched gaps
    gap_left (int): correctly identified gap to the left
    gap_right (int): correctly identified gap to the right
"""
