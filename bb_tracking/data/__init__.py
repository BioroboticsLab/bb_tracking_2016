"""Classes and data structures to access, evaluate and validate data.
"""
from .datastructures import Detection, Score, ScoreMetrics, Track

from .datawrapper import DataWrapper, DataWrapperTruth
from .datawrapper_binary import DataWrapperBinary, DataWrapperTruthBinary
from .datawrapper_pandas import DataWrapperPandas, DataWrapperTruthPandas
from .datawrapper_tracks import DataWrapperTracks, DataWrapperTruthTracks

__all__ = ['DataWrapper', 'DataWrapperTruth', 'DataWrapperBinary', 'DataWrapperPandas',
           'DataWrapperTracks', 'DataWrapperTruthBinary', 'DataWrapperTruthPandas',
           'DataWrapperTruthTracks', 'Detection', 'Score', 'ScoreMetrics', 'Track']
