# -*- coding: utf-8 -*-
"""
Constants that are used e.g. as keys in the meta dictionary in :obj:`.Detection` and :obj:`.Track`.
"""

CAMKEY = 'camId'
"""This key is used in :obj:`.Detection` to mark the camera the detection is associated with."""

DETKEY = 'detections'
"""This key is used in :obj:`.Track` to store the :obj:`.Detection` object while tracking."""

TRUTHKEY = 'truthId'
"""This key is used in :obj:`.Track` and :obj:`.Detection` to store the truth id of the frame object
for training and validation."""

FRAMEIDXKEY = 'frameIdx'
"""This key is used on :obj:`.Track` to save the index of the frame the track is associated with."""

FPKEY = 'fp_track'
"""This key is used in :obj:`.Track` to mark tracks that are identified as entirely consisting of
False Positives while validation."""
