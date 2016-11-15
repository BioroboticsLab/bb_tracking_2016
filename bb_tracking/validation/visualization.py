# -*- coding: utf-8 -*-
"""
Provides functions to visualize the validation of tracks.

Warning:
    The visualizations are **not** tested by the continuous integration server!
"""
# pylint:disable=too-many-arguments,too-many-locals
from __future__ import division
from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from .validation import calc_fragments, convert_validated_to_pandas, track_statistics


def validate_plot(tracks, scores, validator, gap=0, cam_gap=True, metric_keys=None):
    """Visualizes the validation of tracks.

    Arguments:
        tracks (list of :obj:`.Track`): Iterable with tracks to validate
        scores (dict or :obj:`pd.Dataframe`): Result of :func:`.Validator.validate()`
        validator (:class:`.Validator`): Validator class to calculate track statistics

    Keyword Arguments:
        gap (int): the gap the algorithm should be capable to overcome
        cam_gap (bool): flag indicating that a camera switch is a insurmountable gap
        metric_keys (list): Iterable with metric keys that should be shown.
<<<<<<< HEAD
            Default is ``['detections', 'fragments', 'tracks']``.
=======
            Default is ``['detections', 'fragments' 'tracks', 'truth_ids']``.
>>>>>>> 7b6f08c50dd64fdf0248e0472874ef113bf30acf

    Returns:
        tuple: tuple containing figure and axes object from matplotlib
    """
    if isinstance(scores, dict):
        scores = convert_validated_to_pandas(scores)
    if len(tracks) != scores.shape[0]:
        tracks = validator.remove_false_positives(tracks)
    assert len(tracks) == scores.shape[0], "You might have false positives in tracks."
    bar_dict = track_statistics(tracks, scores, validator, gap=gap, cam_gap=cam_gap)
    col_dict = {"detections": "orange",
                "fragments": "red",
                "tracks": "green",
                "truth_ids": "limegreen"}

    fig, axs = plt.subplots()
    colors, keys, values = [], [], {}
    metric_keys = metric_keys or ("detections", "fragments", "tracks", "truth_ids")
    for key in metric_keys:
        metrics = bar_dict[key]
        colors.extend([col_dict[key], ] * len(metrics))
        metrics_types = list(metrics.keys())
        metrics_types.sort()
        values.update({metrics_key: metrics[metrics_key] for metrics_key in metrics_types})
        keys.extend(metrics_types)

    plt.barh(range(len(values)), [values[key][0] / values[key][1] for key in keys],
             align='center', height=0.2, color=colors)

    for i, rect in enumerate(axs.patches):
        value = rect.get_width()
        text = "{:10.2f}% ({})".format(value * 100, values[keys[i]][0])
        axs.text(value + 0.05, rect.get_y(), text, ha='center', va='bottom')

    plt.yticks(range(len(keys)), keys)
    plt.ylim(-0.5, len(keys) - 0.5)
    plt.xlim(0, 1.12)

    axs.set_xticks(np.arange(0.0, 1.1, 0.1))
    minor_locator = AutoMinorLocator(2)
    axs.xaxis.set_minor_locator(minor_locator)

    plt.grid(which='both')
    plt.xlabel("Relative Proportion")
    plt.title("Proportions in percent for fragments with gaps <= {})".format(gap))

    return fig, axs


def plot_fragments(scores, validator, gap, cam_gap=True, interval=5, max_x=None):
    """Visualizes distribution of fragments.

    Arguments:
        scores (dict or :obj:`pd.Dataframe`: Result of :func:`.Validator.validate()`
        validator (:class:`.Validator`): Validator class to calculate track statistics

    Keyword Arguments:
        gap (int): the gap the algorithm should be capable to overcome
        cam_gap (bool): flag indicating that a camera switch is a insurmountable gap
        interval (int): the interval / size of bins in the histogram
        max_x (int): upper limit for x axis

    Returns:
        tuple: tuple containing figure and axes object from matplotlib
    """
    if isinstance(scores, dict):
        scores = convert_validated_to_pandas(scores)
    _, _, _, fragment_lengths = calc_fragments(validator.truth, gap=gap, cam_gap=cam_gap)
    error = scores[scores.value < 1]
    wrong_id = scores[scores.truth_id != scores.calc_id]
    lengths = [fragment_lengths, scores.track_length, wrong_id.track_length, error.track_length]
    bins = np.arange(0, max(max(fragment_lengths), scores.track_length.max()) + 1, interval)
    max_x = max_x or max(bins)
    labels = ["Truth", "Test", "Wrong Id", "Tracking Errors"]
    colors = ["green", "blue", "red", "darkorange"]
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.hist(lengths, bins=bins, normed=True, cumulative=True, histtype='step',
             label=labels, color=colors)

    ax1.set_ylim([0, 1])
    ax1.set_yticks(np.arange(0, 1 + 0.1, 0.1))
    ax1.grid(True)
    ax1.legend(loc='lower right')
    ax1.set_ylabel("Frequency")

    ax2.hist(lengths, bins=bins, label=labels, color=colors)
    ax2.grid(True)
    ax2.set_ylabel("Number")

    plt.setp((ax1, ax2), xticks=bins, xlim=[min(bins), max_x])
    plt.xlabel("Fragment lengths")
    ax1.set_title("Distribution of fragment lengths with gaps <= {}".format(gap))

    plt.tight_layout()
    return fig, ax1, ax2
