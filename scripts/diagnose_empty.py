"""Describe existing empty-room prediction errors without training or relabeling."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from csi_pipeline import find_matching_files, make_split, read_csi_session


def main():
    output = ROOT / 'data_audit'
    output.mkdir(exist_ok=True)
    predictions = pd.read_csv(ROOT / 'csi_test_window_predictions.csv')
    empty = predictions[predictions['Class'] == 'Empty']
    session_results = pd.crosstab(
        [empty['Scenario'], empty['Model'], empty['Session']], empty['Predicted_Class'],
    ).reindex(columns=['Auth', 'Unauth', 'Empty'], fill_value=0)
    session_results['Windows'] = session_results.sum(axis=1)
    session_results['Empty_Recall'] = session_results['Empty'] / session_results['Windows']
    session_results.to_csv(output / 'empty_session_predictions.csv', encoding='utf-8-sig')

    files = {person: find_matching_files(person) for person in 'ABCD'}
    split = make_split(files, 'A')
    training = [path for name, paths in split.items() if name.startswith('train_') for path in paths]
    test_empty = split['test_empty']
    profiles = {}
    record_info = []
    for path in training + test_empty:
        session = read_csi_session(path)
        profiles[path] = np.median(session.amplitude, axis=0)
        if Path(path).parent.name == 'D':
            record_info.append({'file': Path(path).name,
                                'median_amplitude': float(np.median(profiles[path][2:])),
                                'mean_profile_amplitude': float(np.mean(profiles[path][2:]))})
    normalization = np.load(ROOT / 'csi_preprocessing_A.npz')
    scale = normalization['std'][2:]
    train_profiles = np.stack([profiles[path][2:] for path in training])
    nearest = []
    for path in test_empty:
        # Descriptive distances between median amplitude profiles, scaled with
        # baseline A's training standard deviations. This is not a classifier.
        distances = np.sqrt(np.mean(((train_profiles - profiles[path][2:]) / scale) ** 2, axis=1))
        order = np.argsort(distances)[:5]
        nearest.append({'file': Path(path).name,
                        'nearest_training_profiles': [
                            {'file': f'{Path(training[index]).parent.name}/{Path(training[index]).name}',
                             'scaled_rms_distance': float(distances[index])} for index in order],
                        'minimum_distance_by_group': {
                            person: float(min(distances[index] for index, ref in enumerate(training)
                                              if Path(ref).parent.name == person)) for person in 'ABCD'}})

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for name in ('D16.txt', 'D17.txt', 'D18.txt', 'D19.txt', 'D20.txt'):
        path = str(ROOT / 'data' / 'D' / name)
        axes[0].plot(np.arange(2, len(profiles[path])), profiles[path][2:], label=name, linewidth=1.5)
    axes[0].set(title='Empty-room recordings: median CSI amplitude profiles',
                xlabel='Amplitude feature index (first two excluded)', ylabel='Median amplitude')
    axes[0].legend(ncol=5)
    axes[0].grid(alpha=0.2)
    for name, path in [('D18.txt', str(ROOT / 'data/D/D18.txt')),
                       ('D19.txt', str(ROOT / 'data/D/D19.txt')),
                       ('D20.txt', str(ROOT / 'data/D/D20.txt')),
                       ('A04.txt', str(ROOT / 'data/A/A04.txt'))]:
        axes[1].plot(np.arange(2, len(profiles[path])), profiles[path][2:], label=name, linewidth=1.5)
    axes[1].set(title='Descriptive comparison with one A training recording (not label verification)',
                xlabel='Amplitude feature index (first two excluded)', ylabel='Median amplitude')
    axes[1].legend(ncol=4)
    axes[1].grid(alpha=0.2)
    fig.savefig(output / 'empty_amplitude_profiles.png', dpi=180)
    plt.close(fig)
    report = {'model_training_performed': False,
              'session_predictions': session_results.reset_index().to_dict(orient='records'),
              'D_amplitude_summary': record_info,
              'test_D_nearest_training_profiles': nearest,
              'limitation': 'Profile similarity does not establish the measured person or prove a mislabeled recording.'}
    (output / 'empty_diagnosis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
