"""Compare file-level split seeds with frozen Optuna settings and training seed."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from csi_pipeline import find_matching_files, make_split, validate_unique_recordings


def verify_results(output, completed):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    manifest = json.loads((output / 'experiment_manifest.json').read_text())
    settings = {(row['target'], row['model']): row for row in manifest['settings']}
    classes = ['Auth', 'Unauth', 'Empty']
    models_checked = 0
    for recording in manifest['recordings']:
        digest = hashlib.sha256()
        with open(recording['file'], 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != recording['sha256']:
            raise ValueError(f'실험 중 원본 데이터가 변경되었습니다: {recording["file"]}')
    for run in completed:
        directory = output / run['name']
        evaluations = json.loads((directory / 'evaluation_manifest.json').read_text())
        if len(evaluations) != 6:
            raise ValueError('각 분할의 모델 결과가 6개가 아닙니다.')
        summaries = pd.read_csv(directory / 'final_test_summary.csv')
        for job in evaluations:
            target, model = job['authorized_person'], job['model']
            frozen = settings[target, model]
            assert job['params'] == frozen['params'] and job['epochs'] == frozen['epochs']
            assert job['training_determinism'] is True and job['training_seed'] == 42
            assert job['split_seed'] == run['seed'] and not job['hyperparameters_reoptimized']
            train, test = set(job['train_files']), set(job['test_files'])
            assert len(train) == 48 and len(test) == 12 and not train & test
            prefix = directory / f'{target}_{model}'
            history = pd.read_csv(f'{prefix}_training_history.csv')
            assert len(history) == frozen['epochs'] and np.isfinite(history['loss']).all()
            for partition, paths in [('train', train), ('test', test)]:
                prediction = pd.read_csv(f'{prefix}_{partition}_predictions.csv')
                observed = set(zip(prediction['Person'], prediction['Session']))
                assert observed == {(Path(p).parent.name, Path(p).name) for p in paths}
                probabilities = prediction[[f'Probability_{name}' for name in classes]].to_numpy()
                assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
                labels = [classes[int(index)] for index in probabilities.argmax(axis=1)]
                assert labels == prediction['Predicted_Class'].tolist()
                expected = np.where(prediction['Person'] == 'D', 'Empty',
                                    np.where(prediction['Person'] == target, 'Auth', 'Unauth'))
                assert np.array_equal(expected, prediction['Class'])
                if partition == 'test':
                    true = prediction['Class'].map({name: i for i, name in enumerate(classes)})
                    pred = prediction['Predicted_Class'].map({name: i for i, name in enumerate(classes)})
                    matrix = confusion_matrix(true, pred, labels=[0, 1, 2])
                    saved = pd.read_csv(directory / f'cm_{target}_{model}_final_test_counts.csv', index_col=0)
                    assert np.array_equal(matrix, saved.to_numpy())
                    row = summaries[(summaries['Scenario'] == f'{target} Auth') &
                                    (summaries['Model'] == ('1D-CNN' if model == 'cnn' else 'LSTM'))].iloc[0]
                    assert np.isclose(row['Accuracy'], accuracy_score(true, pred))
                    assert np.isclose(row['Macro_F1'], f1_score(true, pred, labels=[0, 1, 2], average='macro'))
            models_checked += 1
    report = {'status': 'passed', 'recordings_unchanged': len(manifest['recordings']),
              'models_checked': models_checked, 'training_test_file_overlap': 0,
              'checks': ['frozen settings and epoch counts', 'training seed and deterministic mode',
                         'original file hashes', 'file membership and folder labels',
                         'probability and prediction alignment', 'confusion counts and test metrics'],
              'parameter_selection_test_overlap': 'Recorded separately; old tuning data may overlap new tests.'}
    (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def collect_results(output, completed):
    rows = []
    session_rows = []
    for run in completed:
        directory = output / run['name']
        summary = pd.read_csv(directory / 'final_test_summary.csv')
        empty = pd.read_csv(directory / 'final_test_class_metrics.csv')
        empty = empty[empty['Class'] == 'Empty'][['Scenario', 'Model', 'Recall', 'F1']]
        merged = summary.merge(empty.rename(columns={'Recall': 'Empty_Recall', 'F1': 'Empty_F1'}),
                               on=['Scenario', 'Model'], validate='one_to_one')
        merged.insert(0, 'Split_Seed', run['seed'])
        merged.insert(0, 'Split', run['name'])
        rows.append(merged)
        for target in 'ABC':
            for model in ['cnn', 'lstm']:
                sessions = pd.read_csv(directory / f'{target}_{model}_test_session_metrics.csv')
                sessions = sessions[sessions['Class'] == 'Empty'].copy()
                sessions.insert(0, 'Model', model)
                sessions.insert(0, 'Target', target)
                sessions.insert(0, 'Split', run['name'])
                session_rows.append(sessions)
    if not rows:
        return
    frame = pd.concat(rows, ignore_index=True)
    frame.to_csv(output / 'all_results.csv', index=False, encoding='utf-8-sig')
    pd.concat(session_rows, ignore_index=True).to_csv(
        output / 'empty_test_sessions.csv', index=False, encoding='utf-8-sig')
    random = frame[frame['Split'] != 'ordered']
    if not len(random):
        return
    statistics = random.groupby(['Scenario', 'Model'])[
        ['Accuracy', 'Macro_F1', 'Empty_Recall', 'Empty_F1']].agg(['mean', 'std', 'min', 'max'])
    statistics.columns = ['_'.join(column) for column in statistics.columns]
    statistics.to_csv(output / 'random_seed_statistics.csv', encoding='utf-8-sig')


def finalize_visuals(output, completed):
    # Scientific figures use Matplotlib; runs are repeated holdouts, not folds.
    from cha_gpt import plt, save_confusion_matrices
    frame = pd.read_csv(output / 'all_results.csv')
    random = frame[frame['Split'] != 'ordered']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title in zip(axes, ['Macro_F1', 'Empty_Recall'],
                                  ['Test Macro F1', 'Empty recall']):
        labels, means, errors, ordered = [], [], [], []
        for target in 'ABC':
            for model in ['1D-CNN', 'LSTM']:
                subset = random[(random['Scenario'] == f'{target} Auth') & (random['Model'] == model)]
                reference = frame[(frame['Split'] == 'ordered') &
                                  (frame['Scenario'] == f'{target} Auth') & (frame['Model'] == model)]
                labels.append(f'{target}\n{model}')
                means.append(subset[metric].mean())
                errors.append(subset[metric].std(ddof=1))
                ordered.append(reference[metric].iloc[0])
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=errors, capsize=4, alpha=.75, label='Random: mean +/- SD')
        ax.scatter(x, ordered, marker='D', color='darkred', label='Ordered (deterministic)')
        ax.set_xticks(x, labels)
        ax.set_ylim(0, max(1.06, max(np.asarray(means) + np.asarray(errors)) + .035))
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'split_comparison.png', dpi=200)
    plt.close(fig)
    for model in ['cnn', 'lstm']:
        total = np.zeros((3, 3), dtype=np.int64)
        for run in completed:
            if run['seed'] is not None:
                total += pd.read_csv(output / run['name'] /
                    f'cm_{model}_combined_final_test_counts.csv', index_col=0).to_numpy(dtype=np.int64)
        save_confusion_matrices(total, f'{model.upper()} - pooled random holdouts (repeated files)',
                                output / f'cm_{model}_random_pooled.png')


def main(args):
    if args.output_dir.exists():
        raise FileExistsError('기존 결과를 덮어쓰지 않도록 새 output-dir을 사용하세요.')
    files = {person: find_matching_files(person) for person in 'ABCD'}
    validate_unique_recordings(files)
    runs = [{'name': 'ordered', 'seed': None}] + [
        {'name': f'seed_{seed}', 'seed': seed} for seed in args.seeds]
    # Validate source settings against their ORIGINAL selection split before
    # allocating output or starting fitting. Random tests can overlap the old
    # tuning data; this protocol measures split sensitivity, not unbiased tuning.
    from evaluate_optuna import load_best_result, recording_hashes
    settings = []
    for target in 'ABC':
        original = make_split(files, target)
        for model in ['cnn', 'lstm']:
            path = args.best_dir / f'{target}_{model}_best.json'
            best, epochs = load_best_result(path, target, model, original)
            settings.append({'target': target, 'model': model, 'params': best['params'],
                             'epochs': epochs, 'source_file': str(path.resolve()),
                             'source_sha256': recording_hashes([str(path)])[0]['sha256'],
                             'optimization_training_determinism': best.get('training_determinism', False)})
    partitions = []
    for run in runs:
        for target in 'ABC':
            partitions.append({**run, 'target': target,
                               'files': make_split(files, target, split_seed=run['seed'])})
    args.output_dir.mkdir(parents=True)
    manifest = {'sessions_per_person': 20, 'split_seeds': args.seeds,
                'training_seed': 42, 'deterministic_ops': True, 'folds_applied': False,
                'hyperparameters_reoptimized': False, 'epochs_reselected': False,
                'interpretation': 'Split-sensitivity exploration; previous tuning files may '
                                  'occur in randomized tests. Repeated tests are not independent.',
                'settings': settings, 'partitions': partitions,
                'recordings': recording_hashes([p for paths in files.values() for p in paths])}
    (args.output_dir / 'experiment_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    completed = []
    started = time.monotonic()
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'TF_CPP_MIN_LOG_LEVEL': '2'}
    for index, run in enumerate(runs):
        print(f'RUN {index + 1}/{len(runs)}: {run["name"]}', flush=True)
        progress = {'status': 'running', 'current_run': run['name'],
                    'completed_runs': completed, 'total_runs': len(runs),
                    'elapsed_seconds': time.monotonic() - started}
        (args.output_dir / 'progress.json').write_text(json.dumps(progress, indent=2))
        command = [sys.executable, '-u', str(ROOT / 'evaluate_optuna.py'),
                   '--best-dir', str(args.best_dir), '--output-dir', str(args.output_dir / run['name']),
                   '--sessions-per-person', '20', '--deterministic']
        if run['seed'] is not None:
            command.extend(['--split-seed', str(run['seed'])])
        with (args.output_dir / f'{run["name"]}.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding='utf-8')
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            return_code = process.wait()
        if return_code:
            progress.update(status='failed', return_code=return_code)
            (args.output_dir / 'progress.json').write_text(json.dumps(progress, indent=2))
            raise RuntimeError(f'{run["name"]} 실행 실패: {return_code}')
        completed.append(run)
        collect_results(args.output_dir, completed)
    verify_results(args.output_dir, completed)
    finalize_visuals(args.output_dir, completed)
    progress.update(status='complete', current_run=None, completed_runs=completed,
                    elapsed_seconds=time.monotonic() - started)
    (args.output_dir / 'progress.json').write_text(json.dumps(progress, indent=2))
    print(f'완료: {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='기존 Optuna 설정으로 번호순/무작위 파일 분할 비교')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--best-dir', type=Path, default=ROOT / 'optuna_results_20261007_new')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2 or min(args.seeds) < 0:
        parser.error('서로 다른 0 이상 seed를 최소 2개 지정하세요.')
    args.output_dir = args.output_dir.resolve()
    args.best_dir = args.best_dir.resolve()
    main(args)
