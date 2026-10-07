"""Diagnose excluded D recordings with saved models, without fitting or selection."""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from csi_pipeline import PREPROCESSING_VERSION, find_matching_files, parse_csi_to_windows
from evaluate_optuna import MODEL_NAMES, recording_hashes, save_prediction_diagnostics, tf


def main(args):
    evaluation_dir = Path(args.evaluation_dir).resolve()
    output = Path(args.output_dir).resolve()
    manifest = json.loads((evaluation_dir / 'evaluation_manifest.json').read_text(encoding='utf-8'))
    available = find_matching_files('D')
    all_rows, source_records = [], []
    for job in manifest:
        if job.get('sessions_per_person') != 15:
            raise ValueError('이 진단은 대상별 15개 실험의 저장된 모델을 사용해야 합니다.')
        used = {str(Path(path).resolve()) for path in job['train_files'] + job['test_files']}
        excluded = [path for path in available if str(Path(path).resolve()) not in used]
        if not excluded:
            raise ValueError('실험에서 제외된 D 파일이 없습니다.')
        target, model_name = job['authorized_person'], job['model']
        tag = f'{target}_{model_name}'
        with np.load(evaluation_dir / f'{tag}_preprocessing.npz') as stored:
            if int(stored['preprocessing_version']) != PREPROCESSING_VERSION:
                raise ValueError('저장된 모델과 현재 전처리 버전이 다릅니다.')
            features, metadata = parse_csi_to_windows(
                excluded, int(stored['feature_dim']), window_sec=float(stored['window_sec']),
                stride_sec=float(stored['stride_sec']), target_fps=int(stored['target_fps']),
                max_gap_sec=float(stored['max_gap_sec']), is_empty=True)
            # Reuse training statistics; never estimate them from the excluded data.
            features = (features - stored['mean']) / stored['std']
        for row in metadata:
            row['Class'] = 'Empty'
        labels = np.full(len(features), 2, dtype=np.int64)
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(evaluation_dir / job['model_file'], compile=False)
        probabilities = None
        try:
            probabilities = model.predict(features, batch_size=job['params']['batch_size'], verbose=0)
            output.mkdir(parents=True, exist_ok=True)
            prefix = output / tag
            save_prediction_diagnostics(prefix, 'excluded_empty', labels, metadata,
                                        probabilities, target, MODEL_NAMES[model_name])
            sessions = pd.read_csv(f'{prefix}_excluded_empty_session_metrics.csv')
            sessions.insert(0, 'Model', MODEL_NAMES[model_name])
            sessions.insert(0, 'Scenario', f'{target} Auth')
            all_rows.extend(sessions.to_dict('records'))
            source_records.append({'scenario': target, 'model': model_name,
                                   'recordings': recording_hashes(excluded)})
            for row in sessions.to_dict('records'):
                print(f'{tag} {row["Session"]}: Empty recall={row["Accuracy"]:.4f}', flush=True)
        finally:
            del model, features, probabilities
            tf.keras.backend.clear_session()
            gc.collect()
    pd.DataFrame(all_rows).to_csv(output / 'excluded_empty_by_file.csv', index=False, encoding='utf-8-sig')
    (output / 'diagnosis_manifest.json').write_text(json.dumps({
        'source_evaluation': str(evaluation_dir),
        'model_training_performed': False, 'used_for_parameter_selection': False,
        'separate_from_15_session_final_test': True, 'sources': source_records,
    }, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='15개 실험에서 제외된 빈방의 추가 예측 진단')
    parser.add_argument('--evaluation-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    main(parser.parse_args())
