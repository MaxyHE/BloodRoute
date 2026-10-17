"""Verify saved follow-up predictions; optionally reproduce routing resampling."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / 'results/validation-20261014'
ROUTING = ROOT / 'results/routing-controls-20261016'
CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'immature_granulocytes',
           'lymphocyte', 'monocyte', 'neutrophil', 'platelet']


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    result = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    assert len({r['source_id'] for r in result}) == len(result)
    return result


def metrics(labels, predictions, classes):
    nclass = len(classes)
    cm = [[0] * (nclass + 1) for _ in classes]
    for label, pred in zip(labels, predictions):
        assert label in classes
        cm[classes.index(label)][classes.index(pred) if pred in classes else nclass] += 1
    n = sum(map(sum, cm))
    correct = sum(cm[i][i] for i in range(nclass))
    f1, recall = [], []
    for i in range(nclass):
        true = sum(cm[i])
        predicted = sum(row[i] for row in cm)
        recall.append(cm[i][i] / true if true else 0)
        f1.append(2 * cm[i][i] / (true + predicted) if true + predicted else 0)
    return dict(n=n, correct=correct, accuracy=correct / n, macro_f1=mean(f1),
                balanced_accuracy=mean(recall), invalid=sum(row[-1] for row in cm))


def compare(actual, expected):
    for key in ('n', 'accuracy', 'macro_f1', 'balanced_accuracy', 'invalid'):
        if key in expected:
            assert abs(actual[key] - expected[key]) < 1e-10, (key, actual[key], expected[key])


def validate_images():
    reference = load(VALIDATION / 'results.json')
    exact = set(reference['data_audit']['exact_test_duplicates'])
    near = set(reference['data_audit']['near_candidates_including_exact'])
    all_predictions = {}
    for variant in ('base', 'lora'):
        predictions = {}
        for condition in ('normal', 'swapped', 'blank', 'text_only'):
            sample = rows(VALIDATION / variant / (condition + '.jsonl'))
            assert len(sample) == 800
            labels = [CLASSES[r['label']] for r in sample]
            assert all(labels.count(c) == 100 for c in CLASSES)
            pred = [r['parsed'] for r in sample]
            expected = reference['metrics'][variant][condition]
            compare(metrics(labels, pred, CLASSES), expected['original_label'])
            if condition in ('blank', 'text_only'):
                assert len(set(pred)) == 1 and expected['actual_unique_inferences'] == 1
            if condition == 'swapped':
                assert len({r['donor_source_id'] for r in sample}) == 800
                assert {r['donor_source_id'] for r in sample} == {r['source_id'] for r in sample}
                assert all(r['label'] != r['donor_label'] for r in sample)
                donor_labels = [CLASSES[r['donor_label']] for r in sample]
                compare(metrics(donor_labels, pred, CLASSES), expected['donor_label'])
            predictions[condition] = {r['source_id']: r for r in sample}
            actual = metrics(labels, pred, CLASSES)
            print(f"{variant} {condition}: Accuracy={actual['accuracy']:.4%}, Macro-F1={actual['macro_f1']:.4%}")
        normal = predictions['normal']
        for sid, r in predictions['swapped'].items():
            assert r['parsed'] == normal[r['donor_source_id']]['parsed']
        sample = rows(VALIDATION / variant / 'visa_predictions.jsonl')
        assert len(sample) == 723
        compare(metrics([r['label'] for r in sample], [r['parsed'] for r in sample], ['no', 'yes']),
                reference['metrics'][variant]['visa'])
        all_predictions[variant] = normal
    assert set(all_predictions['base']) == set(all_predictions['lora'])
    corrected = harmed = 0
    for sid, r in all_predictions['base'].items():
        truth = CLASSES[r['label']]
        a, b = r['parsed'] == truth, all_predictions['lora'][sid]['parsed'] == truth
        corrected += not a and b
        harmed += a and not b
    assert corrected == reference['normal_pair']['corrected'] == 289
    assert harmed == reference['normal_pair']['harmed'] == 5
    sensitivity = load(VALIDATION / 'historical_duplicate_sensitivity.json')
    for public_name, filename in [('lora', 'vlm'), ('frozen_cnn', 'cnn'), ('route', 'route')]:
        sample = rows(ROOT / 'results/fulltrain-20261005' / (filename + '-predictions.jsonl'))
        for name, excluded in [('historical_full', set()), ('exclude_exact_duplicates', exact),
                               ('exclude_all_phash_candidates_sensitivity', near)]:
            filtered = [r for r in sample if r['source_id'] not in excluded]
            compare(metrics([r['label'] for r in filtered], [r['prediction'] for r in filtered], list(range(8))),
                    sensitivity[public_name][name])
    audit = load(VALIDATION / 'data_audit.json')
    assert audit['exact_cross_split_group_count'] == len(audit['exact_cross_split_groups']) == 6
    assert all(not overlap for overlap in audit['id_overlap'].values())
    print('Verified image interventions, paired corrections, VisA and duplicate sensitivity; no retraining.')


def validate_routes(resample):
    reference = load(ROUTING / 'results.json')
    sample = rows(ROUTING / 'per_image_replay.jsonl')
    assert len(sample) == 3421 and [r['source_id'] for r in sample] == sorted(r['source_id'] for r in sample)
    threshold = reference['development_selection_reproduced']['main']['threshold']
    assert all(r['called'] == (r['confidence'] < threshold) for r in sample)
    assert all(r['route'] == (r['vlm'] if r['called'] else r['cnn']) for r in sample)
    masks = {
        'cnn_only': [False] * len(sample), 'full_vlm': [True] * len(sample),
        'low_confidence': [r['called'] for r in sample],
        'high_confidence_diagnostic': [r['high_confidence_control_called'] for r in sample],
    }
    truth = [r['label'] for r in sample]
    for arm, mask in masks.items():
        actual = metrics(truth, [r['vlm'] if take else r['cnn'] for r, take in zip(sample, mask)], list(range(8)))
        compare(actual, reference['arms'][arm])
        assert sum(mask) == reference['arms'][arm]['calls']
        corrected = sum(take and r['cnn'] != r['label'] and r['vlm'] == r['label'] for r, take in zip(sample, mask))
        harmed = sum(take and r['cnn'] == r['label'] and r['vlm'] != r['label'] for r, take in zip(sample, mask))
        assert corrected == reference['arms'][arm]['corrected'] and harmed == reference['arms'][arm]['harmed']
    with (ROUTING / 'random_replicates.csv').open() as handle:
        draws = list(csv.DictReader(handle))
    random = reference['random_control']
    assert len(draws) == random['repetitions'] == 10000
    assert abs(mean(int(r['correct']) / len(sample) for r in draws) - random['accuracy_mean']) < 1e-12
    assert abs(mean(float(r['macro_f1']) for r in draws) - random['macro_f1_mean']) < 1e-12
    exact = reference['arms']['cnn_only']['accuracy'] + sum(masks['low_confidence']) / len(sample) * (
        reference['arms']['full_vlm']['accuracy'] - reference['arms']['cnn_only']['accuracy'])
    assert abs(exact - random['accuracy_expected_exact']) < 1e-12
    print(f"Verified equal-budget controls: low-confidence=97.4861%, random mean={random['accuracy_mean']:.4%}.")
    if resample:
        reproduce_resampling(sample, draws, reference)


def reproduce_resampling(sample, draws, reference):
    import numpy as np
    protocol = load(ROUTING / 'protocol.json')
    y, cnn, vlm, route = [np.array([r[k] for r in sample]) for k in ('label', 'cnn', 'vlm', 'route')]
    n, k = len(y), sum(r['called'] for r in sample)

    def score(labels, pred):
        cm = np.bincount(labels * 8 + pred, minlength=64).reshape(8, 8)
        denom = cm.sum(0) + cm.sum(1)
        f1 = np.divide(2 * cm.diagonal(), denom, out=np.zeros(8), where=denom > 0)
        return cm.trace() / len(labels), f1.mean()

    rng = np.random.default_rng(protocol['random_seed'])
    for archived in draws:
        mask = np.zeros(n, dtype=bool)
        mask[rng.choice(n, size=k, replace=False)] = True
        pred = np.where(mask, vlm, cnn)
        accuracy, f1 = score(y, pred)
        assert round(accuracy * n) == int(archived['correct'])
        assert abs(f1 - float(archived['macro_f1'])) < 1e-12
    rng = np.random.default_rng(protocol['paired_bootstrap_seed'])
    boot = {key: [] for key in ('cnn_only', 'full_vlm')}
    for _ in range(protocol['paired_bootstrap_replicates']):
        indices = rng.integers(0, n, size=n)
        a = np.array(score(y[indices], route[indices]))
        for key, pred in [('cnn_only', cnn), ('full_vlm', vlm)]:
            boot[key].append(a - score(y[indices], pred[indices]))
    for key, values in boot.items():
        for j, field in enumerate(('accuracy', 'macro_f1')):
            np.testing.assert_allclose(np.quantile(np.array(values)[:, j], [0.025, 0.975]),
                                       reference['paired_intervals'][key][field]['paired_image_bootstrap_95'], atol=1e-12)
    print('Reproduced 10000 random subsets and 5000 paired image bootstrap samples with frozen seeds.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resample', action='store_true', help='Requires NumPy; reproduce original routing random draws and bootstrap.')
    args = parser.parse_args()
    validate_images()
    validate_routes(args.resample)
