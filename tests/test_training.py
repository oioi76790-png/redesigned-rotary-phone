import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run_cifar_sequential as app


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        app.seed_all(17)
        self.data = TensorDataset(torch.randn(10, 3, 32, 32), torch.arange(10))

    def test_partition_coverage_repeatability_and_empty_clients(self):
        parts = app.dirichlet_partition(np.arange(10), 20, .01, 1)
        self.assertTrue(any(not ids for ids in parts))
        self.assertEqual(sorted(i for p in parts for i in p), list(range(10)))
        self.assertEqual(parts, app.dirichlet_partition(np.arange(10), 20, .01, 1))
        selected = app.select_participants(parts, 1., np.random.default_rng(1))
        self.assertTrue(all(parts[i] for i in selected))

    def test_empty_round_is_noop(self):
        model = nn.Linear(2, 2)
        before = app.load_state(model)
        selected, updates, weights = app.train_round(
            model, self.data, [[], []], 'cpu', seed=1, round_id=1)
        self.assertEqual((selected, updates, len(weights)), ([], [], 0))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_label_mapping_is_derangement_repeatable_and_nonmutating(self):
        first = app.LabelFlippedDataset(self.data, 11)
        second = app.LabelFlippedDataset(self.data, 11)
        np.testing.assert_array_equal(first.mapping, second.mapping)
        self.assertEqual(sorted(first.mapping.tolist()), list(range(10)))
        self.assertTrue(np.all(first.mapping != np.arange(10)))
        for index in reversed(range(10)):
            image, label = first[index]
            self.assertNotEqual(label, int(self.data[index][1]))
            self.assertEqual(label, second[index][1])
            torch.testing.assert_close(image, self.data[index][0])
        torch.testing.assert_close(self.data.tensors[1], torch.arange(10))

    def test_label_attack_changes_actual_local_gradient(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(3*32*32, 10))
        clean = app.local_train(model, DataLoader(self.data, batch_size=10), 'cpu')
        poisoned = app.local_train(model, DataLoader(app.LabelFlippedDataset(self.data, 3), batch_size=10), 'cpu')
        self.assertTrue(any(not torch.equal(clean[k], poisoned[k]) for k in clean))
        repeat = app.local_train(model, DataLoader(app.LabelFlippedDataset(self.data, 3), batch_size=10), 'cpu')
        for key in poisoned:
            torch.testing.assert_close(poisoned[key], repeat[key], rtol=0, atol=0)

    def test_attack_schedule_and_distinct_start(self):
        self.assertEqual([app.attack_active(t, 3, 'intermittent', 4, 2) for t in range(1, 10)],
                         [False, False, True, True, False, False, True, True, False])
        self.assertFalse(app.attack_active(2, 3))
        self.assertTrue(app.attack_active(20, 3))

    def test_round_applies_attacks_and_keeps_empty_clients_out(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(3*32*32, 10))
        parts = [list(range(5)), [], list(range(5, 10))]
        def run(attack):
            return app.train_round(copy.deepcopy(model), self.data, parts, 'cpu',
                                   seed=2, round_id=1, attack=attack, attack_start=1,
                                   malicious_fraction=1.)
        selected, clean, weights = run('none')
        self.assertEqual(selected, [0, 2])
        np.testing.assert_allclose(weights, [.5, .5])
        _, flipped, _ = run('signflip')
        _, labels, _ = run('labelflip')
        for c, f in zip(clean, flipped):
            for key in c:
                torch.testing.assert_close(f[key], -c[key])
        self.assertTrue(any(not torch.equal(clean[0][k], labels[0][k]) for k in clean[0]))

    def test_participation_normalizes_actual_sample_counts(self):
        parts = [list(range(2)), list(range(2, 5)), [], list(range(5, 10))]
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()
        with patch.object(app, 'local_train', side_effect=lambda *args: {'weight': torch.ones(1, 1)}):
            for round_id, fraction in enumerate([1., .5, .1, 1.], start=1):
                before = model.weight.item()
                selected, _, weights = app.train_round(
                    model, self.data, parts, 'cpu', seed=3, round_id=round_id,
                    participation=fraction, attack='none')
                expected = np.array([len(parts[c]) for c in selected], dtype=float)
                np.testing.assert_allclose(weights, expected/expected.sum())
                self.assertAlmostEqual(model.weight.item()-before, 1.)

    def test_fedavg_matches_weighted_client_endpoints(self):
        model = nn.Linear(2, 1, bias=False)
        model.weight.data.fill_(3.)
        weights = app.apply_aggregate(model, [{'weight':torch.tensor([[1., 2.]])},
                                             {'weight':torch.tensor([[5., 6.]])}], [2, 6])
        np.testing.assert_allclose(weights, [.25, .75])
        torch.testing.assert_close(model.weight, torch.tensor([[7., 8.]]))

    def test_invalid_aggregate_is_atomic(self):
        model = nn.Linear(1, 1, bias=False)
        before = model.weight.detach().clone()
        for updates, weights in [([{'weight':torch.ones(1,1)}], []),
                                 ([{'weight':torch.ones(1,1)}], [0]),
                                 ([{'weight':torch.ones(1,1)}], [-1]),
                                 ([{'weight':torch.ones(1,1)}], [float('nan')]),
                                 ([{'weight':torch.ones(2,1)}], [1]),
                                 ([{'weight':torch.full((1,1), float('nan'))}], [1]),
                                 ([{'bad':torch.ones(1,1)}], [1])]:
            with self.assertRaises(ValueError):
                app.apply_aggregate(model, updates, weights)
            torch.testing.assert_close(model.weight, before, rtol=0, atol=0)

    def test_local_training_global_unchanged_and_cpu_updates(self):
        model = app.SmallCNN()
        before = app.load_state(model)
        update = app.local_train(model, DataLoader(self.data, batch_size=5), 'cpu', prox=.01)
        self.assertTrue(any(torch.count_nonzero(value) for value in update.values()))
        for key, value in update.items():
            self.assertEqual(value.device.type, 'cpu')
            self.assertTrue(torch.isfinite(value).all())
            torch.testing.assert_close(model.state_dict()[key], before[key], rtol=0, atol=0)
        app.apply_aggregate(model, [update], [10])
        loss, accuracy = app.evaluate(model, DataLoader(self.data, batch_size=5), 'cpu')
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(0 <= accuracy <= 1)

    def test_signed_features_detect_opposite_direction(self):
        update = {'w':torch.linspace(.01, .1, 64)}
        clean = app.update_features(update)
        flip = app.update_features({'w':-update['w']})
        np.testing.assert_allclose(clean, -flip)
        self.assertLessEqual(np.linalg.norm(clean), 1.)
        self.assertGreater(np.dot(clean, clean), 0.)

    def test_cli_minimal_pipeline_with_trusted_history_and_no_download(self):
        data = self.data
        class LocalCIFAR:
            def __init__(self, *args, **kwargs):
                self.targets = list(range(10))
            def __len__(self):
                return len(data)
            def __getitem__(self,index):
                return data[index]
        original_predict = app.ClientHistory.predict
        original_round = app.train_round
        predictions = []
        def predict(history,*args,**kwargs):
            prediction = original_predict(history,*args,**kwargs)
            predictions.append(prediction)
            return prediction
        def train(*args,**kwargs):
            # All clients' snapshots must precede each current-round training.
            self.assertEqual(len(predictions),3*kwargs['round_id'])
            return original_round(*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root/'trusted.json'
            history.write_text(json.dumps({str(c):[dict(features=[.1]*16,provenance='test external source') for _ in range(8)] for c in range(3)}))
            argv = ['run','--clients','3','--rounds','2','--participation','.5',
                    '--attack','labelflip','--attack-start','1','--out',str(root),
                    '--trusted-history',str(history)]
            with patch.object(app.datasets,'CIFAR10',LocalCIFAR), patch('sys.argv',argv), \
                    patch.object(app.ClientHistory,'predict',predict), patch.object(app,'train_round',train):
                app.main()
            logs = app.pd.read_csv(root/'clients_seed1.csv')
            self.assertTrue(logs.detector_enabled.all())
            self.assertTrue((logs.trusted_count==8).all())
            np.testing.assert_allclose(logs.groupby('round').weight.sum(),[1.,1.])
            self.assertEqual(len(app.pd.read_csv(root/'rounds_seed1.csv')),2)

    def test_nonfloating_buffers_preserved(self):
        model = nn.BatchNorm1d(2)
        updates = {k:torch.ones_like(v) for k,v in model.state_dict().items()}
        before = model.num_batches_tracked.clone()
        app.apply_aggregate(model, [updates], [1])
        torch.testing.assert_close(model.num_batches_tracked, before)

    def test_bad_partition_and_empty_evaluation(self):
        for clients, alpha in [(0,.5), (2,0), (2,float('nan'))]:
            with self.assertRaises(ValueError):
                app.dirichlet_partition(np.arange(10), clients, alpha, 1)
        with self.assertRaises(ValueError):
            app.evaluate(app.SmallCNN(), [], 'cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable: actual GPU path not executed')
    def test_cuda_train_flatten_aggregate_and_evaluate(self):
        model = app.SmallCNN().cuda()
        before = app.load_state(model)
        update = app.local_train(model, DataLoader(self.data, batch_size=5), 'cuda')
        app.apply_aggregate(model, [update], [10])
        self.assertTrue(all(value.device.type == 'cuda' for value in model.state_dict().values()))
        vector = app.flat_update(model.state_dict(), before)
        self.assertEqual(vector.device.type, 'cpu')
        self.assertTrue(torch.isfinite(vector).all())
        loss, _ = app.evaluate(model, DataLoader(self.data, batch_size=5), 'cuda')
        self.assertTrue(np.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
