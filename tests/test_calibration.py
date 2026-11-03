import copy
import unittest
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from calibration import (BatchEProcess, ExternalReference, ExternalCleanWindow,
                         fixed_history_bound, rolling_history_bound,
                         conservative_bound, hoeffding_radius)
from seta_statistics import BoundedMeanEProcess
import run_cifar_sequential as app


class CalibrationTests(unittest.TestCase):
    def test_width_and_union_parameters(self):
        a=hoeffding_radius(100,.01,width=2.)
        self.assertAlmostEqual(hoeffding_radius(100,.01,width=1.),a/2)
        self.assertGreater(hoeffding_radius(100,.01,comparisons=240),a)
        self.assertGreater(fixed_history_bound(100,.01),a)

    def test_conservative_bound_covers_all_means(self):
        for estimate in np.linspace(-1,1,21):
            self.assertGreaterEqual(conservative_bound(estimate),max(abs(-1-estimate),abs(1-estimate)))

    def test_rolling_includes_context_shift(self):
        a=rolling_history_bound(1000,.005,240,.2,0.)
        b=rolling_history_bound(1000,.005,240,.2,1.)
        self.assertAlmostEqual(float(b-a),.2)

    def test_delayed_clean_reference_only(self):
        window=ExternalCleanWindow(2)
        reference=ExternalReference(.5,0.,1,2,'independent external validation')
        with self.assertRaises(ValueError):
            window.admit(reference,current_round=1)
        with self.assertRaises(ValueError):
            window.admit(.5,current_round=2)
        window.admit(reference,current_round=2)
        frozen=window.snapshot()
        window.admit(ExternalReference(.6,1.,2,3,'clean'),current_round=3)
        self.assertEqual(len(frozen),1)
        with self.assertRaises(ValueError):
            ExternalReference(.5,0.,1,1,'clean')

    def test_batch_matches_scalar_process(self):
        rng=np.random.default_rng(1)
        batch=BatchEProcess((3,2))
        scalar=[[BoundedMeanEProcess() for _ in range(2)] for _ in range(3)]
        for _ in range(30):
            x=rng.uniform(-1,1,(3,1)); mean=rng.uniform(-1,1,(3,2)); bound=rng.uniform(0,.5,(3,2))
            actual=batch.update(x,mean,bound)
            for i in range(3):
                for j in range(2):
                    expected,_=scalar[i][j].update(float(x[i,0]),float(mean[i,j]),float(bound[i,j]))
                    self.assertAlmostEqual(actual[i,j],expected,places=11)

    def test_filter_precedes_aggregation_and_all_rejected_is_noop(self):
        from unittest.mock import patch
        model=nn.Linear(1,1,bias=False)
        before=model.weight.detach().clone()
        dataset=TensorDataset(torch.ones(2,1),torch.zeros(2,dtype=torch.long))
        seen=[]
        def observer(client,update):
            torch.testing.assert_close(model.weight,before)
            seen.append(client)
            return client==0
        with patch.object(app,'local_train',return_value={'weight':torch.ones(1,1)}):
            _,_,weights=app.train_round(model,dataset,[[0],[1]],'cpu',seed=1,round_id=1,
                                       update_observer=observer,aggregation='evidence-filter')
        self.assertEqual(seen,[0,1])
        np.testing.assert_array_equal(weights,[0.,1.])
        torch.testing.assert_close(model.weight,before+1.)
        before=model.weight.detach().clone()
        with patch.object(app,'local_train',return_value={'weight':torch.ones(1,1)}):
            _,_,weights=app.train_round(model,dataset,[[0],[1]],'cpu',seed=1,round_id=1,
                                       update_observer=lambda *args:True,aggregation='evidence-filter')
        np.testing.assert_array_equal(weights,[0.,0.])
        torch.testing.assert_close(model.weight,before)

    def test_stress_pairing_and_no_detector_metric_semantics(self):
        import json
        from pathlib import Path
        from stress_validation import run_cell
        config=json.loads(Path('configs/stage3.json').read_text())
        config.update(trials_per_cell=10,horizon=12,onset=5,reputation_onset=9)
        frame,summary,_=run_cell(config,'normal_dynamic_drift',0.)
        baseline=next(row for row in summary if row['method']=='conditional')
        self.assertIsNone(baseline['false_alarm_rate'])
        for alpha in config['alphas']:
            fixed=frame[(frame.method=='conditional_seq_fixed')&(frame.alpha==alpha)]
            full=frame[(frame.method=='full_seta_candidate')&(frame.alpha==alpha)]
            np.testing.assert_array_equal(fixed.first_alarm,full.first_alarm)
        self.assertEqual(len(frame),10*3*7)

    def test_unobserved_drift_is_not_a_valid_fixed_envelope(self):
        import json
        from pathlib import Path
        from stress_validation import run_cell
        config=json.loads(Path('configs/stage3_misspecification.json').read_text())
        config.update(trials_per_cell=10,horizon=12,onset=5)
        _,summary,_=run_cell(config,'normal_unobserved_dynamic_drift',0.)
        fixed=next(row for row in summary if row['method']=='conditional_seq_fixed')
        conservative=next(row for row in summary if row['method']=='conditional_seq_conservative')
        self.assertEqual(fixed['benign_envelope_violation_rate'],1.)
        self.assertEqual(conservative['benign_envelope_violation_rate'],0.)

    def test_local_archive_rejects_bad_checksum_before_extraction(self):
        import tempfile
        from pathlib import Path
        from cifar_smoke import import_official_archive
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory)/'bad.tar.gz'
            archive.write_bytes(b'not CIFAR-10')
            root=Path(directory)/'output'
            with self.assertRaisesRegex(ValueError,'MD5'):
                import_official_archive(archive,root)
            self.assertFalse(root.exists())


if __name__=='__main__':
    unittest.main()
