import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

import pilot_cifar
from run_cifar_sequential import local_train,apply_aggregate


class PilotGateTests(unittest.TestCase):
    def config(self):
        return json.loads(Path('configs/stage4_pilot.json').read_text())

    def test_complete_registry_and_warmup_attack_separation(self):
        config=self.config();runs=pilot_cifar.plan_runs(config)
        self.assertEqual(len(runs),36)
        self.assertEqual(len({r['run_id'] for r in runs}),36)
        for run in runs:
            command=pilot_cifar.run_command(config,run,'data',Path('out'))
            self.assertIn('--offline',command)
            self.assertEqual('--declare-clean-warmup' in command,run['policy']=='clean-warmup')
            self.assertEqual(command[command.index('--family-runs')+1],'36')
            self.assertEqual(command[command.index('--rounds')+1],'20')

    def test_failed_data_gate_starts_no_training_and_records_all_unrun(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            def fail(command,**kwargs):
                gate=Path(command[command.index('--out')+1])
                gate.write_text(json.dumps(dict(smoke_passed=False,batch_md5_verified=False,error='fixture: unavailable real data')))
                return subprocess.CompletedProcess(command,2)
            with patch('sys.argv',['pilot','--out',str(out)]),patch.object(pilot_cifar.subprocess,'run',side_effect=fail) as process:
                with self.assertRaises(SystemExit) as error:
                    pilot_cifar.main()
                self.assertEqual(error.exception.code,2)
                self.assertEqual(process.call_count,1)
            status=json.loads((out/'pilot_status.json').read_text())
            self.assertEqual(status['runs_started'],0)
            self.assertFalse(status['metrics_available'])
            self.assertFalse((out/'pilot_runs').exists())
            registry=pilot_cifar.pd.read_csv(out/'run_registry.csv')
            self.assertTrue((registry.status=='NOT_RUN').all())

    def test_existing_started_family_cannot_be_silently_restarted(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            prior=json.dumps(dict(runs_started=1,status='COMPLETED_OR_FAILED'))
            (out/'pilot_status.json').write_text(prior)
            with patch('sys.argv',['pilot','--out',str(out)]),patch.object(pilot_cifar.subprocess,'run') as process:
                with self.assertRaisesRegex(RuntimeError,'restart'):
                    pilot_cifar.main()
                process.assert_not_called()
            self.assertEqual((out/'pilot_status.json').read_text(),prior)

    def test_actual_sgd_step_instrumentation_on_tensor_fixture(self):
        model=nn.Linear(2,2)
        fixture=TensorDataset(torch.ones(16,2),torch.zeros(16,dtype=torch.long))
        stats={}
        local_train(model,DataLoader(fixture,batch_size=8),'cpu',stats=stats)
        self.assertEqual(stats['optimizer_steps'],2)
        self.assertEqual(len(stats['losses']),2)
        self.assertEqual(stats['model_device'],stats['image_device'])
        self.assertEqual(stats['model_device'],stats['label_device'])

    def test_overflow_rejected_before_server_mutation(self):
        model=nn.Linear(1,1,bias=False)
        model.weight.data.fill_(3e38)
        before=model.weight.detach().clone()
        with self.assertRaises(FloatingPointError):
            apply_aggregate(model,[{'weight':torch.tensor([[3e38]])}],[1.])
        torch.testing.assert_close(model.weight,before)


if __name__=='__main__':
    unittest.main()
