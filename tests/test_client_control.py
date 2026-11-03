import unittest
import numpy as np
import torch
from torch import nn

from client_control import (ErrorBudget,ClientController,ClientState,ContextualHistory,
                            RoundContext,model_rms,pre_update_context)
from run_cifar_sequential import apply_aggregate


class ClientControlTests(unittest.TestCase):
    def controller(self,**kwargs):
        return ClientController(ErrorBudget(.05,2),**kwargs)

    def context(self,round_id,value=0.):
        return RoundContext(round_id,value,1.,1.)

    def seed_history(self,controller,client=0):
        for context in np.linspace(-1,1,8):
            controller.add_external(client,[.4+.1*context],float(context),provenance='external test source')

    def test_family_budget_clients_features_runs(self):
        budget=ErrorBudget(.05,10,features=3,runs=36)
        self.assertAlmostEqual((budget.alpha_e+budget.delta_calibration)*10*3*36,.05)
        self.assertAlmostEqual(budget.as_dict()['family_sum'],.05)
        for kwargs in [dict(clients=0),dict(clients=1,features=0),dict(clients=1,runs=True),
                       dict(clients=1,calibration_fraction=1.)]:
            with self.assertRaises(ValueError):
                ErrorBudget(.05,**kwargs)

    def test_context_precedes_and_does_not_follow_current_update(self):
        model=nn.Linear(2,1,bias=False)
        model.weight.data.fill_(1.)
        before=pre_update_context(model,1,model_rms(model))
        model.weight.data.fill_(2.)
        self.assertEqual(before.value,0.)
        self.assertGreater(pre_update_context(model,2,1.).value,0.)
        self.assertEqual(before.current_global_rms,1.)

    def test_unknown_is_not_normal_or_auto_calibrated(self):
        controller=self.controller()
        controller.prepare_round(self.context(1))
        controller.observe(0,[.5])
        self.assertEqual(controller.states[0],ClientState.UNCALIBRATED)
        self.assertEqual(len(controller.histories[0].references),0)
        self.assertEqual(len(controller.histories[0].quarantine),1)
        self.assertFalse(controller.row(0,selected=True,weight=0.)['detector_enabled'])
        with self.assertRaises(ValueError):
            controller.admit_declared_warmup(0,[.5])
        with self.assertRaises(ValueError):
            controller.add_external(0,[.5],0.,provenance='late')

    def test_three_state_transitions_and_alarm_latch(self):
        controller=self.controller(assumed_bound=0.)
        self.seed_history(controller)
        self.assertEqual(controller.states[0],ClientState.UNCALIBRATED)
        for round_id in range(1,31):
            controller.prepare_round(self.context(round_id))
            controller.observe(0,[-1.])
        self.assertEqual(controller.states[0],ClientState.ALARMED)
        self.assertEqual(controller.states[1],ClientState.UNCALIBRATED)
        controller.prepare_round(self.context(31))
        controller.observe(0,[.5])
        self.assertEqual(controller.states[0],ClientState.ALARMED)
        self.assertEqual(len(controller.histories[0].references),8)

    def test_defer_excludes_unknown_and_retains_server(self):
        controller=self.controller(policy='defer')
        controller.prepare_round(self.context(1))
        weights,server=controller.allocate([0,1],[10,30],[False,False])
        np.testing.assert_array_equal(weights,[0.,0.]);self.assertEqual(server,1.)

    def test_cap_not_cancelled_by_aggregation_renormalization(self):
        controller=self.controller(policy='capped',weight_cap=.02,group_cap=.2)
        controller.prepare_round(self.context(1))
        weights,server=controller.allocate([0,1],[10,30],[False,False])
        np.testing.assert_allclose(weights,[.02,.02]);self.assertAlmostEqual(server,.96)
        model=nn.Linear(1,1,bias=False);model.weight.data.zero_()
        actual=apply_aggregate(model,[{'weight':torch.ones(1,1)}]*2,weights,server_weight=server)
        np.testing.assert_allclose(actual,weights)
        self.assertAlmostEqual(model.weight.item(),.04,places=7)

    def test_monitored_clients_get_remaining_mass(self):
        controller=self.controller(policy='capped')
        self.seed_history(controller,client=0)
        controller.prepare_round(self.context(1))
        weights,server=controller.allocate([0,1],[10,30],[False,False])
        np.testing.assert_allclose(weights,[.98,.02]);self.assertEqual(server,0.)
        weights,server=controller.allocate([0,1],[10,30],[True,False])
        np.testing.assert_allclose(weights,[0.,.02]);self.assertAlmostEqual(server,.98)

    def test_warmup_requires_declaration_and_does_not_auto_admit(self):
        with self.assertRaises(ValueError):
            self.controller(policy='clean-warmup')
        controller=self.controller(policy='clean-warmup',warmup_rounds=8,declare_clean_warmup=True)
        for round_id in range(1,9):
            controller.prepare_round(self.context(round_id,.01*round_id))
            controller.observe(0,[.5])
            self.assertEqual(len(controller.histories[0].references),round_id-1)
            controller.admit_declared_warmup(0,[.5])
            self.assertEqual(controller.states[0],ClientState.UNCALIBRATED)
            with self.assertRaises(ValueError):
                controller.admit_declared_warmup(0,[.5])
        controller.prepare_round(self.context(9,.09))
        self.assertEqual(controller.states[0],ClientState.MONITORED)
        controller.observe(0,[-.5])
        with self.assertRaises(ValueError):
            controller.admit_declared_warmup(0,[-.5])
        self.assertEqual(len(controller.histories[0].references),8)

    def test_context_model_really_uses_context_and_keeps_snapshot(self):
        history=ContextualHistory()
        for context in np.linspace(-1,1,16):
            history.admit([.4+.2*context],float(context),provenance='test')
        low,_=history.predict(-.5)
        high,details=history.predict(.5)
        self.assertGreater(high.mean,low.mean+.1)
        self.assertEqual(details['certificate'],'ASSUMED_NOT_CERTIFIED')
        previous=high.mean
        history.observe([-.9],1)
        self.assertEqual(history.predict(.5)[0].mean,previous)
        self.assertEqual(len(history.references),16)
        conservative,_=history.predict(.5,bound_mode='conservative')
        self.assertEqual(conservative.mean_error_bound,1+abs(conservative.mean))

    def test_missing_context_rejected(self):
        with self.assertRaises(ValueError):
            ContextualHistory().admit([.5],float('nan'),provenance='test')
        controller=self.controller()
        controller.prepare_round(self.context(1))
        with self.assertRaises(ValueError):
            controller.prepare_round(self.context(1))
        controller.observe(0,[.5])
        with self.assertRaises(ValueError):
            controller.observe(0,[.5])

    def test_logged_evidence_replays_without_reset(self):
        import tempfile
        from pathlib import Path
        import pandas as pd
        from monitoring_audit import replay_client_logs
        controller=self.controller(assumed_bound=0.)
        self.seed_history(controller)
        records=[]
        for round_id in range(1,21):
            controller.prepare_round(self.context(round_id))
            controller.observe(0,[-1.])
            records.append(controller.row(0,selected=True,weight=0.))
            records.append(controller.row(1,selected=False,weight=0.))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'clients.csv'
            pd.DataFrame(records).to_csv(path,index=False)
            result=replay_client_logs(path)
            self.assertEqual(result['client_processes_with_observations'],1)
            self.assertGreater(result['scalar_bets_replayed'],0)
            frame=pd.read_csv(path)
            index=frame[frame.detector_enabled&frame.selected].index[0]
            frame.loc[index,'log_e']+=1.
            frame.to_csv(path,index=False)
            with self.assertRaises(ValueError):
                replay_client_logs(path)


if __name__=='__main__':
    unittest.main()
