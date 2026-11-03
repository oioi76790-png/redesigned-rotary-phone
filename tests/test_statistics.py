import math
import unittest
import numpy as np
from seta_statistics import BoundedMeanEProcess, ClientHistory, logmeanexp


class StatisticsTests(unittest.TestCase):
    def test_old_formula_degenerates(self):
        lambdas = np.array([.1,.2,.4])
        for z in np.linspace(-1,1,1001):
            factor = np.mean(np.exp(-lambdas**2/2)*np.cosh(lambdas*z))
            self.assertLess(factor, 1.)

    def test_exact_conditional_expectation_on_two_point_null(self):
        # Enumerate the full conditional distribution, not an empirical check.
        for true_mean in [-1., -.8, 0., .6, 1.]:
            for predicted in [-1., -.3, .7, 1.]:
                bias = abs(true_mean-predicted)
                for lam in [-2., -1., -.25, .25, 1., 2.]:
                    def factor(x):
                        return math.exp(lam*(x-predicted)-abs(lam)*bias-.5*lam**2)
                    expectation = ((1+true_mean)/2*factor(1.)
                                   +(1-true_mean)/2*factor(-1.))
                    self.assertLessEqual(expectation, 1.+1e-12)

    def test_cumulative_mixture_matches_closed_form(self):
        process = BoundedMeanEProcess(lambdas=(.2,.7))
        observations = [.4,-.7,.9]
        centers = [.1,.2,-.3]
        bounds = [.1,.2,.05]
        for index, (x, mu, b) in enumerate(zip(observations, centers, bounds), start=1):
            actual, _ = process.update(x,mu,b)
            closed = (process.lambdas*sum(np.array(observations[:index])-centers[:index])
                      -abs(process.lambdas)*sum(bounds[:index])
                      -.5*index*process.lambdas**2)
            self.assertAlmostEqual(actual, logmeanexp(closed))

    def test_signal_can_grow_and_alarm_without_threshold_change(self):
        process = BoundedMeanEProcess(alpha=.05)
        for _ in range(50):
            log_e, alarm = process.update(-.8, .8, .1)
        self.assertGreater(log_e, math.log(20))
        self.assertTrue(alarm)

    def test_alarm_latches_without_reset(self):
        process = BoundedMeanEProcess()
        for _ in range(50):
            process.update(-1., 1., 0.)
        self.assertTrue(process.alarmed)
        for _ in range(1000):
            process.update(1., 1., 0.)
        self.assertTrue(process.alarmed)
        self.assertEqual(process.t, 1050)

    def test_no_auto_admission_and_frozen_prediction(self):
        history = ClientHistory()
        self.assertIsNone(history.predict(.1))
        for _ in range(8):
            history.add_trusted([.6,.1], provenance='independent clean source')
        prediction = history.predict(.1)
        for alarm in [False, True]:
            history.observe([-.9,0.], round_id=1, alarm=alarm)
        self.assertEqual(len(history.trusted),8)
        self.assertEqual(len(history.quarantine),2)
        self.assertEqual(history.predict(.1),prediction)
        history.add_trusted([.1,0.], provenance='later independent validation')
        self.assertEqual(prediction.trusted_count,8)
        self.assertNotEqual(history.predict(.1),prediction)
        self.assertGreater(prediction.score([.6,.1]), prediction.score([-.6,-.1]))

    def test_histories_independent_and_quarantine_bounded(self):
        a, b = ClientHistory(8), ClientHistory(8)
        for _ in range(10):
            a.add_trusted([.5], provenance='clean')
            a.observe([-.5], round_id=1, alarm=True)
        self.assertEqual(len(a.trusted),8)
        self.assertEqual(len(a.quarantine),8)
        self.assertEqual(len(b.trusted),0)

    def test_invalid_statistics_inputs(self):
        for kwargs in [dict(alpha=0),dict(alpha=1),dict(lambdas=()),dict(lambdas=(-1,)),dict(lambdas=(float('nan'),))]:
            with self.assertRaises(ValueError):
                BoundedMeanEProcess(**kwargs)
        for args in [(2.,0.,0.),(0.,2.,0.),(0.,0.,-1.),(float('nan'),0.,0.)]:
            with self.assertRaises(ValueError):
                BoundedMeanEProcess().update(*args)
        with self.assertRaises(ValueError):
            ClientHistory().add_trusted([.5],provenance='')
        with self.assertRaises(ValueError):
            ClientHistory().add_trusted([2.],provenance='clean')

    def test_simulation_seed_and_drift_assumptions(self):
        from simulate_statistics import run_trial
        a = run_trial('normal_iid',1,1)
        self.assertEqual(a,run_trial('normal_iid',1,1))
        self.assertTrue(a['envelope_holds'])
        self.assertTrue(run_trial('normal_non_iid',1,1)['envelope_holds'])
        uncovered = run_trial('normal_drift_uncovered',1,1)
        covered = run_trial('normal_drift_covered',1,1)
        self.assertFalse(uncovered['envelope_holds'])
        self.assertTrue(covered['envelope_holds'])
        self.assertAlmostEqual(uncovered['max_actual_mean_error'],covered['max_actual_mean_error'])

    def test_missed_detections_are_reported_as_censored(self):
        from simulate_statistics import summarize
        result = summarize([dict(scenario='persistent_signflip',false_alarm=False,
                                 at_risk=True,detected=False,delay=None,restricted_delay=200)],240,41)[0]
        self.assertEqual(result['detection_rate'],0.)
        self.assertIsNone(result['mean_delay_detected'])
        self.assertEqual(result['restricted_mean_delay'],200.)
        self.assertEqual(result['censored_trials'],1)


if __name__ == '__main__':
    unittest.main()
