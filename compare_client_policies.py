"""Deterministic state/weight contract comparison, NOT a data experiment."""
import json
from pathlib import Path
from client_control import ErrorBudget,ClientController,RoundContext


def main():
    rows=[]
    for policy in ('defer','capped','clean-warmup'):
        controller=ClientController(ErrorBudget(.05,10),policy=policy,
                                    declare_clean_warmup=policy=='clean-warmup')
        controller.prepare_round(RoundContext(1,0.,1.,1.))
        weights,server=controller.allocate(list(range(10)),[100]*10,[False]*10)
        rows.append(dict(policy=policy,states=[s.value for s in controller.states],
                         weights=weights.tolist(),total_client_mass=float(weights.sum()),server_mass=server,
                         detector_enabled=False,normality_certified=False,
                         reason='explicit declared warmup only' if policy=='clean-warmup' else 'uncalibrated handling'))
    result=dict(scope='Deterministic allocation contract only; no images, training, FPR, detection or accuracy measurements.',
                comparison=rows)
    out=Path('reports/stage4/policy_contracts.json');out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
