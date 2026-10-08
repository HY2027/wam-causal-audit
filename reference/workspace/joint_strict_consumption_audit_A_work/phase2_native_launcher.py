"""Explicit namespace adapter, before any environment/model execution.
The project has two unrelated modules named protocol; bind donor geometry to
the existing counterfactual protocol without modifying either module.
"""
from wam_causal_audit.paths import resolve as _release_path
import argparse,sys,time
from pathlib import Path
import phase2_native_validation as V
from phase2_local_future import sha,dump

def prepare_imports():
    import run_experiment_a
    import confirmatory_common as Q
    original=sys.modules.get('protocol')
    sys.path.insert(0,_release_path('@WORKSPACE@/experiments/wam_control_state_v3'))
    try:
        sys.modules['protocol']=Q.OLD
        import generate_group1_donors
        assert generate_group1_donors.old_protocol is Q.OLD
    finally:
        if original is not None:sys.modules['protocol']=original
        else:sys.modules.pop('protocol',None)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--task',type=int);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    target=V.OUT/'import_adapter_freeze.json'
    if a.freeze:
        if target.exists():raise RuntimeError('already frozen')
        prepare_imports()
        dump(target,dict(created_unix=time.time(),code_sha256=sha(__file__),base_code_sha256=sha(V.__file__),reason="initial launches failed before model/env loading: week1 protocol module collided with geometry protocol; explicit import binding only",no_scientific_change=True,no_native_attempt_consumed=True,old_import_failure='AttributeError protocol.CLOSE_COMMAND_THRESHOLD missing',tasks_affected=[0,1]))
        print('IMPORT_ADAPTER_FROZEN')
    else:
        assert V.load(target)['code_sha256']==sha(__file__)
        prepare_imports();V.acquire(a.task,a.gpu)
