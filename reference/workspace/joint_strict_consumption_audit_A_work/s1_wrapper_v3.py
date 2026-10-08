"""Lifecycle-safe callback and frozen current-node operator. No WAM imports."""
import hashlib
import json

_ACTIVE={}
class CallbackGuard:
    def __init__(self,target,name,transform):
        self.target=target;self.name=name;self.transform=transform
        self.installs=0;self.removals=0;self.calls=0;self.before={}
        self.original=None;self.wrapper=None
    def install(self):
        key=(id(self.target),self.name)
        if self.installs or key in _ACTIVE:raise RuntimeError('Duplicate or nested callback installation')
        original_callback=getattr(self.target,self.name)
        self.original=original_callback
        transform=self.transform
        def observe(*args,**kwargs):
            assert getattr(self.target,self.name) is observe,'Wrapper replaced during execution'
            out=original_callback(*args,**kwargs)
            self.before['out']=out;self.calls+=1
            return transform(out,args,kwargs)
        self.wrapper=observe
        assert getattr(self.target,self.name)==original_callback
        setattr(self.target,self.name,observe);_ACTIVE[key]=self;self.installs+=1
        assert getattr(self.target,self.name) is observe
    def uninstall(self):
        key=(id(self.target),self.name)
        if self.installs!=1 or self.removals or _ACTIVE.get(key) is not self:raise RuntimeError('Invalid callback removal')
        was_ours=getattr(self.target,self.name) is self.wrapper
        # Restore the exact saved callable object even when a foreign replacement was detected.
        setattr(self.target,self.name,self.original);del _ACTIVE[key];self.removals+=1
        assert getattr(self.target,self.name) is self.original
        if not was_ours:raise RuntimeError('Callback changed externally during installed lifetime')
    def summary(self):
        return dict(installs=self.installs,removals=self.removals,calls=self.calls,before_populated='out' in self.before,
                    original_identity=id(self.original),restored_identity=id(getattr(self.target,self.name)),
                    restored_exact_object=getattr(self.target,self.name) is self.original,active_registry_empty=not _ACTIVE)

def tensor_hash(t):
    import torch
    a=t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(a)).hexdigest()

class CurrentNode:
    """Original QKVController semantic K/V current group replacement; future runtime values untouched."""
    def __init__(self,model,cache):
        self.model=model;self.cache=cache;self.step=-1;self.tokens_per_group=98
        self.block_map={id(b):('video',i) for i,b in enumerate(model.video_expert.blocks)}
        self.block_map.update({id(b):('action',i) for i,b in enumerate(model.action_expert.blocks)})
        self.events=[];self.action_events=[]
        self.guard=CallbackGuard(model.mot,'_build_expert_attention_io',self.transform)
    def transform(self,result,args,kwargs):
        import torch
        block=kwargs['block'] if 'block' in kwargs else args[1]
        modality,layer=self.block_map[id(block)]
        if modality=='action':self.action_events.append((self.step,layer));return result
        key=(self.step,layer);source=self.cache[key];tpg=self.tokens_per_group
        assert tpg==98 and result[1].ndim==3 and result[1].shape[1]==294
        selected=torch.arange(tpg).to(result[1].device)
        changed=list(result);entry=dict(step=self.step,layer=layer,modality='video',groups=[0],future_clamped=False)
        for component,slot in [('k',1),('v',2)]:
            assert source[component].shape==result[slot].shape
            assert source[component].dtype==result[slot].dtype and source[component].device==result[slot].device
            # Identical operations to the registered QKVController semantic current-only branch.
            replacement=source[component].index_select(1,selected).clone()
            injected=result[slot].clone();injected[:,selected]=replacement
            assert torch.equal(injected[:,selected],replacement)
            assert torch.equal(injected[:,tpg:],result[slot][:,tpg:])
            source_hash=tensor_hash(replacement);written_hash=tensor_hash(injected[:,selected])
            assert source_hash==written_hash
            entry[component]=dict(shape=list(replacement.shape),dtype=str(replacement.dtype),device=str(replacement.device),
                                  donor_current_tensor_hash=source_hash,written_current_tensor_hash=written_hash,elementwise_equal=True)
            changed[slot]=injected
        self.events.append(entry)
        return tuple(changed)
    def install(self):self.guard.install()
    def uninstall(self):self.guard.uninstall()
    def summary(self):
        expected=[(s,l) for s in range(10) for l in range(30)]
        order=[(r['step'],r['layer']) for r in self.events]
        assert order==expected and self.action_events==expected
        g=self.guard.summary();assert g['installs']==g['removals']==1 and g['calls']==600 and g['restored_exact_object']
        digest=lambda x:hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
        return dict(expected_current_write_events=300,observed_current_write_events=len(self.events),component_writes=600,
                    action_events=len(self.action_events),ordering_digest=digest(order),events=self.events,lifecycle=g,
                    donor_current_tensor_hash=digest([[r[k]['donor_current_tensor_hash'] for k in ('k','v')] for r in self.events]),
                    written_current_tensor_hash=digest([[r[k]['written_current_tensor_hash'] for k in ('k','v')] for r in self.events]),
                    future_clamp_active=False,downstream='Original callback and full MoT forward execute at every event; future slots untouched, not required to change numerically')

def unit_tests():
    from types import SimpleNamespace
    checks={};log=[];output=object()
    def original(*a,**kw):log.append('original');return output
    target=SimpleNamespace(callback=original)
    def transform(out,a,kw):log.append('observe');assert out is output;return out
    guard=CallbackGuard(target,'callback',transform);guard.install()
    try:
        checks['active_during_execution']=target.callback is guard.wrapper
        got=target.callback()
        checks.update(original_called=log==['original','observe'],observe_called=guard.calls==1,
                      before_populated=guard.before['out'] is output,returned_output_unchanged=got is output,
                      not_restored_early=target.callback is guard.wrapper)
        try:CallbackGuard(target,'callback',transform).install()
        except RuntimeError:checks['nested_install_rejected']=True
        else:checks['nested_install_rejected']=False
        try:guard.install()
        except RuntimeError:checks['duplicate_install_rejected']=True
        else:checks['duplicate_install_rejected']=False
    finally:guard.uninstall()
    checks['restored_after_execution']=target.callback is original
    for error_origin in ['callback','observer']:
        def raising():raise ValueError('forced callback exception')
        target.callback=raising if error_origin=='callback' else original
        saved=target.callback
        def throw(out,a,kw):raise ValueError('forced observer exception')
        g=CallbackGuard(target,'callback',throw);g.install()
        try:target.callback()
        except ValueError:pass
        finally:g.uninstall()
        checks['exception_restore_'+error_origin]=target.callback is saved
    target.callback=original
    for _ in range(5):
        g=CallbackGuard(target,'callback',transform);g.install()
        try:assert target.callback() is output
        finally:g.uninstall()
        assert target.callback is original
    checks['repeated_cycles_no_stack']=not _ACTIVE
    # Exercise the actual controller's complete 30x10 event path on CPU tensor stand-ins.
    # No WAM class, weight, model forward, environment or simulator is involved.
    import torch
    vb=[object() for _ in range(30)];ab=[object() for _ in range(30)]
    raw=(torch.zeros(1,294,8,dtype=torch.bfloat16),torch.ones(1,294,8,dtype=torch.bfloat16),torch.ones(1,294,8,dtype=torch.bfloat16)*2)
    def io(*a,**kw):return raw
    fake=SimpleNamespace(video_expert=SimpleNamespace(blocks=vb),action_expert=SimpleNamespace(blocks=ab),mot=SimpleNamespace(_build_expert_attention_io=io))
    cache={(s,l):dict(k=raw[1].clone(),v=raw[2].clone()) for s in range(10) for l in range(30)}
    ctl=CurrentNode(fake,cache);ctl.install()
    try:
        for s in range(10):
            ctl.step=s
            for l in range(30):
                out=fake.mot._build_expert_attention_io(expert=None,block=vb[l]);assert all(torch.equal(x,y) for x,y in zip(raw,out))
                assert fake.mot._build_expert_attention_io(expert=None,block=ab[l]) is raw
    finally:ctl.uninstall()
    summary=ctl.summary();checks['actual_controller_standin_300_events']=summary['observed_current_write_events']==300
    checks['actual_controller_restored']=fake.mot._build_expert_attention_io is io
    assert all(checks.values()),checks
    return dict(WRAPPER_UNIT_TEST='PASS',checks=checks,standin_summary={k:v for k,v in summary.items() if k!='events'},new_wam_forwards=0)
