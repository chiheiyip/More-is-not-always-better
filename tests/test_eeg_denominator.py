from __future__ import annotations

import json
import subprocess
import zipfile
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from paper_analysis.teacher.eeg_audit import outcome_spec, prepare_inputs, FACTOR_LEVELS, TRIMS, METRICS
from paper_analysis.teacher.eeg_denominator import (EFFECTS, cache_identity, compare_tables,
    factor_families, freeze_history, paired_inputs, run_models, verify_cache)
from paper_analysis.teacher.state import StageBlockedError, file_sha256
from test_eeg_audit import synthetic_config, coefficient_fixture


def history_fixture(tmp_path):
    cfg=synthetic_config(tmp_path,n=4)
    frames,_=prepare_inputs(cfg)
    archive=tmp_path/'historic.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for trim,frame in frames.items():
            for metric in METRICS:
                frame[f'{metric}_relative'] += 1
            frame['view_start_s']=frame.GlobalTrialOrder*70
            frame['view_end_s']=frame.view_start_s+60
            frame['Cond']=frame.Complexity
            frame.loc[frame.Complexity.eq('C0'),'Complexity']=np.nan
            frame.loc[frame.PreviousComplexity.eq('C0'),'PreviousComplexity']=np.nan
            z.writestr(f'filesource_flat/SRC__06_eeg_primary_onset_window_models_trim_{trim}s__eeg_onset_order_model_input.csv',frame.to_csv(index=False))
    config={'denominator_sensitivity':{'historical_package':str(archive),'expected_sample':{'participants':4,'trials':48}}}
    return config


def test_freeze_decodes_c0_and_preserves_predecessor(tmp_path):
    config=history_fixture(tmp_path);frames,_=freeze_history(config)
    assert frames[0].Complexity.notna().all()
    assert frames[0].loc[frames[0].PreviousWWR.notna(),'PreviousComplexity'].notna().all()
    assert frames[0].loc[frames[0].PositionWithinBlock.eq(1),'PreviousComplexity'].isna().all()
    config['denominator_sensitivity']['expected_sample']['trials']=47
    with pytest.raises(StageBlockedError):freeze_history(config)


def test_paired_denominators_and_frozen_fields(tmp_path):
    frames,_=freeze_history(history_fixture(tmp_path))
    powers=pd.DataFrame([{'Participant':p,'GlobalTrialOrder':g,'onset_trim_s':t,'roi':r,
        'total_1_45':100+g,'total_1_40':99+g,'power_40_45':.6,'theta':3.,'alpha':5.,'beta':7.}
        for t,f in frames.items() for p,g in f[['Participant','GlobalTrialOrder']].itertuples(index=False,name=None) for r in ('F','P','O')])
    versions,integrals=paired_inputs(frames,powers)
    b,c=versions['B'][10],versions['C'][10]
    assert b.F_alpha_absolute.equals(c.F_alpha_absolute)
    assert b.hf_ratio_20_40Hz.equals(c.hf_ratio_20_40Hz) if 'hf_ratio_20_40Hz' in b else b.bad_eeg_quality.equals(c.bad_eeg_quality)
    np.testing.assert_allclose(c.F_alpha_relative/b.F_alpha_relative,(100+b.GlobalTrialOrder)/(99+b.GlobalTrialOrder))
    assert (integrals.denominator_reduction_percent>integrals.percent_40_45).all()
    with pytest.raises(StageBlockedError):paired_inputs(frames,powers.iloc[1:])
    bad=powers.copy();bad.loc[0,'theta']=np.nan
    with pytest.raises(StageBlockedError):paired_inputs(frames,bad)


def factors_fixture():
    return pd.DataFrame([{'onset_trim_s':t,'outcome':s['outcome'],'model':'Model1','term':e,
        'p.value':.01+.0001*i,'estimate':np.nan if e.startswith('WWR') else .02,
        'Fstat':5.,'df_num':2 if e.startswith('WWR') else 1,'df_denom':30.,'inference_valid':True}
        for t in TRIMS for i,s in enumerate(outcome_spec().to_dict('records')) for e in EFFECTS])


def test_factor_complete_bh_and_missing_inference():
    f=factors_fixture();table,status=factor_families(f)
    assert status.loc[status.scope.eq('joint'),'expected_tests'].tolist()==[96,216,216]
    for _,group in table.groupby('family_id'):
        np.testing.assert_allclose(group.joint_q,multipletests(group['p.value'],method='fdr_bh')[1])
    f.loc[0,'df_denom']=np.nan
    table,status=factor_families(f)
    assert table.loc[table.family_id.eq('factor_expanded_relative'),'joint_q'].isna().all()
    assert table.loc[table.family_id.eq('factor_expanded_absolute'),'joint_q'].notna().all()
    assert status.loc[status.family_id.eq('factor_expanded_relative') & status.scope.eq('joint'),'valid_tests'].iloc[0]==215


def test_cache_identity_and_tampering(tmp_path):
    trials=pd.DataFrame([{'Participant':'S1','start':10}])
    k=cache_identity([{'source_hash':'abc'}],trials,{'code':'1'})
    assert k!=cache_identity([{'source_hash':'def'}],trials,{'code':'1'})
    assert k!=cache_identity([{'source_hash':'abc'}],trials,{'code':'2'})
    with pytest.raises(StageBlockedError):verify_cache(tmp_path,k)
    p=tmp_path/'spectrum.mat';p.write_bytes(b'synthetic cache')
    manifest={'cache_key':k,'status':'complete','output_hashes':[{'path':p.name,'sha256':file_sha256(p)}]}
    (tmp_path/'cache_manifest.json').write_text(json.dumps(manifest))
    verify_cache(tmp_path,k)
    p.write_bytes(b'changed')
    with pytest.raises(StageBlockedError):verify_cache(tmp_path,k)


def test_omnibus_beta_is_unavailable_and_near_zero_percentage():
    left=factors_fixture();left,_=factor_families(left)
    left.loc[left.term.eq('Complexity'),'estimate']=1e-9
    right=left.copy();right.loc[right.term.eq('Complexity'),'estimate']=2e-9
    c=compare_tables(left,right,'B→C','factor')
    assert c.loc[c.term.eq('WWR'),'estimate_from'].isna().all()
    assert c.loc[c.term.eq('Complexity'),'estimate_percent_change'].isna().all()
    assert not c.flip_joint_q.any()


def test_sensitivity_command_read_only_and_requires_r(tmp_path,monkeypatch,capsys):
    root=Path(__file__).resolve().parents[1]
    module=runpy.run_path(str(root/'scripts/run_teacher_analysis.py'))
    monkeypatch.setitem(module['main'].__globals__,'denominator_preflight',lambda *args: ({},{},{'common_trials':461}))
    config=tmp_path/'config.json';config.write_text('{}')
    out=tmp_path/'new-run'
    assert module['main'](['eeg-denominator-sensitivity','--config',str(config),'--outdir',str(out),'--dry-run'])==0
    assert not out.exists()
    assert json.loads(capsys.readouterr().out)['common_trials']==461
    from paper_analysis.teacher.eeg_denominator import run_eeg_denominator
    with pytest.raises(StageBlockedError):run_eeg_denominator({},config_path=config,outdir=out,repo_root=root,r_required=False)
    assert not out.exists()


def test_report_complete_pairs_without_inventing_unadjusted_q(tmp_path):
    from paper_analysis.teacher.eeg_denominator import finish_report
    from paper_analysis.teacher.eeg_audit import family_tables
    frames,_=freeze_history(history_fixture(tmp_path))
    coef=coefficient_fixture();ct,cs=family_tables(coef);ft,fs=factor_families(factors_fixture())
    raw=coef.copy();raw['family_id']='all_model_coefficients_unadjusted';raw['within_q']=np.nan;raw['joint_q']=np.nan
    result={'inputs':frames,'diagnostics':pd.DataFrame({'inference_valid':[True]}),
        'family_status':pd.concat([cs,fs]),'family_coefficients':ct,'family_factors':ft,'all_model_coefficients':raw}
    powers=pd.DataFrame([{'onset_trim_s':t,'roi':r,'total_1_45':100.,'power_40_45':.6,
        'percent_40_45':.6,'denominator_reduction_percent':1.,'relative_increase_percent':1.01}
        for t in TRIMS for r in ('F','P','O')])
    reg={'max_beta_error':0.,'max_p_error':0.,'factor_max_p_error':0.}
    status=finish_report(tmp_path,{v:result for v in ('A','B','C')},powers,reg,{'common_participants':4,'common_trials':48})
    assert status=='complete'
    summary=json.loads((tmp_path/'summary.json').read_text(encoding='utf-8'))
    assert len(summary['comparison_summary'])==30
    assert all(r['joint_q_flips']==0 for r in summary['comparison_summary'])
    assert all(r['max_abs_delta_joint_q'] is None for r in summary['comparison_summary'] if r['family_id']=='all_model_coefficients_unadjusted')
    assert not pd.read_csv(tmp_path/'model_comparisons.csv').loc[lambda d:d.family_id.eq('all_model_coefficients_unadjusted'),'joint_q_to'].notna().any()


def test_real_r_marginal_contrasts_and_absolute_reuse(tmp_path):
    root=Path(__file__).resolve().parents[1]
    config=synthetic_config(tmp_path,unicode_ids=True)
    frames,_=prepare_inputs(config)
    contract=tmp_path/'contract.json';contract.write_text(json.dumps({'factor_levels':FACTOR_LEVELS,'seed':20260906}))
    result=run_models(frames,config,root,tmp_path/'B',contract)
    reused=run_models(frames,config,root,tmp_path/'C',contract,reuse_absolute=result)
    assert result['family_status'].status.eq('complete').all()
    for name in ('coefficients','factor_tests'):
        a=result[name].loc[result[name].outcome.str.startswith('log10_')].reset_index(drop=True)
        b=reused[name].loc[reused[name].outcome.str.startswith('log10_')].reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_exact=True)
    matrices=result['contrast_matrices']
    c=matrices.loc[matrices.term.eq('Complexity') & matrices.coefficient.eq('WWRWWR45:ComplexityC1')]
    assert np.allclose(c.weight,1/3)
    e=matrices.loc[matrices.term.eq('Complexity') & matrices.coefficient.eq('ComplexityC1:ExperienceGroupLow')]
    assert np.allclose(e.weight,.5)
    script=root/'tests/r/test_eeg_factor_tests.R'
    subprocess.run([config['rscript'],str(script),str(tmp_path/'B/trim_0s/input.csv'),str(contract)],cwd=root,check=True)
