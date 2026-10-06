"""Additional evidence from frozen independent outputs; never production helpers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t, norm
from statsmodels.stats.multitest import multipletests
from .independent_eye import KEY, CORE, save, dump, sha, unique


def bh_checks(frame, kind):
    rows=[]
    for (version,family),g in frame.groupby(['version','family_id']):
        complete=np.isfinite(g.raw_p).all() and g.inference_valid.fillna(False).all()
        expected=multipletests(g.raw_p,method='fdr_bh')[1] if complete else np.full(len(g),np.nan)
        match=np.allclose(expected,g.joint_q,rtol=1e-10,atol=1e-12,equal_nan=True)
        if not match: raise ValueError('Python/R current EEG joint BH differs')
        for trim,h in g.groupby('onset_trim_s'):
            valid=np.isfinite(h.raw_p).all() and h.inference_valid.fillna(False).all()
            q=multipletests(h.raw_p,method='fdr_bh')[1] if valid else np.full(len(h),np.nan)
            if not np.allclose(q,h.within_q,rtol=1e-10,atol=1e-12,equal_nan=True): raise ValueError('Python/R current EEG window BH differs')
        rows.append(dict(version=version,family_id=family,kind=kind,FullFamilySize=len(g),Complete=bool(complete),PythonRMatch=bool(match)))
    return pd.DataFrame(rows)


def pair_statistics(a,b,keys):
    unique(a,keys);unique(b,keys)
    merged=a.merge(b,on=keys,how='outer',suffixes=('_from','_to'),indicator=True,validate='one_to_one')
    for c in ['estimate','SE','CR2_SE','df','CI_low','CI_high','raw_p','within_q','joint_q','Fstat','df_num','df_denom']:
        if c+'_from' in merged and c+'_to' in merged: merged[c+'_difference']=merged[c+'_to']-merged[c+'_from']
    for c in ['raw_p','within_q','joint_q']:
        if c+'_from' in merged and c+'_to' in merged:
            valid=np.isfinite(merged[c+'_from'])&np.isfinite(merged[c+'_to'])
            merged[c+'_crosses_005']=valid & merged[c+'_from'].lt(.05).ne(merged[c+'_to'].lt(.05))
    if 'estimate_from' in merged and 'estimate_to' in merged:
        merged['DirectionChanged']=np.isfinite(merged.estimate_from)&np.isfinite(merged.estimate_to)&np.sign(merged.estimate_from).ne(np.sign(merged.estimate_to))
        merged['NearZeroDirectionChange']=merged.DirectionChanged & merged[['estimate_from','estimate_to']].abs().max(axis=1).lt(1e-8)
    return merged


def eeg_evidence(config,root):
    root=Path(root);target=root/'eeg';source=Path(config['sensitivity_run'])
    current=pd.read_csv(target/'independent_eeg_trial_QC.csv',encoding='utf-8-sig')
    inventory=pd.read_csv(target/'independent_eeg_source_inventory.csv',encoding='utf-8-sig')
    # Mean of every average-referenced channel can be a quantization residual.
    # Measure this directly without changing the prespecified historical QC.
    waveform=[]
    selected=inventory.head(10).copy()
    selected=pd.concat([selected,inventory.loc[inventory.Participant.isin(['侯集瀚','李泽豪','刘泽雨'])]]).drop_duplicates('Participant')
    for record in selected.to_dict('records'):
        path=Path(record['FDT']);before=sha(path)
        data=np.memmap(path,dtype='<f4',mode='r',shape=(int(record['Channels']),int(record['Samples'])),order='F')
        for row in current.loc[current.Participant.eq(record['Participant']) & current.onset_trim_s.eq(0)].head(2).itertuples():
            x=np.asarray(data[:,int(row.AnalysisStartSample)-1:int(row.EndSample)],dtype=np.float64)
            channel=np.sqrt(np.nanmean(x*x,axis=1)).mean();average=np.sqrt(np.nanmean(np.nanmean(x,axis=0)**2))
            waveform.append(dict(Participant=record['Participant'],GlobalTrialOrder=row.GlobalTrialOrder,MeanAcrossChannelsRMSUV=average,ChannelRMSMeanUV=channel,ResidualToChannelRMSRatio=average/channel,Interpretation='Diagnostic only; historical all-channel-mean HF QC is preserved'))
        del data
        if sha(path)!=before: raise ValueError('FDT changed during reference residual check')
    save(pd.DataFrame(waveform),target/'average_reference_QC_signal_audit.csv')
    raw=pd.read_csv(target/'independent_raw_trigger_events.csv',dtype={'Marker':str});epochs=pd.read_csv(target/'raw_to_SET_epoch_endpoint_matching.csv')
    details=[]
    for row in epochs.loc[epochs.EndpointMatchStatus.eq('right_only')].itertuples():
        events=raw.loc[raw.Participant.eq(row.Participant)]
        start=events.Marker.eq('7')&events.LatencySample.eq(row.StartSample);end=events.Marker.eq('8')&events.LatencySample.eq(row.EndSample)
        inside=events.loc[events.LatencySample.gt(row.StartSample)&events.LatencySample.lt(row.EndSample)]
        details.append(dict(Participant=row.Participant,SETOrdinal=row.SETOrdinal,StartSample=row.StartSample,EndSample=row.EndSample,RawStartPresent=bool(start.any()),RawEndPresent=bool(end.any()),IntermediateTriggers=len(inside),IntermediateMarkers=';'.join(inside.Marker.astype(str)),Interpretation='Original endpoints independently checked; intermediate triggers explain adjacency differences only when both endpoints exist'))
    save(pd.DataFrame(details),target/'raw_SET_nonadjacent_endpoint_investigation.csv')
    checks=[];differences=[]
    for trim in [0,5,10,15]:
        a=current.loc[current.onset_trim_s.eq(trim)]
        b=pd.read_csv(source/'B'/f'trim_{trim}s/input.csv');j=a.merge(b,on=KEY,suffixes=('_now','_frozen'),validate='one_to_one')
        j['EndpointSame']=j.AnalysisStartSample.eq(np.rint(j.analysis_start_s*j.srate))&j.EndSample.eq(np.rint(j.analysis_end_s*j.srate))
        for roi in ['F','P','O']:
            for band in ['theta','alpha','beta']:
                col=roi+'_'+band+'_relative';delta=j[col+'_now']-j[col+'_frozen'];match=np.isclose(j[col+'_now'],j[col+'_frozen'],rtol=1e-10,atol=1e-12)
                same=j.EndpointSame
                checks.append(dict(onset_trim_s=trim,ROI=roi,band=band,UnchangedEndpointsTrials=int(same.sum()),UnchangedEndpointsPowerMatch=bool(match[same].all()),ChangedEndpointsTrials=int((~same).sum()),MaxErrorUnchangedEndpoints=float(delta[same].abs().max())))
                for i in np.flatnonzero(~match):
                    row=j.iloc[i];differences.append(dict(Participant=row.Participant,GlobalTrialOrder=row.GlobalTrialOrder,onset_trim_s=trim,ROI=roi,band=band,CurrentStartSample=int(row.AnalysisStartSample),FrozenStartSample=int(round(row.analysis_start_s*row.srate)),EndSample=int(row.EndSample),CurrentRelative=row[col+'_now'],FrozenRelative=row[col+'_frozen'],Difference=float(delta.iloc[i]),EndpointSame=bool(row.EndpointSame)))
    save(pd.DataFrame(checks),target/'waveform_power_unchanged_endpoint_checks.csv');save(pd.DataFrame(differences),target/'waveform_power_endpoint_differences.csv')
    all_pairs=[];all_checks=[]
    for kind in ['coefficient','factor']:
        new=pd.read_csv(target/'current_QC_models'/f'current_QC_{kind}_families.csv')
        all_checks.append(bh_checks(new,kind))
        for oldversion,version in [('B','D_current_QC_1_45'),('C','E_current_QC_1_40')]:
            old=pd.read_csv(source/oldversion/f'family_{"coefficients" if kind=="coefficient" else "factors"}.csv').rename(columns={'p.value':'raw_p','std.error':'CR2_SE'})
            old.family_id=old.family_id.str.replace('condition_','coefficient_',regex=False)
            paired=pair_statistics(old,new.loc[new.version.eq(version)],['onset_trim_s','outcome','model','term','family_id']);paired['pair']=oldversion+'→'+version;paired['test_kind']=kind;all_pairs.append(paired)
        paired=pair_statistics(new.loc[new.version.eq('D_current_QC_1_45')],new.loc[new.version.eq('E_current_QC_1_40')],['onset_trim_s','outcome','model','term','family_id']);paired['pair']='D→E';paired['test_kind']=kind;all_pairs.append(paired)
    save(pd.concat(all_pairs,ignore_index=True),target/'current_QC_model_comparisons.csv');save(pd.concat(all_checks,ignore_index=True),target/'current_QC_Python_R_BH_checks.csv')
    co=pd.read_csv(target/'current_QC_models/current_QC_coefficients.csv');expected=2*t.sf(np.abs(co.estimate/co.CR2_SE),co.df)
    if not np.allclose(expected,co.raw_p,atol=1e-6,rtol=1e-10):raise ValueError('Independent EEG coefficient/CR2/df/p relation differs')
    dump(target/'supplementary_evidence_summary.json',{'unchanged_endpoints_power_match':bool(pd.DataFrame(checks).UnchangedEndpointsPowerMatch.all()),'changed_endpoint_power_records':len(differences),'current_QC_Python_R_BH_verified':True,'reference_residual_spot_checks':len(waveform),'reference_residual_max_ratio':float(pd.DataFrame(waveform).ResidualToChannelRMSRatio.max()),'nonadjacent_SET_epochs':len(details),'nonadjacent_original_endpoints_present':sum(r['RawStartPresent'] and r['RawEndPresent'] for r in details)})


def eye_arithmetic(root):
    target=Path(root)/'eye';d=pd.read_csv(target/'independent_eye_trials.csv');selected=d.loc[d.QC60]
    participant=selected.groupby(['Participant','WWR','Complexity'],as_index=False)[CORE].mean()
    save(participant,target/'descriptive_participant_condition_means.csv')
    description=[]
    for (w,c),g in participant.groupby(['WWR','Complexity']):
        for outcome in CORE:
            values=g[outcome].dropna();n=len(values);sd=values.std(ddof=1);half=t.ppf(.975,n-1)*sd/np.sqrt(n) if n>1 else np.nan
            description.append(dict(WWR=w,Complexity=c,outcome=outcome,NParticipants=n,Mean=values.mean(),SD=sd,Median=values.median(),CI_low=values.mean()-half,CI_high=values.mean()+half,Unit='participant condition mean, descriptive only'))
    save(pd.DataFrame(description),target/'descriptive_participant_condition_summary.csv')
    contrasts=pd.read_csv(target/'independent_contrasts.csv');checks=[]
    for (variant,outcome,omitted,inference),g in contrasts.groupby(['variant','outcome','omitted_participant','inference'],dropna=False):
        g=g.loc[g.contrast.str.startswith('WWR')]
        if len(g)==3 and np.isfinite(g.raw_p).all():
            q=multipletests(g.raw_p,method='holm')[1]
            if not np.allclose(q,g.Holm_p,rtol=1e-10,atol=1e-12):raise ValueError('Independent eye Python/R Holm differs')
            checks.append(dict(variant=variant,outcome=outcome,omitted_participant=omitted,inference=inference,FamilySize=3,PythonRMatch=True))
    save(pd.DataFrame(checks),target/'Python_R_Holm_checks.csv')
    co=pd.read_csv(target/'independent_coefficients.csv');valid=co.raw_p.notna()&co.estimate.notna()
    coef_checks=[]
    for (variant,term,family,omitted),g in co.groupby(['variant','term','family','omitted_participant'],dropna=False):
        if family not in ['A','B']:continue
        expected_members=CORE[:3] if family=='A' else CORE[3:]
        complete=len(g)==3 and set(g.outcome)==set(expected_members) and np.isfinite(g.raw_p).all() and g.converged.all()
        q=multipletests(g.raw_p,method='fdr_bh')[1] if complete else np.full(len(g),np.nan)
        if not np.allclose(q,g.q,atol=1e-12,rtol=1e-10,equal_nan=True):raise ValueError('Eye coefficient Python/R BH differs')
        coef_checks.append(dict(variant=variant,term=term,family=family,omitted_participant=omitted,ExpectedMembers=3,ObservedMembers=len(g),Complete=bool(complete),PythonRMatch=True))
    save(pd.DataFrame(coef_checks),target/'Python_R_coefficient_BH_checks.csv')
    expected=np.where(co.inference.eq('GLMM_Wald_z'),2*norm.sf(np.abs(co.estimate/co.SE)),2*t.sf(np.abs(co.estimate/co.CR2_SE),co.df))
    if not np.allclose(expected[valid],co.raw_p[valid],atol=1e-6,rtol=1e-10):raise ValueError('Eye coefficient/SE/df/p relation differs')
    return dict(descriptive_rows=len(description),HolmFamiliesVerified=len(checks),CoefficientRelationsVerified=int(valid.sum()))


def normalize_eye_term(value):
    return str(value).replace('WWRWWR','WWR').replace('ExperienceGroup','ExerciseFrequency')


def eye_models_comparison(config,root):
    """Compare like inference layers, keeping primary and companion separate."""
    root=Path(root);target=root/'eye';formal=Path(config['formal_eye_run'])/'02_eye_stage2'
    independent=pd.read_csv(target/'independent_coefficients.csv');tables=[]
    old=pd.concat([pd.read_csv(formal/'09_familyA_primary_models.csv'),pd.read_csv(formal/'10_familyB_primary_models.csv')],ignore_index=True)
    old['term']=old.term.map(normalize_eye_term)
    old['raw_p']=np.where(old.outcome.isin(['TableShare','WindowShare']),old['p.value.likelihood'],old['p.value.CR2'])
    old['SE']=old['std.error.likelihood'];old['CI_low']=old['conf.low'];old['CI_high']=old['conf.high']
    new=independent.loc[independent.variant.eq('main')].copy()
    new['CI_low']=new.estimate-1.95996398454005*new.SE;new['CI_high']=new.estimate+1.95996398454005*new.SE
    # The original report uses likelihood Wald CI and separate CR2 p evidence.
    primary=pair_statistics(old[['outcome','term','estimate','SE','CI_low','CI_high','raw_p']],new[['outcome','term','estimate','SE','CI_low','CI_high','raw_p']],['outcome','term'])
    primary['comparison_layer']='primary_estimate_likelihood_CI_and_corresponding_p';tables.append(primary)
    robust=pd.read_csv(formal/'12_CR2_robust_results.csv').rename(columns={'std.error':'CR2_SE','p.value':'raw_p','p.value.BH':'joint_q'})
    robust['term']=robust.term.map(normalize_eye_term)
    new=independent.loc[independent.variant.eq('CR2_companion_family')|independent.variant.eq('main')&independent.family.eq('B')].rename(columns={'q':'joint_q'})
    joined=pair_statistics(robust[['outcome','term','estimate','CR2_SE','df','raw_p','joint_q']],new[['outcome','term','estimate','CR2_SE','df','raw_p','joint_q']],['outcome','term'])
    joined['comparison_layer']='Gaussian_participant_CR2_complete_three_outcome_BH';tables.append(joined)
    result=pd.concat(tables,ignore_index=True)
    exact=result._merge.eq('both') & result.estimate_difference.abs().le(1e-8) & result.raw_p_difference.abs().le(1e-6)
    substantive=result._merge.ne('both')|result.raw_p_crosses_005.fillna(False)|result.joint_q_crosses_005.fillna(False)
    result['Agreement']=np.where(substantive,'substantive discrepancy',np.where(exact,'near-exact agreement','minor numerical discrepancy'))
    save(result,target/'12_eye_tracking_independent_replication_audit.csv')
    pairs=pd.read_csv(formal/'11_WWR_posthoc_Holm.csv').rename(columns={'p.value':'Holm_p','SE':'SE'})
    directions={'WWR15 - WWR45':('WWR45-WWR15',-1),'WWR15 - WWR75':('WWR75-WWR15',-1),'WWR45 - WWR75':('WWR75-WWR45',-1)}
    if not set(pairs.contrast).issubset(directions):raise ValueError('Unknown original WWR contrast direction')
    pairs['estimate']=[value*directions[label][1] for value,label in zip(pairs.estimate,pairs.contrast)]
    pairs['contrast']=[directions[label][0] for label in pairs.contrast]
    newpairs=pd.read_csv(target/'independent_contrasts.csv');newpairs=newpairs.loc[newpairs.variant.eq('main')&newpairs.inference.isin(['equal_weight_primary_likelihood','equal_weight_GLMM_Wald'])&newpairs.contrast.str.startswith('WWR')]
    post=pairs[['outcome','contrast','estimate','SE','df','Holm_p']].merge(newpairs[['outcome','contrast','estimate','SE','df','Holm_p']],on=['outcome','contrast'],suffixes=('_from','_to'),how='outer',validate='one_to_one',indicator=True)
    post['HolmThresholdCrossing']=post.Holm_p_from.notna()&post.Holm_p_to.notna()&post.Holm_p_from.lt(.05).ne(post.Holm_p_to.lt(.05))
    save(post,target/'WWR_primary_Holm_replication_comparison.csv')
    changed=result.loc[substantive].copy();save(changed,target/'eye_threshold_discrepancy_investigation.csv')
    return dict(PrimaryAndCR2RowsCompared=len(result),NearExactRows=int(exact.sum()),RawPThresholdCrossings=int(result.raw_p_crosses_005.fillna(False).sum()),RobustBHThresholdCrossings=int(result.joint_q_crosses_005.fillna(False).sum()),WWRHolmThresholdCrossings=int(post.HolmThresholdCrossing.sum()),OriginalRobustNonInterceptPositive=int((robust.term.ne('(Intercept)')&robust.joint_q.lt(.05)).sum()),IndependentRobustNonInterceptPositive=int((new.term.ne('(Intercept)')&new.joint_q.lt(.05)).sum()),InitialIndependence='First calculations sealed before comparison; corrected validity/primary-model revision after comparison is disclosed',PixelRule='Independent rounded coordinates/cv2 polygon masks; original floor coordinates/PIL masks. Interpret boundary differences explicitly, not as proof of original error.')
