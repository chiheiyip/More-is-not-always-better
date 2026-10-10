"""Actual R Stage 3 must retain Python CSV booleans and fit nonempty samples."""
from pathlib import Path
import os, subprocess
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def test_stage3_retains_title_case_csv_flags_and_numeric_visited(tmp_path):
    from paper_analysis.teacher.runtime_lock import locked_r_environment
    rng=np.random.default_rng(20261010)
    records=[]
    for person in range(24):
        for trial in range(12):
            row=dict(Participant=f'P{person:02}',WWR=['WWR15','WWR45','WWR75'][trial%3],
                     Complexity=['C0','C1'][(trial//3)%2],ExperienceGroup=['Low','High'][person%2],
                     Gender=['Male','Female'][(person//2)%2],OrderGroup=['order1','order2'][(person//4)%2],
                     Block=trial//6+1,PositionWithinBlockCentered=trial%6-2.5,
                     IncludedPrimary=trial!=11,RawCompetition=rng.normal(),AdjustedCompetition=rng.normal())
            records.append(row)
    data=pd.DataFrame(records);data.to_csv(tmp_path/'input.csv',index=False)
    aoi=pd.concat([data.assign(AOICategory=label,TFD=rng.uniform(10,1000,len(data)),
                              Visited=rng.random(len(data))>.25) for label in ['Table','Window']])
    aoi.to_csv(tmp_path/'aoi.csv',index=False)
    # Exact title-case text emitted by pandas is the regression trigger.
    assert ',True,' in (tmp_path/'aoi.csv').read_text()
    env=locked_r_environment(ROOT)
    result=subprocess.run([str(ROOT/'scripts/portable_rscript.cmd'),str(ROOT/'analysis/r/eye_stage3_analysis.R'),
        str(tmp_path/'input.csv'),str(tmp_path/'out'),'I,G',str(tmp_path/'aoi.csv'),'','3'],
        cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    diagnostics=pd.read_csv(tmp_path/'out/17_stage3_model_diagnostics.csv')
    assert len(diagnostics)==6
    assert diagnostics.status.eq('fit').all(),diagnostics.to_string()
    assert diagnostics['n'].eq(24*11).all()
    repeat=subprocess.run([str(ROOT/'scripts/portable_rscript.cmd'),str(ROOT/'analysis/r/eye_stage3_analysis.R'),
        str(tmp_path/'input.csv'),str(tmp_path/'repeat'),'G','','','3'],
        cwd=ROOT,env=env,capture_output=True,text=True)
    assert repeat.returncode==0,repeat.stdout+repeat.stderr
    for name in ['08d_experience_cluster_bootstrap_5000.csv','bootstrap_draw_provenance.json']:
        assert (tmp_path/'out'/name).read_bytes()==(tmp_path/'repeat'/name).read_bytes()
    check=subprocess.run([str(ROOT/'scripts/portable_rscript.cmd'),str(ROOT/'analysis/r/verify_bootstrap_draws.R'),
        str(tmp_path/'out/bootstrap_draw_provenance.json')],cwd=ROOT,env=env,capture_output=True,text=True)
    assert check.returncode==0,check.stdout+check.stderr

def test_boolean_parser_rejects_unknown_values_and_preserves_missing(tmp_path):
    from paper_analysis.teacher.runtime_lock import locked_r_environment
    expression=('source("analysis/r/common.R");stopifnot(identical(read_teacher_boolean('
        'c("True","FALSE","1","0",NA),"flag"),c(TRUE,FALSE,TRUE,FALSE,NA)));'
        'stopifnot(inherits(try(read_teacher_boolean("not-a-boolean","flag"),silent=TRUE),"try-error"))')
    result=subprocess.run([str(ROOT/'scripts/portable_rscript.cmd'),'-e',expression],
        cwd=ROOT,env=locked_r_environment(ROOT),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
