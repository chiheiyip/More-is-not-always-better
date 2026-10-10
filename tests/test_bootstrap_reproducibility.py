import json
from pathlib import Path
import subprocess
import pandas as pd
import pytest
from test_eeg_audit import synthetic_config
from paper_analysis.teacher.eeg_audit import prepare_inputs
from paper_analysis.teacher.r_runner import invoke_r

def test_real_production_bootstrap_draws_and_estimates_repeat(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    if not (repo/'.r-env/Lib/R/bin/Rscript.exe').is_file():pytest.skip('Registered R required')
    config=synthetic_config(tmp_path,n=24,unicode_ids=True)
    frames,_=prepare_inputs(config);source=tmp_path/'model_input.csv'
    frames[10].to_csv(source,index=False,encoding='utf-8-sig')
    artifacts=[]
    for name in ('a','b'):
        out=tmp_path/name
        invoke_r(config['rscript'],repo/'analysis/r/eeg_primary_analysis.R',
            [str(source),str(out),'F_theta_relative','log10_F_theta_absolute','12','O_alpha_relative','P_beta_relative'])
        proof=out/'bootstrap_draw_provenance.json'
        record=json.loads(proof.read_text(encoding='utf-8'))
        assert record['outcomes'] and all(v['requested']==12 for v in record['outcomes'].values())
        subprocess.run([config['rscript'],str(repo/'analysis/r/verify_bootstrap_draws.R'),str(proof)],check=True)
        artifacts.append(out)
    assert (artifacts[0]/'bootstrap_draw_provenance.json').read_bytes()==(artifacts[1]/'bootstrap_draw_provenance.json').read_bytes()
    pd.testing.assert_frame_equal(pd.read_csv(artifacts[0]/'07_eeg_cluster_bootstrap_5000.csv'),
                                  pd.read_csv(artifacts[1]/'07_eeg_cluster_bootstrap_5000.csv'),check_exact=True)
