import pandas as pd
import pytest

from paper_analysis.teacher.manuscript_figure import PANELS, panel_data


def data():
    rows=[]
    for _,suffix,term,family,_ in PANELS:
        for roi in ['F','P','O']:
            for trim in [0,5,10,15]:
                rows.append(dict(outcome=('log10_' if suffix.endswith('absolute') else '')+roi+'_'+suffix,
                                 model='Model1',term=term,family_id=family,onset_trim_s=trim,
                                 estimate=.02,CI_low=-.01,CI_high=.05,joint_q=.06,n_trials=24,
                                 n_participants=8,inference_valid=True))
    return pd.DataFrame(rows)


def test_all_planned_panels_keep_every_roi_window_and_nonsignificant_member():
    frame=data();panels=panel_data(frame)
    assert len(panels)==5 and sum(len(p[3]) for p in panels)==60
    assert all(p[3].joint_q.eq(.06).all() for p in panels)


@pytest.mark.parametrize('failure',['missing','duplicate','invalid','interval','probability'])
def test_incomplete_invalid_or_duplicate_figure_members_stop(failure):
    frame=data()
    if failure=='missing':frame=frame.iloc[1:]
    if failure=='duplicate':frame=pd.concat([frame,frame.iloc[:1]])
    if failure=='invalid':frame.loc[0,'inference_valid']=False
    if failure=='interval':frame.loc[0,'CI_high']=-.5
    if failure=='probability':frame.loc[0,'joint_q']=1.2
    with pytest.raises(ValueError):panel_data(frame)
