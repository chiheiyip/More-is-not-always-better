"""Render the manuscript's existing five EEG temporal panels from sealed CSVs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

PANELS = [
    ('Alpha relative power', 'alpha_relative', 'Block', 'temporal_relative', 'Viewing round'),
    ('log10 absolute alpha power', 'alpha_absolute', 'Block', 'temporal_absolute', 'Viewing round'),
    ('Alpha relative power', 'alpha_relative', 'PositionWithinBlockCentered', 'temporal_relative', 'Within-round position'),
    ('log10 absolute alpha power', 'alpha_absolute', 'PositionWithinBlockCentered', 'temporal_absolute', 'Within-round position'),
    ('Theta relative power', 'theta_relative', 'Block', 'temporal_relative', 'Viewing round'),
]


def panel_data(frame):
    result = []
    for title, suffix, term, family, axis_title in PANELS:
        sub = frame.loc[frame.family_id.eq(family) & frame.model.eq('Model1') & frame.term.eq(term)].copy()
        names = [('log10_' if suffix.endswith('absolute') else '')+roi+'_'+suffix for roi in ['F', 'P', 'O']]
        sub = sub.loc[sub.outcome.isin(names)].copy()
        if len(sub) != 12 or sub.duplicated(['outcome', 'onset_trim_s']).any():
            raise ValueError('Missing or duplicated temporal figure members')
        if any(set(sub.loc[sub.outcome.eq(name), 'onset_trim_s']) != {0, 5, 10, 15} for name in names):
            raise ValueError('Four parallel windows required for every ROI')
        numeric = sub[['estimate', 'CI_low', 'CI_high', 'joint_q', 'n_trials', 'n_participants']]
        if not np.isfinite(numeric).all().all() or not sub.inference_valid.eq(True).all():
            raise ValueError('Invalid temporal figure inference')
        if not sub.joint_q.between(0, 1).all() or not (sub.CI_low.le(sub.estimate) & sub.estimate.le(sub.CI_high)).all():
            raise ValueError('Invalid interval or adjusted probability')
        if sub[['n_trials', 'n_participants']].drop_duplicates().shape[0] != 1:
            raise ValueError('Temporal figure samples differ')
        result.append((title, term, axis_title, sub.sort_values(['outcome', 'onset_trim_s'])))
    return result


def render(input_path, output_prefix, alignment_script):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    frame = pd.read_csv(input_path); panels = panel_data(frame)
    matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 7.5, 'pdf.fonttype': 42,
                               'svg.fonttype': 'none', 'axes.spines.right': False, 'axes.spines.top': False})
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.65))
    fig.subplots_adjust(left=.075, right=.98, bottom=.105, top=.91, wspace=.4, hspace=.69)
    styles = [('F', '#2869ad', 'o', '-'), ('P', '#b8741a', 's', '--'), ('O', '#087f82', '^', '-.')]
    for index, (title, term, axis_title, data) in enumerate(panels):
        ax = axes.flat[index]
        for offset, (roi, color, marker, line) in zip([-.23, 0, .23], styles):
            subset = data.loc[data.outcome.str.contains('_'+roi+'_') if title.startswith('log10') else data.outcome.str.startswith(roi+'_')].sort_values('onset_trim_s')
            x = np.arange(4)+offset
            ax.plot(x, subset.estimate, color=color, linestyle=line, linewidth=.8)
            for xx, (_, row) in zip(x, subset.iterrows()):
                ax.errorbar(xx, row.estimate, yerr=[[row.estimate-row.CI_low], [row.CI_high-row.estimate]],
                            color=color, fmt=marker, markersize=4.2, capsize=2, linewidth=.8,
                            markerfacecolor=color if row.joint_q < .05 else 'white', markeredgewidth=.8)
        ax.axhline(0, color='#666666', linewidth=.65, linestyle=':')
        ax.set_xticks(range(4), ['0', '5', '10', '15']);ax.set_xlabel('Onset trim (s)', fontsize=7.2)
        ax.set_ylabel('Coefficient', fontsize=7.2)
        ax.set_title(axis_title+'\n'+title, loc='left', fontsize=8, pad=9)
        ax.text(-.17, 1.14, chr(65+index), transform=ax.transAxes, fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
    legend = axes.flat[5];legend.axis('off')
    legend.legend([Line2D([0],[0],color=c,marker=m,linestyle=l,markersize=4) for _,c,m,l in styles],
                  ['Frontal ROI', 'Parietal ROI', 'Occipital ROI'],loc='upper left',fontsize=8,frameon=False)
    first = panels[0][3].iloc[0]
    legend.text(.02,.30,f"{int(first.n_participants)} participants, {int(first.n_trials)} trials\n95% CR2 confidence intervals\nFilled: joint BH q < 0.05\nOpen: joint BH q ≥ 0.05",transform=legend.transAxes,fontsize=7.1,va='top',linespacing=1.6)
    prefix = Path(output_prefix);prefix.parent.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('figure_alignment', alignment_script)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.require_matplotlib_panel_alignment(fig, json_out=str(prefix)+'.alignment.json',
        overlay_svg=str(prefix)+'.alignment.svg', exclude_axes=[legend], strict=True,
        tolerance_pt=1.5, gutter_tolerance_pt=1.5)
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(str(prefix)+'.'+suffix, dpi=300)
    pd.concat([p[3].assign(figure_panel=chr(65+i)) for i,p in enumerate(panels)]).to_csv(str(prefix)+'.source_data.csv', index=False, encoding='utf-8-sig')
    Path(str(prefix)+'.selection.json').write_text(json.dumps({'source_rows':len(frame),'plotted_rows':60,
        'selection':'Preserve the five manuscript panels, all three ROIs and four parallel windows; use joint 72-test BH families independently for relative and absolute temporal terms.',
        'intervals':'Unadjusted 95% participant-clustered CR2 confidence intervals; filled markers denote adjusted inference.'},indent=2),encoding='utf-8')
    plt.close(fig)
