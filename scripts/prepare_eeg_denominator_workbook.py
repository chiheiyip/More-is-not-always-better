"""Prepare typed, participant-free workbook tables from a completed EEG run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def typed_table(name, frame):
    numeric=[c for c in frame if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
    return {"name":name,"columns":frame.columns.tolist(),"numeric":numeric,
            "rows":json.loads(frame.to_json(orient="values",force_ascii=False,double_precision=15))}


def prepare(run):
    manifest=json.loads((run/"run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"]!="complete":raise ValueError("Only complete analyses can be exported")
    focus=pd.read_csv(run/"manuscript_comparisons.csv")
    rows=[]
    for (pair,claim),g in focus.groupby(["pair","manuscript_claim"],sort=False):
        valid=g.joint_q_from.notna().all() and g.joint_q_to.notna().all()
        changed=g.flip_joint_q.fillna(False).any() or g.direction_changed.fillna(False).any()
        rows.append({"比较":pair,"正文结论":claim,"检验数":len(g),
            "原联合显著数":int(g.joint_q_from.lt(.05).sum()),"新联合显著数":int(g.joint_q_to.lt(.05).sum()),
            "联合q翻转数":int(g.flip_joint_q.sum()),"方向变化数":int(g.direction_changed.sum()),
            "判断":"无法判断" if not valid else "改变" if changed else "方向及联合显著性未改变"})
    claims=pd.DataFrame(rows)
    claims.to_csv(run/"manuscript_conclusion_summary.csv",index=False,encoding="utf-8-sig")
    tables=[typed_table("主要结论",claims)]
    comparison=pd.read_csv(run/"model_comparisons.csv")
    fields=["onset_trim_s","outcome","model","term","family_id","test_kind",
        "estimate_from","estimate_to","CI_low_from","CI_high_from","CI_low_to","CI_high_to",
        "std.error_from","std.error_to","Fstat_from","Fstat_to","df_num_from","df_num_to",
        "df_denom_from","df_denom_to","df_from","df_to","p.value_from","p.value_to",
        "within_q_from","within_q_to","joint_q_from","joint_q_to","delta_estimate",
        "estimate_percent_change","delta_p.value","delta_within_q","delta_joint_q",
        "flip_p.value","flip_within_q","flip_joint_q","direction_changed","inference_valid_from","inference_valid_to"]
    for pair,name in (("B→C","B到C分母效应"),("A→B","A到B输入复现"),("A→C","A到C论文对照")):
        tables.append(typed_table(name,comparison.loc[comparison.pair.eq(pair),[c for c in fields if c in comparison]]))
    focus_columns=["pair","manuscript_claim",*fields[:6],"estimate_from","estimate_to","Fstat_from","Fstat_to",
        "p.value_from","p.value_to","within_q_from","within_q_to","joint_q_from","joint_q_to","flip_joint_q","direction_changed"]
    tables.append(typed_table("正文数值",focus[[c for c in focus_columns if c in focus]]))
    tables.append(typed_table("功率占比",pd.read_csv(run/"power_fraction_summary.csv")))
    tables.append(typed_table("输入复现差异",pd.read_csv(run/"input_reproduction_differences.csv")))
    for filename,name in (("diagnostics","模型诊断"),("family_status","校正检验族")):
        frames=[]
        for version in ("A","B","C"):
            f=pd.read_csv(run/version/f"{filename}.csv");f.insert(0,"version",version);frames.append(f)
        tables.append(typed_table(name,pd.concat(frames,ignore_index=True)))
    notes=[
        ("样本","0805冻结42人、461共同试次；0/5/10/15 s，参与者聚类CR2。"),
        ("A","0805历史1–45 Hz功率及模型输入。"),
        ("B","当前预处理.set/.fdt按原Welch参数复现1–45 Hz。"),
        ("C","同B的PSD、分子及absolute功率，仅将relative分母改为1–40 Hz。"),
        ("主要比较","B到C判断分母效应；A到B单列源文件复现差异；A到C核对论文。"),
        ("factor-level","等权边际CR2/HTZ重新实现，已核对独立参考表。多自由度总体F没有单一β。"),
        ("上一场景","PreviousWWR + PreviousComplexity，保持0805执行结构；论文交互项描述与代码有差异。"),
        ("校正","各版本独立计算窗口内及四窗口联合BH。最终判断用joint_q；完整族失败时q不发布。"),
        ("功率比例","40–45功率占比与实际分母减少比例独立积分，不使用统一缩放。比例单位为%。"),
        ("空白","多自由度β/CI、接近零β的百分比，以及未校正全系数的q留空，表示不适用。"),
        ("数值","底层保留数值精度；科学计数显示p/q。原始CSV、对比矩阵与会话信息保存在完整运行目录。"),
        ("代码提交",manifest["git_sha"]),
        ("历史回归",json.dumps(manifest["regression"],ensure_ascii=False)),
    ]
    tables.append(typed_table("说明",pd.DataFrame(notes,columns=["项目","说明"])))
    target=run/"workbook_tables.json"
    target.write_text(json.dumps({"tables":tables},ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print(json.dumps({"tables":len(tables),"comparison_rows":len(comparison),"output":str(target)},ensure_ascii=False))
    return target


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("run",type=Path)
    prepare(parser.parse_args().run)
