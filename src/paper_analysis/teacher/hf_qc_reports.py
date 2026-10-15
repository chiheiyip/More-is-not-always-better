"""Evidence-driven reports for the paired HF QC audit."""
from __future__ import annotations

import pandas as pd

from .hf_qc_revision import write_csv


def markdown_table(frame):
    def cell(value):
        if pd.isna(value):
            return "无法推断"
        if isinstance(value, float):
            return f"{value:.8g}"
        return str(value).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                       "| " + " | ".join("---" for _ in frame.columns) + " |",
                       *["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]])


def classification_summary(frame):
    rows = []
    for family, group in frame.groupby("family_id", sort=False):
        row = {"family_id": family, "tests": len(group)}
        for metric in ("p.value", "within_q", "joint_q"):
            row[f"original_{metric}_lt05"] = int(group[f"{metric}_from"].lt(.05).sum())
            row[f"revised_{metric}_lt05"] = int(group[f"{metric}_to"].lt(.05).sum())
            row[f"classification_flips_{metric}"] = int(group[f"flip_{metric}"].fillna(False).sum())
            row[f"unavailable_{metric}"] = int((group[f"{metric}_from"].isna() | group[f"{metric}_to"].isna()).sum())
        row["original_min_joint_q"] = group.joint_q_from.min()
        row["revised_min_joint_q"] = group.joint_q_to.min()
        row["direction_changes"] = int(group.direction_changed.fillna(False).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def audit_report(config, source, run, decision):
    qc, models = run / "qc", run / "audit_models"
    overlap = pd.read_csv(qc / "HF_QC_flag_overlap_by_variant.csv")
    people = pd.read_csv(qc / "HF_QC_participant_impact.csv")
    flow = pd.read_csv(qc / "HF_QC_sample_comparison.csv")
    coef = pd.read_csv(models / "HF_QC_all_coefficient_comparison.csv")
    factor = pd.read_csv(models / "HF_QC_factor_comparison.csv")
    summary = pd.concat([classification_summary(coef), classification_summary(factor)], ignore_index=True)
    write_csv(summary, models / "HF_QC_classification_summary.csv")
    affected_people = people.loc[people.category.eq("A_original_excluded_nohf_retained")]
    core = summary.loc[summary.family_id.eq("condition_core_relative")].iloc[0]
    key = pd.read_csv(models / "HF_QC_key_effects_comparison.csv")
    theta = pd.read_csv(models / "HF_QC_occipital_theta_complexity.csv")
    theta_flip = int(theta["flip_joint_q"].fillna(False).sum())
    temporal = summary.loc[summary.family_id.str.startswith("temporal_")]
    lines = ["# HF ratio 对 QC 样本和 EEG 结论的影响审计", "",
        "## 首页回答", "",
        f"1. HF 标记试次（按窗口分别计数）：{'；'.join(f'{r.onset_trim}s：{r.n_hf_flag}' for r in overlap.itertuples())}。四窗口计数包含同一试次的重复观测。",
        f"2. HF-only：{'；'.join(f'{r.onset_trim}s：{r.n_hf_only}' for r in overlap.itertuples())}。",
        f"3. 删除 HF 后不再触发整人排除：{affected_people.participant_id.nunique()} 人、{len(affected_people)} 个参与者×窗口记录；具体身份见 HF_QC_participant_impact.csv。",
        "4. 四窗口 Original/No-HF 人数与试次数见下表。",
        "5. 四窗口共同样本变化见下表 common 行，具体新增身份见 HF_QC_trial_difference.csv。",
        f"6. 核心 144 项：联合 q 分类翻转 {int(core['classification_flips_joint_q'])} 项；窗口内 q 分类翻转 {int(core['classification_flips_within_q'])} 项；无法比较 {int(core['unavailable_joint_q'])} 项。",
        f"7. 枕区 θ 复杂度效应：四窗口联合 q 分类翻转 {theta_flip} 项，估计值、CI、F、自由度及 p/q 见专表。",
        f"8. 时间效应：联合 q 分类翻转共 {int(temporal['classification_flips_joint_q'].sum())} 项；两类 72 项检验族分别报告。", "",
        "## 样本流", "", markdown_table(flow), "", "## 各判据贡献", "", markdown_table(overlap), "",
        "HF-only 和各指标贡献使用整人排除前的标记。HF-overlap 不代表 HF 对该试次具有独立排除贡献。",
        "实际 boundary 状态仅记录，没有进入正式剔除函数；本轮未新增 boundary 剔除规则。", "",
        "## 检验族与结论变化", "", markdown_table(summary), "",
        "校正族成员无效时不缩小 BH 范围；空值代表无法有效推断或比较，不能解释为不显著。", "",
        "## 额区 θ：WWR45 × C1", "",
        markdown_table(key[["onset_trim_s", "estimate_from", "estimate_to", "CI_low_from", "CI_high_from", "CI_low_to", "CI_high_to", "p.value_from", "p.value_to", "within_q_from", "within_q_to", "joint_q_from", "joint_q_to"]]), "",
        "## 枕区 θ：等权边际 C1−C0", "",
        markdown_table(theta[["onset_trim_s", "estimate_from", "estimate_to", "CI_low_from", "CI_high_from", "CI_low_to", "CI_high_to", "Fstat_from", "Fstat_to", "df_num_from", "df_num_to", "df_denom_from", "df_denom_to", "p.value_from", "p.value_to", "joint_q_from", "joint_q_to"]]), "",
        "## 正式修订条件", "", markdown_table(pd.DataFrame([{"condition": c["condition"], "pass": c["pass"], "evidence": str(c["evidence"])} for c in decision["checks"]])), "",
        f"方法学条件是否全部通过：{decision['adopt_NoHF']}。判定不使用任何 p/q 或显著结果数量。", "",
        "## 来源与解释边界", "", f"Original 来源运行：{source}",
        "HF 的实际实现和真实波形数值核对复用已登记的 MATLAB/Python 核查证据；可复算不等于作为高频污染指标有效。",
        "factor-level 为等权边际 CR2/HTZ 重新实现，历史脚本来源缺口继续保留。previous-scene 使用 PreviousWWR + PreviousComplexity。",
        "经验组按 Q1.4 乒乓球参与频率前两档／后两档划分。本轮未新增经验分类变量。"]
    (run / "HF_RATIO_QC_IMPACT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def formal_report(run):
    summary = pd.read_csv(run / "audit_models/HF_QC_classification_summary.csv")
    flow = pd.read_csv(run / "qc/HF_QC_sample_comparison.csv")
    sensitivity = pd.read_csv(run / "10_eeg_denominator_sensitivity/model_comparisons.csv")
    lines = ["# Original QC 与 Revised No-HF QC 追踪报告", "",
        "Original 使用 HF、RMS、峰峰值等实际判据；Revised 仅删除 HF，其余参数、阈值计算人群及整人排除规则保持一致。",
        "删除原因：平均参考后全部八通道平均波形近乎相消，原比值主要描述微小残差的频谱构成；数值可复现不能证明其污染检测效度。五项方法学条件见 methodological_gate.json。", "",
        "## 样本流", "", markdown_table(flow), "", "## 所有正式检验族", "", markdown_table(summary), "",
        "## 分母敏感性", "",
        f"在修订后相同样本、相同 PSD 上，1–45→1–40 Hz 联合 q 分类翻转 {int(sensitivity.flip_joint_q.fillna(False).sum())} 项（含各独立校正族；重叠族不作为唯一发现计数）。",
        "原始 QC 和结果未覆盖；Original 不再作为本轮正式推断结果。", "",
        "Methods、四窗口样本、3.3、受影响的3.4、表图、附录及结论的具体修改见 manuscript_changes.csv。",
        "本文件是内部追踪报告，投稿正文不叙述内部代码审计过程。"]
    (run / "ORIGINAL_VS_REVISED_EEG_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
