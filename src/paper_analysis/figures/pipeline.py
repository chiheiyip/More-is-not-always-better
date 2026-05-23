from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paper_analysis.utils.io import read_table, write_table, write_text


DEFAULT_WIDTH_MM = 183
DEFAULT_HEIGHT_MM = 120
DEFAULT_DPI = 600

SUPPORT_SCORE = {
    "unsupported_no_data": 0.0,
    "unsupported_no_model": 0.0,
    "unsupported_no_term": 0.0,
    "exploratory_incomplete_modalities": 1.0,
    "exploratory": 1.0,
    "exploratory_only_three_levels": 1.0,
    "see_diagnostics": 1.5,
    "moderate_pending_effect_tests": 2.0,
    "moderate": 2.0,
}


def run_figure_pipeline(
    outputs_root: str | Path = "outputs",
    figure_contracts_config: str | Path = "configs/figure_contracts.json",
    outdir: str | Path | None = None,
) -> dict[str, Path]:
    """Build Nature-style figure exports, source data, manifests, and QA tables."""
    plt = _setup_matplotlib()
    outputs_root = Path(outputs_root)
    outdir = Path(outdir) if outdir is not None else outputs_root / "10_figures"
    config = _load_json(figure_contracts_config)
    export_contract = config.get("journal_export_contract", {})
    width_mm = float(export_contract.get("width_mm", DEFAULT_WIDTH_MM))
    height_mm = float(export_contract.get("height_mm", DEFAULT_HEIGHT_MM))
    dpi = int(export_contract.get("dpi", DEFAULT_DPI))
    formats = _export_formats(export_contract)

    figure_dir = outdir / "figures"
    source_dir = outdir / "source_data"
    manifest_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    legend_blocks: list[str] = ["# Figure Legends And QA Notes", ""]

    for contract in config.get("figures", []):
        figure_id = str(contract["figure_id"])
        source = build_figure_source(figure_id, outputs_root)
        source_path = write_table(source, source_dir / f"{figure_id}_source.csv")
        fig = build_figure(figure_id, source, contract, width_mm=width_mm, height_mm=height_mm)
        export_paths = save_publication_figure(fig, figure_dir / figure_id, formats=formats, dpi=dpi)
        plt.close(fig)

        missing_inputs = missing_source_files(contract, outputs_root)
        qa = figure_qa_row(
            figure_id=figure_id,
            contract=contract,
            source=source,
            source_path=source_path,
            export_paths=export_paths,
            missing_inputs=missing_inputs,
        )
        qa_rows.append(qa)
        manifest_rows.append({
            "figure_id": figure_id,
            "core_conclusion": contract.get("core_conclusion", ""),
            "archetype": contract.get("archetype", ""),
            "backend": "python_matplotlib",
            "width_mm": width_mm,
            "height_mm": height_mm,
            "dpi": dpi,
            "source_csv": str(source_path),
            "svg": str(export_paths.get("svg", "")),
            "pdf": str(export_paths.get("pdf", "")),
            "tiff": str(export_paths.get("tiff", "")),
            "png": str(export_paths.get("png", "")),
            "qa_status": qa["qa_status"],
        })
        legend_blocks.extend(figure_legend_block(contract, source, source_path, export_paths))

    manifest = write_table(pd.DataFrame(manifest_rows), outdir / "figure_manifest.csv")
    qa = write_table(pd.DataFrame(qa_rows), outdir / "figure_qa.csv")
    legends = write_text("\n".join(legend_blocks) + "\n", outdir / "figure_legends.md")
    return {
        "figure_manifest": manifest,
        "figure_qa": qa,
        "figure_legends": legends,
    }


def build_figure_source(figure_id: str, outputs_root: Path) -> pd.DataFrame:
    builders = {
        "Fig1_design_and_sample": _source_fig1,
        "Fig2_questionnaire_effects": _source_fig2,
        "Fig3_eye_aoi_validity": _source_fig3,
        "Fig4_eeg_eye_fusion": _source_fig4,
        "Fig5_robustness_and_claims": _source_fig5,
    }
    builder = builders.get(figure_id)
    source = builder(outputs_root) if builder else pd.DataFrame()
    if source.empty:
        return pd.DataFrame([{
            "figure_id": figure_id,
            "panel_id": "NA",
            "metric": "no_source_data",
            "source_file": "",
            "n": 0,
            "error_bar_definition": "not_applicable_no_data",
        }])
    source.insert(0, "figure_id", figure_id)
    return source


def build_figure(
    figure_id: str,
    source: pd.DataFrame,
    contract: dict[str, Any],
    width_mm: float = DEFAULT_WIDTH_MM,
    height_mm: float = DEFAULT_HEIGHT_MM,
):
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    builders = {
        "Fig1_design_and_sample": _plot_fig1,
        "Fig2_questionnaire_effects": _plot_fig2,
        "Fig3_eye_aoi_validity": _plot_fig3,
        "Fig4_eeg_eye_fusion": _plot_fig4,
        "Fig5_robustness_and_claims": _plot_fig5,
    }
    figsize = (width_mm / 25.4, height_mm / 25.4)
    builder = builders.get(figure_id)
    if builder is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.text(0.5, 0.5, f"No plot builder for {figure_id}", ha="center", va="center")
        ax.set_axis_off()
        return fig
    fig = builder(source, figsize)
    fig.suptitle(contract.get("core_conclusion", figure_id), x=0.01, y=1.02, ha="left", fontsize=7.5, fontweight="bold")
    return fig


def save_publication_figure(fig, basename: Path, formats: list[str], dpi: int = DEFAULT_DPI) -> dict[str, Path]:
    basename.parent.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for fmt in formats:
        if fmt == "source_csv":
            continue
        path = basename.with_suffix(f".{fmt}")
        save_kwargs = {"bbox_inches": "tight"}
        if fmt in {"png", "tiff", "tif"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)
        paths[fmt] = path
    return paths


def figure_qa_row(
    figure_id: str,
    contract: dict[str, Any],
    source: pd.DataFrame,
    source_path: Path,
    export_paths: dict[str, Path],
    missing_inputs: list[str],
) -> dict[str, Any]:
    file_status = {fmt: path.exists() and path.stat().st_size > 0 for fmt, path in export_paths.items()}
    required_formats = {"svg", "pdf", "tiff", "png"}
    missing_exports = sorted(fmt for fmt in required_formats if not file_status.get(fmt, False))
    source_rows = int(len(source))
    has_n = "n" in source.columns and pd.to_numeric(source["n"], errors="coerce").fillna(0).gt(0).any()
    has_error_definition = "error_bar_definition" in source.columns and source["error_bar_definition"].astype(str).str.len().gt(0).any()
    qa_status = "pass"
    if missing_exports or missing_inputs or source_rows == 0 or not has_n or not has_error_definition:
        qa_status = "warning"
    return {
        "figure_id": figure_id,
        "qa_status": qa_status,
        "backend": "python_matplotlib",
        "archetype": contract.get("archetype", ""),
        "source_csv": str(source_path),
        "source_rows": source_rows,
        "panel_count": int(source["panel_id"].nunique()) if "panel_id" in source.columns else 0,
        "has_n": bool(has_n),
        "has_error_bar_definition": bool(has_error_definition),
        "editable_text_policy": "svg.fonttype=none; pdf.fonttype=42",
        "image_integrity_note": contract.get("image_integrity_note", ""),
        "statistics_note": contract.get("statistics_note", ""),
        "missing_input_files": ";".join(missing_inputs),
        "missing_exports": ";".join(missing_exports),
        "review_risk": contract.get("review_risk", ""),
    }


def figure_legend_block(
    contract: dict[str, Any],
    source: pd.DataFrame,
    source_path: Path,
    export_paths: dict[str, Path],
) -> list[str]:
    figure_id = str(contract.get("figure_id", "Figure"))
    panel_rows = []
    for panel in contract.get("panel_map", []):
        panel_rows.append(f"Panel {panel.get('panel_id')}: {panel.get('claim_role')}.")
    n_summary = ""
    if "n" in source.columns and "panel_id" in source.columns:
        n_by_panel = source.groupby("panel_id")["n"].sum(numeric_only=True).to_dict()
        n_summary = "; ".join(f"{panel}=n{int(n)}" for panel, n in n_by_panel.items())
    return [
        f"## {figure_id}",
        "",
        f"Conclusion: {contract.get('core_conclusion', '')}",
        "",
        " ".join(panel_rows) if panel_rows else "Panel map: not specified.",
        "",
        f"Statistics: {contract.get('statistics_note', '')} {n_summary}".strip(),
        "",
        f"Source data: `{source_path}`.",
        "",
        "Exports: " + ", ".join(f"`{path}`" for path in export_paths.values()) + ".",
        "",
        f"Review risk: {contract.get('review_risk', '')}",
        "",
    ]


def missing_source_files(contract: dict[str, Any], outputs_root: Path) -> list[str]:
    missing = []
    for source in contract.get("source_data", []):
        path = _resolve_output_path(source, outputs_root)
        if not path.exists():
            missing.append(str(path))
    return missing


def _source_fig1(outputs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    flow_path = _resolve_output_path("outputs/01_sample_qc/participant_flow.csv", outputs_root)
    flow = _read_optional(flow_path)
    for _, row in flow.iterrows():
        rows.append(_row("A", "participant_count", flow_path, group=row.get("stage"), value=row.get("n"), n=row.get("n"), error="not_applicable_count"))

    balance_path = _resolve_output_path("outputs/01_sample_qc/group_balance_before_after.csv", outputs_root)
    balance = _read_optional(balance_path)
    for _, row in balance.iterrows():
        rows.append(_row("B", "group_balance", balance_path, group=row.get("factor"), subgroup=row.get("level"), value=row.get("n"), n=row.get("n"), percent=row.get("percent"), error="not_applicable_count"))

    design_path = _resolve_output_path("outputs/01_sample_qc/scene_design_balance.csv", outputs_root)
    design = _read_optional(design_path)
    for _, row in design.iterrows():
        rows.append(_row("C", "scene_design_balance", design_path, group=row.get("factor"), subgroup=row.get("level"), value=row.get("trial_count"), n=row.get("trial_count"), error="not_applicable_count"))
    return pd.DataFrame(rows)


def _source_fig2(outputs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    s_path = _resolve_output_path("outputs/02_questionnaire/s_items_descriptives.csv", outputs_root)
    s_desc = _read_optional(s_path)
    if not s_desc.empty and {"item", "WWR", "mean"}.issubset(s_desc.columns):
        for (item, wwr), sub in s_desc.groupby(["item", "WWR"], dropna=False):
            values = pd.to_numeric(sub["mean"], errors="coerce").dropna()
            n_total = int(pd.to_numeric(sub.get("n", pd.Series(1, index=sub.index)), errors="coerce").fillna(0).sum())
            mean = _weighted_mean(sub, "mean", "n")
            sem = float(values.std() / math.sqrt(len(values))) if len(values) > 1 else np.nan
            rows.append(_row("A", "s_item_mean_by_wwr", s_path, group=item, subgroup=wwr, value=mean, mean=mean, sem=sem, n=n_total, error="SEM across available subgroup means"))

    contrast_path = _resolve_output_path("outputs/06_models/emmeans_contrasts.csv", outputs_root)
    contrasts = _read_optional(contrast_path)
    if not contrasts.empty and {"outcome", "contrast", "estimate"}.issubset(contrasts.columns):
        selected = contrasts.loc[contrasts["contrast"].astype(str).eq("WWR45_minus_mean_WWR15_WWR75")]
        for _, row in selected.iterrows():
            rows.append(_row("B", "wwr45_planned_contrast", contrast_path, group=row.get("outcome"), subgroup=row.get("contrast"), value=row.get("estimate"), n=1, error="model contrast estimate; no error bar in this audit figure"))
    return pd.DataFrame(rows)


def _source_fig3(outputs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    aoi_path = _resolve_output_path("outputs/03_eye_tracking/aoi_validation_summary.csv", outputs_root)
    aoi = _read_optional(aoi_path)
    if not aoi.empty and "class_name" in aoi.columns:
        for class_name, sub in aoi.groupby("class_name", dropna=False):
            visited = pd.to_numeric(sub.get("visited_rate"), errors="coerce")
            area = pd.to_numeric(sub.get("polygon_area_px2"), errors="coerce")
            rows.append(_row("A", "visited_rate", aoi_path, group=class_name, value=visited.mean(), mean=visited.mean(), sem=_sem(visited), n=visited.notna().sum(), error="SEM across AOI validation rows"))
            rows.append(_row("B", "polygon_area_px2", aoi_path, group=class_name, value=area.mean(), mean=area.mean(), sem=_sem(area), n=area.notna().sum(), error="SEM across AOI validation rows"))

    qc_path = _resolve_output_path("outputs/03_eye_tracking/eye_qc.csv", outputs_root)
    qc = _read_optional(qc_path)
    if not qc.empty:
        for col in [c for c in ["missing_eye_file", "missing_aoi_file"] if c in qc.columns]:
            count = qc[col].astype(str).str.lower().isin({"true", "1", "yes", "y"}).sum()
            rows.append(_row("C", col, qc_path, group=col, value=count, n=len(qc), error="not_applicable_count"))
        if "eye_sample_count" in qc.columns:
            values = pd.to_numeric(qc["eye_sample_count"], errors="coerce")
            rows.append(_row("C", "eye_sample_count", qc_path, group="eye_sample_count", value=values.mean(), mean=values.mean(), sem=_sem(values), n=values.notna().sum(), error="SEM across scene files"))
    return pd.DataFrame(rows)


def _source_fig4(outputs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    aligned_path = _resolve_output_path("outputs/05_multimodal_fusion/aligned_scene_table.csv", outputs_root)
    aligned = _read_optional(aligned_path)
    theta_col = _first_col(aligned, ["eeg_O_theta", "O_theta"])
    if not aligned.empty and theta_col and "attention_share" in aligned.columns:
        for _, row in aligned.iterrows():
            rows.append(_row("A", "eeg_eye_scene_projection", aligned_path, group=row.get("class_name"), subgroup=row.get("WWR"), x=row.get("attention_share"), y=row.get(theta_col), value=row.get(theta_col), n=1, error="raw scene-level point"))

    sync_path = _resolve_output_path("outputs/05_multimodal_fusion/sync_qc.csv", outputs_root)
    sync = _read_optional(sync_path)
    if not sync.empty and "duration_delta_s" in sync.columns:
        for _, row in sync.iterrows():
            rows.append(_row("B", "duration_delta_s", sync_path, group=row.get("participant_id"), subgroup=row.get("scene_id"), value=row.get("duration_delta_s"), n=1, error="raw trial-level QC value"))

    support_path = _resolve_output_path("outputs/05_multimodal_fusion/claim_support_matrix.csv", outputs_root)
    support = _read_optional(support_path)
    if not support.empty and "claim_id" in support.columns:
        for _, row in support.iterrows():
            level = str(row.get("support_level", ""))
            rows.append(_row("C", "claim_support_score", support_path, group=row.get("claim_id"), subgroup=level, value=_support_score(level), n=1, error="categorical score for visual audit"))
    return pd.DataFrame(rows)


def _source_fig5(outputs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    order_path = _resolve_output_path("outputs/06_robustness/order_fatigue_effects.csv", outputs_root)
    order = _read_optional(order_path)
    if not order.empty and "correlation" in order.columns:
        selected = order.dropna(subset=["correlation"]).copy()
        selected["abs_correlation"] = pd.to_numeric(selected["correlation"], errors="coerce").abs()
        selected = selected.sort_values("abs_correlation", ascending=False).head(12)
        for _, row in selected.iterrows():
            rows.append(_row("A", "order_abs_correlation", order_path, group=row.get("outcome"), subgroup=row.get("order_variable"), value=row.get("abs_correlation"), n=row.get("n"), error="diagnostic correlation; no error bar"))

    nonlinear_path = _resolve_output_path("outputs/06_robustness/nonlinear_wwr_sensitivity.csv", outputs_root)
    nonlinear = _read_optional(nonlinear_path)
    if not nonlinear.empty and "outcome" in nonlinear.columns:
        for _, row in nonlinear.iterrows():
            rows.append(_row("B", "wwr45_peak_index", nonlinear_path, group=row.get("outcome"), subgroup=row.get("claim_strength"), value=row.get("wwr45_peak_index"), n=1, error="diagnostic index; no error bar"))

    claim_path = _resolve_output_path("outputs/07_paper_tables/claim_strength_table.csv", outputs_root)
    claim = _read_optional(claim_path)
    if not claim.empty and "claim_id" in claim.columns:
        for _, row in claim.iterrows():
            level = str(row.get("support_level", ""))
            rows.append(_row("C", "claim_strength_score", claim_path, group=row.get("claim_id"), subgroup=level, value=_support_score(level), n=1, error="categorical score for visual audit"))
    return pd.DataFrame(rows)


def _plot_fig1(source: pd.DataFrame, figsize: tuple[float, float]):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    _bar_axis(axes[0], _panel(source, "A"), title="A Participant flow", label_col="group", value_col="value", color="#4C78A8")
    _barh_axis(axes[1], _panel(source, "B").head(10), title="B Group balance", label_cols=["group", "subgroup"], value_col="value", color="#72B7B2")
    design = _panel(source, "C")
    design = design.loc[design["group"].astype(str).isin(["WWR", "Complexity", "position", "condition_id"])].head(12)
    _barh_axis(axes[2], design, title="C Scene design", label_cols=["group", "subgroup"], value_col="value", color="#F58518")
    return fig


def _plot_fig2(source: pd.DataFrame, figsize: tuple[float, float]):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    panel = _panel(source, "A")
    for item, sub in panel.groupby("group", dropna=False):
        sub = sub.copy()
        sub["_x"] = pd.to_numeric(sub["subgroup"], errors="coerce")
        sub = sub.sort_values("_x")
        axes[0].errorbar(sub["_x"], pd.to_numeric(sub["mean"], errors="coerce"), yerr=pd.to_numeric(sub["sem"], errors="coerce"), marker="o", linewidth=1.2, label=str(item), capsize=2)
    axes[0].set_title("A S-item means by WWR")
    axes[0].set_xlabel("WWR")
    axes[0].set_ylabel("Mean score")
    axes[0].legend(fontsize=5, ncol=2)
    _barh_axis(axes[1], _panel(source, "B"), title="B WWR45 planned contrast", label_cols=["group"], value_col="value", color="#E45756")
    axes[1].axvline(0, color="#333333", linewidth=0.6)
    return fig


def _plot_fig3(source: pd.DataFrame, figsize: tuple[float, float]):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    _bar_axis(axes[0], _panel(source, "A"), title="A AOI visited rate", label_col="group", value_col="mean", error_col="sem", color="#54A24B")
    axes[0].set_ylim(0, 1.05)
    _bar_axis(axes[1], _panel(source, "B"), title="B AOI area", label_col="group", value_col="mean", error_col="sem", color="#B279A2")
    _barh_axis(axes[2], _panel(source, "C"), title="C Eye QC", label_cols=["group"], value_col="value", color="#FF9DA6")
    return fig


def _plot_fig4(source: pd.DataFrame, figsize: tuple[float, float]):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax = axes[0, 0]
    panel = _panel(source, "A")
    for group, sub in panel.groupby("group", dropna=False):
        ax.scatter(pd.to_numeric(sub["x"], errors="coerce"), pd.to_numeric(sub["y"], errors="coerce"), s=18, alpha=0.8, label=str(group))
    ax.set_title("A EEG-eye projection")
    ax.set_xlabel("Attention share")
    ax.set_ylabel("O theta")
    ax.legend(fontsize=5)
    _barh_axis(axes[0, 1], _panel(source, "B"), title="B Duration delta", label_cols=["group", "subgroup"], value_col="value", color="#4C78A8")
    axes[0, 1].axvline(0, color="#333333", linewidth=0.6)
    _barh_axis(axes[1, 0], _panel(source, "C"), title="C Claim support", label_cols=["group", "subgroup"], value_col="value", color="#72B7B2")
    axes[1, 1].axis("off")
    axes[1, 1].text(0.0, 0.8, "Interpret EEG only with\nsync QC and modality\nconvergence.", fontsize=8, va="top")
    return fig


def _plot_fig5(source: pd.DataFrame, figsize: tuple[float, float]):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    _barh_axis(axes[0], _panel(source, "A"), title="A Order/fatigue", label_cols=["group", "subgroup"], value_col="value", color="#F58518")
    _barh_axis(axes[1], _panel(source, "B").head(12), title="B WWR sensitivity", label_cols=["group"], value_col="value", color="#E45756")
    axes[1].axvline(0, color="#333333", linewidth=0.6)
    _barh_axis(axes[2], _panel(source, "C"), title="C Claim strength", label_cols=["group", "subgroup"], value_col="value", color="#72B7B2")
    return fig


def _bar_axis(ax, data: pd.DataFrame, title: str, label_col: str, value_col: str, color: str, error_col: str | None = None) -> None:
    if data.empty:
        _no_data(ax, title)
        return
    labels = data[label_col].astype(str).tolist()
    values = pd.to_numeric(data[value_col], errors="coerce").fillna(0)
    errors = pd.to_numeric(data[error_col], errors="coerce") if error_col and error_col in data.columns else None
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, color=color, edgecolor="#222222", linewidth=0.4, capsize=2 if errors is not None else 0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(value_col)


def _barh_axis(ax, data: pd.DataFrame, title: str, label_cols: list[str], value_col: str, color: str) -> None:
    if data.empty:
        _no_data(ax, title)
        return
    labels = data.apply(lambda row: " | ".join(str(row.get(col, "")) for col in label_cols), axis=1).tolist()
    values = pd.to_numeric(data[value_col], errors="coerce").fillna(0)
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, edgecolor="#222222", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(value_col)


def _no_data(ax, title: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_axis_off()


def _setup_matplotlib():
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    return plt


def _panel(source: pd.DataFrame, panel_id: str) -> pd.DataFrame:
    return source.loc[source["panel_id"].astype(str).eq(panel_id)].copy() if "panel_id" in source.columns else pd.DataFrame()


def _row(
    panel_id: str,
    metric: str,
    source_file: Path,
    group: Any = "",
    subgroup: Any = "",
    value: Any = np.nan,
    x: Any = np.nan,
    y: Any = np.nan,
    mean: Any = np.nan,
    sem: Any = np.nan,
    n: Any = np.nan,
    percent: Any = np.nan,
    error: str = "",
) -> dict[str, Any]:
    return {
        "panel_id": panel_id,
        "metric": metric,
        "source_file": str(source_file),
        "group": group,
        "subgroup": subgroup,
        "value": _number(value),
        "x": _number(x),
        "y": _number(y),
        "mean": _number(mean),
        "sem": _number(sem),
        "n": _number(n),
        "percent": _number(percent),
        "error_bar_definition": error,
    }


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_table(path)


def _resolve_output_path(path: str | Path, outputs_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "outputs":
        return outputs_root.joinpath(*parts[1:])
    return path


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _export_formats(export_contract: dict[str, Any]) -> list[str]:
    formats = [str(fmt).lower() for fmt in export_contract.get("formats", ["svg", "pdf", "tiff", "png", "source_csv"])]
    for required in ["svg", "pdf", "tiff", "png", "source_csv"]:
        if required not in formats:
            formats.append(required)
    return formats


def _weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    values = pd.to_numeric(df[value_col], errors="coerce")
    weights = pd.to_numeric(df.get(weight_col, pd.Series(1, index=df.index)), errors="coerce").fillna(0)
    mask = values.notna() & weights.gt(0)
    if not mask.any():
        return float(values.mean()) if values.notna().any() else np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def _sem(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return np.nan
    return float(numeric.std() / math.sqrt(len(numeric)))


def _number(value: Any) -> float | str:
    if value is None:
        return np.nan
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return np.nan
    except TypeError:
        return str(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _support_score(level: str) -> float:
    text = str(level)
    if text in SUPPORT_SCORE:
        return SUPPORT_SCORE[text]
    for key, score in SUPPORT_SCORE.items():
        if key in text:
            return score
    return np.nan
