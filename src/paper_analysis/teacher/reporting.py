from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from docx import Document


def write_docx_report(
    path: str | Path,
    *,
    title: str,
    paragraphs: Iterable[str],
    tables: Iterable[tuple[str, pd.DataFrame]] = (),
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    document.add_heading(title, 0)
    for paragraph in paragraphs:
        document.add_paragraph(str(paragraph))
    for heading, frame in tables:
        document.add_heading(heading, level=1)
        preview = frame.head(30)
        if preview.empty:
            document.add_paragraph("No rows.")
            continue
        table = document.add_table(rows=1, cols=len(preview.columns))
        table.style = "Table Grid"
        for index, column in enumerate(preview.columns):
            table.rows[0].cells[index].text = str(column)
        for _, row in preview.iterrows():
            cells = table.add_row().cells
            for index, value in enumerate(row):
                cells[index].text = "" if pd.isna(value) else str(value)
    document.save(target)
    return target


def save_condition_plot(
    frame: pd.DataFrame,
    *,
    outcome: str,
    path: str | Path,
    title: str,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    if not frame.empty and outcome in frame:
        summary = (
            frame.groupby(["WWR", "Complexity"], dropna=False)[outcome]
            .agg(["mean", "sem"])
            .reset_index()
        )
        for complexity, sub in summary.groupby("Complexity", dropna=False):
            ax.errorbar(
                sub["WWR"].astype(str), sub["mean"], yerr=1.96 * sub["sem"],
                marker="o", capsize=3, label=str(complexity),
            )
        ax.legend(title="Complexity", frameon=False)
    else:
        ax.text(.5, .5, "No estimable data", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("WWR (categorical)")
    ax.set_ylabel(outcome)
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target


def save_histogram(
    values: pd.Series,
    *,
    path: str | Path,
    title: str,
    xlabel: str,
    thresholds: Iterable[float] = (),
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        ax.text(.5, .5, "No estimable data", ha="center", va="center")
    else:
        ax.hist(numeric, bins=min(20, max(5, int(numeric.nunique()))), color="#4C78A8")
    for threshold in thresholds:
        ax.axvline(float(threshold), linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target
