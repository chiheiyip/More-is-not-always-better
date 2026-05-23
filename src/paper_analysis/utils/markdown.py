from __future__ import annotations

import pandas as pd


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    data = df.copy()
    data = data.fillna("")
    columns = [str(c) for c in data.columns]
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in data.iterrows():
        rows.append("| " + " | ".join(_escape(row[c]) for c in data.columns) + " |")
    return "\n".join(rows)


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
