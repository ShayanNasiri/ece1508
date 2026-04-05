"""Streamlit dashboard for Recipe-MPR QA evaluation results."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from recipe_mpr_qa.data.constants import QUERY_TYPE_NAMES as QUERY_TYPES

RESULTS_DIR = Path("llm_evaluation/results")


def load_result_files():
    """Load full-results JSON files (with 'metrics' and 'results' keys)."""
    results = []
    if not RESULTS_DIR.exists():
        return results

    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)

        # Only load full results files (must have nested metrics + results array)
        if "metrics" not in data or "results" not in data:
            continue

        metrics = data["metrics"]
        entry = {
            "file": path.name,
            "model": data.get("model", path.stem),
            "split": data.get("split", "unknown"),
            "overall": metrics.get("overall", 0),
            "total_correct": metrics.get("total_correct", 0),
            "total": metrics.get("total", 0),
            "parse_failures": metrics.get("parse_failures", 0),
            "per_question": data["results"],
        }

        for qt in QUERY_TYPES:
            qt_data = metrics.get(qt, {})
            entry[f"{qt}_accuracy"] = qt_data.get("accuracy", 0)
            entry[f"{qt}_correct"] = qt_data.get("correct", 0)
            entry[f"{qt}_total"] = qt_data.get("total", 0)

        results.append(entry)

    return results


def main():
    st.set_page_config(page_title="Recipe-MPR QA Dashboard", layout="wide")
    st.title("Recipe-MPR QA — Evaluation Dashboard")

    results = load_result_files()
    if not results:
        st.warning(
            f"No full-result files found in `{RESULTS_DIR}/`. "
            "Run an evaluation first (see CLAUDE.md for commands)."
        )
        return

    # --- Summary Table ---
    st.header("Summary")

    summary_rows = []
    for r in results:
        row = {
            "File": r["file"],
            "Model": r["model"],
            "Split": r["split"],
            "Overall Accuracy": f"{r['overall']:.1%}",
            "Correct / Total": f"{r['total_correct']} / {r['total']}",
            "Parse Failures": r["parse_failures"],
        }
        for qt in QUERY_TYPES:
            acc = r[f"{qt}_accuracy"]
            total = r[f"{qt}_total"]
            row[qt] = f"{acc:.1%} ({total})" if total > 0 else "—"
        summary_rows.append(row)

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # --- Bar Chart: Accuracy by Query Type ---
    st.header("Accuracy by Query Type")

    chart_rows = []
    for r in results:
        label = f"{r['model']} ({r['split']})"
        for qt in QUERY_TYPES:
            if r[f"{qt}_total"] > 0:
                chart_rows.append(
                    {
                        "Model": label,
                        "Query Type": qt,
                        "Accuracy": r[f"{qt}_accuracy"],
                    }
                )

    if chart_rows:
        chart_df = pd.DataFrame(chart_rows)
        fig = px.bar(
            chart_df,
            x="Query Type",
            y="Accuracy",
            color="Model",
            barmode="group",
            range_y=[0, 1],
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # --- Drill-Down: Per-Question Results ---
    st.header("Per-Question Drill-Down")

    options_map = {
        f"{r['model']} ({r['split']}) — {r['file']}": r for r in results
    }

    selected = st.selectbox("Select a result file", list(options_map.keys()))
    r = options_map[selected]
    questions = r["per_question"]

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        filter_correct = st.selectbox(
            "Filter", ["All", "Correct only", "Incorrect only", "Parse failures"]
        )
    with col2:
        filter_query = st.text_input("Search query text")

    filtered = questions
    if filter_correct == "Correct only":
        filtered = [q for q in filtered if q.get("is_correct")]
    elif filter_correct == "Incorrect only":
        filtered = [q for q in filtered if not q.get("is_correct") and q.get("predicted_letter")]
    elif filter_correct == "Parse failures":
        filtered = [q for q in filtered if not q.get("predicted_letter")]

    if filter_query:
        filtered = [
            q for q in filtered if filter_query.lower() in q.get("query", "").lower()
        ]

    st.write(f"Showing {len(filtered)} of {len(questions)} questions")

    detail_rows = []
    for q in filtered:
        detail_rows.append(
            {
                "ID": q.get("example_id", ""),
                "Query": q.get("query", ""),
                "Correct": q.get("correct_letter", ""),
                "Predicted": q.get("predicted_letter", "—"),
                "Result": "Correct" if q.get("is_correct") else "Wrong",
            }
        )

    if detail_rows:
        st.dataframe(
            pd.DataFrame(detail_rows), use_container_width=True, hide_index=True
        )

    # Expandable raw response viewer
    st.subheader("Raw Responses")
    for q in filtered:
        with st.expander(
            f"{q.get('example_id', '')} — {'Correct' if q.get('is_correct') else 'Wrong'}"
        ):
            st.markdown(f"**Query:** {q.get('query', '')}")
            st.markdown(
                f"**Correct:** {q.get('correct_letter', '')} | "
                f"**Predicted:** {q.get('predicted_letter', '—')}"
            )
            st.code(q.get("raw_response", ""), language=None)


if __name__ == "__main__":
    main()
