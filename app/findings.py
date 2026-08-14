"""
Findings Builder — the "Antara Intelligence Brain" layer.
Takes MediCore's raw analytics output and turns it into an ordered findings[] list:
each finding has a KPI, a chart, and an explanation — ready for the step-confirm UI
(KPI -> chart draws in -> explanation -> next finding).
"""

import pandas as pd
import numpy as np


def _chart(chart_type, labels, values, x_label="", y_label=""):
    return {
        "type": chart_type,             # "bar" | "line" | "pie"
        "labels": [str(l) for l in labels],
        "values": [round(float(v), 2) for v in values],
        "x_label": x_label,
        "y_label": y_label,
    }


def build_findings(df, cm, hs, quality, maturity, clinical_domains, top_n=8):
    """Returns an ordered list of findings, highest severity first."""
    findings = []
    cond_col = cm.get("condition")
    cost_col = cm.get("cost")
    los_col = cm.get("length_of_stay")
    sat_col = cm.get("satisfaction")
    age_col = cm.get("age")

    # 1. Readmission
    if "Readmission_Bin" in df.columns:
        rr = hs["rr"]
        severity = "critical" if rr > 20 else "warning" if rr > 15 else "good"
        chart = None
        if cond_col and cond_col in df.columns:
            by_cond = (df.groupby(cond_col)["Readmission_Bin"].mean() * 100).sort_values(ascending=False).head(6)
            chart = _chart("bar", by_cond.index, by_cond.values, y_label="Readmission Rate (%)")
        findings.append({
            "id": "readmission",
            "severity": severity,
            "kpi": {"label": "Readmission Rate", "value": f"{rr:.1f}%", "benchmark": "≤15%"},
            "chart": chart,
            "explanation": (
                f"Readmission rate is {rr:.1f}%, "
                f"{'above' if rr > 15 else 'within'} the 15% benchmark. "
                + (f"Highest-risk condition: {by_cond.index[0]} ({by_cond.iloc[0]:.1f}%)."
                   if chart else "")
            ),
        })

    # 2. Cost
    if cost_col and cost_col in df.columns:
        avg_cost = df[cost_col].mean()
        severity = "critical" if avg_cost > 12000 else "warning" if avg_cost > 9000 else "good"
        chart = None
        if cond_col and cond_col in df.columns:
            by_cond = df.groupby(cond_col)[cost_col].mean().sort_values(ascending=False).head(6)
            chart = _chart("bar", by_cond.index, by_cond.values, y_label="Avg Cost (KES)")
        findings.append({
            "id": "cost",
            "severity": severity,
            "kpi": {"label": "Avg Treatment Cost", "value": f"KES {avg_cost:,.0f}", "benchmark": "KES 9,000"},
            "chart": chart,
            "explanation": (
                f"Average treatment cost is KES {avg_cost:,.0f} vs a KES 9,000 benchmark. "
                + (f"Highest-cost condition: {by_cond.index[0]} (KES {by_cond.iloc[0]:,.0f})."
                   if chart else "")
            ),
        })

    # 3. Satisfaction
    if sat_col and sat_col in df.columns:
        sat = df[sat_col].mean()
        severity = "critical" if sat < 3.5 else "warning" if sat < 4.0 else "good"
        chart = None
        if cond_col and cond_col in df.columns:
            by_cond = df.groupby(cond_col)[sat_col].mean().sort_values().head(6)
            chart = _chart("bar", by_cond.index, by_cond.values, y_label="Avg Satisfaction (/5)")
        findings.append({
            "id": "satisfaction",
            "severity": severity,
            "kpi": {"label": "Avg Satisfaction", "value": f"{sat:.2f}/5", "benchmark": "≥4.0"},
            "chart": chart,
            "explanation": (
                f"Patient satisfaction averages {sat:.2f}/5 against a 4.0 target. "
                + (f"Lowest-scoring condition: {by_cond.index[0]} ({by_cond.iloc[0]:.2f}/5)."
                   if chart else "")
            ),
        })

    # 4. Length of stay
    if los_col and los_col in df.columns:
        avg_los = df[los_col].mean()
        severity = "warning" if avg_los > 7 else "good"
        chart = _chart("bar",
                        ["Mean", "Median", "75th pct"],
                        [df[los_col].mean(), df[los_col].median(), df[los_col].quantile(0.75)],
                        y_label="Days")
        findings.append({
            "id": "length_of_stay",
            "severity": severity,
            "kpi": {"label": "Avg Length of Stay", "value": f"{avg_los:.1f} d", "benchmark": "≤7 d"},
            "chart": chart,
            "explanation": f"Average length of stay is {avg_los:.1f} days against a 7-day benchmark.",
        })

    # 5. Admission timing anomaly
    if "Admission_DayOfWeek" in df.columns:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = df["Admission_DayOfWeek"].value_counts().reindex(dow_order).fillna(0)
        peak_day, peak_val = dow.idxmax(), dow.max()
        findings.append({
            "id": "admission_timing",
            "severity": "info",
            "kpi": {"label": "Peak Admission Day", "value": peak_day, "benchmark": "—"},
            "chart": _chart("bar", dow.index, dow.values, y_label="Admissions"),
            "explanation": f"{peak_day} sees the highest admission volume ({peak_val:.0f} admissions). Consider staffing alignment.",
        })

    # 6. Data quality
    dq = quality["quality_score"]
    severity = "critical" if dq < 70 else "warning" if dq < 85 else "good"
    findings.append({
        "id": "data_quality",
        "severity": severity,
        "kpi": {"label": "Data Quality Score", "value": f"{dq:.0f}/100", "benchmark": "≥85"},
        "chart": None,
        "explanation": quality["recommendations"][0] if quality["recommendations"] else "Data quality is within acceptable range.",
    })

    # 7. Clinical domain flags (e.g. HbA1c, CD4)
    if "Diabetes" in clinical_domains:
        hba1c_col = cm.get("hba1c")
        if hba1c_col and hba1c_col in df.columns:
            uncontrolled_pct = (
                (df["Glycaemic_Control"] == "Uncontrolled DM").mean() * 100
                if "Glycaemic_Control" in df.columns else None
            )
            findings.append({
                "id": "diabetes_control",
                "severity": "warning" if (uncontrolled_pct or 0) > 20 else "good",
                "kpi": {"label": "Uncontrolled Diabetes", "value": f"{uncontrolled_pct:.1f}%" if uncontrolled_pct is not None else "N/A", "benchmark": "≤20%"},
                "chart": None,
                "explanation": "Share of diabetic patients with HbA1c above 8% (poor glycaemic control).",
            })

    if "HIV / ART" in clinical_domains:
        cd4_col = cm.get("cd4_count")
        if cd4_col and cd4_col in df.columns and "CD4_Category" in df.columns:
            severe_pct = (df["CD4_Category"] == "Severe (<200)").mean() * 100
            findings.append({
                "id": "hiv_severity",
                "severity": "critical" if severe_pct > 15 else "warning" if severe_pct > 5 else "good",
                "kpi": {"label": "Severe Immunosuppression", "value": f"{severe_pct:.1f}%", "benchmark": "≤5%"},
                "chart": None,
                "explanation": "Share of HIV+ patients with CD4 count below 200 cells/μL.",
            })

    # Sort by severity: critical > warning > info > good
    order = {"critical": 0, "warning": 1, "info": 2, "good": 3}
    findings.sort(key=lambda f: order.get(f["severity"], 4))

    return findings[:top_n]
