"""
Findings Builder — the "Antara Intelligence Brain" layer.
Takes MediCore's raw analytics output and turns it into an ordered findings[] list:
each finding has a KPI, a chart, and an explanation — ready for the step-confirm UI
(KPI -> chart draws in -> explanation -> next finding).

Deliberately broader than the original 6-finding version: every clinical domain panel,
demographics, financial breakdowns (procedure/insurance/cost-per-day), and patient
segmentation (KMeans, previously computed in core.py but never surfaced) each get a
finding when the relevant columns exist. Findings are additive -- a dataset with only
age/cost/condition still gets the core set; a rich dataset with HIV+diabetes+TB+cancer
markers gets a finding for each domain actually present.
"""

import pandas as pd
import numpy as np

from app import core


def _chart(chart_type, labels, values, x_label="", y_label=""):
    return {
        "type": chart_type,             # "bar" | "line" | "pie"
        "labels": [str(l) for l in labels],
        "values": [round(float(v), 2) for v in values],
        "x_label": x_label,
        "y_label": y_label,
    }


def _top_by(df, group_col, value_col, n=6, ascending=False):
    if not group_col or group_col not in df.columns or value_col not in df.columns:
        return None
    s = df.groupby(group_col)[value_col].mean().sort_values(ascending=ascending).head(n)
    return s if len(s) else None


def build_findings(df, cm, hs, quality, maturity, clinical_domains, top_n=20):
    findings = []
    cond_col = cm.get("condition")
    cost_col = cm.get("cost")
    los_col = cm.get("length_of_stay")
    sat_col = cm.get("satisfaction")
    age_col = cm.get("age")
    gender_col = cm.get("gender")
    ins_col = cm.get("insurance")
    proc_col = cm.get("procedure")
    ward_col = cm.get("ward")
    outcome_col = cm.get("outcome")

    # -- 1. Readmission --
    if "Readmission_Bin" in df.columns:
        rr = hs["rr"]
        severity = "critical" if rr > 20 else "warning" if rr > 15 else "good"
        chart = None
        by_cond = None
        if cond_col and cond_col in df.columns:
            by_cond = (df.groupby(cond_col)["Readmission_Bin"].mean() * 100).sort_values(ascending=False).head(6)
            chart = _chart("bar", by_cond.index, by_cond.values, y_label="Readmission Rate (%)")
        findings.append({
            "id": "readmission", "severity": severity,
            "kpi": {"label": "Readmission Rate", "value": f"{rr:.1f}%", "benchmark": "<=15%"},
            "chart": chart,
            "explanation": (
                f"Readmission rate is {rr:.1f}%, {'above' if rr > 15 else 'within'} the 15% benchmark. "
                + (f"Highest-risk condition: {by_cond.index[0]} ({by_cond.iloc[0]:.1f}%)." if by_cond is not None else "")
            ),
        })

    # -- 2. Cost --
    if cost_col and cost_col in df.columns:
        avg_cost = df[cost_col].mean()
        severity = "critical" if avg_cost > 12000 else "warning" if avg_cost > 9000 else "good"
        by_cond = _top_by(df, cond_col, cost_col)
        chart = _chart("bar", by_cond.index, by_cond.values, y_label="Avg Cost (KES)") if by_cond is not None else None
        findings.append({
            "id": "cost", "severity": severity,
            "kpi": {"label": "Avg Treatment Cost", "value": f"KES {avg_cost:,.0f}", "benchmark": "KES 9,000"},
            "chart": chart,
            "explanation": (
                f"Average treatment cost is KES {avg_cost:,.0f} vs a KES 9,000 benchmark. "
                + (f"Highest-cost condition: {by_cond.index[0]} (KES {by_cond.iloc[0]:,.0f})." if by_cond is not None else "")
            ),
        })

    # -- 3. Satisfaction --
    if sat_col and sat_col in df.columns:
        sat = df[sat_col].mean()
        severity = "critical" if sat < 3.5 else "warning" if sat < 4.0 else "good"
        by_cond = _top_by(df, cond_col, sat_col, ascending=True)
        chart = _chart("bar", by_cond.index, by_cond.values, y_label="Avg Satisfaction (/5)") if by_cond is not None else None
        findings.append({
            "id": "satisfaction", "severity": severity,
            "kpi": {"label": "Avg Satisfaction", "value": f"{sat:.2f}/5", "benchmark": ">=4.0"},
            "chart": chart,
            "explanation": (
                f"Patient satisfaction averages {sat:.2f}/5 against a 4.0 target. "
                + (f"Lowest-scoring condition: {by_cond.index[0]} ({by_cond.iloc[0]:.2f}/5)." if by_cond is not None else "")
            ),
        })

    # -- 4. Length of stay --
    if los_col and los_col in df.columns:
        avg_los = df[los_col].mean()
        severity = "warning" if avg_los > 7 else "good"
        chart = _chart("bar", ["Mean", "Median", "75th pct"],
                        [df[los_col].mean(), df[los_col].median(), df[los_col].quantile(0.75)], y_label="Days")
        findings.append({
            "id": "length_of_stay", "severity": severity,
            "kpi": {"label": "Avg Length of Stay", "value": f"{avg_los:.1f} d", "benchmark": "<=7 d"},
            "chart": chart,
            "explanation": f"Average length of stay is {avg_los:.1f} days against a 7-day benchmark.",
        })

    # -- 5. Admission timing --
    if "Admission_DayOfWeek" in df.columns:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = df["Admission_DayOfWeek"].value_counts().reindex(dow_order).fillna(0)
        peak_day, peak_val = dow.idxmax(), dow.max()
        findings.append({
            "id": "admission_timing", "severity": "info",
            "kpi": {"label": "Peak Admission Day", "value": peak_day, "benchmark": "-"},
            "chart": _chart("bar", dow.index, dow.values, y_label="Admissions"),
            "explanation": f"{peak_day} sees the highest admission volume ({peak_val:.0f} admissions). Consider staffing alignment.",
        })

    # -- 6. Data quality --
    dq = quality["quality_score"]
    findings.append({
        "id": "data_quality",
        "severity": "critical" if dq < 70 else "warning" if dq < 85 else "good",
        "kpi": {"label": "Data Quality Score", "value": f"{dq:.0f}/100", "benchmark": ">=85"},
        "chart": None,
        "explanation": quality["recommendations"][0] if quality["recommendations"] else "Data quality is within acceptable range.",
    })

    # -- 7. Demographics - age distribution --
    if age_col and age_col in df.columns and "Age_Group" in df.columns:
        elderly_pct = (df[age_col] >= 65).mean() * 100
        ag_counts = df["Age_Group"].value_counts().reindex(["<18","18-35","35-50","50-65","65+"]).fillna(0)
        findings.append({
            "id": "demographics_age", "severity": "warning" if elderly_pct > 30 else "info",
            "kpi": {"label": "Patients 65+", "value": f"{elderly_pct:.1f}%", "benchmark": "-"},
            "chart": _chart("bar", ag_counts.index, ag_counts.values, y_label="Patients"),
            "explanation": (
                f"{elderly_pct:.1f}% of the cohort is 65 or older."
                + (" Consider dedicated geriatric assessment capacity." if elderly_pct > 30 else "")
            ),
        })

    # -- 8. Demographics - gender disparity in outcomes --
    if gender_col and gender_col in df.columns and cost_col and cost_col in df.columns:
        by_gender = df.groupby(gender_col)[cost_col].mean().sort_values(ascending=False)
        if len(by_gender) >= 2:
            gap_pct = (by_gender.iloc[0] - by_gender.iloc[-1]) / by_gender.iloc[-1] * 100
            findings.append({
                "id": "gender_cost_gap", "severity": "warning" if gap_pct > 15 else "info",
                "kpi": {"label": "Cost Gap by Gender", "value": f"{gap_pct:.1f}%", "benchmark": "-"},
                "chart": _chart("bar", by_gender.index, by_gender.values, y_label="Avg Cost (KES)"),
                "explanation": (
                    f"{by_gender.index[0]} patients average {gap_pct:.1f}% higher treatment cost than "
                    f"{by_gender.index[-1]} patients."
                ),
            })

    # -- 9. Outcome distribution --
    if outcome_col and outcome_col in df.columns:
        odist = df[outcome_col].value_counts()
        top_outcome, top_pct = odist.index[0], odist.iloc[0] / len(df) * 100
        findings.append({
            "id": "outcomes", "severity": "info",
            "kpi": {"label": "Most Common Outcome", "value": top_outcome, "benchmark": "-"},
            "chart": _chart("bar", odist.index[:6], odist.values[:6], y_label="Patients"),
            "explanation": f"{top_pct:.1f}% of patients had outcome '{top_outcome}'.",
        })

    # -- 10. Financial - cost by insurance/payer --
    if ins_col and ins_col in df.columns and cost_col and cost_col in df.columns:
        by_ins = df.groupby(ins_col)[cost_col].mean().sort_values(ascending=False).head(6)
        if len(by_ins) >= 2:
            findings.append({
                "id": "cost_by_insurance", "severity": "info",
                "kpi": {"label": "Highest-Cost Payer", "value": by_ins.index[0], "benchmark": "-"},
                "chart": _chart("bar", by_ins.index, by_ins.values, y_label="Avg Cost (KES)"),
                "explanation": f"{by_ins.index[0]} patients have the highest average cost at KES {by_ins.iloc[0]:,.0f}.",
            })

    # -- 11. Financial - cost by procedure --
    if proc_col and proc_col in df.columns and cost_col and cost_col in df.columns:
        by_proc = df.groupby(proc_col)[cost_col].mean().sort_values(ascending=False).head(6)
        if len(by_proc) >= 2:
            findings.append({
                "id": "cost_by_procedure", "severity": "info",
                "kpi": {"label": "Highest-Cost Procedure", "value": by_proc.index[0], "benchmark": "-"},
                "chart": _chart("bar", by_proc.index, by_proc.values, y_label="Avg Cost (KES)"),
                "explanation": f"{by_proc.index[0]} averages KES {by_proc.iloc[0]:,.0f} per patient, the highest of any procedure recorded.",
            })

    # -- 12. Financial - high-cost patient share --
    if "High_Cost" in df.columns and cost_col and cost_col in df.columns:
        high_cost_pct = df["High_Cost"].mean() * 100
        high_cost_avg = df.loc[df["High_Cost"], cost_col].mean()
        findings.append({
            "id": "high_cost_patients", "severity": "warning" if high_cost_pct > 30 else "info",
            "kpi": {"label": "High-Cost Patients", "value": f"{high_cost_pct:.1f}%", "benchmark": "top quartile ~25%"},
            "chart": None,
            "explanation": f"{high_cost_pct:.1f}% of patients fall in the high-cost band, averaging KES {high_cost_avg:,.0f} each.",
        })

    # -- 13. Ward / department volume --
    if ward_col and ward_col in df.columns:
        by_ward = df[ward_col].value_counts().head(6)
        if len(by_ward) >= 2:
            findings.append({
                "id": "ward_volume", "severity": "info",
                "kpi": {"label": "Busiest Ward", "value": by_ward.index[0], "benchmark": "-"},
                "chart": _chart("bar", by_ward.index, by_ward.values, y_label="Patients"),
                "explanation": f"{by_ward.index[0]} handles the highest patient volume ({by_ward.iloc[0]:,} patients).",
            })

    # -- 14. Diabetes domain --
    if "Diabetes" in clinical_domains:
        hba1c_col = cm.get("hba1c")
        if hba1c_col and hba1c_col in df.columns:
            uncontrolled_pct = ((df["Glycaemic_Control"] == "Uncontrolled DM").mean() * 100
                                 if "Glycaemic_Control" in df.columns else None)
            chart = None
            if "Glycaemic_Control" in df.columns:
                gc = df["Glycaemic_Control"].value_counts()
                chart = _chart("bar", gc.index, gc.values, y_label="Patients")
            findings.append({
                "id": "diabetes_control",
                "severity": "warning" if (uncontrolled_pct or 0) > 20 else "good",
                "kpi": {"label": "Uncontrolled Diabetes", "value": f"{uncontrolled_pct:.1f}%" if uncontrolled_pct is not None else "N/A", "benchmark": "<=20%"},
                "chart": chart,
                "explanation": "Share of diabetic patients with HbA1c above 8% (poor glycaemic control).",
            })

    # -- 15. HIV / ART domain --
    if "HIV / ART" in clinical_domains:
        cd4_col = cm.get("cd4_count")
        if cd4_col and cd4_col in df.columns and "CD4_Category" in df.columns:
            severe_pct = (df["CD4_Category"] == "Severe (<200)").mean() * 100
            cd4_dist = df["CD4_Category"].value_counts()
            findings.append({
                "id": "hiv_severity",
                "severity": "critical" if severe_pct > 15 else "warning" if severe_pct > 5 else "good",
                "kpi": {"label": "Severe Immunosuppression", "value": f"{severe_pct:.1f}%", "benchmark": "<=5%"},
                "chart": _chart("bar", cd4_dist.index, cd4_dist.values, y_label="Patients"),
                "explanation": "Share of HIV+ patients with CD4 count below 200 cells/uL.",
            })

    # -- 16. Hypertension domain --
    if "Hypertension" in clinical_domains and "BP_Category" in df.columns:
        bp_dist = df["BP_Category"].value_counts()
        stage2_pct = (df["BP_Category"] == "Stage 2 HTN").mean() * 100
        findings.append({
            "id": "hypertension_severity",
            "severity": "critical" if stage2_pct > 20 else "warning" if stage2_pct > 10 else "good",
            "kpi": {"label": "Stage 2 Hypertension", "value": f"{stage2_pct:.1f}%", "benchmark": "<=10%"},
            "chart": _chart("bar", bp_dist.index, bp_dist.values, y_label="Patients"),
            "explanation": f"{stage2_pct:.1f}% of patients with BP readings fall in Stage 2 hypertension (>=140/90 mmHg).",
        })

    # -- 17. Tuberculosis domain --
    if "Tuberculosis" in clinical_domains:
        dr_col = cm.get("drug_resistance")
        if dr_col and dr_col in df.columns:
            resistant_pct = (~df[dr_col].astype(str).str.lower().isin(["negative", "no", "none", "nan"])).mean() * 100
            dr_dist = df[dr_col].value_counts().head(6)
            findings.append({
                "id": "tb_drug_resistance",
                "severity": "critical" if resistant_pct > 10 else "warning" if resistant_pct > 3 else "good",
                "kpi": {"label": "Drug-Resistant TB", "value": f"{resistant_pct:.1f}%", "benchmark": "<=3%"},
                "chart": _chart("bar", dr_dist.index, dr_dist.values, y_label="Patients"),
                "explanation": f"{resistant_pct:.1f}% of TB patients show drug-resistant results (MDR/XDR-TB), which changes treatment protocol and isolation requirements.",
            })

    # -- 18. Cancer / Oncology domain --
    if "Cancer / Oncology" in clinical_domains:
        stage_col = cm.get("cancer_stage")
        if stage_col and stage_col in df.columns:
            stage_dist = df[stage_col].value_counts().head(6)
            late_stage_pct = None
            if "Cancer_Stage_Num" in df.columns:
                late_stage_pct = (df["Cancer_Stage_Num"] >= 3).mean() * 100
            findings.append({
                "id": "cancer_stage",
                "severity": "critical" if (late_stage_pct or 0) > 40 else "warning" if (late_stage_pct or 0) > 20 else "info",
                "kpi": {"label": "Late-Stage Diagnoses (III/IV)", "value": f"{late_stage_pct:.1f}%" if late_stage_pct is not None else "N/A", "benchmark": "<=20%"},
                "chart": _chart("bar", stage_dist.index, stage_dist.values, y_label="Patients"),
                "explanation": "Share of cancer patients diagnosed at Stage III or IV, when treatment is more intensive and prognosis is worse -- often a screening/early-detection signal.",
            })

    # -- 19. Vitals & Labs domain - abnormal readings --
    if "Vitals & Labs" in clinical_domains:
        spo2_col = cm.get("oxygen_saturation")
        if spo2_col and spo2_col in df.columns:
            low_spo2_pct = (pd.to_numeric(df[spo2_col], errors="coerce") < 92).mean() * 100
            if low_spo2_pct > 0:
                findings.append({
                    "id": "low_oxygen_saturation",
                    "severity": "critical" if low_spo2_pct > 10 else "warning" if low_spo2_pct > 3 else "good",
                    "kpi": {"label": "Low SpO2 (<92%)", "value": f"{low_spo2_pct:.1f}%", "benchmark": "<=3%"},
                    "chart": None,
                    "explanation": f"{low_spo2_pct:.1f}% of patients had an oxygen saturation reading below 92%, a threshold typically requiring clinical attention.",
                })

    # -- 20. Patient segmentation (KMeans) - previously computed, never surfaced --
    cluster_features = [c for c in [age_col, cost_col, los_col, sat_col] if c and c in df.columns]
    if len(cluster_features) >= 2:
        try:
            clust = core.run_clustering(df, cluster_features)
        except Exception:
            clust = None
        if clust is not None:
            cluster_sizes = clust["df"]["Cluster"].value_counts().sort_index()
            findings.append({
                "id": "patient_segmentation", "severity": "info",
                "kpi": {"label": "Patient Segments Found", "value": str(clust["best_k"]), "benchmark": "-"},
                "chart": _chart("bar", [f"Cluster {i}" for i in cluster_sizes.index], cluster_sizes.values, y_label="Patients"),
                "explanation": (
                    f"KMeans clustering on {', '.join(cluster_features)} found {clust['best_k']} distinct patient "
                    f"segments (silhouette score {clust['sil']:.2f}, where closer to 1 means more distinct groups)."
                ),
            })

    # Sort by severity: critical > warning > info > good
    order = {"critical": 0, "warning": 1, "info": 2, "good": 3}
    findings.sort(key=lambda f: order.get(f["severity"], 4))

    return findings[:top_n]
