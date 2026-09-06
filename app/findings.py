"""
Findings Builder -- the "Antara Intelligence Brain" layer.
Takes MediCore's raw analytics output and turns it into an ordered findings[] list:
each finding has a KPI, a chart, and an explanation -- ready for the step-confirm UI
(KPI -> chart draws in -> explanation -> next finding).

v3: adds statistical inference (correlation/t-test/ANOVA), analytics maturity +
improvement roadmap, additional recommendation-style findings (geriatric, staffing),
and depth on existing clinical domains (viral load, ART-vs-CD4, glucose, recovery
rate by condition, readmission by age group, admission hour, condition x gender).
Each finding is still purely additive -- it only appears when its source columns exist.
"""

import pandas as pd
import numpy as np

from app import core

try:
    from scipy import stats as scipy_stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


def _chart(chart_type, labels, values, x_label="", y_label=""):
    return {
        "type": chart_type,
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


def _compare_all(series, formatter):
    """
    Turns a sorted (label -> value) series into a comparison sentence.
    Four or fewer categories: names all of them. More than four: shows the
    first two and last two (the extremes on both ends of the sort) with a
    "then N more" bridge, so the walkthrough still points at real bars
    without reading out a long list. `formatter` turns a raw value into its
    display string, e.g. lambda v: f"{v:.1f}%".
    """
    if series is None or len(series) == 0:
        return ""
    n = len(series)
    if n <= 4:
        parts = [f"{label} {formatter(val)}" for label, val in series.items()]
        listed = parts[0] if n == 1 else ", ".join(parts[:-1]) + f", and {parts[-1]}"
    else:
        top = [f"{label} {formatter(val)}" for label, val in series.iloc[:2].items()]
        bottom = [f"{label} {formatter(val)}" for label, val in series.iloc[-2:].items()]
        skipped = n - 4
        listed = f"{top[0]} and {top[1]} lead, then {skipped} more, down to {bottom[0]} and {bottom[1]}"
    if n == 1:
        return f"{listed}."
    spread = abs(series.iloc[0] - series.iloc[-1])
    return f"By category: {listed} -- a {formatter(spread)} spread from lowest to highest."


def build_findings(df, cm, hs, quality, maturity, clinical_domains, top_n=40):
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

    # ==================== CORE (readmission, cost, satisfaction, LOS) ====================

    if "Readmission_Bin" in df.columns:
        rr = hs["rr"]
        severity = "critical" if rr > 20 else "warning" if rr > 15 else "good"
        by_cond = None
        chart = None
        if cond_col and cond_col in df.columns:
            by_cond = (df.groupby(cond_col)["Readmission_Bin"].mean() * 100).sort_values(ascending=False).head(6)
            chart = _chart("bar", by_cond.index, by_cond.values, y_label="Readmission Rate (%)")
        findings.append({
            "id": "readmission", "severity": severity,
            "kpi": {"label": "Readmission Rate", "value": f"{rr:.1f}%", "benchmark": "<=15%"},
            "chart": chart,
            "explanation": (
                f"Readmission rate is {rr:.1f}%, {'above' if rr > 15 else 'within'} the 15% benchmark. "
                + (_compare_all(by_cond, lambda v: f"{v:.1f}%") if by_cond is not None else "")
            ),
        })

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
                + (_compare_all(by_cond, lambda v: f"KES {v:,.0f}") if by_cond is not None else "")
            ),
        })

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
                + (_compare_all(by_cond, lambda v: f"{v:.2f}/5") if by_cond is not None else "")
            ),
        })

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

    if "Long_Stay" in df.columns:
        long_stay_pct = df["Long_Stay"].mean() * 100
        findings.append({
            "id": "long_stay_share", "severity": "warning" if long_stay_pct > 30 else "info",
            "kpi": {"label": "Long-Stay Patients", "value": f"{long_stay_pct:.1f}%", "benchmark": "top quartile ~25%"},
            "chart": None,
            "explanation": f"{long_stay_pct:.1f}% of patients stayed longer than the 75th-percentile length of stay.",
        })

    # ==================== ADMISSIONS & TIMING ====================

    if "Admission_DayOfWeek" in df.columns:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = df["Admission_DayOfWeek"].value_counts().reindex(dow_order).fillna(0)
        peak_day, peak_val = dow.idxmax(), dow.max()
        findings.append({
            "id": "admission_timing", "severity": "info",
            "kpi": {"label": "Peak Admission Day", "value": peak_day, "benchmark": "--"},
            "chart": _chart("bar", dow.index, dow.values, y_label="Admissions"),
            "explanation": (
                f"{peak_day} sees the highest admission volume. "
                + _compare_all(dow.sort_values(ascending=False), lambda v: f"{int(v):,} admissions")
                + " Consider staffing alignment."
            ),
        })

    if "Admission_Hour" in df.columns:
        hour_counts = df["Admission_Hour"].value_counts().sort_index()
        # Guard: if every admission falls in one hour, the source data almost certainly
        # stores dates without real time-of-day (defaults to midnight) -- not a real pattern.
        if len(hour_counts) > 1 and hour_counts.max() / hour_counts.sum() < 0.9:
            peak_hour = int(hour_counts.idxmax())
            findings.append({
                "id": "admission_hour_peak", "severity": "info",
                "kpi": {"label": "Peak Admission Hour", "value": f"{peak_hour}:00", "benchmark": "--"},
                "chart": _chart("bar", [f"{h}:00" for h in hour_counts.index], hour_counts.values, y_label="Admissions"),
                "explanation": f"Most admissions occur around {peak_hour}:00 -- useful for aligning shift handover and triage capacity.",
            })

    # ==================== DEMOGRAPHICS ====================

    if age_col and age_col in df.columns and "Age_Group" in df.columns:
        elderly_pct = (df[age_col] >= 65).mean() * 100
        ag_counts = df["Age_Group"].value_counts().reindex(["<18","18-35","35-50","50-65","65+"]).fillna(0)
        findings.append({
            "id": "demographics_age", "severity": "warning" if elderly_pct > 30 else "info",
            "kpi": {"label": "Patients 65+", "value": f"{elderly_pct:.1f}%", "benchmark": "--"},
            "chart": _chart("bar", ag_counts.index, ag_counts.values, y_label="Patients"),
            "explanation": (
                f"{elderly_pct:.1f}% of the cohort is 65 or older. "
                + _compare_all(ag_counts, lambda v: f"{int(v):,} patients")
                + (" Consider dedicated geriatric assessment capacity." if elderly_pct > 30 else "")
            ),
        })

    if gender_col and gender_col in df.columns and cost_col and cost_col in df.columns:
        by_gender = df.groupby(gender_col)[cost_col].mean().sort_values(ascending=False)
        if len(by_gender) >= 2:
            gap_pct = (by_gender.iloc[0] - by_gender.iloc[-1]) / by_gender.iloc[-1] * 100
            findings.append({
                "id": "gender_cost_gap", "severity": "warning" if gap_pct > 15 else "info",
                "kpi": {"label": "Cost Gap by Gender", "value": f"{gap_pct:.1f}%", "benchmark": "--"},
                "chart": _chart("bar", by_gender.index, by_gender.values, y_label="Avg Cost (KES)"),
                "explanation": (
                    f"{by_gender.index[0]} patients average {gap_pct:.1f}% higher treatment cost than "
                    f"{by_gender.index[-1]} patients."
                ),
            })

    if gender_col and gender_col in df.columns and cond_col and cond_col in df.columns:
        gc = df.groupby([gender_col, cond_col]).size().reset_index(name="count")
        if not gc.empty:
            top_combo = gc.loc[gc["count"].idxmax()]
            findings.append({
                "id": "condition_gender_prevalence", "severity": "info",
                "kpi": {
                    "label": "Most Common Condition x Gender",
                    "value": f"{top_combo[cond_col]} ({top_combo[gender_col]})",
                    "benchmark": "--",
                },
                "chart": None,
                "explanation": f"{top_combo[cond_col]} in {top_combo[gender_col]} patients is the single most frequent condition/gender combination ({int(top_combo['count'])} cases).",
            })

    # ==================== OUTCOMES ====================

    if outcome_col and outcome_col in df.columns:
        odist = df[outcome_col].value_counts()
        top_outcome, top_pct = odist.index[0], odist.iloc[0] / len(df) * 100
        findings.append({
            "id": "outcomes", "severity": "info",
            "kpi": {"label": "Most Common Outcome", "value": top_outcome, "benchmark": "--"},
            "chart": _chart("bar", odist.index[:6], odist.values[:6], y_label="Patients"),
            "explanation": (
                f"{top_pct:.1f}% of patients had outcome '{top_outcome}'. "
                + _compare_all(odist.head(6), lambda v: f"{int(v):,} patients")
            ),
        })

    if outcome_col and outcome_col in df.columns and cond_col and cond_col in df.columns:
        recovered_mask = df[outcome_col].astype(str).str.lower() == "recovered"
        if recovered_mask.any():
            recovery_by_cond = (
                df.assign(_rec=recovered_mask.astype(int))
                .groupby(cond_col)["_rec"].mean() * 100
            ).sort_values()
            if len(recovery_by_cond) >= 2:
                worst_cond = recovery_by_cond.index[0]
                findings.append({
                    "id": "recovery_rate_by_condition", "severity": "warning" if recovery_by_cond.iloc[0] < 70 else "info",
                    "kpi": {"label": "Lowest Recovery Rate", "value": f"{worst_cond} ({recovery_by_cond.iloc[0]:.1f}%)", "benchmark": "--"},
                    "chart": _chart("bar", recovery_by_cond.index[:6], recovery_by_cond.values[:6], y_label="Recovery Rate (%)"),
                    "explanation": (
                        f"{worst_cond} has the lowest recovery rate among recorded conditions. "
                        + _compare_all(recovery_by_cond.head(6), lambda v: f"{v:.1f}%")
                    ),
                })

    if "Readmission_Bin" in df.columns and "Age_Group" in df.columns:
        ra = (df.groupby("Age_Group", observed=True)["Readmission_Bin"].mean() * 100).round(1)
        if len(ra) >= 2 and ra.max() > 0:
            worst_age = ra.idxmax()
            findings.append({
                "id": "readmission_by_age_group", "severity": "warning" if ra.max() > 20 else "info",
                "kpi": {"label": "Highest-Readmission Age Group", "value": f"{worst_age} ({ra.max():.1f}%)", "benchmark": "<=15%"},
                "chart": _chart("bar", ra.index, ra.values, y_label="Readmission Rate (%)"),
                "explanation": (
                    f"The {worst_age} age group has the highest readmission rate. "
                    + _compare_all(ra.sort_values(ascending=False), lambda v: f"{v:.1f}%")
                ),
            })

    # ==================== FINANCIAL ====================

    if ins_col and ins_col in df.columns and cost_col and cost_col in df.columns:
        by_ins = df.groupby(ins_col)[cost_col].mean().sort_values(ascending=False).head(6)
        if len(by_ins) >= 2:
            findings.append({
                "id": "cost_by_insurance", "severity": "info",
                "kpi": {"label": "Highest-Cost Payer", "value": by_ins.index[0], "benchmark": "--"},
                "chart": _chart("bar", by_ins.index, by_ins.values, y_label="Avg Cost (KES)"),
                "explanation": (
                    f"{by_ins.index[0]} patients have the highest average cost. "
                    + _compare_all(by_ins, lambda v: f"KES {v:,.0f}")
                ),
            })

    if proc_col and proc_col in df.columns and cost_col and cost_col in df.columns:
        by_proc = df.groupby(proc_col)[cost_col].mean().sort_values(ascending=False).head(6)
        if len(by_proc) >= 2:
            findings.append({
                "id": "cost_by_procedure", "severity": "info",
                "kpi": {"label": "Highest-Cost Procedure", "value": by_proc.index[0], "benchmark": "--"},
                "chart": _chart("bar", by_proc.index, by_proc.values, y_label="Avg Cost (KES)"),
                "explanation": (
                    f"{by_proc.index[0]} is the highest-cost procedure per patient. "
                    + _compare_all(by_proc, lambda v: f"KES {v:,.0f}")
                ),
            })

    if proc_col and proc_col in df.columns:
        proc_counts = df[proc_col].value_counts().head(6)
        if len(proc_counts) >= 2:
            findings.append({
                "id": "procedure_volume", "severity": "info",
                "kpi": {"label": "Most Common Procedure", "value": proc_counts.index[0], "benchmark": "--"},
                "chart": _chart("bar", proc_counts.index, proc_counts.values, y_label="Patients"),
                "explanation": (
                    f"{proc_counts.index[0]} is the most frequently performed procedure. "
                    + _compare_all(proc_counts, lambda v: f"{int(v):,} patients")
                ),
            })

    if "High_Cost" in df.columns and cost_col and cost_col in df.columns:
        high_cost_pct = df["High_Cost"].mean() * 100
        high_cost_avg = df.loc[df["High_Cost"], cost_col].mean()
        findings.append({
            "id": "high_cost_patients", "severity": "warning" if high_cost_pct > 30 else "info",
            "kpi": {"label": "High-Cost Patients", "value": f"{high_cost_pct:.1f}%", "benchmark": "top quartile ~25%"},
            "chart": None,
            "explanation": f"{high_cost_pct:.1f}% of patients fall in the high-cost band, averaging KES {high_cost_avg:,.0f} each.",
        })

    if "Cost_Per_Day" in df.columns:
        avg_cpd = df["Cost_Per_Day"].mean()
        findings.append({
            "id": "cost_per_day", "severity": "info",
            "kpi": {"label": "Avg Cost per Inpatient Day", "value": f"KES {avg_cpd:,.0f}", "benchmark": "--"},
            "chart": None,
            "explanation": f"Average cost per inpatient day is KES {avg_cpd:,.0f} across the cohort.",
        })

    if ward_col and ward_col in df.columns:
        by_ward = df[ward_col].value_counts().head(6)
        if len(by_ward) >= 2:
            findings.append({
                "id": "ward_volume", "severity": "info",
                "kpi": {"label": "Busiest Ward", "value": by_ward.index[0], "benchmark": "--"},
                "chart": _chart("bar", by_ward.index, by_ward.values, y_label="Patients"),
                "explanation": (
                    f"{by_ward.index[0]} handles the highest patient volume. "
                    + _compare_all(by_ward, lambda v: f"{int(v):,} patients")
                ),
            })

    # ==================== DATA QUALITY ====================

    dq = quality["quality_score"]
    findings.append({
        "id": "data_quality",
        "severity": "critical" if dq < 70 else "warning" if dq < 85 else "good",
        "kpi": {"label": "Data Quality Score", "value": f"{dq:.0f}/100", "benchmark": ">=85"},
        "chart": None,
        "explanation": quality["recommendations"][0] if quality["recommendations"] else "Data quality is within acceptable range.",
    })

    # ==================== CLINICAL DOMAINS ====================

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
        glucose_col = cm.get("blood_glucose")
        if glucose_col and glucose_col in df.columns:
            avg_glucose = pd.to_numeric(df[glucose_col], errors="coerce").mean()
            findings.append({
                "id": "blood_glucose_avg", "severity": "warning" if avg_glucose > 130 else "good",
                "kpi": {"label": "Avg Blood Glucose", "value": f"{avg_glucose:.0f}", "benchmark": "<=130"},
                "chart": None,
                "explanation": f"Average fasting blood glucose across diabetic patients is {avg_glucose:.0f}, {'above' if avg_glucose > 130 else 'within'} the typical target range.",
            })

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
        vl_col = cm.get("viral_load")
        if vl_col and vl_col in df.columns:
            vl_numeric = pd.to_numeric(df[vl_col], errors="coerce")
            suppressed_pct = (vl_numeric < 1000).mean() * 100
            findings.append({
                "id": "viral_load_suppression", "severity": "warning" if suppressed_pct < 80 else "good",
                "kpi": {"label": "Viral Suppression Rate", "value": f"{suppressed_pct:.1f}%", "benchmark": ">=90%"},
                "chart": None,
                "explanation": f"{suppressed_pct:.1f}% of HIV+ patients have viral load under 1,000 copies/mL (virally suppressed).",
            })
        art_col = cm.get("art_status")
        if art_col and art_col in df.columns and cd4_col and cd4_col in df.columns:
            art_cd4 = pd.to_numeric(df[cd4_col], errors="coerce").groupby(df[art_col]).mean()
            if len(art_cd4) >= 2:
                findings.append({
                    "id": "art_status_cd4", "severity": "info",
                    "kpi": {"label": "CD4 by ART Status", "value": art_cd4.idxmax(), "benchmark": "--"},
                    "chart": _chart("bar", art_cd4.index, art_cd4.values, y_label="Avg CD4 Count"),
                    "explanation": f"Patients on ART status '{art_cd4.idxmax()}' have the highest average CD4 count, consistent with treatment effectiveness.",
                })

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

    if "Cancer / Oncology" in clinical_domains:
        stage_col = cm.get("cancer_stage")
        if stage_col and stage_col in df.columns:
            stage_dist = df[stage_col].value_counts().head(6)
            late_stage_pct = (df["Cancer_Stage_Num"] >= 3).mean() * 100 if "Cancer_Stage_Num" in df.columns else None
            findings.append({
                "id": "cancer_stage",
                "severity": "critical" if (late_stage_pct or 0) > 40 else "warning" if (late_stage_pct or 0) > 20 else "info",
                "kpi": {"label": "Late-Stage Diagnoses (III/IV)", "value": f"{late_stage_pct:.1f}%" if late_stage_pct is not None else "N/A", "benchmark": "<=20%"},
                "chart": _chart("bar", stage_dist.index, stage_dist.values, y_label="Patients"),
                "explanation": "Share of cancer patients diagnosed at Stage III or IV, when treatment is more intensive and prognosis is worse -- often a screening/early-detection signal.",
            })

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

    # ==================== PATIENT SEGMENTATION ====================

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
                "kpi": {"label": "Patient Segments Found", "value": str(clust["best_k"]), "benchmark": "--"},
                "chart": _chart("bar", [f"Cluster {i}" for i in cluster_sizes.index], cluster_sizes.values, y_label="Patients"),
                "explanation": (
                    f"KMeans clustering on {', '.join(cluster_features)} found {clust['best_k']} distinct patient "
                    f"segments (silhouette score {clust['sil']:.2f}, where closer to 1 means more distinct groups)."
                ),
            })

    # ==================== STATISTICAL INFERENCE ====================

    if SCIPY_OK:
        num_candidates = [c for c in [cost_col, los_col, sat_col, age_col] if c and c in df.columns]

        if "Readmission_Bin" in df.columns and len(num_candidates) >= 1:
            corrs = {}
            for c in num_candidates:
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().sum() > 10:
                    corrs[c] = df["Readmission_Bin"].corr(s)
            if corrs:
                top_feat = max(corrs, key=lambda k: abs(corrs[k]))
                top_val = corrs[top_feat]
                findings.append({
                    "id": "readmission_correlation", "severity": "info",
                    "kpi": {"label": "Strongest Readmission Correlate", "value": f"{top_feat} ({top_val:+.2f})", "benchmark": "--"},
                    "chart": _chart("bar", list(corrs.keys()), list(corrs.values()), y_label="Correlation with Readmission"),
                    "explanation": f"{top_feat} has the strongest linear relationship with readmission (r = {top_val:+.2f}). Values near +/-1 indicate a strong relationship; near 0, little to none.",
                })

        if "Readmission_Bin" in df.columns:
            best_ttest = None
            for c in num_candidates:
                s = pd.to_numeric(df[c], errors="coerce")
                g1 = s[df["Readmission_Bin"] == 1].dropna()
                g0 = s[df["Readmission_Bin"] == 0].dropna()
                if len(g1) > 5 and len(g0) > 5:
                    t_stat, p_val = scipy_stats.ttest_ind(g1, g0, equal_var=False)
                    if best_ttest is None or p_val < best_ttest[3]:
                        best_ttest = (c, g1.mean(), g0.mean(), p_val)
            if best_ttest:
                var, mean1, mean0, p_val = best_ttest
                significant = p_val < 0.05
                findings.append({
                    "id": "ttest_readmission", "severity": "warning" if significant else "info",
                    "kpi": {"label": f"{var}: Readmitted vs Not", "value": f"p = {p_val:.4f}", "benchmark": "p < 0.05"},
                    "chart": _chart("bar", ["Readmitted", "Not Readmitted"], [mean1, mean0], y_label=f"Avg {var}"),
                    "explanation": (
                        f"{'Statistically significant' if significant else 'No statistically significant'} difference in {var} "
                        f"between readmitted (avg {mean1:.1f}) and non-readmitted (avg {mean0:.1f}) patients (p = {p_val:.4f})."
                    ),
                })

        if cond_col and cond_col in df.columns:
            best_anova = None
            for c in num_candidates:
                groups = [pd.to_numeric(g[c], errors="coerce").dropna().values
                          for _, g in df.groupby(cond_col) if len(g[c].dropna()) > 2]
                if len(groups) >= 2:
                    try:
                        f_stat, p_val = scipy_stats.f_oneway(*groups)
                        if best_anova is None or p_val < best_anova[2]:
                            best_anova = (c, f_stat, p_val)
                    except Exception:
                        continue
            if best_anova:
                var, f_stat, p_val = best_anova
                significant = p_val < 0.05
                findings.append({
                    "id": "anova_by_condition", "severity": "info",
                    "kpi": {"label": f"{var} Variance by Condition", "value": f"p = {p_val:.4f}", "benchmark": "p < 0.05"},
                    "chart": None,
                    "explanation": (
                        f"{'Condition significantly affects' if significant else 'Condition does not significantly affect'} "
                        f"{var} (ANOVA F = {f_stat:.2f}, p = {p_val:.4f})."
                    ),
                })

    # ==================== ANALYTICS MATURITY & RECOMMENDATIONS ====================

    findings.append({
        "id": "analytics_maturity", "severity": "warning" if maturity["score"] < 70 else "good",
        "kpi": {"label": "Analytics Maturity", "value": f"{maturity['score']}/100 (Grade {maturity['grade']})", "benchmark": ">=80"},
        "chart": _chart(
            "bar",
            ["Data Quality", "Clinical", "Financial", "Satisfaction", "Readmission"],
            [maturity["dq"], maturity["clinical"], maturity["financial"], maturity["satisfaction"], maturity["readmission"]],
            y_label="Score /100",
        ),
        "explanation": f"Overall analytics maturity is {maturity['score']}/100 (Grade {maturity['grade']}), a composite of data quality, clinical outcomes, financial performance, satisfaction, and readmission control.",
    })

    maturity_dims = {
        "Data Quality": maturity["dq"], "Clinical": maturity["clinical"],
        "Financial": maturity["financial"], "Satisfaction": maturity["satisfaction"],
        "Readmission": maturity["readmission"],
    }
    weakest_dim = min(maturity_dims, key=maturity_dims.get)
    if maturity_dims[weakest_dim] < 80:
        findings.append({
            "id": "maturity_improvement_area", "severity": "warning",
            "kpi": {"label": "Weakest Maturity Dimension", "value": f"{weakest_dim} ({maturity_dims[weakest_dim]:.0f})", "benchmark": ">=80"},
            "chart": None,
            "explanation": f"{weakest_dim} is the lowest-scoring maturity dimension at {maturity_dims[weakest_dim]:.0f}/100 -- the highest-leverage area to improve overall maturity.",
        })

    if age_col and age_col in df.columns:
        elderly_pct = (df[age_col] >= 65).mean() * 100
        if elderly_pct > 30:
            findings.append({
                "id": "rec_geriatric_care", "severity": "warning",
                "kpi": {"label": "Geriatric Care Recommendation", "value": f"{elderly_pct:.0f}% aged 65+", "benchmark": "--"},
                "chart": None,
                "explanation": f"With {elderly_pct:.0f}% of the cohort aged 65+, establishing a dedicated geriatric assessment team and transitional care unit could reduce LOS and readmissions.",
            })

    if "Admission_DayOfWeek" in df.columns and "Admission_Hour" in df.columns:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = df["Admission_DayOfWeek"].value_counts().reindex(dow_order).fillna(0)
        hour_counts = df["Admission_Hour"].value_counts()
        hour_has_variance = len(hour_counts) > 1 and hour_counts.max() / hour_counts.sum() < 0.9
        if hour_has_variance:
            peak_hour = int(hour_counts.idxmax())
            value_str = f"{dow.idxmax()} @ {peak_hour}:00"
            explanation = f"Peak admissions cluster around {dow.idxmax()} at {peak_hour}:00 -- review staffing rotas and consider staggering discharges ahead of this window."
        else:
            value_str = dow.idxmax()
            explanation = f"Peak admissions cluster on {dow.idxmax()} -- review staffing rotas for that day. (Admission timestamps in this dataset don't carry reliable time-of-day detail, so hour-level alignment isn't shown.)"
        findings.append({
            "id": "rec_staffing_alignment", "severity": "info",
            "kpi": {"label": "Staffing Alignment", "value": value_str, "benchmark": "--"},
            "chart": None,
            "explanation": explanation,
        })

    shap_importance = core.compute_shap_feature_importance(df, cm)
    if shap_importance is not None and len(shap_importance):
        top = shap_importance.iloc[0]
        direction = "raises" if top["mean_signed_shap"] > 0 else "lowers"
        chart = _chart(
            "bar", shap_importance["label"].tolist(), [round(v, 4) for v in shap_importance["mean_abs_shap"]],
            y_label="Impact on Readmission Risk (mean |SHAP|)",
        )
        driver_series = shap_importance.set_index("label")["mean_abs_shap"]
        explanation = (
            f"The strongest single driver of readmission risk this model finds is "
            f"{top['label']}, which on average {direction} the predicted risk. "
            + _compare_all(driver_series, lambda v: f"{v:.3f} impact")
            + " This is an exploratory model trained fresh on this dataset for narrative "
              "purposes -- not a validated or clinically-approved risk score."
        )
        findings.append({
            "id": "shap_readmission_drivers", "severity": "info",
            "kpi": {"label": "Top Readmission Driver", "value": top["label"], "benchmark": "--"},
            "chart": chart,
            "explanation": explanation,
        })

    anomaly_info = core.compute_anomalies(df, cm)
    if anomaly_info:
        labels = [f.replace("_", " ").title() for f in anomaly_info["feature_cols"]]
        deltas = {
            label: round(anomaly_info["means_anomaly"][f] - anomaly_info["means_normal"][f], 2)
            for label, f in zip(labels, anomaly_info["feature_cols"])
        }
        chart = _chart("bar", list(deltas.keys()), list(deltas.values()),
                        y_label="Anomalous vs Typical Patient (difference)")
        severity = "warning" if anomaly_info["pct"] > 8 else "info"
        explanation = (
            f"{anomaly_info['n_anomalies']} patients ({anomaly_info['pct']}% of the cohort) show an unusual "
            f"combination of {', '.join(labels).lower()} relative to the rest of the population -- not "
            f"necessarily extreme on any single metric alone, but statistically atypical together. "
            + _compare_all(pd.Series(deltas), lambda v: f"{v:+.2f} vs typical patient")
            + " These records may be worth a manual chart review -- data entry errors and genuinely "
              "unusual cases both tend to surface here."
        )
        findings.append({
            "id": "cohort_anomalies", "severity": severity,
            "kpi": {"label": "Anomalous Patients", "value": f"{anomaly_info['pct']}%", "benchmark": "--"},
            "chart": chart,
            "explanation": explanation,
        })

    CATEGORY_MAP = {
        "readmission": "Overview", "cost": "Overview", "satisfaction": "Overview",
        "length_of_stay": "Overview", "long_stay_share": "Overview",
        "demographics_age": "Demographics & Timing", "gender_cost_gap": "Demographics & Timing",
        "condition_gender_prevalence": "Demographics & Timing", "admission_timing": "Demographics & Timing",
        "admission_hour_peak": "Demographics & Timing",
        "outcomes": "Clinical Outcomes", "recovery_rate_by_condition": "Clinical Outcomes",
        "readmission_by_age_group": "Clinical Outcomes",
        "cost_by_insurance": "Financial", "cost_by_procedure": "Financial", "procedure_volume": "Financial",
        "high_cost_patients": "Financial", "cost_per_day": "Financial", "ward_volume": "Financial",
        "data_quality": "Data Quality",
        "diabetes_control": "Clinical Domains", "blood_glucose_avg": "Clinical Domains",
        "hiv_severity": "Clinical Domains", "viral_load_suppression": "Clinical Domains",
        "art_status_cd4": "Clinical Domains", "hypertension_severity": "Clinical Domains",
        "tb_drug_resistance": "Clinical Domains", "cancer_stage": "Clinical Domains",
        "low_oxygen_saturation": "Clinical Domains",
        "patient_segmentation": "Statistics", "readmission_correlation": "Statistics",
        "ttest_readmission": "Statistics", "anova_by_condition": "Statistics",
        "shap_readmission_drivers": "Predictive Insights", "cohort_anomalies": "Predictive Insights",
        "analytics_maturity": "Maturity & Recommendations", "maturity_improvement_area": "Maturity & Recommendations",
        "rec_geriatric_care": "Maturity & Recommendations", "rec_staffing_alignment": "Maturity & Recommendations",
    }
    for f in findings:
        f["category"] = CATEGORY_MAP.get(f["id"], "Overview")

    order = {"critical": 0, "warning": 1, "info": 2, "good": 3}
    findings.sort(key=lambda f: order.get(f["severity"], 4))

    return findings[:top_n]
