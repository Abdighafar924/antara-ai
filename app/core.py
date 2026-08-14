"""
MediCore AI — Core Analytics Engine
Framework-agnostic port of the original Streamlit app's data pipeline.
No st.* calls. Pure pandas/numpy/sklearn functions that FastAPI (or anything else) can call.
"""

import io
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SK_OK = True
except ImportError:
    SK_OK = False

try:
    from scipy import stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


BENCHMARKS_DEFAULT = {
    "Readmission Rate (%)":   {"target": 15.0, "dir": "lower"},
    "Avg Length of Stay (d)": {"target": 7.0,  "dir": "lower"},
    "Avg Satisfaction (/5)":  {"target": 4.0,  "dir": "higher"},
    "Avg Cost (kes)":         {"target": 9000, "dir": "lower"},
    "Mortality Rate (%)":     {"target": 5.0,  "dir": "lower"},
}

COLUMN_ALIASES = {
    "patient_id":       ["patient_id","patientid","pid","record_id","mrn","patient_no",
                          "id","pt_id","chart_no","file_no","encounter_id"],
    "age":              ["age","patient_age","pt_age","age_years","age_yrs","patient_years"],
    "gender":           ["gender","sex","patient_sex","pt_sex","biological_sex"],
    "condition":        ["condition","diagnosis","primary_condition","disease","primary_diagnosis",
                          "main_diagnosis","presenting_complaint","icd_description",
                          "admitting_diagnosis","chief_complaint"],
    "secondary_condition": ["secondary_diagnosis","secondary_condition","comorbidity",
                             "comorbidities","co_morbidity"],
    "procedure":        ["procedure","treatment","intervention","surgery","main_procedure",
                          "primary_procedure","operation"],
    "cost":             ["cost","total_cost","treatment_cost","charges","amount",
                          "bill_amount","total_charges","billing_amount","invoice_amount",
                          "total_bill","payment_amount"],
    "length_of_stay":   ["length_of_stay","los","stay_days","days_admitted","hospital_days",
                          "inpatient_days","bed_days","duration"],
    "readmission":      ["readmission","readmitted","re_admission","readmit",
                          "was_readmitted","30day_readmission","readmission_flag"],
    "outcome":          ["outcome","result","discharge_status","status","discharge_outcome",
                          "final_outcome","patient_outcome","discharge_disposition"],
    "satisfaction":     ["satisfaction","satisfaction_score","rating","nps","survey_score",
                          "patient_rating","patient_satisfaction","experience_score",
                          "feedback_score","overall_rating"],
    "admission_dt":     ["admission_datetime","admit_date","admission_date","admitted",
                          "date_admitted","admit_datetime","admission_time","date_of_admission"],
    "discharge_dt":     ["discharge_datetime","discharge_date","discharged","release_date",
                          "date_discharged","discharge_time","date_of_discharge"],
    "ward":             ["ward","department","unit","ward_name","inpatient_ward",
                          "bed_type","admission_type","ward_type"],
    "doctor":           ["doctor","physician","clinician","attending","provider",
                          "attending_physician","doctor_name","treating_physician"],
    "insurance":        ["insurance","payer","payment_type","insurance_type",
                          "coverage","payment_method","scheme"],
    "disease_category": ["disease_category","disease_type","category","condition_category",
                          "icd_chapter","diagnosis_category"],
    "comorbidity_index":["comorbidity_index","charlson_index","cci","comorbidity_score",
                          "burden_score"],
    "risk_group":       ["risk_group","risk_level","risk_category","risk_tier"],
    "cd4_count":        ["cd4_count","cd4","cd4count","cd4_cells"],
    "viral_load":       ["viral_load","vl","hiv_viral_load","viral_load_copies"],
    "art_status":       ["art_status","art","antiretroviral","on_art","hiv_treatment"],
    "who_stage":        ["who_stage","hiv_stage","who_clinical_stage"],
    "hba1c":            ["hba1c","hba1c_level","glycated_haemoglobin","a1c","hemoglobin_a1c"],
    "blood_glucose":    ["blood_glucose","fasting_glucose","glucose","blood_sugar",
                          "fasting_blood_sugar","glucose_fasting"],
    "insulin_use":      ["insulin_use","on_insulin","insulin","insulin_dependent"],
    "diabetes_type":    ["diabetes_type","dm_type","type_of_diabetes"],
    "systolic_bp":      ["systolic_bp","systolic","sbp","systolic_blood_pressure"],
    "diastolic_bp":     ["diastolic_bp","diastolic","dbp","diastolic_blood_pressure"],
    "tb_type":          ["tb_type","tuberculosis_type","tb_classification"],
    "sputum_result":    ["sputum_result","sputum","afb_result","tb_sputum"],
    "drug_resistance":  ["drug_resistance","drug_resistant_tb","mdr_tb","xdr_tb"],
    "cancer_type":      ["cancer_type","tumour_type","tumor_type","cancer_diagnosis"],
    "cancer_stage":     ["cancer_stage","stage","tumour_stage","tumor_stage","disease_stage"],
    "temperature":      ["temperature","temp","body_temperature","temp_celsius"],
    "oxygen_saturation":["oxygen_saturation","spo2","o2_sat","pulse_oximetry"],
    "heart_rate":       ["heart_rate","pulse","hr","pulse_rate"],
    "bmi":              ["bmi","body_mass_index","weight_status"],
    "creatinine":       ["creatinine","serum_creatinine","kidney_function"],
    "haemoglobin":      ["haemoglobin","hemoglobin","hb","hgb"],
}

CLINICAL_DOMAINS = {
    "HIV / ART":         ["cd4_count","viral_load","art_status","who_stage"],
    "Diabetes":          ["hba1c","blood_glucose","insulin_use","diabetes_type"],
    "Hypertension":      ["systolic_bp","diastolic_bp"],
    "Tuberculosis":      ["tb_type","sputum_result","drug_resistance"],
    "Cancer / Oncology": ["cancer_type","cancer_stage"],
    "Vitals & Labs":     ["temperature","oxygen_saturation","heart_rate","bmi",
                           "haemoglobin","creatinine"],
}

FIELD_FILL_RULES = {
    "gender": ("constant", "Unknown"), "insurance": ("constant", "Unknown"),
    "ward": ("constant", "Unknown"), "doctor": ("constant", "Unknown"),
    "condition": ("constant", "Unclassified"), "disease_category": ("constant", "Unclassified"),
    "risk_group": ("constant", "Unclassified"), "secondary_condition": ("constant", "None"),
    "comorbidity_index": ("zero", None), "procedure": ("constant", "No Procedure"),
    "cd4_count": ("zero", None), "viral_load": ("zero", None),
    "art_status": ("constant", "N/A"), "who_stage": ("constant", "N/A"),
    "hba1c": ("demographic_mean", None), "blood_glucose": ("demographic_mean", None),
    "insulin_use": ("constant", "None"), "diabetes_type": ("constant", "None"),
    "tb_type": ("constant", "Negative"), "sputum_result": ("constant", "Negative"),
    "drug_resistance": ("constant", "Negative"), "cancer_type": ("constant", "No Malignancy"),
    "cancer_stage": ("constant", "No Malignancy"), "systolic_bp": ("demographic_mean", None),
    "diastolic_bp": ("demographic_mean", None), "temperature": ("demographic_mean", None),
    "oxygen_saturation": ("demographic_mean", None), "heart_rate": ("demographic_mean", None),
    "bmi": ("demographic_mean", None), "haemoglobin": ("demographic_mean", None),
}


def resolve_columns(df):
    def normalise(s):
        return str(s).lower().replace(" ", "_").replace(".", "_").replace("-", "_")
    cols_norm = {normalise(c): c for c in df.columns}
    mapping = {}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in cols_norm:
                mapping[field] = cols_norm[alias]
                break
    return mapping


def detect_clinical_domains(cm):
    found = {}
    for domain, fields in CLINICAL_DOMAINS.items():
        present = [f for f in fields if f in cm]
        if present:
            found[domain] = present
    return found


def infer_column_type(df, col):
    s = df[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "binary" if s.nunique() <= 2 else "numeric"
    coerced = pd.to_numeric(s, errors="coerce")
    if coerced.notna().sum() / max(len(s), 1) > 0.7:
        return "numeric"
    try:
        pd.to_datetime(s.dropna().iloc[:20], errors="raise")
        return "datetime"
    except Exception:
        pass
    return "categorical" if s.nunique() <= max(20, len(s) * 0.05) else "text"


def safe_mean(df, col):
    if col and col in df.columns and df[col].notna().any():
        return df[col].mean()
    return None


def _apply_field_fill_rules(df, cm, gender_col=None):
    log = []
    for field, (strategy, const_val) in FIELD_FILL_RULES.items():
        col = cm.get(field)
        if not col or col not in df.columns:
            continue
        n_null = df[col].isna().sum()
        if n_null == 0:
            continue
        if strategy == "constant":
            df[col] = df[col].fillna(const_val)
            log.append(f"Filled {n_null} missing value(s) in '{col}' with '{const_val}'.")
        elif strategy == "zero":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            log.append(f"Filled {n_null} missing value(s) in '{col}' with 0.")
        elif strategy == "demographic_mean":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n_null = df[col].isna().sum()
            if n_null == 0:
                continue
            if gender_col and gender_col in df.columns:
                try:
                    grp_mean = df.groupby(gender_col)[col].transform("mean")
                    df[col] = df[col].fillna(grp_mean)
                except Exception:
                    pass
            overall_mean = df[col].mean()
            df[col] = df[col].fillna(overall_mean if pd.notna(overall_mean) else 0)
            log.append(f"Filled {n_null} missing value(s) in '{col}' with demographic mean.")
    return df, log


def _validate_raw(raw, cm):
    errors = []
    n = len(raw)
    for field in ("age", "cost", "length_of_stay"):
        col = cm.get(field)
        if col is None:
            errors.append(f"Field '{field}' not found — some analytics will be limited.")
        elif col in raw.columns:
            numeric = pd.to_numeric(raw[col], errors="coerce")
            n_neg, n_null = (numeric < 0).sum(), numeric.isna().sum()
            if n_neg:
                errors.append(f"RAW: '{col}' has {n_neg} negative value(s).")
            if n_null > n * 0.5:
                errors.append(f"RAW: '{col}' is >50% missing.")
    adm, dis = cm.get("admission_dt"), cm.get("discharge_dt")
    if adm and dis and adm in raw.columns and dis in raw.columns:
        bad = (pd.to_datetime(raw[dis], errors="coerce") < pd.to_datetime(raw[adm], errors="coerce")).sum()
        if bad:
            errors.append(f"RAW: {bad} record(s) have discharge before admission date.")
    n_dups = raw.duplicated().sum()
    if n_dups:
        errors.append(f"RAW: {n_dups} exact duplicate row(s) detected.")
    return errors


def _clean(df, cm):
    log = []
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if n_before - len(df):
        log.append(f"Removed {n_before - len(df)} duplicate row(s).")

    for field in ("age", "cost", "length_of_stay", "satisfaction"):
        col = cm.get(field)
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for field in ("admission_dt", "discharge_dt"):
        col = cm.get(field)
        if col and col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for field in ("age", "cost", "length_of_stay"):
        col = cm.get(field)
        if col and col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg:
                df[col] = df[col].abs()
                log.append(f"Auto-fixed {n_neg} negative value(s) in '{col}'.")

    adm, dis = cm.get("admission_dt"), cm.get("discharge_dt")
    if adm and dis and adm in df.columns and dis in df.columns:
        bad_mask = df[dis] < df[adm]
        if bad_mask.sum():
            df.loc[bad_mask, [adm, dis]] = df.loc[bad_mask, [dis, adm]].values
            log.append(f"Swapped {bad_mask.sum()} reversed admission/discharge date pair(s).")

    gender_col = cm.get("gender")
    if gender_col and gender_col in df.columns:
        _male = {"male","m","1","man","men","boy","gents","masculine"}
        _female = {"female","f","0","2","woman","women","girl","ladies","feminine"}
        def _norm(v):
            if pd.isna(v): return np.nan
            s = str(v).strip().lower()
            if s in _male: return "M"
            if s in _female: return "F"
            return str(v).strip()
        df[gender_col] = df[gender_col].map(_norm)
        log.append("Normalised gender values to M/F.")

    readm_col = cm.get("readmission")
    if readm_col and readm_col in df.columns:
        _yes, _no = {"yes","y","1","true","readmitted"}, {"no","n","0","false","not readmitted"}
        def _norm_readm(v):
            if pd.isna(v): return "No"
            s = str(v).strip().lower()
            if s in _yes: return "Yes"
            if s in _no: return "No"
            return str(v).strip()
        df[readm_col] = df[readm_col].map(_norm_readm)
        log.append("Normalised readmission values to Yes/No.")

    for field in ("cost", "length_of_stay"):
        col = cm.get(field)
        if col and col in df.columns and df[col].notna().any():
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 3*iqr, q3 + 3*iqr
            n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
            if n_clipped:
                df[col] = df[col].clip(lower=lower, upper=upper)
                log.append(f"Clipped {n_clipped} extreme outlier(s) in '{col}'.")

    df, rule_log = _apply_field_fill_rules(df, cm, gender_col=gender_col)
    log.extend(rule_log)
    rule_cols = {cm.get(f) for f in FIELD_FILL_RULES if cm.get(f)}

    for col in df.select_dtypes(include=np.number).columns:
        if col in rule_cols: continue
        n_null = df[col].isna().sum()
        if n_null:
            df[col] = df[col].fillna(df[col].median())
            log.append(f"Filled {n_null} missing numeric value(s) in '{col}' with median.")

    for col in df.select_dtypes(include="object").columns:
        if col in rule_cols: continue
        n_null = df[col].isna().sum()
        if n_null:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
                log.append(f"Filled {n_null} missing categorical value(s) in '{col}' with mode.")

    if gender_col and gender_col in df.columns and df[gender_col].isna().any():
        df[gender_col] = df[gender_col].fillna("Unknown")

    for col in df.select_dtypes(include="datetime64").columns:
        if df[col].isna().any():
            df[col] = df[col].ffill().bfill()

    return df, log


def _engineer(df, cm):
    age_col, cost_col, los_col = cm.get("age"), cm.get("cost"), cm.get("length_of_stay")
    adm_col, readm_col = cm.get("admission_dt"), cm.get("readmission")

    if age_col and age_col in df.columns:
        df["Age_Group"] = pd.cut(df[age_col], bins=[0,18,35,50,65,200],
                                  labels=["<18","18–35","35–50","50–65","65+"])

    if cost_col and los_col and cost_col in df.columns and los_col in df.columns:
        df["Cost_Per_Day"] = (df[cost_col] / df[los_col].replace(0, np.nan)).round(0)
        q1, q3 = df["Cost_Per_Day"].quantile([0.25, 0.75])
        iqr = q3 - q1
        df["Cost_Per_Day"] = df["Cost_Per_Day"].clip(lower=q1-3*iqr, upper=q3+3*iqr)
        df["High_Cost"] = df[cost_col] > df[cost_col].quantile(0.75)

    if los_col and los_col in df.columns:
        df["Long_Stay"] = df[los_col] > df[los_col].quantile(0.75)

    if adm_col and adm_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[adm_col]):
        df["Admission_Hour"] = df[adm_col].dt.hour
        df["Admission_DayOfWeek"] = df[adm_col].dt.day_name()
        df["Admission_Month"] = df[adm_col].dt.to_period("M").astype(str)

    if readm_col and readm_col in df.columns:
        s = df[readm_col].astype(str).str.lower().str.strip()
        df["Readmission_Bin"] = s.isin(["yes","1","true","y","readmitted"]).astype(int)

    sys_col, dia_col = cm.get("systolic_bp"), cm.get("diastolic_bp")
    if sys_col and dia_col and sys_col in df.columns and dia_col in df.columns:
        conditions = [
            (df[sys_col] < 120) & (df[dia_col] < 80),
            (df[sys_col] < 130) & (df[dia_col] < 80),
            ((df[sys_col] >= 130) | (df[dia_col] >= 80)) & ((df[sys_col] < 140) | (df[dia_col] < 90)),
            (df[sys_col] >= 140) | (df[dia_col] >= 90),
        ]
        df["BP_Category"] = np.select(conditions, ["Normal","Elevated","Stage 1 HTN","Stage 2 HTN"], default="Unknown")

    hba1c_col = cm.get("hba1c")
    if hba1c_col and hba1c_col in df.columns:
        df["Glycaemic_Control"] = pd.cut(df[hba1c_col], bins=[0,5.7,6.5,8.0,100],
                                          labels=["Normal","Prediabetes","Controlled DM","Uncontrolled DM"])

    cd4_col = cm.get("cd4_count")
    if cd4_col and cd4_col in df.columns:
        df["CD4_Category"] = pd.cut(df[cd4_col], bins=[0,200,350,500,10000],
                                     labels=["Severe (<200)","Moderate (200–350)","Mild (350–500)","Healthy (>500)"])

    return df


def load_and_prepare(file_bytes: bytes):
    """
    Returns (raw_df, cleaned_df, col_map, validation_errors_raw, cleaning_log, attrs)
    attrs = dict with size_cat, clinical_domains, all_col_types
    """
    try:
        raw = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Cannot read CSV: {e}")

    raw = raw.reset_index(drop=True)
    cm = resolve_columns(raw)
    val_errors_raw = _validate_raw(raw, cm)
    df, cleaning_log = _clean(raw.copy(), cm)
    df = _engineer(df, cm)

    n = len(df)
    attrs = {
        "size_cat": ("small" if n < 100 else "medium" if n < 500
                      else "predictive" if n < 5000 else "enterprise"),
        "clinical_domains": detect_clinical_domains(cm),
        "all_col_types": {c: infer_column_type(df, c) for c in df.columns},
    }
    return raw, df, cm, val_errors_raw, cleaning_log, attrs


def compute_quality(df):
    n = len(df)
    missing = df.isnull().sum()
    missing_pct = (missing / n * 100).round(1)
    completeness = (1 - missing / n) * 100
    n_dups = int(df.duplicated().sum())

    outlier_counts = {}
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].nunique() <= 2: continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = int(((df[col] < q1-3*iqr) | (df[col] > q3+3*iqr)).sum())
        if n_out: outlier_counts[col] = n_out

    col_health = {}
    for col in df.columns:
        score = 100.0 - missing_pct.get(col, 0) * 0.5
        if col in outlier_counts:
            score -= min(20, outlier_counts[col] / n * 100 * 2)
        col_health[col] = max(0, round(score, 1))

    comp_score = completeness.mean()
    dup_penalty = min(10, n_dups / n * 100)
    out_penalty = min(10, sum(outlier_counts.values()) / max(n, 1) * 100)
    quality_score = round(comp_score*0.60 + (100-dup_penalty)*0.20 + (100-out_penalty)*0.20, 1)

    recs = []
    for col, pct in sorted(missing_pct[missing_pct > 10].items(), key=lambda x: -x[1]):
        recs.append(f"Column '{col}' has {pct:.1f}% missing.")
    if n_dups:
        recs.append(f"{n_dups} duplicate rows detected.")
    for col, cnt in sorted(outlier_counts.items(), key=lambda x: -x[1]):
        recs.append(f"Column '{col}' has {cnt} extreme outliers.")
    if not recs:
        recs.append("Dataset is clean — no issues detected.")

    return dict(
        quality_score=quality_score, completeness_score=round(comp_score, 1),
        missing=missing[missing > 0].to_dict(), missing_pct=missing_pct[missing_pct > 0].to_dict(),
        completeness_per_col=completeness.to_dict(), n_dups=n_dups, outliers=outlier_counts,
        col_health=col_health, n_rows=n, n_cols=len(df.columns), recommendations=recs,
    )


def compute_hospital_score(df, quality_score, cm):
    sat_col, los_col, cost_col = cm.get("satisfaction"), cm.get("length_of_stay"), cm.get("cost")
    sat_score = min(100, (safe_mean(df, sat_col) or 3.5) / 5 * 100)
    rr = df["Readmission_Bin"].mean() * 100 if "Readmission_Bin" in df.columns else 20
    readm_score = max(0, min(100, (1 - rr/30) * 100))
    avg_los = safe_mean(df, los_col) or 10
    los_score = max(0, min(100, 100 - (avg_los-7)/7*20))
    avg_cost = safe_mean(df, cost_col) or 9000
    cost_score = max(0, min(100, 100 - (avg_cost-9000)/9000*30))
    overall = round(sat_score*0.30 + readm_score*0.25 + los_score*0.20 + cost_score*0.15 + quality_score*0.10, 1)
    grade = ("A+" if overall>=95 else "A" if overall>=90 else "A-" if overall>=87 else
             "B+" if overall>=83 else "B" if overall>=80 else "B-" if overall>=77 else
             "C+" if overall>=73 else "C" if overall>=70 else "C-" if overall>=67 else
             "D" if overall>=60 else "F")
    risk = "Low" if overall>=80 else "Moderate" if overall>=65 else "High" if overall>=50 else "Critical"
    return dict(score=overall, grade=grade, risk=risk, sat_score=round(sat_score,1),
                readm_score=round(readm_score,1), los_score=round(los_score,1),
                cost_score=round(cost_score,1), quality_score=round(quality_score,1), rr=round(rr,1))


def compute_maturity(quality, hs):
    dq_score = min(100, quality["quality_score"])
    clin_score = hs["readm_score"]*0.5 + hs["sat_score"]*0.3 + hs["los_score"]*0.2
    overall = round(dq_score*0.20 + clin_score*0.25 + hs["cost_score"]*0.20 +
                     hs["sat_score"]*0.20 + hs["readm_score"]*0.15, 1)
    grade = "A" if overall>=90 else "B" if overall>=80 else "C" if overall>=70 else "D" if overall>=60 else "F"
    return dict(score=overall, grade=grade, dq=round(dq_score,1), clinical=round(clin_score,1),
                financial=round(hs["cost_score"],1), satisfaction=round(hs["sat_score"],1),
                readmission=round(hs["readm_score"],1))


def run_clustering(df, feature_cols, max_k=8):
    if not SK_OK:
        return None
    df_m = df[feature_cols].dropna().copy()
    if len(df_m) < 10:
        return None
    for col in df_m.columns:
        if not pd.api.types.is_numeric_dtype(df_m[col]):
            df_m[col] = LabelEncoder().fit_transform(df_m[col].astype(str))
    X = StandardScaler().fit_transform(df_m.values)
    max_k = min(max_k, len(df_m) - 1)
    inertias, silhouettes = [], []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    best_k = silhouettes.index(max(silhouettes)) + 2
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km_final.fit_predict(X)
    df_out = df.loc[df_m.index].copy()
    df_out["Cluster"] = labels
    return dict(df=df_out, best_k=best_k, sil=round(silhouette_score(X, labels), 3),
                inertias=inertias, silhouettes=silhouettes, k_range=list(range(2, max_k+1)))
