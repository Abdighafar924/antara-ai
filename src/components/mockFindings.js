// Matches the exact shape returned by POST /api/findings on the Render backend.
// Use this to build/test the UI before wiring the real fetch call.
export const mockFindingsResponse = {
  patients: 1204,
  findings: [
    {
      id: "readmission",
      severity: "critical",
      kpi: { label: "Readmission Rate", value: "22.4%", benchmark: "≤15%" },
      chart: {
        type: "bar",
        labels: ["Diabetes", "Hypertension", "Pneumonia", "TB"],
        values: [28.1, 19.3, 15.2, 11.4],
        y_label: "Readmission Rate (%)",
      },
      explanation:
        "Readmission rate is 22.4%, above the 15% benchmark. Highest-risk condition: Diabetes (28.1%).",
    },
    {
      id: "cost",
      severity: "warning",
      kpi: { label: "Avg Treatment Cost", value: "KES 10,450", benchmark: "KES 9,000" },
      chart: {
        type: "bar",
        labels: ["TB", "Diabetes", "Malaria", "HTN"],
        values: [13200, 11800, 8900, 8100],
        y_label: "Avg Cost (KES)",
      },
      explanation:
        "Average treatment cost is KES 10,450 vs a KES 9,000 benchmark. Highest-cost condition: TB (KES 13,200).",
    },
    {
      id: "satisfaction",
      severity: "warning",
      kpi: { label: "Avg Satisfaction", value: "3.7/5", benchmark: "≥4.0" },
      chart: null,
      explanation: "Patient satisfaction averages 3.7/5 against a 4.0 target.",
    },
    {
      id: "data_quality",
      severity: "good",
      kpi: { label: "Data Quality Score", value: "91/100", benchmark: "≥85" },
      chart: null,
      explanation: "Dataset is clean — no major issues detected.",
    },
  ],
};
