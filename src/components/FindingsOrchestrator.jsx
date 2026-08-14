import { useState, useEffect, useRef } from "react";
import "./FindingsOrchestrator.css";

/**
 * FindingsOrchestrator
 * Consumes the exact JSON shape returned by MediCore's POST /api/findings:
 *   { findings: [{ id, severity, kpi, chart, explanation }], patients }
 *
 * Flow per finding: KPI number counts up -> chart bars draw in one by one ->
 * explanation fades in -> "Next finding" becomes clickable. The user is the
 * pacing mechanism, not a timer or audio track.
 */

const SEVERITY_COLOR = {
  critical: "#ef4444",
  warning: "#f59e0b",
  good: "#0d9488",
  info: "#1a3a6b",
};

const SEVERITY_LABEL = {
  critical: "Needs attention",
  warning: "Watch",
  good: "On target",
  info: "For context",
};

function useCountUp(target, active, duration = 700) {
  const [value, setValue] = useState(0);
  const rafRef = useRef(null);

  useEffect(() => {
    if (!active) {
      setValue(0);
      return;
    }
    const start = performance.now();
    const from = 0;
    const numericTarget = typeof target === "number" ? target : parseFloat(target) || 0;

    function tick(now) {
      const elapsed = now - start;
      const progress = Math.min(1, elapsed / duration);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      setValue(from + (numericTarget - from) * eased);
      if (progress < 1) rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [target, active, duration]);

  return value;
}

function Chart({ chart, phase }) {
  if (!chart) return null;
  const { labels, values, y_label } = chart;
  const maxV = Math.max(...values, 1);

  return (
    <div className="mc-chart">
      {y_label && <div className="mc-chart-label">{y_label}</div>}
      <div className="mc-chart-bars">
        {values.map((v, i) => {
          const heightPct = (v / maxV) * 100;
          const drawn = phase === "chart" || phase === "explain";
          return (
            <div className="mc-bar-col" key={labels[i] + i}>
              <div className="mc-bar-value">{drawn ? v.toLocaleString() : ""}</div>
              <div className="mc-bar-track">
                <div
                  className="mc-bar-fill"
                  style={{
                    height: drawn ? `${heightPct}%` : "0%",
                    transitionDelay: `${i * 90}ms`,
                  }}
                />
              </div>
              <div className="mc-bar-label">{labels[i]}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function FindingCard({ finding, phase }) {
  const color = SEVERITY_COLOR[finding.severity] || SEVERITY_COLOR.info;
  const kpiActive = phase === "kpi" || phase === "chart" || phase === "explain";
  const rawValue = finding.kpi.value;
  const numericPart = parseFloat(String(rawValue).replace(/[^0-9.]/g, ""));
  const suffix = String(rawValue).replace(/^[^0-9]*[0-9.,]+/, "");
  const prefix = String(rawValue).match(/^[^0-9]*/)?.[0] || "";
  const animated = useCountUp(numericPart, kpiActive);
  const displayValue = isNaN(numericPart)
    ? rawValue
    : `${prefix}${Number.isInteger(numericPart) ? Math.round(animated) : animated.toFixed(2)}${suffix}`;

  return (
    <div className="mc-card">
      <div className="mc-card-top" style={{ borderLeftColor: color }}>
        <span className="mc-severity-badge" style={{ background: color }}>
          {SEVERITY_LABEL[finding.severity] || finding.severity}
        </span>
        <div className="mc-kpi-label">{finding.kpi.label}</div>
        <div className="mc-kpi-value" style={{ color }}>
          {displayValue}
        </div>
        {finding.kpi.benchmark && finding.kpi.benchmark !== "—" && (
          <div className="mc-kpi-benchmark">benchmark {finding.kpi.benchmark}</div>
        )}
      </div>

      <Chart chart={finding.chart} phase={phase} />

      <div className={`mc-explanation ${phase === "explain" ? "mc-explanation-visible" : ""}`}>
        {finding.explanation}
      </div>
    </div>
  );
}

export default function FindingsOrchestrator({ data }) {
  const findings = data?.findings || [];
  const [step, setStep] = useState(0);
  const [phase, setPhase] = useState("kpi"); // kpi -> chart -> explain

  const current = findings[step];

  useEffect(() => {
    setPhase("kpi");
    if (!current) return;
    const t1 = setTimeout(() => setPhase("chart"), 500);
    const chartDuration = current.chart ? current.chart.values.length * 90 + 500 : 300;
    const t2 = setTimeout(() => setPhase("explain"), 500 + chartDuration);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [step, current]);

  if (!current) {
    return <div className="mc-empty">No findings to show.</div>;
  }

  const isLast = step === findings.length - 1;
  const canAdvance = phase === "explain";

  return (
    <div className="mc-orchestrator">
      <div className="mc-header">
        <div className="mc-header-title">MediCore AI — findings walkthrough</div>
        <div className="mc-header-sub">{data.patients?.toLocaleString()} patients analyzed</div>
      </div>

      <div className="mc-progress">
        {findings.map((f, i) => (
          <button
            key={f.id}
            className={`mc-dot ${i === step ? "mc-dot-active" : ""} ${i < step ? "mc-dot-done" : ""}`}
            style={i <= step ? { background: SEVERITY_COLOR[f.severity] } : {}}
            onClick={() => i < step && setStep(i)}
            disabled={i > step}
            aria-label={`Finding ${i + 1} of ${findings.length}`}
          />
        ))}
      </div>

      <FindingCard finding={current} phase={phase} />

      <div className="mc-controls">
        <span className="mc-step-count">
          Finding {step + 1} of {findings.length}
        </span>
        <button
          className="mc-next-btn"
          disabled={!canAdvance}
          onClick={() => {
            if (isLast) return;
            setStep((s) => s + 1);
          }}
        >
          {isLast ? (canAdvance ? "Walkthrough complete" : "…") : canAdvance ? "Next finding →" : "…"}
        </button>
      </div>
    </div>
  );
}
