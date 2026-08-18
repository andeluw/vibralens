"use client";

import { ChangeEvent, DragEvent, FormEvent, useCallback, useEffect, useRef, useState } from "react";

type Health = {
  status: "ready" | "unavailable";
  model_version?: string;
};

type ModelInfo = {
  model_version: string;
  supported_condition_ids: number[];
  feature_set: string;
  feature_names: string[];
  limitations: string[];
};

type RulInterval = {
  pessimistic: number;
  median: number;
  optimistic: number;
};

type Prediction = {
  model_version: string;
  rul_minutes: RulInterval | null;
  planned_break_minutes: number;
  break_risk: "low" | "uncertain" | "high" | "unknown";
  advisory: string;
  support_status: "supported" | "unsupported";
  limitations: string[];
};

const API_BASE = (process.env.NEXT_PUBLIC_VIBRALENS_API_URL || "http://localhost:8000").replace(/\/$/, "");

const advisoryCopy: Record<string, { title: string; detail: string }> = {
  proceed: {
    title: "Continue to planned stop",
    detail: "The estimated interval supports continuing to the entered maintenance window.",
  },
  inspect_first: {
    title: "Inspect before proceeding",
    detail: "The estimate overlaps your planned stop. Confirm condition before continued operation.",
  },
  stop: {
    title: "Prioritize intervention",
    detail: "The conservative estimate falls before the planned stop. Escalate for maintenance review.",
  },
};

function formatMinutes(value: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
}

function titleCase(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [snapshot, setSnapshot] = useState<File | null>(null);
  const [age, setAge] = useState("100");
  const [condition, setCondition] = useState("1");
  const [plannedBreak, setPlannedBreak] = useState("60");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const refreshStatus = useCallback(async () => {
    try {
      const [healthResponse, modelResponse] = await Promise.all([
        fetch(`${API_BASE}/health`),
        fetch(`${API_BASE}/model`),
      ]);
      const nextHealth = (await healthResponse.json()) as Health;
      setHealth(nextHealth);
      if (modelResponse.ok) {
        setModel((await modelResponse.json()) as ModelInfo);
      }
    } catch {
      setHealth({ status: "unavailable" });
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
  }, [refreshStatus]);

  function acceptFile(file?: File) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError("Choose a CSV vibration snapshot.");
      return;
    }
    setSnapshot(file);
    setPrediction(null);
    setError(null);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!snapshot) {
      setError("Add a vibration snapshot before running the estimate.");
      inputRef.current?.focus();
      return;
    }

    setSubmitting(true);
    setPrediction(null);
    setError(null);

    const body = new FormData();
    body.append("snapshot", snapshot);
    body.append("bearing_age_minutes", age);
    body.append("condition_id", condition);
    body.append("planned_break_minutes", plannedBreak);

    try {
      const response = await fetch(`${API_BASE}/predict`, { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) {
        const detail = typeof payload.detail === "string" ? payload.detail : "The snapshot could not be analyzed.";
        throw new Error(detail);
      }
      setPrediction(payload as Prediction);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The analysis service could not be reached.");
    } finally {
      setSubmitting(false);
    }
  }

  const advice = prediction ? advisoryCopy[prediction.advisory] || advisoryCopy.inspect_first : null;
  const interval = prediction?.rul_minutes;
  const medianPosition = interval && interval.optimistic > 0
    ? Math.min(100, Math.max(0, (interval.median / interval.optimistic) * 100))
    : 0;

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="VibraLens home">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /><i /></span>
          <span>VibraLens</span>
        </a>
        <div className="product-label"><span>BEARING INTELLIGENCE</span><b>XJTU-SY / RUL</b></div>
        <button className={`status-pill ${health?.status === "ready" ? "is-ready" : ""}`} onClick={() => void refreshStatus()}>
          <span aria-hidden="true" />
          {health === null ? "Connecting" : health.status === "ready" ? "System online" : "Service offline"}
        </button>
      </header>

      <section className="hero" id="top">
        <div>
          <p className="eyebrow"><span>01</span> Predictive maintenance workspace</p>
          <h1>Know what the<br />bearing is telling you.</h1>
        </div>
        <p className="hero-copy">
          Turn a two-channel vibration snapshot into a transparent remaining-life interval and a clear maintenance recommendation.
        </p>
      </section>

      <section className="workspace" aria-label="Bearing analysis workspace">
        <form className="intake-panel" onSubmit={handleSubmit}>
          <div className="panel-heading">
            <div><span className="step-number">01</span><h2>Snapshot input</h2></div>
            <span className="required-note">ALL FIELDS REQUIRED</span>
          </div>

          <div
            className={`drop-zone ${dragging ? "is-dragging" : ""} ${snapshot ? "has-file" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
          >
            <input ref={inputRef} type="file" accept=".csv,text/csv" onChange={handleFileChange} aria-label="Vibration snapshot CSV" />
            <span className="upload-icon" aria-hidden="true"><i /></span>
            {snapshot ? (
              <div className="file-summary">
                <strong>{snapshot.name}</strong>
                <span>{(snapshot.size / 1024 / 1024).toFixed(2)} MB · Ready to analyze</span>
              </div>
            ) : (
              <div>
                <strong>Drop vibration snapshot</strong>
                <span>or click to browse · CSV, 32,768 rows</span>
              </div>
            )}
            <em>{snapshot ? "REPLACE" : "BROWSE"}</em>
          </div>

          <div className="field-grid">
            <label>
              <span>Bearing age <small>MINUTES</small></span>
              <input type="number" min="0" step="1" value={age} onChange={(event) => setAge(event.target.value)} required />
            </label>
            <label>
              <span>Operating condition</span>
              <select value={condition} onChange={(event) => setCondition(event.target.value)}>
                {(model?.supported_condition_ids || [1, 2, 3]).map((id) => <option key={id} value={id}>Condition {id}</option>)}
              </select>
            </label>
            <label>
              <span>Next planned stop <small>MINUTES</small></span>
              <input type="number" min="0" step="1" value={plannedBreak} onChange={(event) => setPlannedBreak(event.target.value)} required />
            </label>
          </div>

          {error && <div className="form-error" role="alert"><span>!</span>{error}</div>}

          <button className="analyze-button" type="submit" disabled={submitting || health?.status !== "ready"}>
            <span>{submitting ? "Analyzing snapshot…" : "Run life estimate"}</span>
            <b aria-hidden="true">↗</b>
          </button>
          <p className="privacy-note"><span aria-hidden="true">⌁</span> Snapshot is processed for this estimate and not retained.</p>
        </form>

        <section className={`result-panel ${prediction ? "has-result" : ""}`} aria-live="polite">
          <div className="panel-heading result-heading">
            <div><span className="step-number">02</span><h2>Life estimate</h2></div>
            {prediction && <span className={`risk-badge risk-${prediction.break_risk}`}>{prediction.break_risk} risk</span>}
          </div>

          {!prediction && !submitting && (
            <div className="empty-result">
              <div className="radar" aria-hidden="true"><i /><i /><i /></div>
              <div>
                <p>Awaiting snapshot</p>
                <span>Your RUL interval and maintenance guidance will appear here.</span>
              </div>
            </div>
          )}

          {submitting && (
            <div className="loading-result">
              <span className="loader" aria-hidden="true" />
              <p>Reading vibration signature</p>
              <small>Extracting deterministic time and frequency features…</small>
            </div>
          )}

          {prediction && prediction.support_status === "unsupported" && (
            <div className="unsupported-result">
              <span>OUTSIDE MODEL SUPPORT</span>
              <h3>Inspection required</h3>
              <p>This operating condition is outside the evaluated model boundary, so VibraLens has abstained from estimating remaining life.</p>
            </div>
          )}

          {prediction && interval && (
            <div className="prediction-result">
              <div className="estimate-lead">
                <p>Estimated remaining life</p>
                <div><strong>{formatMinutes(interval.median)}</strong><span>minutes<br />median estimate</span></div>
              </div>

              <div className="interval-chart" aria-label={`Estimated remaining life from ${formatMinutes(interval.pessimistic)} to ${formatMinutes(interval.optimistic)} minutes, median ${formatMinutes(interval.median)} minutes`}>
                <div className="chart-labels"><span>Conservative</span><span>Optimistic</span></div>
                <div className="range-track">
                  <span className="range-fill" />
                  <i className="median-marker" style={{ left: `${medianPosition}%` }}><b>{formatMinutes(interval.median)}</b></i>
                </div>
                <div className="range-values"><strong>{formatMinutes(interval.pessimistic)} min</strong><strong>{formatMinutes(interval.optimistic)} min</strong></div>
              </div>

              <div className={`advisory-card risk-${prediction.break_risk}`}>
                <span className="advisory-icon" aria-hidden="true">{prediction.break_risk === "low" ? "✓" : "!"}</span>
                <div><small>MAINTENANCE ADVISORY</small><h3>{advice?.title}</h3><p>{advice?.detail}</p></div>
              </div>

              <div className="result-meta">
                <div><span>Planned stop</span><b>{formatMinutes(prediction.planned_break_minutes)} min</b></div>
                <div><span>Support</span><b>{titleCase(prediction.support_status)}</b></div>
                <div><span>Model</span><b>{prediction.model_version}</b></div>
              </div>
            </div>
          )}
        </section>
      </section>

      <section className="model-strip">
        <div><span className="mini-pulse" aria-hidden="true" /><p><b>Evaluated model</b><small>{model?.model_version || "Loading model profile"}</small></p></div>
        <div><p><b>{model?.feature_names.length ?? 28} inputs</b><small>Time + frequency domain</small></p></div>
        <div><p><b>Transparent interval</b><small>Conservative to optimistic</small></p></div>
        <p className="disclaimer">Decision support, not a safety certification. Always combine model output with qualified inspection.</p>
      </section>

      <footer>
        <span>VIBRALENS / 2026</span>
        <p>Leakage-safe bearing prognostics.</p>
        <a href="https://github.com/andeluw/vibralens" target="_blank" rel="noreferrer">Documentation ↗</a>
      </footer>
    </main>
  );
}
