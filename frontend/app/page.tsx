"use client";

import { useRef, type ChangeEvent, type DragEvent, type FormEvent } from "react";
import { DecisionTimeline } from "../components/decision-timeline";
import { advisoryCopy, formatMinutes, titleCase, type Prediction } from "../lib/vibralens";
import { useVibraLensAssessment } from "../lib/use-vibralens-assessment";

const panelPadding = "min-w-0 p-[clamp(26px,2.8vw,40px)] max-[760px]:p-[24px_19px]";
const panelHeader = "mb-[30px] flex min-h-[30px] items-center justify-between gap-[18px] max-[760px]:mb-[23px]";
const stepLabel = "text-xs font-bold text-[var(--signal)]";
const fieldLabel = "block min-h-[34px] text-xs font-semibold text-[var(--muted)]";
const fieldHelp = "mt-1 block text-[11px] font-normal";
const fieldControl = "h-[39px] w-full min-w-0 border-0 border-b border-[#aaa99f] bg-transparent p-0 text-[17px] font-bold text-[var(--ink)] [font-variant-numeric:tabular-nums] outline-none focus:border-[var(--signal)] focus:shadow-[0_2px_0_color-mix(in_srgb,var(--signal)_24%,transparent)] max-[760px]:min-h-11";

const operatingConditionLabels: Record<number, string> = {
  1: "2,100 rpm · 12 kN",
  2: "2,250 rpm · 11 kN",
  3: "2,400 rpm · 10 kN",
};

const riskVariable: Record<Prediction["break_risk"], string> = {
  low: "[--risk:var(--green)]",
  uncertain: "[--risk:var(--amber)]",
  high: "[--risk:#a33f29]",
  unknown: "[--risk:var(--muted)]",
};

const riskText: Record<Prediction["break_risk"], string> = {
  low: "text-[var(--green)]",
  uncertain: "text-[var(--amber)]",
  high: "text-[#a33f29]",
  unknown: "text-[var(--muted)]",
};

function ModelDisclosure({ prediction }: { prediction: Prediction }) {
  return (
    <details className="group mt-5 border-t border-[#bdbbb0]">
      <summary className="flex min-h-[45px] cursor-pointer list-none items-center justify-between text-xs font-semibold text-[#4f514b] focus-visible:outline-3 focus-visible:outline-offset-2 focus-visible:outline-[var(--signal)] [&::-webkit-details-marker]:hidden">
        Model scope and limitations
        <span aria-hidden="true" className="text-base font-normal">
          <span className="group-open:hidden">+</span>
          <span className="hidden group-open:inline">−</span>
        </span>
      </summary>
      <dl className="m-0 grid grid-cols-[1fr_2fr] border border-[#c2bfb3] max-[760px]:grid-cols-1">
        <div className="min-w-0 p-3">
          <dt className="text-[11px] text-[var(--muted)]">Support</dt>
          <dd className="mt-[5px] overflow-hidden text-xs text-ellipsis">{titleCase(prediction.support_status)}</dd>
        </div>
        <div className="min-w-0 border-l border-[#c2bfb3] p-3 max-[760px]:border-t max-[760px]:border-l-0">
          <dt className="text-[11px] text-[var(--muted)]">Model</dt>
          <dd className="mt-[5px] overflow-hidden font-mono text-[11px] text-ellipsis">{prediction.model_version}</dd>
        </div>
      </dl>
      <ul className="mt-[14px] mb-[5px] list-disc pl-[17px] text-[11px] leading-[1.65] text-[#5f6159] [&>li+li]:mt-[5px]">
        {prediction.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}
      </ul>
    </details>
  );
}

export default function MaintenanceAssessment() {
  const {
    health,
    model,
    snapshot,
    age,
    condition,
    plannedBreak,
    prediction,
    submitting,
    dragging,
    error,
    setAge,
    setCondition,
    setPlannedBreak,
    setDragging,
    acceptFile,
    refreshStatus,
    runAssessment,
  } = useVibraLensAssessment();
  const inputRef = useRef<HTMLInputElement>(null);
  const advice = prediction
    ? advisoryCopy[prediction.advisory] || advisoryCopy.inspect_first
    : null;
  const interval = prediction?.rul_minutes;
  const isReady = health?.status === "ready";
  const statusDot = isReady
    ? "bg-[#3a995f]"
    : health === null
      ? "bg-[#a7a79e]"
      : "bg-[#92928b]";

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!snapshot) inputRef.current?.focus();
    void runAssessment();
  }

  return (
    <main aria-label="Single bearing maintenance assessment" className="min-h-dvh bg-[var(--paper)] text-[var(--ink)]" id="main-content">
      <header className="sticky top-0 z-20 grid min-h-[73px] grid-cols-[1fr_auto_1fr] items-center border-b border-[var(--line)] bg-[var(--paper)] px-[max(24px,4vw)] max-[760px]:min-h-[67px] max-[760px]:grid-cols-[1fr_auto] max-[760px]:px-[18px]">
        <a className="inline-flex w-fit items-center gap-3 text-[19px] font-extrabold tracking-[-.045em] text-[var(--ink)] no-underline max-[760px]:text-[17px]" href="#main-content" aria-label="VibraLens maintenance assessment">
          <span className="flex h-[25px] w-[31px] items-center gap-[3px] border-2 border-[var(--ink)] px-[3px] max-[760px]:h-6 max-[760px]:w-[29px]" aria-hidden="true">
            <i className="block h-[7px] w-[3px] bg-[var(--signal)]" />
            <i className="block h-[15px] w-[3px] bg-[var(--signal)]" />
            <i className="block h-5 w-[3px] bg-[var(--signal)]" />
            <i className="block h-[10px] w-[3px] bg-[var(--signal)]" />
          </span>
          <span>VibraLens</span>
        </a>
        <div className="flex items-center gap-[11px] text-xs max-[760px]:hidden">
          <span className="text-[var(--muted)]">Maintenance</span>
          <i className="font-normal text-[var(--line)] not-italic">/</i>
          <b className="font-semibold">Assessment</b>
        </div>
        <button
          className={`inline-flex min-h-[38px] items-center justify-self-end gap-[9px] rounded-[3px] border border-[var(--line)] bg-[var(--panel)] px-[13px] py-2 text-xs font-medium text-[var(--muted)] transition-[border-color,color,transform] duration-200 hover:border-[var(--ink)] hover:text-[var(--ink)] active:translate-y-px max-[760px]:min-h-11 max-[760px]:max-w-[150px] max-[760px]:px-[9px] max-[760px]:py-[7px] max-[760px]:text-[11px] ${isReady ? "text-[var(--ink)]" : ""}`}
          onClick={() => void refreshStatus()}
          type="button"
        >
          <span className={`h-[7px] w-[7px] shrink-0 rounded-full ${statusDot}`} aria-hidden="true" />
          {health === null ? "Connecting" : isReady ? "Analysis service online" : "Service unavailable"}
        </button>
      </header>

      <div className="mx-auto w-[min(1440px,calc(100%-72px))] max-[1040px]:w-[min(calc(100%-48px),900px)] max-[760px]:w-[calc(100%-32px)]">
        <section className="pt-[clamp(54px,7vw,96px)] pb-10 max-[760px]:pt-[42px] max-[760px]:pb-[29px]" aria-labelledby="assessment-title">
          <h1 className="m-0 max-w-[1050px] text-[clamp(42px,5vw,72px)] leading-[.98] font-bold tracking-[-.06em] text-balance max-[760px]:text-[clamp(38px,12vw,55px)]" id="assessment-title">Bearing maintenance assessment</h1>
          <p className="mt-[22px] mb-0 max-w-[650px] text-[15px] leading-[1.65] text-[#5d5f57] text-pretty max-[760px]:mt-[15px] max-[760px]:text-[13px]">Evaluate a vibration snapshot against the next planned maintenance window.</p>
        </section>

        <section className="grid grid-cols-[minmax(340px,.78fr)_minmax(520px,1.22fr)] border border-[var(--ink)] bg-[var(--panel)] shadow-[9px_9px_0_#dedacd] max-[1040px]:grid-cols-1 max-[760px]:shadow-[6px_6px_0_#dedacd]" aria-label="Bearing maintenance assessment workspace">
          <form className={`${panelPadding} border-r border-[var(--ink)] bg-[var(--panel)] max-[1040px]:border-r-0 max-[1040px]:border-b`} onSubmit={handleSubmit}>
            <header className={panelHeader}>
              <div className="flex items-baseline gap-3">
                <span className={stepLabel}>01</span>
                <h2 className="m-0 text-xl font-semibold tracking-[-.035em]">Observation details</h2>
              </div>
              <small className="text-[11px] font-medium text-[var(--muted)] max-[760px]:hidden">All fields required</small>
            </header>

            <label
              className={`relative grid min-h-[112px] cursor-pointer grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-[17px] border border-dashed border-[#98998f] bg-[#f1efe6] p-[21px] transition-[transform,border-color,background] duration-200 hover:-translate-y-0.5 hover:border-[var(--signal)] hover:bg-[#fff5ec] focus-within:outline-3 focus-within:outline-offset-2 focus-within:outline-[var(--signal)] max-[760px]:min-h-[104px] max-[760px]:grid-cols-[auto_minmax(0,1fr)] max-[760px]:p-[17px] ${dragging ? "-translate-y-0.5 border-[var(--signal)] bg-[#fff5ec]" : ""} ${snapshot ? "border-solid border-[var(--green)] bg-[#eff6e9]" : ""}`}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <input
                ref={inputRef}
                className="absolute h-px w-px overflow-hidden opacity-0"
                type="file"
                accept=".csv,text/csv"
                onChange={handleFileChange}
                aria-label="Vibration snapshot CSV"
              />
              <span className="grid h-[42px] w-[42px] place-items-center rounded-full border border-[var(--ink)]" aria-hidden="true">
                <svg className="h-[22px] w-[22px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4"><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 19.5h14" /></svg>
              </span>
              {snapshot ? (
                <span className="min-w-0 not-italic">
                  <strong className="mb-[5px] block overflow-hidden text-sm text-ellipsis whitespace-nowrap">{snapshot.name}</strong>
                  <small className="block text-xs leading-[1.45] text-[var(--muted)]">{(snapshot.size / 1024 / 1024).toFixed(2)} MB · Ready for assessment</small>
                </span>
              ) : (
                <span className="min-w-0 not-italic">
                  <strong className="mb-[5px] block text-sm">Vibration snapshot</strong>
                  <small className="block text-xs leading-[1.45] text-[var(--muted)]">Drop or browse · Two-channel vibration CSV</small>
                </span>
              )}
              <em className="border-b border-[var(--ink)] pb-0.5 text-xs font-medium not-italic max-[760px]:hidden">{snapshot ? "Replace" : "Browse"}</em>
            </label>

            <div className="mt-5 grid grid-cols-2 border border-[var(--line)] max-[760px]:grid-cols-1">
              <label className="min-w-0 p-[16px_17px_14px]">
                <span className={fieldLabel}>Bearing age <small className={fieldHelp}>Minutes observed</small></span>
                <input className={fieldControl} type="number" min="0" step="1" value={age} onChange={(event) => setAge(event.target.value)} required />
              </label>
              <label className="min-w-0 border-l border-[var(--line)] p-[16px_17px_14px] max-[760px]:border-t max-[760px]:border-l-0">
                <span className={fieldLabel}>Operating condition <small className={fieldHelp}>Speed · Radial load</small></span>
                <select className={`${fieldControl} text-[13px]`} value={condition} onChange={(event) => setCondition(event.target.value)}>
                  {(model?.supported_condition_ids || [1, 2, 3]).map((id) => (
                    <option key={id} value={id}>{operatingConditionLabels[id]}</option>
                  ))}
                </select>
              </label>
              <label className="col-span-2 min-w-0 border-t border-[var(--line)] p-[16px_17px_14px] max-[760px]:col-span-1">
                <span className={fieldLabel}>Next planned stop <small className={fieldHelp}>Minutes from now</small></span>
                <input className={fieldControl} type="number" min="0" step="1" value={plannedBreak} onChange={(event) => setPlannedBreak(event.target.value)} required />
              </label>
            </div>

            {error ? (
              <div className="mt-4 grid grid-cols-[auto_1fr] items-center gap-[10px] text-xs text-[#963d27]" role="alert">
                <span className="grid h-5 w-5 place-items-center rounded-full bg-[#963d27] font-extrabold text-white">!</span>
                <p className="m-0 leading-normal">{error}</p>
              </div>
            ) : null}
            {!isReady ? <p className="mt-[15px] mb-0 text-xs leading-normal text-[var(--muted)]">The analysis service must be online before an assessment can run.</p> : null}
            <button className="mt-[22px] flex min-h-[53px] w-full items-center justify-between border border-[var(--ink)] bg-[var(--ink)] px-[18px] py-[14px] font-bold text-white transition-[transform,background,box-shadow] duration-200 hover:not-disabled:-translate-x-0.5 hover:not-disabled:-translate-y-0.5 hover:not-disabled:bg-[var(--signal)] hover:not-disabled:shadow-[4px_4px_0_var(--ink)] active:not-disabled:translate-y-px active:not-disabled:shadow-none disabled:cursor-not-allowed disabled:opacity-40" type="submit" disabled={submitting || !isReady}>
              <span>{submitting ? "Analyzing snapshot…" : "Run assessment"}</span><b className="text-[21px] font-normal" aria-hidden="true">↗</b>
            </button>
            <p className="mt-3 mb-0 text-center text-[11px] leading-normal text-[var(--muted)]">The uploaded snapshot is processed for this request and then removed.</p>
          </form>

          <section className={`${panelPadding} flex flex-col bg-[#e7e4d8]`} aria-live="polite" aria-labelledby="decision-title">
            <header className={panelHeader}>
              <div className="flex items-baseline gap-3"><span className={stepLabel}>02</span><h2 className="m-0 text-xl font-semibold tracking-[-.035em]" id="decision-title">Decision summary</h2></div>
              {prediction ? <small className={`rounded-[2px] border border-current px-2 py-1.5 text-[11px] font-medium ${riskText[prediction.break_risk]}`}>{prediction.break_risk} risk</small> : <small className="text-[11px] font-medium text-[var(--muted)] max-[760px]:hidden">Awaiting assessment</small>}
            </header>

            {!prediction && !submitting ? (
              <div className="flex min-h-[440px] flex-1 flex-col justify-center px-[6%] pt-[5%] pb-[6%] max-[1040px]:min-h-[390px] max-[760px]:px-1 max-[760px]:pt-6 max-[760px]:pb-[31px]">
                <h3 className="m-0 max-w-[520px] text-[clamp(32px,3vw,44px)] leading-none font-bold tracking-[-.055em] text-balance max-[760px]:text-[34px]">No assessment yet</h3>
                <p className="mt-4 mb-0 max-w-[520px] text-sm leading-[1.65] text-[var(--muted)]">Upload a vibration snapshot and enter its operating context to generate a decision summary.</p>
              </div>
            ) : null}

            {submitting ? (
              <div className="flex min-h-[440px] flex-1 animate-pulse flex-col justify-center px-[4%] py-[5%] max-[1040px]:min-h-[390px] max-[760px]:px-0" aria-label="Analyzing vibration snapshot">
                <span className="h-[9px] w-[105px] bg-[#d5d2c6]" /><span className="mt-[19px] h-11 w-[min(76%,430px)] bg-[#d5d2c6]" /><span className="mt-[15px] h-[14px] w-[min(92%,600px)] bg-[#d5d2c6]" /><span className="mt-[42px] h-[118px] w-full bg-[#d5d2c6]" />
                <div className="mt-6 grid grid-cols-4 gap-px">{[0, 1, 2, 3].map((item) => <span className="h-[58px] bg-[#d5d2c6]" key={item} />)}</div>
              </div>
            ) : null}

            {prediction && prediction.support_status === "unsupported" ? (
              <article className="flex min-h-[440px] flex-1 flex-col justify-center px-[5%] py-[4%] max-[1040px]:min-h-[390px] max-[760px]:px-0">
                <p className="m-0 text-[11px] font-semibold text-[var(--signal)]">Recommended action</p><h3 className="mt-[13px] mb-[11px] text-[clamp(34px,4vw,54px)] leading-[.95] font-bold tracking-[-.055em]">Manual inspection required</h3><p className="m-0 max-w-[560px] text-[13px] leading-[1.65] text-[var(--muted)]">This operating condition is outside the evaluated model boundary. VibraLens has withheld a remaining-life estimate.</p><ModelDisclosure prediction={prediction} />
              </article>
            ) : null}

            {prediction && interval ? (
              <article className={`${riskVariable[prediction.break_risk]} text-[var(--ink)]`}>
                <header className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-[26px] border-b border-b-[#bdbbb0] border-l-4 border-l-[var(--risk)] pt-1 pr-0 pb-[26px] pl-5 max-[760px]:grid-cols-1 max-[760px]:gap-6 max-[760px]:pl-[14px]">
                  <div><p className="m-0 text-[11px] font-semibold text-[var(--risk)]">Recommended action</p><h3 className="my-[7px] text-[clamp(30px,3.3vw,48px)] leading-[.98] font-bold tracking-[-.055em] text-balance max-[760px]:text-[34px]">{advice?.title}</h3><span className="block max-w-[570px] text-[13px] leading-[1.55] text-[#5e6058]">{advice?.detail}</span></div>
                  <strong className="text-right text-[clamp(42px,4.5vw,66px)] leading-[.8] font-bold tracking-[-.065em] [font-variant-numeric:tabular-nums] max-[760px]:text-left">{formatMinutes(interval.median)}<small className="mt-[10px] block font-mono text-[11px] font-normal tracking-[.04em] text-[var(--muted)]">min median</small></strong>
                </header>
                <DecisionTimeline interval={interval} plannedBreak={prediction.planned_break_minutes} />
                <dl className="mt-[27px] mb-0 grid grid-cols-4 border-y border-[#bdbbb0] max-[760px]:grid-cols-2">
                  {[["Conservative", interval.pessimistic], ["Median estimate", interval.median], ["Optimistic", interval.optimistic], ["Planned stop", prediction.planned_break_minutes]].map(([label, value], index) => (
                    <div className={`min-w-0 py-[15px] pr-[10px] pb-[14px] ${index > 0 ? "border-l border-[#bdbbb0] pl-[14px]" : ""} ${index === 2 ? "max-[760px]:border-t max-[760px]:border-l-0 max-[760px]:pl-0" : ""} ${index === 3 ? "max-[760px]:border-t" : ""}`} key={label}>
                      <dt className="text-[11px] text-[var(--muted)]">{label}</dt><dd className="mt-[7px] mb-0 font-mono text-[17px] font-semibold [font-variant-numeric:tabular-nums]">{formatMinutes(value as number)}<span className="ml-[3px] text-[11px] font-normal text-[var(--muted)]">min</span></dd>
                    </div>
                  ))}
                </dl>
                <ModelDisclosure prediction={prediction} />
              </article>
            ) : null}
          </section>
        </section>

        <footer className="mt-9 flex min-h-[94px] items-center justify-between gap-[30px] pb-[22px] font-sans text-[11px] text-[var(--muted)] max-[760px]:grid max-[760px]:min-h-[110px] max-[760px]:gap-[15px] max-[760px]:py-6">
          <p className="m-0 block max-w-[680px] leading-[1.65]">Advisory output only. Combine the assessment with qualified inspection and operating context.</p><a className="border-b border-[var(--ink)] text-[var(--ink)] no-underline" href="https://github.com/andeluw/vibralens" target="_blank" rel="noreferrer">Technical documentation ↗</a>
        </footer>
      </div>
    </main>
  );
}
