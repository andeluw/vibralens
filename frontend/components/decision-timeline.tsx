import { type CSSProperties } from "react";
import { formatMinutes, timelinePercent, type RulInterval } from "../lib/vibralens";

type DecisionTimelineProps = {
  interval: RulInterval;
  plannedBreak: number;
};

export function DecisionTimeline({ interval, plannedBreak }: DecisionTimelineProps) {
  const maximum = Math.max(interval.optimistic, plannedBreak, 1);
  const lower = timelinePercent(interval.pessimistic, maximum);
  const upper = timelinePercent(interval.optimistic, maximum);
  const styles = {
    "--range-start": `${lower}%`,
    "--range-width": `${Math.max(upper - lower, 0)}%`,
    "--median-position": `${timelinePercent(interval.median, maximum)}%`,
    "--stop-position": `${timelinePercent(plannedBreak, maximum)}%`,
  } as CSSProperties;

  return (
    <figure
      className="mt-[31px] mb-0 px-[3px] max-[760px]:mt-[27px]"
      style={styles}
      aria-label={`Remaining-life interval ${formatMinutes(interval.pessimistic)} to ${formatMinutes(interval.optimistic)} minutes; median ${formatMinutes(interval.median)} minutes; planned stop ${formatMinutes(plannedBreak)} minutes.`}
    >
      <figcaption className="m-0 text-[11px] font-semibold">Remaining life compared with planned stop</figcaption>
      <div className="mt-[-13px] flex justify-end gap-4 text-[11px] text-[var(--muted)] max-[760px]:hidden" aria-hidden="true">
        <span className="inline-flex items-center gap-1.5"><i className="h-[3px] w-[14px] bg-[var(--risk)]" />Empirical interval</span>
        <span className="inline-flex items-center gap-1.5"><i className="h-[11px] w-0.5 bg-[var(--signal)]" />Planned stop</span>
      </div>
      <div className="relative mt-12 mb-3 h-[5px] bg-[#c2bfb3] max-[760px]:mt-[55px]" aria-hidden="true">
        <span className="absolute left-[var(--range-start)] h-full w-[var(--range-width)] bg-[var(--risk)]" />
        <i className="absolute top-[-9px] left-[var(--median-position)] h-[23px] w-0.5 -translate-x-px bg-[var(--ink)] not-italic">
          <b className="absolute bottom-[29px] left-1/2 -translate-x-1/2 bg-[var(--ink)] px-1.5 py-1 text-[11px] font-medium whitespace-nowrap text-white max-[760px]:left-0 max-[760px]:translate-x-0">Median</b>
        </i>
        <i className="absolute top-[-9px] left-[var(--stop-position)] h-[23px] w-0.5 -translate-x-px bg-[var(--signal)] not-italic">
          <b className="absolute right-0 bottom-[29px] bg-[var(--signal)] px-1.5 py-1 text-[11px] font-medium whitespace-nowrap text-white">Planned stop</b>
        </i>
      </div>
      <div className="flex justify-between font-mono text-[11px] [font-variant-numeric:tabular-nums]" aria-hidden="true">
        <span>0 min</span>
        <span>{formatMinutes(maximum)} min</span>
      </div>
    </figure>
  );
}
