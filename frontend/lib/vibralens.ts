export type Health = {
  status: "ready" | "unavailable";
  model_version?: string;
};

export type ModelInfo = {
  model_version: string;
  supported_condition_ids: number[];
  feature_set: string;
  feature_names: string[];
  limitations: string[];
};

export type RulInterval = {
  pessimistic: number;
  median: number;
  optimistic: number;
};

export type Prediction = {
  model_version: string;
  rul_minutes: RulInterval | null;
  planned_break_minutes: number;
  break_risk: "low" | "uncertain" | "high" | "unknown";
  advisory: string;
  support_status: "supported" | "unsupported";
  limitations: string[];
};

export type AdvisoryCopy = {
  title: string;
  detail: string;
};

export const API_BASE = (
  process.env.NEXT_PUBLIC_VIBRALENS_API_URL || "http://localhost:8000"
).replace(/\/$/, "");

export const advisoryCopy: Record<string, AdvisoryCopy> = {
  safe_to_wait: {
    title: "Continue to planned stop",
    detail: "The estimated interval supports continuing to the entered maintenance window.",
  },
  inspect_first: {
    title: "Inspect before proceeding",
    detail: "The estimate overlaps your planned stop. Confirm condition before continued operation.",
  },
  maintenance_urgent: {
    title: "Prioritize intervention",
    detail: "The estimated interval ends before the planned stop. Escalate for maintenance review.",
  },
};

export function formatMinutes(value: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
}

export function titleCase(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function timelinePercent(value: number, maximum: number) {
  if (!Number.isFinite(value) || !Number.isFinite(maximum) || maximum <= 0) {
    return 0;
  }
  return Math.min(100, Math.max(0, (value / maximum) * 100));
}
