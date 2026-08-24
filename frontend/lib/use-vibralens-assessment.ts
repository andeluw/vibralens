"use client";

import { useCallback, useEffect, useState } from "react";
import {
  assessmentErrorMessage,
  loadServiceProfile,
  submitPrediction,
} from "./vibralens-api";
import { type Health, type ModelInfo, type Prediction } from "./vibralens";

export type VibraLensAssessmentController = {
  health: Health | null;
  model: ModelInfo | null;
  snapshot: File | null;
  age: string;
  condition: string;
  plannedBreak: string;
  prediction: Prediction | null;
  submitting: boolean;
  dragging: boolean;
  error: string | null;
  setAge(value: string): void;
  setCondition(value: string): void;
  setPlannedBreak(value: string): void;
  setDragging(value: boolean): void;
  acceptFile(file?: File): void;
  refreshStatus(): Promise<void>;
  runAssessment(): Promise<void>;
};

export function useVibraLensAssessment(): VibraLensAssessmentController {
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

  const refreshStatus = useCallback(async () => {
    try {
      const profile = await loadServiceProfile();
      setHealth(profile.health);
      setModel(profile.model);
    } catch {
      setHealth({ status: "unavailable" });
      setModel(null);
    }
  }, []);

  useEffect(() => {
    let active = true;

    void loadServiceProfile()
      .then((profile) => {
        if (!active) return;
        setHealth(profile.health);
        setModel(profile.model);
      })
      .catch(() => {
        if (!active) return;
        setHealth({ status: "unavailable" });
        setModel(null);
      });

    return () => {
      active = false;
    };
  }, []);

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

  async function runAssessment() {
    if (!snapshot) {
      setError("Add a vibration snapshot before running the assessment.");
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
      const nextPrediction = await submitPrediction(body);
      setPrediction(nextPrediction);
    } catch (caught) {
      setError(assessmentErrorMessage(caught));
    } finally {
      setSubmitting(false);
    }
  }

  return {
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
  };
}
