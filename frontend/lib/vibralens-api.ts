import axios from "axios";
import {
  API_BASE,
  type Health,
  type ModelInfo,
  type Prediction,
} from "./vibralens";

type ApiErrorPayload = {
  detail?: unknown;
};

export type ServiceProfile = {
  health: Health;
  model: ModelInfo | null;
};

const api = axios.create({ baseURL: API_BASE });

export async function loadServiceProfile(): Promise<ServiceProfile> {
  const healthRequest = api.get<Health>("/health");
  const modelRequest = api
    .get<ModelInfo>("/model")
    .then((response) => response.data)
    .catch(() => null);
  const [healthResponse, model] = await Promise.all([
    healthRequest,
    modelRequest,
  ]);

  return { health: healthResponse.data, model };
}

export async function submitPrediction(body: FormData): Promise<Prediction> {
  const response = await api.post<Prediction>("/predict", body);
  return response.data;
}

export function assessmentErrorMessage(error: unknown): string {
  if (axios.isAxiosError<ApiErrorPayload>(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }

  return "The analysis service could not be reached.";
}
