import type { DashboardData, DayUpdates, LibraryStatus, OperatorStatus, Reference } from "./types";

export class ApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function readJson(response: Response): Promise<any> {
  try {
    return await response.json();
  } catch {
    return {};
  }
}

async function postJson(url: string, payload: unknown): Promise<void> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await readJson(response);
  if (!response.ok || body.ok === false) {
    throw new ApiError(body.error || `Request failed with HTTP ${response.status}`, response.status);
  }
}

export async function getDashboard(): Promise<DashboardData> {
  const response = await fetch("/api/progress");
  if (!response.ok) {
    throw new ApiError(`Unable to load dashboard data: HTTP ${response.status}`, response.status);
  }
  return response.json() as Promise<DashboardData>;
}

export function saveDay(day: number, updates: DayUpdates): Promise<void> {
  return postJson("/api/day", {
    day,
    updates: { ...updates, auto_status: updates.auto_status ?? true },
  });
}

export function saveOperator(
  operator: string,
  updates: { status?: OperatorStatus; artifacts?: Record<string, boolean>; notes?: string },
): Promise<void> {
  return postJson("/api/operator", { operator, updates });
}

export function saveLibrary(
  library: string,
  updates: { status?: LibraryStatus; evidence?: string[] },
): Promise<void> {
  return postJson("/api/library", { library, updates });
}

export function addReference(ref: Omit<Reference, "id">): Promise<void> {
  return postJson("/api/reference", { action: "add", ...ref });
}

export function updateReference(ref: Reference): Promise<void> {
  return postJson("/api/reference", { action: "update", ...ref });
}

export function deleteReference(id: string): Promise<void> {
  return postJson("/api/reference", { action: "delete", id });
}

export interface ModuleInfo {
  name: string;
  file_count: number;
  files: string[];
}

export async function getModules(): Promise<ModuleInfo[]> {
  const response = await fetch("/api/modules");
  if (!response.ok) {
    throw new ApiError(`Unable to load modules: HTTP ${response.status}`, response.status);
  }
  return response.json() as Promise<ModuleInfo[]>;
}

export async function getFileContent(path: string): Promise<string> {
  const response = await fetch(`/api/file?path=${encodeURIComponent(path)}`);
  if (!response.ok) {
    throw new ApiError(`Unable to load file: HTTP ${response.status}`, response.status);
  }
  const data = await response.json();
  return data.content as string;
}
