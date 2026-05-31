export type DayStatus = "not_started" | "in_progress" | "done" | "blocked" | "skipped";
export type OperatorStatus =
  | "not_started"
  | "correctness_stage"
  | "benchmark_stage"
  | "profile_stage"
  | "complete"
  | "blocked";
export type LibraryStatus = "not_started" | "in_progress" | "complete" | "blocked";

export type Checklist = Record<string, boolean | string>;

export interface DashboardDay {
  num: number;
  week: number;
  title: string;
  date?: string;
  status: DayStatus;
  daily_check?: number;
  weekly_check_score?: number;
  stage_check_score?: number;
  jd_tags?: string[];
  tasks?: Checklist;
  artifacts?: Checklist;
  verification?: string;
  weaknesses?: string;
  next_fix?: string;
  notes?: string;
  task_done: number;
  task_total: number;
  artifact_done: number;
  artifact_total: number;
  completion_pct: number;
  guide?: DayGuide;
}

export interface DashboardWeek {
  num: number;
  days: DashboardDay[];
}

export interface OperatorInfo {
  status: OperatorStatus;
  artifacts?: Checklist;
  notes?: string;
}

export interface GpuLibraryInfo {
  status: LibraryStatus;
  evidence?: string[];
}

export interface Maturity {
  done: number;
  total: number;
  pct: number;
  status: string;
}

export interface Reference {
  id: string;
  title: string;
  url: string;
  category: string;
  notes?: string;
}

export interface DashboardData {
  meta: { title?: string; [key: string]: unknown };
  weeks: DashboardWeek[];
  operators: Record<string, OperatorInfo>;
  gpu_libraries: Record<string, GpuLibraryInfo>;
  tag_coverage: Record<string, { done: number; planned: number }>;
  operator_maturity: Record<string, Maturity>;
  status_counts: Record<string, number>;
  summary: {
    done_days: number;
    total_days: number;
    done_tasks: number;
    total_tasks: number;
    done_artifacts: number;
    total_artifacts: number;
    average_daily_check: number | null;
    latest_stage_score?: number | null;
  };
  current_day: DashboardDay | null;
  risks: string[];
  references: Reference[];
  options: {
    day_statuses: DayStatus[];
    operator_statuses: OperatorStatus[];
    library_statuses: LibraryStatus[];
    tags: string[];
  };
}

export interface TaskGuide {
  summary: string;
  steps: string[];
  done_when: string;
  time_minutes: number;
  depends_on: string[];
  refs?: { title: string; url: string }[];
}

export interface ArtifactGuide {
  summary: string;
  done_when: string;
  time_minutes: number;
  depends_on: string[];
  refs?: { title: string; url: string }[];
}

export interface DayGuide {
  tasks: Record<string, TaskGuide>;
  artifacts: Record<string, ArtifactGuide>;
  total_time_minutes: number;
}

export type DayUpdates = Partial<Pick<
  DashboardDay,
  | "status"
  | "date"
  | "daily_check"
  | "weekly_check_score"
  | "stage_check_score"
  | "verification"
  | "weaknesses"
  | "next_fix"
  | "notes"
>> & {
  tasks?: Record<string, boolean>;
  artifacts?: Record<string, boolean>;
  auto_status?: boolean;
};
