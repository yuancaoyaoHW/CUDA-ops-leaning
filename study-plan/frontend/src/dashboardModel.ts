import type { DashboardData, DashboardDay, DashboardWeek } from "./types";

export interface DashboardFilters {
  week: "all" | "current" | string;
  status: "all" | string;
  tag: "all" | string;
  query: string;
}

const STATUS_LABELS: Record<string, string> = {
  not_started: "Not Started",
  in_progress: "In Progress",
  done: "Done",
  blocked: "Blocked",
  skipped: "Skipped",
  correctness_stage: "Correctness",
  benchmark_stage: "Benchmark",
  profile_stage: "Profile",
  complete: "Complete",
};

export function truthy(value: unknown): boolean {
  return value === true || value === "true" || value === "complete";
}

export function formatStatus(status?: string): string {
  return STATUS_LABELS[status || ""] || status || "Unknown";
}

export function flattenDays(data: Pick<DashboardData, "weeks">): DashboardDay[] {
  return data.weeks.flatMap((week) => week.days);
}

export function currentWeek(data: DashboardData): number {
  return data.current_day?.week || 8;
}

export function filterWeeks(data: DashboardData, filters: DashboardFilters): DashboardWeek[] {
  const targetCurrentWeek = currentWeek(data);
  const query = filters.query.trim().toLowerCase();
  return data.weeks
    .map((week) => {
      const days = week.days.filter((day) => {
        if (filters.week !== "all") {
          const targetWeek = filters.week === "current" ? targetCurrentWeek : Number(filters.week);
          if (day.week !== targetWeek) return false;
        }
        if (filters.status !== "all" && day.status !== filters.status) return false;
        if (filters.tag !== "all" && !(day.jd_tags || []).includes(filters.tag)) return false;
        if (query) {
          const text = [
            day.title,
            day.status,
            day.verification,
            day.weaknesses,
            day.next_fix,
            day.notes,
            ...(day.jd_tags || []),
          ]
            .join(" ")
            .toLowerCase();
          if (!text.includes(query)) return false;
        }
        return true;
      });
      return { ...week, days };
    })
    .filter((week) => week.days.length > 0);
}
