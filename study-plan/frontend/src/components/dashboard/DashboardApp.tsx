import { useCallback, useEffect, useState } from "react";
import { Toaster, toast } from "sonner";
import { getDashboard, saveDay, saveLibrary, saveOperator } from "@/api";
import { filterWeeks, currentWeek, type DashboardFilters } from "@/dashboardModel";
import type { DashboardData } from "@/types";
import { CurrentFocusPanel } from "./CurrentFocusPanel";
import { EditDrawer, type EditTarget } from "./EditDrawer";
import { EmptyState } from "./EmptyState";
import { RisksTile, OperatorsTile, LibraryTile, TagCoverageTile } from "./InsightTiles";
import { LoadingState } from "./LoadingState";
import { PlanFilters } from "./PlanFilters";
import { KpiTile, DailyCheckTile } from "./ProgressOverview";
import { Sidebar, type View } from "./Sidebar";
import { WeekPlanList } from "./WeekPlanList";

export function DashboardApp() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>("focus");
  const [filters, setFilters] = useState<DashboardFilters>({
    week: "all",
    status: "all",
    tag: "all",
    query: "",
  });
  const [editTarget, setEditTarget] = useState<EditTarget>(null);

  const refresh = useCallback(async (): Promise<boolean> => {
    try {
      setError(null);
      const next = await getDashboard();
      setData(next);
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to load dashboard data";
      setError(message);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const handleRefresh = () => {
    void refresh().then((ok) => {
      if (ok) toast.success("Data refreshed");
      else toast.error("Unable to refresh dashboard data");
    });
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <LoadingState />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex h-screen items-center justify-center">
        <EmptyState title="Unable To Load Dashboard" role="alert">
          <p>{error || "Unable to load dashboard data"}</p>
          <p className="mt-2">Start the local dashboard server before editing or refreshing data.</p>
        </EmptyState>
        <Toaster richColors position="bottom-right" />
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar active={view} onChange={setView} onRefresh={handleRefresh} />

      <main className="flex-1 overflow-y-auto bg-slate-50/50 p-6">
        <header className="mb-6">
          <h1 className="text-2xl font-semibold text-slate-950">
            {data.meta.title || "Study Plan"}
          </h1>
          <p className="mt-1 text-sm text-slate-600">
            {viewDescription(view)}
          </p>
        </header>

        {view === "focus" && (
          <div className="grid gap-4">
            {/* KPI row */}
            <section className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              <KpiTile label="Days" done={data.summary.done_days} total={data.summary.total_days} />
              <KpiTile label="Tasks" done={data.summary.done_tasks} total={data.summary.total_tasks} />
              <KpiTile label="Artifacts" done={data.summary.done_artifacts} total={data.summary.total_artifacts} />
              <DailyCheckTile value={data.summary.average_daily_check} />
            </section>

            {/* Focus panel */}
            <CurrentFocusPanel
              day={data.current_day}
              onEditDay={(day) => setEditTarget({ type: "day", day })}
            />
          </div>
        )}

        {view === "plan" && (
          <div className="grid gap-4">
            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="mb-4">
                <PlanFilters data={data} filters={filters} onChange={setFilters} />
              </div>
              <WeekPlanList
                weeks={filterWeeks(data, filters)}
                currentWeek={currentWeek(data)}
                onEditDay={(day) => setEditTarget({ type: "day", day })}
              />
            </section>
          </div>
        )}

        {view === "operators" && (
          <div className="max-w-2xl">
            <OperatorsTile
              operators={data.operator_maturity}
              onEdit={(name) => {
                const operator = data.operators[name];
                if (operator) setEditTarget({ type: "operator", name, operator });
              }}
            />
          </div>
        )}

        {view === "libraries" && (
          <div className="max-w-2xl">
            <LibraryTile
              libraries={data.gpu_libraries}
              onEdit={(name) => {
                const library = data.gpu_libraries[name];
                if (library) setEditTarget({ type: "library", name, library });
              }}
            />
          </div>
        )}

        {view === "risks" && (
          <div className="max-w-2xl">
            <RisksTile risks={data.risks} />
          </div>
        )}

        {view === "tags" && (
          <div className="max-w-2xl">
            <TagCoverageTile tags={data.tag_coverage} />
          </div>
        )}
      </main>

      <EditDrawer
        open={editTarget !== null}
        target={editTarget}
        options={data.options}
        onOpenChange={(open) => {
          if (!open) setEditTarget(null);
        }}
        onSaveDay={async (day, updates) => {
          await saveDay(day, updates);
          await refresh();
          toast.success("Day saved");
        }}
        onSaveOperator={async (name, updates) => {
          await saveOperator(name, updates);
          await refresh();
          toast.success("Operator saved");
        }}
        onSaveLibrary={async (name, updates) => {
          await saveLibrary(name, updates);
          await refresh();
          toast.success("Library saved");
        }}
      />
      <Toaster richColors position="bottom-right" />
    </div>
  );
}

function viewDescription(view: View): string {
  switch (view) {
    case "focus":
      return "当前焦点日和整体进度概览";
    case "plan":
      return "按周浏览和筛选所有计划日";
    case "operators":
      return "Operator 成熟度和 artifact 覆盖";
    case "libraries":
      return "GPU 加速库学习进度";
    case "risks":
      return "高优先级风险项";
    case "tags":
      return "JD 标签覆盖率";
  }
}
