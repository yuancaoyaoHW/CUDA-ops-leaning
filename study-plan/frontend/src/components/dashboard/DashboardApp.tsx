import { useCallback, useEffect, useState } from "react";
import { Toaster, toast } from "sonner";
import { getDashboard, saveDay, saveLibrary, saveOperator } from "@/api";
import { filterWeeks, currentWeek, type DashboardFilters } from "@/dashboardModel";
import type { DashboardData } from "@/types";
import { CurrentFocusPanel } from "./CurrentFocusPanel";
import { DashboardLayout } from "./DashboardLayout";
import { EditDrawer, type EditTarget } from "./EditDrawer";
import { EmptyState } from "./EmptyState";
import { RisksTile, OperatorsTile, LibraryTile, TagCoverageTile } from "./InsightTiles";
import { LoadingState } from "./LoadingState";
import { PlanFilters } from "./PlanFilters";
import { KpiTile, DailyCheckTile } from "./ProgressOverview";
import { WeekPlanList } from "./WeekPlanList";

export function DashboardApp() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
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

  if (loading) {
    return (
      <main className="mx-auto max-w-7xl px-5 py-6 sm:px-7">
        <LoadingState />
      </main>
    );
  }

  if (!data) {
    return (
      <main className="mx-auto max-w-7xl px-5 py-6 sm:px-7">
        <EmptyState title="Unable To Load Dashboard" role="alert">
          <p>{error || "Unable to load dashboard data"}</p>
          <p className="mt-2">Start the local dashboard server before editing or refreshing data.</p>
        </EmptyState>
        <Toaster richColors position="bottom-right" />
      </main>
    );
  }

  return (
    <DashboardLayout
      title={data.meta.title || "Study Plan"}
      onRefresh={() => {
        void refresh().then((ok) => {
          if (ok) {
            toast.success("Data refreshed");
          } else {
            toast.error("Unable to refresh dashboard data");
          }
        });
      }}
    >
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        {/* Hero: Current Focus — left column, spans 7 cols and 3 rows on xl */}
        <section className="xl:col-span-7 xl:row-span-3">
          <CurrentFocusPanel day={data.current_day} onEditDay={(day) => setEditTarget({ type: "day", day })} />
        </section>

        {/* KPI tiles — top-right, 2x2 grid */}
        <section className="grid grid-cols-2 gap-4 sm:grid-cols-4 xl:col-span-5 xl:grid-cols-2">
          <KpiTile label="Days" done={data.summary.done_days} total={data.summary.total_days} />
          <KpiTile label="Tasks" done={data.summary.done_tasks} total={data.summary.total_tasks} />
          <KpiTile label="Artifacts" done={data.summary.done_artifacts} total={data.summary.total_artifacts} />
          <DailyCheckTile value={data.summary.average_daily_check} />
        </section>

        {/* Risks — mid-right */}
        <section className="xl:col-span-5">
          <RisksTile risks={data.risks} />
        </section>

        {/* Operators + Libraries — bottom-right left */}
        <section className="grid gap-4 xl:col-span-3">
          <OperatorsTile
            operators={data.operator_maturity}
            onEdit={(name) => {
              const operator = data.operators[name];
              if (operator) setEditTarget({ type: "operator", name, operator });
            }}
          />
          <LibraryTile
            libraries={data.gpu_libraries}
            onEdit={(name) => {
              const library = data.gpu_libraries[name];
              if (library) setEditTarget({ type: "library", name, library });
            }}
          />
        </section>

        {/* Tags — bottom-right right */}
        <section className="xl:col-span-2">
          <TagCoverageTile tags={data.tag_coverage} />
        </section>

        {/* Plan — full width bottom */}
        <section className="col-span-full rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-4 grid gap-3 lg:grid-cols-[minmax(180px,0.45fr)_minmax(0,1.55fr)] lg:items-end">
            <div>
              <h2 className="text-lg font-semibold text-slate-950">Plan</h2>
              <p className="text-sm text-slate-600">Filtered by week, status, tag, and search.</p>
            </div>
            <PlanFilters data={data} filters={filters} onChange={setFilters} />
          </div>
          <WeekPlanList
            weeks={filterWeeks(data, filters)}
            currentWeek={currentWeek(data)}
            onEditDay={(day) => setEditTarget({ type: "day", day })}
          />
        </section>
      </div>
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
    </DashboardLayout>
  );
}
