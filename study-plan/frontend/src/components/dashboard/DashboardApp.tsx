import { useCallback, useEffect, useState } from "react";
import { Toaster, toast } from "sonner";
import { getDashboard } from "@/api";
import { filterWeeks, currentWeek, type DashboardFilters } from "@/dashboardModel";
import type { DashboardData } from "@/types";
import { CurrentFocusPanel } from "./CurrentFocusPanel";
import { DashboardLayout } from "./DashboardLayout";
import { EmptyState } from "./EmptyState";
import { LoadingState } from "./LoadingState";
import { InsightRail } from "./InsightRail";
import { PlanFilters } from "./PlanFilters";
import { ProgressOverview } from "./ProgressOverview";
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
      <div className="grid gap-4">
        <section className="grid gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(280px,0.75fr)]">
          <CurrentFocusPanel day={data.current_day} onEditDay={() => toast.info("Editor arrives in the next slice")} />
          <ProgressOverview summary={data.summary} />
        </section>
        <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
          <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
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
              onEditDay={() => toast.info("Editor arrives in the next slice")}
            />
          </section>
          <InsightRail
            data={data}
            onEditOperator={() => toast.info("Editor arrives in the next slice")}
            onEditLibrary={() => toast.info("Editor arrives in the next slice")}
          />
        </section>
      </div>
      <Toaster richColors position="bottom-right" />
    </DashboardLayout>
  );
}
