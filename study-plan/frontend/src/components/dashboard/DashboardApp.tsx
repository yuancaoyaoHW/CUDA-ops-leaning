import { useCallback, useEffect, useState } from "react";
import { Toaster, toast } from "sonner";
import { getDashboard } from "@/api";
import type { DashboardData } from "@/types";
import { CurrentFocusPanel } from "./CurrentFocusPanel";
import { DashboardLayout } from "./DashboardLayout";
import { EmptyState } from "./EmptyState";
import { LoadingState } from "./LoadingState";
import { ProgressOverview } from "./ProgressOverview";

export function DashboardApp() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
      </div>
      <Toaster richColors position="bottom-right" />
    </DashboardLayout>
  );
}
