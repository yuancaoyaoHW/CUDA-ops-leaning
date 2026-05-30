import { useCallback, useEffect, useState } from "react";
import { Toaster, toast } from "sonner";
import { getDashboard } from "@/api";
import type { DashboardData } from "@/types";
import { DashboardLayout } from "./DashboardLayout";
import { EmptyState } from "./EmptyState";
import { LoadingState } from "./LoadingState";

export function DashboardApp() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setError(null);
      const next = await getDashboard();
      setData(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load dashboard data");
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

  if (error || !data) {
    return (
      <main className="mx-auto max-w-7xl px-5 py-6 sm:px-7">
        <EmptyState title="Unable To Load Dashboard">
          <p>{error}</p>
          <p className="mt-2">Start the local dashboard server before editing or refreshing data.</p>
        </EmptyState>
      </main>
    );
  }

  return (
    <DashboardLayout
      title={data.meta.title || "Study Plan"}
      onRefresh={() => {
        void refresh().then(() => toast.success("Data refreshed"));
      }}
    >
      <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-950">Dashboard Loaded</h2>
      </section>
      <Toaster richColors position="bottom-right" />
    </DashboardLayout>
  );
}
