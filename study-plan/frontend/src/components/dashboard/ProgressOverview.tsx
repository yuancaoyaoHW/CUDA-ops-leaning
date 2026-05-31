import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { DashboardData } from "@/types";

interface ProgressOverviewProps {
  summary: DashboardData["summary"];
}

/**
 * Standalone KPI tile — renders a single metric as its own Card.
 * Designed to be placed independently in a bento grid.
 */
export function KpiTile({ label, done, total }: { label: string; done: number; total: number }) {
  const value = total ? Math.round((done / total) * 100) : 0;
  return (
    <Card className="shadow-sm">
      <CardContent className="p-4">
        <span className="text-xs text-slate-600">{label}</span>
        <strong className="mt-1 block text-2xl font-semibold tabular-nums text-slate-950">
          {done}/{total}
        </strong>
        <Progress value={value} className="mt-2 h-2" aria-label={`${label} progress`} />
      </CardContent>
    </Card>
  );
}

/**
 * Standalone tile for the average daily check metric.
 */
export function DailyCheckTile({ value }: { value: number | null }) {
  return (
    <Card className="shadow-sm">
      <CardContent className="p-4">
        <span className="text-xs text-slate-600">Avg Daily Check</span>
        <strong className="mt-1 block text-2xl font-semibold tabular-nums text-slate-950">
          {value ?? "—"}
        </strong>
      </CardContent>
    </Card>
  );
}

/**
 * Backward-compatible wrapper that renders all KPIs inside a single Card.
 * Kept so existing consumers and tests continue to work unchanged.
 */
export function ProgressOverview({ summary }: ProgressOverviewProps) {
  return (
    <Card className="shadow-sm">
      <CardHeader>
        <p className="text-xs font-bold uppercase tracking-wider text-blue-700">Overview</p>
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-3">
        <KpiTileInner label="Days" done={summary.done_days} total={summary.total_days} />
        <KpiTileInner label="Tasks" done={summary.done_tasks} total={summary.total_tasks} />
        <KpiTileInner label="Artifacts" done={summary.done_artifacts} total={summary.total_artifacts} />
        <div>
          <span className="text-xs text-slate-600">Avg Daily Check</span>
          <strong className="mt-1 block text-xl font-semibold tabular-nums text-slate-950">
            {summary.average_daily_check ?? "—"}
          </strong>
        </div>
      </CardContent>
    </Card>
  );
}

// Internal version without Card wrapper for use inside ProgressOverview
function KpiTileInner({ label, done, total }: { label: string; done: number; total: number }) {
  const value = total ? Math.round((done / total) * 100) : 0;
  return (
    <div>
      <span className="text-xs text-slate-600">{label}</span>
      <strong className="mt-1 block text-xl font-semibold tabular-nums text-slate-950">
        {done}/{total}
      </strong>
      <Progress value={value} className="mt-2 h-2" aria-label={`${label} progress`} />
    </div>
  );
}
