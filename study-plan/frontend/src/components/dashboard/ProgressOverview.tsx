import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { DashboardData } from "@/types";

interface ProgressOverviewProps {
  summary: DashboardData["summary"];
}

export function ProgressOverview({ summary }: ProgressOverviewProps) {
  return (
    <Card className="shadow-sm">
      <CardHeader>
        <p className="text-xs font-bold uppercase tracking-wider text-blue-700">Overview</p>
      </CardHeader>
      <CardContent className="grid gap-4">
        <OverviewRow label="Days" done={summary.done_days} total={summary.total_days} />
        <OverviewRow label="Tasks" done={summary.done_tasks} total={summary.total_tasks} />
        <OverviewRow label="Artifacts" done={summary.done_artifacts} total={summary.total_artifacts} />
        <div>
          <span className="text-xs text-slate-600">Avg Daily Check</span>
          <strong className="block text-xl font-semibold tabular-nums text-slate-950">
            {summary.average_daily_check ?? "—"}
          </strong>
        </div>
      </CardContent>
    </Card>
  );
}

function OverviewRow({ label, done, total }: { label: string; done: number; total: number }) {
  const value = total ? Math.round((done / total) * 100) : 0;
  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs text-slate-600">{label}</span>
        <strong className="tabular-nums text-slate-950">{done}/{total}</strong>
      </div>
      <Progress value={value} aria-label={`${label} progress`} />
    </div>
  );
}
