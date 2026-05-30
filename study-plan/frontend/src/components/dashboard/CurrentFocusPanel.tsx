import { CalendarDays, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { DashboardDay } from "@/types";
import { StatusBadge } from "./StatusBadge";

interface CurrentFocusPanelProps {
  day: DashboardDay | null;
  onEditDay: (day: DashboardDay) => void;
}

export function CurrentFocusPanel({ day, onEditDay }: CurrentFocusPanelProps) {
  if (!day) {
    return (
      <Card className="shadow-sm">
        <CardHeader>
          <h2 className="text-lg font-semibold">All planned days are complete.</h2>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card className="shadow-sm">
      <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs font-bold uppercase tracking-wider text-blue-700">Next Step</p>
          <h2 className="mt-1 text-lg font-semibold text-slate-950">{day.title}</h2>
          <p className="mt-1 flex items-center gap-2 text-sm text-slate-600">
            <CalendarDays className="h-4 w-4" />
            Day {String(day.num).padStart(2, "0")} · Week {day.week}
          </p>
        </div>
        <Button onClick={() => onEditDay(day)}>Edit Day {String(day.num).padStart(2, "0")}</Button>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className="flex flex-wrap gap-2">
          <StatusBadge status={day.status} />
          {(day.jd_tags || []).map((tag) => (
            <span key={tag} className="rounded-full bg-blue-50 px-2.5 py-1 text-xs font-semibold text-blue-700">
              {tag}
            </span>
          ))}
        </div>
        {day.next_fix ? (
          <p className="text-sm text-slate-700"><span className="font-semibold text-slate-950">Next Fix</span>: {day.next_fix}</p>
        ) : null}
        {day.weaknesses ? (
          <p className="text-sm text-slate-700"><span className="font-semibold text-slate-950">Weakness</span>: {day.weaknesses}</p>
        ) : null}
        <div className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 sm:grid-cols-4">
          <Metric label="Tasks" value={`${day.task_done}/${day.task_total}`} />
          <Metric label="Artifacts" value={`${day.artifact_done}/${day.artifact_total}`} />
          <Metric label="Daily Check" value={`${day.daily_check || 0}/3`} />
          <Metric label="Progress" value={`${Math.round(day.completion_pct)}%`} />
        </div>
        <Progress value={day.completion_pct} aria-label="Current day progress" />
        <div className="grid gap-4 md:grid-cols-2">
          <Checklist title="Tasks" items={day.tasks || {}} />
          <Checklist title="Artifacts" items={day.artifacts || {}} />
        </div>
      </CardContent>
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <strong className="block text-xl font-semibold tabular-nums text-slate-950">{value}</strong>
      <span className="text-xs text-slate-600">{label}</span>
    </div>
  );
}

function Checklist({ title, items }: { title: string; items: Record<string, unknown> }) {
  return (
    <div>
      <p className="mb-2 text-xs font-semibold text-slate-600">{title}</p>
      <ul className="grid gap-1.5">
        {Object.entries(items).map(([key, value]) => (
          <li key={key} className="flex items-start gap-2 text-sm text-slate-700">
            <CheckCircle2 className={value ? "mt-0.5 h-4 w-4 text-emerald-600" : "mt-0.5 h-4 w-4 text-slate-300"} />
            <span>{key}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
