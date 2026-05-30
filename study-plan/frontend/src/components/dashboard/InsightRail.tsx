import { AlertTriangle, Cpu, LibraryBig, Tags } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { DashboardData } from "@/types";
import { StatusBadge } from "./StatusBadge";

interface InsightRailProps {
  data: DashboardData;
  onEditOperator: (name: string) => void;
  onEditLibrary: (name: string) => void;
}

export function InsightRail({ data, onEditOperator, onEditLibrary }: InsightRailProps) {
  return (
    <aside className="grid gap-4">
      <Card className="shadow-sm">
        <CardHeader className="flex-row items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-amber-600" />
          <h2 className="text-base font-semibold">Risks</h2>
        </CardHeader>
        <CardContent>
          {data.risks.length ? (
            <ul className="grid gap-2">
              {data.risks.map((risk) => (
                <li key={risk} className="rounded-md border-l-4 border-amber-500 bg-amber-50 px-3 py-2 text-sm text-amber-950">
                  {risk}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-slate-600">No high-priority risks.</p>
          )}
        </CardContent>
      </Card>

      <InsightCard icon={<Cpu className="h-4 w-4" />} title="Operator Maturity">
        {Object.entries(data.operator_maturity).map(([name, item]) => (
          <div key={name} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border-t border-slate-200 py-3 first:border-t-0 first:pt-0">
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold">{name}</p>
              <p className="text-xs text-slate-600">{item.done}/{item.total} artifacts</p>
              <Progress value={item.pct} className="mt-2 h-2" />
            </div>
            <Button variant="outline" size="sm" onClick={() => onEditOperator(name)} aria-label={`Edit operator ${name}`}>Edit</Button>
          </div>
        ))}
      </InsightCard>

      <InsightCard icon={<LibraryBig className="h-4 w-4" />} title="GPU Library Coverage">
        {Object.entries(data.gpu_libraries).map(([name, info]) => (
          <div key={name} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border-t border-slate-200 py-3 first:border-t-0 first:pt-0">
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold">{name}</p>
              <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-600">
                <StatusBadge status={info.status} />
                <span>{(info.evidence || []).length} evidence</span>
              </div>
            </div>
            <Button variant="outline" size="sm" onClick={() => onEditLibrary(name)} aria-label={`Edit library ${name}`}>Edit</Button>
          </div>
        ))}
      </InsightCard>

      <InsightCard icon={<Tags className="h-4 w-4" />} title="JD Tag Coverage">
        {Object.entries(data.tag_coverage).map(([tag, item]) => {
          const percent = item.planned ? Math.round((item.done / item.planned) * 100) : 0;
          return (
            <div key={tag} className="border-t border-slate-200 py-3 first:border-t-0 first:pt-0">
              <div className="mb-2 flex items-center justify-between gap-3">
                <span className="text-sm font-semibold">{tag}</span>
                <span className="text-xs tabular-nums text-slate-600">{item.done}/{item.planned}</span>
              </div>
              <Progress value={percent} className="h-2" />
            </div>
          );
        })}
      </InsightCard>
    </aside>
  );
}

function InsightCard({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <Card className="shadow-sm">
      <CardHeader className="flex-row items-center gap-2">
        <span className="text-blue-700">{icon}</span>
        <h2 className="text-base font-semibold">{title}</h2>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}
