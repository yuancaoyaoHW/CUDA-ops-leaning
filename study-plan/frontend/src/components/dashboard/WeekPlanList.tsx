import { Button } from "@/components/ui/button";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Progress } from "@/components/ui/progress";
import type { DashboardDay, DashboardWeek } from "@/types";
import { StatusBadge } from "./StatusBadge";

interface WeekPlanListProps {
  weeks: DashboardWeek[];
  currentWeek: number;
  onEditDay: (day: DashboardDay) => void;
}

export function WeekPlanList({ weeks, currentWeek, onEditDay }: WeekPlanListProps) {
  if (!weeks.length) {
    return (
      <section className="rounded-lg border border-slate-200 bg-white p-6 text-sm text-slate-600">
        No matching days. Adjust filters to show planned work.
      </section>
    );
  }

  return (
    <Accordion
      type="multiple"
      defaultValue={
        weeks.map((week) => `week-${week.num}`).includes(`week-${currentWeek}`)
          ? [`week-${currentWeek}`]
          : [`week-${weeks[0].num}`]
      }
      className="rounded-lg border border-slate-200 bg-white shadow-sm"
    >
      {weeks.map((week) => {
        const done = week.days.filter((day) => day.status === "done").length;
        const percent = week.days.length ? Math.round((done / week.days.length) * 100) : 0;
        return (
          <AccordionItem value={`week-${week.num}`} key={week.num} className="border-slate-200">
            <AccordionTrigger className="px-4 hover:no-underline">
              <span className="flex w-full items-center justify-between gap-4 pr-3">
                <span className="font-semibold text-slate-950">Week {week.num}</span>
                <span className="hidden min-w-36 items-center gap-2 sm:flex">
                  <Progress value={percent} className="h-2" />
                  <span className="text-xs tabular-nums text-slate-600">{percent}%</span>
                </span>
              </span>
            </AccordionTrigger>
            <AccordionContent>
              <div className="divide-y divide-slate-200">
                {week.days.map((day) => (
                  <div
                    key={day.num}
                    className="grid grid-cols-[44px_minmax(0,1fr)_auto] items-center gap-3 px-4 py-3"
                  >
                    <strong className="text-sm tabular-nums text-slate-700">
                      D{String(day.num).padStart(2, "0")}
                    </strong>
                    <div className="min-w-0">
                      <p className="truncate text-sm font-semibold text-slate-950">{day.title}</p>
                      <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-600">
                        <StatusBadge status={day.status} />
                        <span>
                          {day.task_done}/{day.task_total} tasks
                        </span>
                        <span>
                          {day.artifact_done}/{day.artifact_total} artifacts
                        </span>
                        {(day.jd_tags || []).map((tag) => (
                          <span key={tag} className="rounded-full bg-blue-50 px-2 py-0.5 font-semibold text-blue-700">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      aria-label={`Edit day ${day.num}`}
                      onClick={() => onEditDay(day)}
                    >
                      Edit
                    </Button>
                  </div>
                ))}
              </div>
            </AccordionContent>
          </AccordionItem>
        );
      })}
    </Accordion>
  );
}
