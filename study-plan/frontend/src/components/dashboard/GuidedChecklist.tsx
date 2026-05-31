import { CheckCircle2, Clock, AlertTriangle, ExternalLink, FileText } from "lucide-react";
import { truthy } from "@/dashboardModel";
import type { Checklist as ChecklistType, TaskGuide, ArtifactGuide } from "@/types";

function isExternalUrl(url: string): boolean {
  return /^https?:\/\//.test(url);
}

function humanize(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

interface GuidedChecklistProps {
  title: string;
  items: ChecklistType;
  guides?: Record<string, TaskGuide | ArtifactGuide>;
}

export function GuidedChecklist({ title, items, guides }: GuidedChecklistProps) {
  return (
    <div>
      <p className="mb-3 text-xs font-semibold text-slate-600">{title}</p>
      <ul className="grid gap-4">
        {Object.entries(items).map(([key, value]) => {
          const done = truthy(value);
          const guide = guides?.[key] as TaskGuide | undefined;

          if (!guide) {
            return (
              <li
                key={key}
                className="flex items-start gap-2 text-sm text-slate-700"
                aria-label={`${humanize(key)}: ${done ? "done" : "not done"}`}
              >
                <CheckCircle2
                  aria-hidden="true"
                  className={done ? "mt-0.5 h-4 w-4 text-emerald-600" : "mt-0.5 h-4 w-4 text-slate-300"}
                />
                <span>{humanize(key)}</span>
              </li>
            );
          }

          const blockers = (guide.depends_on || []).filter(
            (dep) => !truthy(items[dep]),
          );

          return (
            <li key={key} className="rounded-lg border border-slate-200 p-3">
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-start gap-2">
                  <CheckCircle2
                    aria-hidden="true"
                    className={
                      done
                        ? "mt-0.5 h-4 w-4 shrink-0 text-emerald-600"
                        : "mt-0.5 h-4 w-4 shrink-0 text-slate-300"
                    }
                  />
                  <div>
                    <span className="text-sm font-medium text-slate-900">
                      {humanize(key)}
                    </span>
                    <p className="mt-0.5 text-xs text-slate-500">
                      {guide.summary}
                    </p>
                  </div>
                </div>
                <span className="flex shrink-0 items-center gap-1 rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
                  <Clock className="h-3 w-3" aria-hidden="true" />
                  ~{guide.time_minutes}min
                </span>
              </div>

              {blockers.length > 0 ? (
                <p className="mt-2 flex items-center gap-1 text-xs text-amber-700">
                  <AlertTriangle className="h-3 w-3" aria-hidden="true" />
                  需要先完成: {blockers.map((b) => humanize(b)).join(", ")}
                </p>
              ) : null}

              {"steps" in guide && guide.steps.length > 0 ? (
                <ol className="mt-2 ml-6 list-decimal space-y-1 text-xs text-slate-700">
                  {guide.steps.map((step, i) => (
                    <li key={i}>{step}</li>
                  ))}
                </ol>
              ) : null}

              <div className="mt-2 rounded border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs text-emerald-800">
                <span aria-hidden="true">✅ </span><span>{guide.done_when}</span>
              </div>

              {guide.refs && guide.refs.length > 0 ? (
                <div className="mt-2 flex flex-wrap gap-2">
                  {guide.refs.map((ref, i) =>
                    isExternalUrl(ref.url) ? (
                      <a
                        key={i}
                        href={ref.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-xs text-blue-600 hover:underline"
                      >
                        <ExternalLink className="h-3 w-3" aria-hidden="true" />
                        {ref.title}
                      </a>
                    ) : (
                      <span
                        key={i}
                        className="inline-flex items-center gap-1 text-xs text-slate-500"
                      >
                        <FileText className="h-3 w-3" aria-hidden="true" />
                        {ref.title} <code className="text-[10px] text-slate-400">{ref.url}</code>
                      </span>
                    ),
                  )}
                </div>
              ) : null}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
