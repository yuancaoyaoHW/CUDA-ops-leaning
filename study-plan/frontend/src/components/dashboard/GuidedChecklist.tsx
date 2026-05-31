import { useState } from "react";
import { CheckCircle2, Clock, AlertTriangle, ExternalLink, FileText, ChevronDown, ChevronRight } from "lucide-react";
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
      <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500">{title}</p>
      <ul className="grid gap-2">
        {Object.entries(items).map(([key, value]) => {
          const done = truthy(value);
          const guide = guides?.[key] as TaskGuide | undefined;

          if (!guide) {
            return (
              <li
                key={key}
                className="flex items-center gap-3 rounded-lg border border-slate-100 bg-white px-3 py-2.5"
                aria-label={`${humanize(key)}: ${done ? "done" : "not done"}`}
              >
                <CheckCircle2
                  aria-hidden="true"
                  className={done ? "h-4 w-4 shrink-0 text-emerald-600" : "h-4 w-4 shrink-0 text-slate-300"}
                />
                <span className={`text-sm ${done ? "text-slate-500 line-through" : "text-slate-800"}`}>
                  {humanize(key)}
                </span>
              </li>
            );
          }

          return <GuidedItem key={key} itemKey={key} done={done} guide={guide} items={items} />;
        })}
      </ul>
    </div>
  );
}

function GuidedItem({
  itemKey,
  done,
  guide,
  items,
}: {
  itemKey: string;
  done: boolean;
  guide: TaskGuide;
  items: ChecklistType;
}) {
  const [expanded, setExpanded] = useState(false);
  const blockers = (guide.depends_on || []).filter((dep) => !truthy(items[dep]));
  const hasSteps = "steps" in guide && guide.steps.length > 0;

  return (
    <li className={`rounded-lg border bg-white transition-shadow ${done ? "border-emerald-200 bg-emerald-50/30" : "border-slate-200"} ${expanded ? "shadow-sm" : ""}`}>
      {/* Header row — always visible */}
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-3 px-3 py-3 text-left"
        aria-expanded={expanded}
      >
        <CheckCircle2
          aria-hidden="true"
          className={done ? "h-4 w-4 shrink-0 text-emerald-600" : "h-4 w-4 shrink-0 text-slate-300"}
        />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className={`text-sm font-medium ${done ? "text-slate-500 line-through" : "text-slate-900"}`}>
              {humanize(itemKey)}
            </span>
            {blockers.length > 0 && (
              <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-700">
                <AlertTriangle className="h-2.5 w-2.5" aria-hidden="true" />
                blocked
              </span>
            )}
          </div>
          <p className="mt-0.5 truncate text-xs text-slate-500">{guide.summary}</p>
        </div>
        <span className="flex shrink-0 items-center gap-1 text-xs tabular-nums text-slate-400">
          <Clock className="h-3 w-3" aria-hidden="true" />
          {guide.time_minutes}m
        </span>
        {hasSteps ? (
          expanded ? (
            <ChevronDown className="h-4 w-4 shrink-0 text-slate-400" aria-hidden="true" />
          ) : (
            <ChevronRight className="h-4 w-4 shrink-0 text-slate-400" aria-hidden="true" />
          )
        ) : (
          <span className="w-4 shrink-0" />
        )}
      </button>

      {/* Expanded details */}
      {expanded && (
        <div className="border-t border-slate-100 px-3 pb-3 pt-2">
          {blockers.length > 0 && (
            <p className="mb-2 flex items-center gap-1 text-xs text-amber-700">
              <AlertTriangle className="h-3 w-3" aria-hidden="true" />
              需要先完成: {blockers.map((b) => humanize(b)).join(", ")}
            </p>
          )}

          {hasSteps && (
            <ol className="mb-2 ml-5 list-decimal space-y-1 text-xs text-slate-700">
              {guide.steps.map((step, i) => (
                <li key={i}>{step}</li>
              ))}
            </ol>
          )}

          <div className="rounded border border-emerald-200 bg-emerald-50 px-2 py-1.5 text-xs text-emerald-800">
            <span aria-hidden="true">✅ </span><span>{guide.done_when}</span>
          </div>

          {guide.refs && guide.refs.length > 0 && (
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
          )}
        </div>
      )}
    </li>
  );
}
