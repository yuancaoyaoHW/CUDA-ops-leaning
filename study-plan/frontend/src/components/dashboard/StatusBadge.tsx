import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { formatStatus } from "@/dashboardModel";

const tone: Record<string, string> = {
  done: "border-emerald-200 bg-emerald-50 text-emerald-700",
  in_progress: "border-amber-200 bg-amber-50 text-amber-700",
  blocked: "border-red-200 bg-red-50 text-red-700",
  skipped: "border-red-200 bg-red-50 text-red-700",
  not_started: "border-slate-200 bg-slate-100 text-slate-600",
};

export function StatusBadge({ status }: { status?: string }) {
  return (
    <Badge variant="outline" className={cn("rounded-full font-medium", tone[status || "not_started"])}>
      {formatStatus(status)}
    </Badge>
  );
}
