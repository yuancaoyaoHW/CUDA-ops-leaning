import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { formatStatus, type DashboardFilters } from "@/dashboardModel";
import type { DashboardData } from "@/types";

interface PlanFiltersProps {
  data: DashboardData;
  filters: DashboardFilters;
  onChange: (filters: DashboardFilters) => void;
}

export function PlanFilters({ data, filters, onChange }: PlanFiltersProps) {
  return (
    <div className="grid gap-3 md:grid-cols-[130px_150px_130px_minmax(220px,1fr)]">
      <FilterSelect
        label="Week"
        value={filters.week}
        values={["all", "current", ...data.weeks.map((week) => String(week.num))]}
        labels={{ all: "All Weeks", current: "Current Week" }}
        onValueChange={(week) => onChange({ ...filters, week })}
      />
      <FilterSelect
        label="Status"
        value={filters.status}
        values={["all", ...data.options.day_statuses]}
        labels={{ all: "All Statuses" }}
        formatter={formatStatus}
        onValueChange={(status) => onChange({ ...filters, status })}
      />
      <FilterSelect
        label="Tag"
        value={filters.tag}
        values={["all", ...data.options.tags]}
        labels={{ all: "All Tags" }}
        onValueChange={(tag) => onChange({ ...filters, tag })}
      />
      <label className="grid gap-1 text-xs font-semibold text-slate-600">
        Search
        <span className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <Input
            className="pl-9"
            value={filters.query}
            placeholder="row_softmax, Nsight, blocked..."
            onChange={(event) => onChange({ ...filters, query: event.target.value })}
          />
        </span>
      </label>
    </div>
  );
}

function FilterSelect({
  label,
  value,
  values,
  labels = {},
  formatter,
  onValueChange,
}: {
  label: string;
  value: string;
  values: string[];
  labels?: Record<string, string>;
  formatter?: (value: string) => string;
  onValueChange: (value: string) => void;
}) {
  return (
    <label className="grid gap-1 text-xs font-semibold text-slate-600">
      {label}
      <Select value={value} onValueChange={onValueChange}>
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {values.map((item) => (
            <SelectItem key={item} value={item}>
              {labels[item] || formatter?.(item) || (label === "Week" ? `Week ${item}` : item)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </label>
  );
}
