import { FormEvent, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Textarea } from "@/components/ui/textarea";
import { formatStatus } from "@/dashboardModel";
import type { DashboardData, DashboardDay, DayUpdates, GpuLibraryInfo, OperatorInfo } from "@/types";

export type EditTarget =
  | { type: "day"; day: DashboardDay }
  | { type: "operator"; name: string; operator: OperatorInfo }
  | { type: "library"; name: string; library: GpuLibraryInfo }
  | null;

interface EditDrawerProps {
  open: boolean;
  target: EditTarget;
  options: DashboardData["options"];
  onOpenChange: (open: boolean) => void;
  onSaveDay: (day: number, updates: DayUpdates) => Promise<void>;
  onSaveOperator: (name: string, updates: any) => Promise<void>;
  onSaveLibrary: (name: string, updates: any) => Promise<void>;
}

export function EditDrawer({
  open,
  target,
  options,
  onOpenChange,
  onSaveDay,
  onSaveOperator,
  onSaveLibrary,
}: EditDrawerProps) {
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setError(null);
    setSaving(false);
  }, [target]);

  async function submitDay(event: FormEvent<HTMLFormElement>, day: DashboardDay) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const updates: DayUpdates = {
      status: String(form.get("status") || day.status) as DayUpdates["status"],
      date: String(form.get("date") || ""),
      daily_check: Number(form.get("daily_check") || 0),
      verification: String(form.get("verification") || ""),
      weaknesses: String(form.get("weaknesses") || ""),
      next_fix: String(form.get("next_fix") || ""),
      notes: String(form.get("notes") || ""),
      tasks: readChecks(form, "tasks", day.tasks || {}),
      artifacts: readChecks(form, "artifacts", day.artifacts || {}),
    };
    try {
      setError(null);
      setSaving(true);
      await onSaveDay(day.num, updates);
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  async function submitOperator(event: FormEvent<HTMLFormElement>, name: string, operator: OperatorInfo) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    try {
      setError(null);
      setSaving(true);
      await onSaveOperator(name, {
        status: String(form.get("status") || operator.status),
        artifacts: readChecks(form, "artifacts", operator.artifacts || {}),
        notes: String(form.get("notes") || ""),
      });
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  async function submitLibrary(event: FormEvent<HTMLFormElement>, name: string, library: GpuLibraryInfo) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    try {
      setError(null);
      setSaving(true);
      await onSaveLibrary(name, {
        status: String(form.get("status") || library.status),
        evidence: String(form.get("evidence") || "")
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean),
      });
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-xl">
        {!target ? null : target.type === "day" ? (
          <form onSubmit={(event) => void submitDay(event, target.day)} className="grid gap-4">
            <SheetHeader>
              <SheetTitle>Day {String(target.day.num).padStart(2, "0")} · {target.day.title}</SheetTitle>
            </SheetHeader>
            {error ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p> : null}
            <Field label="Status">
              <Select name="status" defaultValue={target.day.status}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {options.day_statuses.map((status) => (
                    <SelectItem key={status} value={status}>{formatStatus(status)}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            <Field label="Date"><Input name="date" type="date" defaultValue={target.day.date || ""} /></Field>
            <Field label="Daily Check"><Input name="daily_check" type="number" min={0} max={3} defaultValue={target.day.daily_check || 0} /></Field>
            <Checklist name="tasks" title="Tasks" values={target.day.tasks || {}} />
            <Checklist name="artifacts" title="Artifacts" values={target.day.artifacts || {}} />
            <TextField label="Verification" name="verification" value={target.day.verification || ""} />
            <TextField label="Weaknesses" name="weaknesses" value={target.day.weaknesses || ""} />
            <TextField label="Next Fix" name="next_fix" value={target.day.next_fix || ""} />
            <TextField label="Notes" name="notes" value={target.day.notes || ""} />
            <div className="sticky bottom-0 -mx-6 flex justify-end gap-2 border-t bg-white px-6 py-4">
              <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
              <Button type="submit" disabled={saving}>{saving ? "Saving..." : "Save Day"}</Button>
            </div>
          </form>
        ) : target.type === "operator" ? (
          <form onSubmit={(event) => void submitOperator(event, target.name, target.operator)} className="grid gap-4">
            <SheetHeader><SheetTitle>Operator · {target.name}</SheetTitle></SheetHeader>
            {error ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p> : null}
            <Field label="Status">
              <Select name="status" defaultValue={target.operator.status}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {options.operator_statuses.map((status) => (
                    <SelectItem key={status} value={status}>{formatStatus(status)}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            <Checklist name="artifacts" title="Artifacts" values={target.operator.artifacts || {}} />
            <TextField label="Notes" name="notes" value={target.operator.notes || ""} />
            <div className="sticky bottom-0 -mx-6 flex justify-end gap-2 border-t bg-white px-6 py-4">
              <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
              <Button type="submit" disabled={saving}>{saving ? "Saving..." : "Save Operator"}</Button>
            </div>
          </form>
        ) : (
          <form onSubmit={(event) => void submitLibrary(event, target.name, target.library)} className="grid gap-4">
            <SheetHeader><SheetTitle>GPU Library · {target.name}</SheetTitle></SheetHeader>
            {error ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p> : null}
            <Field label="Status">
              <Select name="status" defaultValue={target.library.status}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {options.library_statuses.map((status) => (
                    <SelectItem key={status} value={status}>{formatStatus(status)}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            <TextField label="Evidence" name="evidence" value={(target.library.evidence || []).join("\n")} />
            <div className="sticky bottom-0 -mx-6 flex justify-end gap-2 border-t bg-white px-6 py-4">
              <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
              <Button type="submit" disabled={saving}>{saving ? "Saving..." : "Save Library"}</Button>
            </div>
          </form>
        )}
      </SheetContent>
    </Sheet>
  );
}

function readChecks(form: FormData, name: string, source: Record<string, unknown>): Record<string, boolean> {
  const checked = new Set(form.getAll(name).map(String));
  return Object.fromEntries(Object.keys(source).map((key) => [key, checked.has(key)]));
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="grid gap-1 text-sm font-medium text-slate-700">{label}{children}</label>;
}

function TextField({ label, name, value }: { label: string; name: string; value: string }) {
  return (
    <label className="grid gap-1 text-sm font-medium text-slate-700">
      {label}
      <Textarea name={name} defaultValue={value} aria-label={label} />
    </label>
  );
}

function Checklist({ name, title, values }: { name: string; title: string; values: Record<string, unknown> }) {
  return (
    <fieldset className="grid gap-2">
      <legend className="text-sm font-medium text-slate-700">{title}</legend>
      {Object.entries(values).map(([key, value]) => (
        <label key={key} className="flex items-start gap-2 text-sm text-slate-700">
          <Checkbox name={name} value={key} defaultChecked={Boolean(value)} aria-label={key} />
          <span>{key}</span>
        </label>
      ))}
    </fieldset>
  );
}
