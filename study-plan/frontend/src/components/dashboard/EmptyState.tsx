import type { ReactNode } from "react";

interface EmptyStateProps {
  title: string;
  children: ReactNode;
  role?: "alert" | "status";
}

export function EmptyState({ title, children, role }: EmptyStateProps) {
  return (
    <section
      className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm"
      role={role}
    >
      <h2 className="text-lg font-semibold text-slate-950">{title}</h2>
      <div className="mt-1 text-sm text-slate-600">{children}</div>
    </section>
  );
}
