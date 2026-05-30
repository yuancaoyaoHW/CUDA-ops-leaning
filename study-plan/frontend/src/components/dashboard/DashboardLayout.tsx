import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";

interface DashboardLayoutProps {
  title: string;
  onRefresh: () => void;
  children: ReactNode;
}

export function DashboardLayout({ title, onRefresh, children }: DashboardLayoutProps) {
  return (
    <main className="mx-auto min-h-screen max-w-7xl px-5 py-6 sm:px-7">
      <header className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs font-bold uppercase tracking-wider text-blue-700">LLM Kernel Lab</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-normal text-slate-950">{title}</h1>
          <p className="mt-1 max-w-3xl text-sm text-slate-600">
            本地可编辑进度面板。启动 Python dashboard server 后可保存到 progress.yaml。
          </p>
        </div>
        <Button variant="outline" onClick={onRefresh}>Refresh Data</Button>
      </header>
      {children}
    </main>
  );
}
