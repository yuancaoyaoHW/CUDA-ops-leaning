import { CalendarDays, ClipboardList, Cpu, LibraryBig, AlertTriangle, Tags, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

export type View = "focus" | "plan" | "operators" | "libraries" | "risks" | "tags";

interface SidebarProps {
  active: View;
  onChange: (view: View) => void;
  onRefresh: () => void;
}

const navItems: { id: View; label: string; icon: React.ReactNode }[] = [
  { id: "focus", label: "Focus", icon: <CalendarDays className="h-5 w-5" /> },
  { id: "plan", label: "Plan", icon: <ClipboardList className="h-5 w-5" /> },
  { id: "operators", label: "Operators", icon: <Cpu className="h-5 w-5" /> },
  { id: "libraries", label: "Libraries", icon: <LibraryBig className="h-5 w-5" /> },
  { id: "risks", label: "Risks", icon: <AlertTriangle className="h-5 w-5" /> },
  { id: "tags", label: "Tags", icon: <Tags className="h-5 w-5" /> },
];

export function Sidebar({ active, onChange, onRefresh }: SidebarProps) {
  return (
    <aside className="flex h-screen w-16 flex-col items-center border-r border-slate-200 bg-slate-50 py-4 lg:w-52">
      <div className="mb-6 px-3">
        <p className="hidden text-xs font-bold uppercase tracking-wider text-blue-700 lg:block">
          LLM Kernel Lab
        </p>
        <p className="text-xs font-bold uppercase tracking-wider text-blue-700 lg:hidden">LKL</p>
      </div>

      <nav className="flex flex-1 flex-col gap-1 px-2">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onChange(item.id)}
            className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
              active === item.id
                ? "bg-blue-100 text-blue-700"
                : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
            }`}
            aria-current={active === item.id ? "page" : undefined}
          >
            {item.icon}
            <span className="hidden lg:inline">{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="px-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={onRefresh}
          className="flex items-center gap-2 text-slate-600"
          aria-label="Refresh data"
        >
          <RefreshCw className="h-4 w-4" />
          <span className="hidden lg:inline">Refresh</span>
        </Button>
      </div>
    </aside>
  );
}
