import { useEffect, useState } from "react";
import { FolderOpen, FileText } from "lucide-react";
import { getModules, type ModuleInfo } from "@/api";
import { MarkdownViewer } from "./MarkdownViewer";

const MODULE_LABELS: Record<string, string> = {
  benchmarks: "Benchmarks 基准测试",
  evaluation: "Evaluation 面试评估",
  interview: "Interview 面试准备",
  plans: "Plans 学习计划",
  projects: "Projects 项目规划",
  third_round: "Third Round 第三轮准备",
  verification: "Verification 验证审计",
};

const VIEWABLE_EXTS = [".md", ".py", ".txt", ".csv", ".yaml", ".yml"];

function isViewable(filename: string) {
  return VIEWABLE_EXTS.some((ext) => filename.endsWith(ext));
}

export function ModulesView() {
  const [modules, setModules] = useState<ModuleInfo[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [viewingFile, setViewingFile] = useState<string | null>(null);

  useEffect(() => {
    getModules().then(setModules).catch(() => {});
  }, []);

  if (viewingFile) {
    return <MarkdownViewer path={viewingFile} onBack={() => setViewingFile(null)} />;
  }

  return (
    <div className="grid gap-4">
      {modules.map((m) => (
        <div key={m.name} className="rounded-lg border border-slate-200 bg-white shadow-sm">
          <button
            className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-slate-50"
            onClick={() => setExpanded(expanded === m.name ? null : m.name)}
          >
            <FolderOpen className="h-5 w-5 text-blue-600" />
            <span className="flex-1 font-medium text-slate-900">
              {MODULE_LABELS[m.name] || m.name}
            </span>
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
              {m.file_count} files
            </span>
          </button>
          {expanded === m.name && (
            <ul className="border-t border-slate-100 px-4 py-2">
              {m.files.map((f) => {
                const fullPath = `${m.name}/${f}`;
                const clickable = isViewable(f);
                return (
                  <li key={f} className="flex items-center gap-2 py-1 text-sm">
                    <FileText className="h-3.5 w-3.5 text-slate-400" />
                    {clickable ? (
                      <button
                        className="text-blue-600 hover:underline"
                        onClick={() => setViewingFile(fullPath)}
                      >
                        {f}
                      </button>
                    ) : (
                      <span className="text-slate-700">{f}</span>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      ))}
    </div>
  );
}
