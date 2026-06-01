import { useEffect, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ArrowLeft } from "lucide-react";
import { getFileContent } from "@/api";

interface Props {
  path: string;
  onBack: () => void;
}

export function MarkdownViewer({ path, onBack }: Props) {
  const [content, setContent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setContent(null);
    setError(null);
    getFileContent(path)
      .then(setContent)
      .catch((e) => setError(e.message));
  }, [path]);

  return (
    <div>
      <button
        onClick={onBack}
        className="mb-4 flex items-center gap-1 text-sm text-blue-600 hover:text-blue-800"
      >
        <ArrowLeft className="h-4 w-4" /> 返回模块列表
      </button>
      <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-lg font-semibold text-slate-800">{path}</h2>
        {error && <p className="text-red-600">{error}</p>}
        {content === null && !error && <p className="text-slate-500">加载中...</p>}
        {content !== null && (
          <article className="prose prose-slate max-w-none prose-pre:overflow-x-auto prose-table:w-full">
            <Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown>
          </article>
        )}
      </div>
    </div>
  );
}
