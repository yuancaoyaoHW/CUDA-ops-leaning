import { useState } from "react";
import { ExternalLink, Plus, Pencil, Trash2, X, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { Reference } from "@/types";

const CATEGORIES = ["paper", "blog", "docs", "video", "repo", "other"];

interface ReferencesViewProps {
  references: Reference[];
  onAdd: (ref: Omit<Reference, "id">) => Promise<void>;
  onUpdate: (ref: Reference) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}

export function ReferencesView({ references, onAdd, onUpdate, onDelete }: ReferencesViewProps) {
  const [adding, setAdding] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [filterCategory, setFilterCategory] = useState("all");

  const filtered = filterCategory === "all"
    ? references
    : references.filter((r) => r.category === filterCategory);

  const grouped = filtered.reduce<Record<string, Reference[]>>((acc, ref) => {
    const cat = ref.category || "other";
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(ref);
    return acc;
  }, {});

  return (
    <div className="grid gap-4">
      {/* Header actions */}
      <div className="flex items-center justify-between gap-3">
        <Select value={filterCategory} onValueChange={setFilterCategory}>
          <SelectTrigger className="w-40">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Categories</SelectItem>
            {CATEGORIES.map((cat) => (
              <SelectItem key={cat} value={cat}>{categoryLabel(cat)}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button onClick={() => setAdding(true)} disabled={adding} size="sm">
          <Plus className="mr-1 h-4 w-4" />
          Add Reference
        </Button>
      </div>

      {/* Add form */}
      {adding && (
        <ReferenceForm
          onSave={async (ref) => {
            await onAdd(ref);
            setAdding(false);
          }}
          onCancel={() => setAdding(false)}
        />
      )}

      {/* Grouped list */}
      {Object.keys(grouped).length === 0 && !adding && (
        <Card className="shadow-sm">
          <CardContent className="py-8 text-center text-sm text-slate-500">
            No references yet. Click "Add Reference" to get started.
          </CardContent>
        </Card>
      )}

      {Object.entries(grouped).map(([category, refs]) => (
        <Card key={category} className="shadow-sm">
          <CardHeader className="pb-2">
            <h3 className="text-sm font-semibold text-slate-700">{categoryLabel(category)}</h3>
          </CardHeader>
          <CardContent className="grid gap-1">
            {refs.map((ref) =>
              editingId === ref.id ? (
                <ReferenceForm
                  key={ref.id}
                  initial={ref}
                  onSave={async (updated) => {
                    await onUpdate({ ...updated, id: ref.id });
                    setEditingId(null);
                  }}
                  onCancel={() => setEditingId(null)}
                />
              ) : (
                <ReferenceRow
                  key={ref.id}
                  ref_={ref}
                  onEdit={() => setEditingId(ref.id)}
                  onDelete={() => onDelete(ref.id)}
                />
              ),
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function ReferenceRow({
  ref_,
  onEdit,
  onDelete,
}: {
  ref_: Reference;
  onEdit: () => void;
  onDelete: () => void;
}) {
  return (
    <div className="group flex items-center gap-3 rounded-lg px-3 py-2 hover:bg-slate-50">
      <a
        href={ref_.url}
        target="_blank"
        rel="noopener noreferrer"
        className="flex min-w-0 flex-1 items-center gap-2 text-sm text-blue-700 hover:underline"
      >
        <ExternalLink className="h-3.5 w-3.5 shrink-0" />
        <span className="truncate">{ref_.title}</span>
      </a>
      {ref_.notes && (
        <span className="hidden truncate text-xs text-slate-500 lg:inline">{ref_.notes}</span>
      )}
      <div className="flex shrink-0 items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
        <Button variant="ghost" size="sm" onClick={onEdit} aria-label={`Edit ${ref_.title}`}>
          <Pencil className="h-3.5 w-3.5" />
        </Button>
        <Button variant="ghost" size="sm" onClick={onDelete} aria-label={`Delete ${ref_.title}`}>
          <Trash2 className="h-3.5 w-3.5 text-red-500" />
        </Button>
      </div>
    </div>
  );
}

function ReferenceForm({
  initial,
  onSave,
  onCancel,
}: {
  initial?: Reference;
  onSave: (ref: Omit<Reference, "id">) => Promise<void>;
  onCancel: () => void;
}) {
  const [title, setTitle] = useState(initial?.title || "");
  const [url, setUrl] = useState(initial?.url || "");
  const [category, setCategory] = useState(initial?.category || "paper");
  const [notes, setNotes] = useState(initial?.notes || "");
  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim() || !url.trim()) return;
    setSaving(true);
    try {
      await onSave({ title: title.trim(), url: url.trim(), category, notes: notes.trim() || undefined });
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="rounded-lg border border-blue-200 bg-blue-50/50 p-3">
      <div className="grid gap-2 sm:grid-cols-[1fr_1fr_120px]">
        <Input
          placeholder="Title"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          required
        />
        <Input
          placeholder="URL"
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          required
        />
        <Select value={category} onValueChange={setCategory}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {CATEGORIES.map((cat) => (
              <SelectItem key={cat} value={cat}>{categoryLabel(cat)}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="mt-2">
        <Input
          placeholder="Notes (optional)"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
        />
      </div>
      <div className="mt-2 flex items-center gap-2">
        <Button type="submit" size="sm" disabled={saving || !title.trim() || !url.trim()}>
          <Check className="mr-1 h-3.5 w-3.5" />
          {initial ? "Update" : "Add"}
        </Button>
        <Button type="button" variant="ghost" size="sm" onClick={onCancel}>
          <X className="mr-1 h-3.5 w-3.5" />
          Cancel
        </Button>
      </div>
    </form>
  );
}

function categoryLabel(cat: string): string {
  const labels: Record<string, string> = {
    paper: "📄 Papers",
    blog: "📝 Blogs",
    docs: "📚 Documentation",
    video: "🎬 Videos",
    repo: "💻 Repositories",
    other: "🔗 Other",
  };
  return labels[cat] || cat;
}
