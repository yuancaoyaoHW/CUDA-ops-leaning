import { RisksTile, OperatorsTile, LibraryTile, TagCoverageTile } from "./InsightTiles";
import type { DashboardData } from "@/types";

interface InsightRailProps {
  data: DashboardData;
  onEditOperator: (name: string) => void;
  onEditLibrary: (name: string) => void;
}

export function InsightRail({ data, onEditOperator, onEditLibrary }: InsightRailProps) {
  return (
    <aside className="grid gap-4">
      <RisksTile risks={data.risks} />
      <OperatorsTile operators={data.operator_maturity} onEdit={onEditOperator} />
      <LibraryTile libraries={data.gpu_libraries} onEdit={onEditLibrary} />
      <TagCoverageTile tags={data.tag_coverage} />
    </aside>
  );
}
