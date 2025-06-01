import React from "react";
import { Card, CardContent } from "@/shared/ui/card";
import type { PerformanceMetrics } from "../core/PerformanceTracker";

interface PerformanceViewerProps {
  metrics: PerformanceMetrics;
  isVisible: boolean;
}

export const PerformanceViewer: React.FC<PerformanceViewerProps> = ({
  metrics,
  isVisible,
}) => {
  if (!isVisible) return null;

  return (
    <Card className="fixed top-4 left-4 z-50 w-32 bg-black/80 text-white border-gray-600">
      <CardContent className="p-3 space-y-1">
        <div className="flex justify-between text-xs font-mono">
          <span>js:</span>
          <span>{metrics.js.toFixed(1)}ms</span>
        </div>
        <div className="flex justify-between text-xs font-mono">
          <span>fps:</span>
          <span>{metrics.fps.toFixed(0)}</span>
        </div>
      </CardContent>
    </Card>
  );
};
