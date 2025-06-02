import React from "react";
import { Card, CardContent } from "@/shared/ui/card";
import type { PerformanceMetrics } from "../core/PerformanceTracker";

interface PerformanceViewerProps {
  metrics: PerformanceMetrics;
  isVisible: boolean;
}

interface MetricConfig {
  key: keyof PerformanceMetrics;
  label: string;
  format: (value: number | boolean, metrics: PerformanceMetrics) => string;
  condition?: (metrics: PerformanceMetrics) => boolean;
  className?:
    | string
    | ((value: number | boolean, metrics: PerformanceMetrics) => string);
}

const METRIC_DISPLAY_CONFIG: MetricConfig[] = [
  {
    key: "fps",
    label: "fps:",
    format: (value: number | boolean) => `${(value as number).toFixed(0)}`,
    className: (value: number | boolean) => {
      const fps = value as number;
      if (fps >= 55) return "text-green-400";
      if (fps >= 30) return "text-yellow-400";
      return "text-red-400";
    },
  },
  {
    key: "js",
    label: "js:",
    format: (value: number | boolean) => `${(value as number).toFixed(1)}ms`,
  },
  {
    key: "gpu",
    label: "gpu:",
    format: (value: number | boolean, metrics: PerformanceMetrics) =>
      metrics.gpuSupported ? `${(value as number).toFixed(1)}µs` : "N/A",
    condition: (metrics) => metrics.gpuSupported || true, // Always show, but with conditional formatting
  },
  {
    key: "timestep",
    label: "dt:",
    format: (value: number | boolean) => `${(value as number).toFixed(1)}ms`,
  },
];

export const PerformanceViewer: React.FC<PerformanceViewerProps> = ({
  metrics,
  isVisible,
}) => {
  if (!isVisible) return null;

  return (
    <Card className="fixed top-4 left-4 z-50 w-32 bg-black/80 text-white border-gray-600">
      <CardContent className="p-3 space-y-1">
        {METRIC_DISPLAY_CONFIG.map((config) => {
          // Check condition if specified
          if (config.condition && !config.condition(metrics)) {
            return null;
          }

          const value = metrics[config.key];
          const formattedValue = config.format(value, metrics);
          const className =
            typeof config.className === "function"
              ? config.className(value, metrics)
              : config.className;

          return (
            <div
              key={config.key}
              className="flex justify-between text-xs font-mono"
            >
              <span>{config.label}</span>
              <span className={className}>{formattedValue}</span>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
};
