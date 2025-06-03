import React from "react";
import { Card, CardContent } from "@/shared/ui/card";
import type { PerformanceMetrics } from "../core/PerformanceTracker";
import type { EventPerformanceMetrics } from "../types/eventMetrics";

interface PerformanceViewerProps {
  metrics: PerformanceMetrics;
  eventMetrics?: EventPerformanceMetrics;
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

interface EventMetricConfig {
  key: keyof EventPerformanceMetrics;
  label: string;
  format: (value: number) => string;
  className?: string | ((value: number) => string);
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

const EVENT_METRIC_DISPLAY_CONFIG: EventMetricConfig[] = [
  {
    key: "mouseMoveCallsPerSecond",
    label: "mouse:",
    format: (value: number) => `${value.toFixed(1)}/s`,
    className: (value: number) => {
      if (value > 100) return "text-red-400";
      if (value > 60) return "text-yellow-400";
      return "text-green-400";
    },
  },
  {
    key: "mouseDownCallsPerSecond",
    label: "click:",
    format: (value: number) => `${value.toFixed(1)}/s`,
  },
];

export const PerformanceViewer: React.FC<PerformanceViewerProps> = React.memo(
  ({ metrics, eventMetrics, isVisible }) => {
    if (!isVisible) return null;

    const hasEventMetrics =
      eventMetrics && Object.values(eventMetrics).some((v) => v > 0);

    return (
      <Card className="fixed top-4 left-4 z-50 w-32 bg-black/80 text-white border-gray-600">
        <CardContent className="p-3 space-y-1">
          {/* System Performance Metrics */}
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

          {/* Event Performance Metrics */}
          {hasEventMetrics && (
            <>
              <div className="border-t border-gray-600 my-2"></div>
              <div className="text-xs font-mono text-gray-300 mb-1">events</div>
              {EVENT_METRIC_DISPLAY_CONFIG.map((config) => {
                if (!eventMetrics || eventMetrics[config.key] === undefined) {
                  return null;
                }

                const value = eventMetrics[config.key] as number;
                const formattedValue = config.format(value);
                const className =
                  typeof config.className === "function"
                    ? config.className(value)
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
            </>
          )}
        </CardContent>
      </Card>
    );
  }
);
