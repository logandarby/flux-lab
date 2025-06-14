import { useCallback, useRef, useEffect } from "react";
import { COLOR_PRESETS } from "../core/app.constants";
import { Checkbox } from "@/shared/ui/checkbox";

interface ColorPickerProps {
  selectedColorIndex: number;
  onColorSelect: (colorIndex: number) => void;
  isVisible: boolean;
  onVisibilityChange: (visible: boolean) => void;
  autoHideDelay?: number;
  lavaLampMode: boolean;
  onLavaLampModeChange: (enabled: boolean) => void;
}

export function ColorPicker({
  selectedColorIndex,
  onColorSelect,
  isVisible,
  onVisibilityChange,
  autoHideDelay = 3000,
  lavaLampMode,
  onLavaLampModeChange,
}: ColorPickerProps) {
  const timeoutRef = useRef<number | null>(null);

  // Show the color picker temporarily
  const showTemporarily = useCallback(() => {
    onVisibilityChange(true);

    // Clear existing timeout
    if (timeoutRef.current) {
      window.clearTimeout(timeoutRef.current);
    }

    // Hide after specified delay
    timeoutRef.current = window.setTimeout(() => {
      onVisibilityChange(false);
    }, autoHideDelay);
  }, [onVisibilityChange, autoHideDelay]);

  // Handle color selection
  const handleColorSelect = useCallback(
    (colorIndex: number) => {
      onColorSelect(colorIndex);
      showTemporarily();
    },
    [onColorSelect, showTemporarily]
  );

  // Handle lava lamp mode toggle
  const handleLavaLampToggle = useCallback(() => {
    onLavaLampModeChange(!lavaLampMode);
    showTemporarily();
  }, [lavaLampMode, onLavaLampModeChange, showTemporarily]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        window.clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  // Expose showTemporarily method to parent component
  useEffect(() => {
    // This effect allows the parent to trigger the show temporarily behavior
    // by changing the isVisible prop to true
    if (isVisible && timeoutRef.current === null) {
      showTemporarily();
    }
  }, [isVisible, showTemporarily]);

  return (
    <div
      className={`fixed bottom-6 left-1/2 transform -translate-x-1/2 z-10 transition-all duration-500 ease-in-out ${
        isVisible
          ? "opacity-100 translate-y-0"
          : "opacity-0 translate-y-4 pointer-events-none"
      }`}
    >
      <div className="bg-black/80 backdrop-blur-md border border-neutral-700 rounded-2xl px-6 py-4 shadow-2xl">
        <div className="space-y-4">
          {/* Color Presets */}
          <div className="grid grid-cols-8 gap-3">
            {COLOR_PRESETS.map((preset, index) => (
              <button
                key={preset.name}
                onClick={() => handleColorSelect(index)}
                className={`w-10 h-10 rounded-full border-2 transition-all duration-200 hover:scale-110 focus:outline-none focus:ring-2 focus:ring-white/50 ${
                  selectedColorIndex === index
                    ? "border-white shadow-lg shadow-white/20"
                    : "border-neutral-600 hover:border-neutral-400"
                }`}
                style={{
                  backgroundColor: `rgb(${preset.color.map((c) => Math.floor(c * 255)).join(", ")})`,
                }}
                title={preset.name}
              />
            ))}
          </div>

          {/* Lava Lamp Mode Toggle */}
          <div className="flex items-center justify-center pt-2 border-t border-neutral-700">
            <label className="flex items-center justify-center gap-2 cursor-pointer">
              <Checkbox
                checked={lavaLampMode}
                onCheckedChange={handleLavaLampToggle}
                className="border-neutral-600 hover:cursor-pointer"
              />
              <span
                className={`text-sm font-medium transition-colors text-neutral-100 vertical-align-middle`}
                style={{ fontFamily: "Baskervville, serif" }}
              >
                Lava Lamp
              </span>
            </label>
          </div>
        </div>
      </div>
    </div>
  );
}
