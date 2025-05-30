import { useState, useEffect } from "react";

/**
 * Utility functions for localStorage operations
 */
export class LocalStorageUtils {
  /**
   * Save a value to localStorage with error handling
   */
  static save<T>(key: string, value: T): void {
    try {
      const serializedValue = JSON.stringify(value);
      localStorage.setItem(key, serializedValue);
    } catch (error) {
      console.warn(`Failed to save to localStorage for key "${key}":`, error);
    }
  }

  /**
   * Load a value from localStorage with error handling
   */
  static load<T>(key: string, defaultValue: T): T {
    try {
      const item = localStorage.getItem(key);
      if (item === null) {
        return defaultValue;
      }
      return JSON.parse(item) as T;
    } catch (error) {
      console.warn(`Failed to load from localStorage for key "${key}":`, error);
      return defaultValue;
    }
  }

  /**
   * Remove a value from localStorage
   */
  static remove(key: string): void {
    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.warn(
        `Failed to remove from localStorage for key "${key}":`,
        error
      );
    }
  }

  /**
   * Check if localStorage is available
   */
  static isAvailable(): boolean {
    try {
      const testKey = "__localStorage_test__";
      localStorage.setItem(testKey, "test");
      localStorage.removeItem(testKey);
      return true;
    } catch {
      return false;
    }
  }
}

/**
 * Custom hook for persisted state using localStorage
 * Automatically saves state changes to localStorage and loads initial state from localStorage
 */
export function usePersistedState<T>(
  key: string,
  defaultValue: T,
  options: {
    serialize?: (value: T) => string;
    deserialize?: (value: string) => T;
    storage?: "localStorage" | "sessionStorage";
  } = {}
): [T, React.Dispatch<React.SetStateAction<T>>] {
  const {
    serialize = JSON.stringify,
    deserialize = JSON.parse,
    storage = "localStorage",
  } = options;

  const storageObject =
    storage === "localStorage" ? localStorage : sessionStorage;

  // Initialize state from storage or use default
  const [state, setState] = useState<T>(() => {
    try {
      const item = storageObject.getItem(key);
      if (item === null) {
        return defaultValue;
      }
      return deserialize(item);
    } catch (error) {
      console.warn(`Failed to load persisted state for key "${key}":`, error);
      return defaultValue;
    }
  });

  // Save to storage whenever state changes
  useEffect(() => {
    try {
      storageObject.setItem(key, serialize(state));
    } catch (error) {
      console.warn(`Failed to persist state for key "${key}":`, error);
    }
  }, [key, state, serialize, storageObject]);

  return [state, setState];
}

/**
 * Configuration manager for multiple related settings
 */
export class ConfigurationManager<T extends Record<string, unknown>> {
  private readonly prefix: string;

  constructor(prefix: string) {
    this.prefix = prefix;
  }

  /**
   * Save entire configuration object
   */
  saveConfig(config: T): void {
    LocalStorageUtils.save(this.prefix, config);
  }

  /**
   * Load entire configuration object
   */
  loadConfig(defaultConfig: T): T {
    return LocalStorageUtils.load(this.prefix, defaultConfig);
  }

  /**
   * Save a specific configuration value
   */
  saveValue<K extends keyof T>(key: K, value: T[K]): void {
    const currentConfig = this.loadConfig({} as T);
    const updatedConfig = { ...currentConfig, [key]: value };
    this.saveConfig(updatedConfig);
  }

  /**
   * Load a specific configuration value
   */
  loadValue<K extends keyof T>(key: K, defaultValue: T[K]): T[K] {
    const config = this.loadConfig({} as T);
    return config[key] !== undefined ? config[key] : defaultValue;
  }

  /**
   * Clear all configuration
   */
  clearConfig(): void {
    LocalStorageUtils.remove(this.prefix);
  }
}
