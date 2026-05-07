import "@testing-library/jest-dom";

// Recharts uses ResizeObserver, which jsdom doesn't ship.
class _ResizeObserver {
  observe() { /* noop */ }
  unobserve() { /* noop */ }
  disconnect() { /* noop */ }
}
// @ts-expect-error — installing on globalThis for jsdom
globalThis.ResizeObserver = globalThis.ResizeObserver ?? _ResizeObserver;
