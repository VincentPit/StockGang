/**
 * Jest config (T3a) — jsdom env, ts-jest preset, @/ path alias mirrors tsconfig.
 * Coverage threshold starts at 50% to ratchet up — keep CI honest while we add
 * tests rather than gating ship on a coverage we don't have yet.
 */
module.exports = {
  preset: "ts-jest",
  testEnvironment: "jsdom",
  setupFilesAfterEnv: ["<rootDir>/__tests__/setup.ts"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/$1",
    "\\.(css|scss|sass)$": "<rootDir>/__tests__/style-stub.js",
  },
  testMatch: ["<rootDir>/__tests__/**/*.test.ts?(x)"],
  transform: {
    "^.+\\.(ts|tsx)$": ["ts-jest", { tsconfig: { jsx: "react-jsx" } }],
  },
  collectCoverageFrom: [
    "components/**/*.{ts,tsx}",
    "lib/**/*.{ts,tsx}",
    "!**/*.d.ts",
  ],
  // T3a target — 50% lines/statements. Ratchet up as more behavioural tests
  // land. Branches/functions stay loose for now since smoke tests don't
  // exercise enough conditional code yet.
  coverageThreshold: {
    global: { branches: 20, functions: 30, lines: 50, statements: 47 },
  },
};
