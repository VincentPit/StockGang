/** @type {import('jest').Config} */
const config = {
  testEnvironment: "jsdom",
  transform: {
    "^.+\\.(ts|tsx)$": ["ts-jest", { tsconfig: { jsx: "react-jsx" } }],
  },
  moduleNameMapper: {
    // Handle @/ path alias used in Next.js
    "^@/(.*)$": "<rootDir>/$1",
    // Stub out CSS modules and static assets
    "\\.(css|less|scss|sass)$": "<rootDir>/__mocks__/styleMock.js",
    "\\.(jpg|jpeg|png|gif|svg|ico)$": "<rootDir>/__mocks__/fileMock.js",
  },
  setupFilesAfterEnv: ["@testing-library/jest-dom"],
  testMatch: [
    "**/__tests__/**/*.{ts,tsx}",
    "**/*.test.{ts,tsx}",
    "**/*.spec.{ts,tsx}",
  ],
  collectCoverageFrom: [
    "components/**/*.{ts,tsx}",
    "lib/**/*.{ts,tsx}",
    "!**/*.d.ts",
  ],
};

module.exports = config;
