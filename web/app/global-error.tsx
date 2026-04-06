"use client";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html>
      <body style={{ background: "#111", color: "#eee", padding: 40, fontFamily: "monospace" }}>
        <h1 style={{ color: "#f87171" }}>Client Error Caught</h1>
        <pre style={{ whiteSpace: "pre-wrap", color: "#fbbf24", fontSize: 14 }}>
          {error.message}
        </pre>
        <pre style={{ whiteSpace: "pre-wrap", color: "#9ca3af", fontSize: 12, marginTop: 16 }}>
          {error.stack}
        </pre>
        {error.digest && (
          <p style={{ color: "#6b7280", marginTop: 12 }}>Digest: {error.digest}</p>
        )}
        <button
          onClick={() => reset()}
          style={{
            marginTop: 24,
            padding: "8px 16px",
            background: "#4f46e5",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
          }}
        >
          Try again
        </button>
      </body>
    </html>
  );
}
