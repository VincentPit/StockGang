/**
 * /api/healthz — Next.js liveness probe (T3c).
 *
 * Returns 200 + a small JSON payload as long as the Node server is up.
 * Used by docker-compose healthcheck for the `web` service so the
 * orchestrator restarts the container if the process becomes unresponsive.
 */
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  return NextResponse.json({ status: "ok", uptime_s: process.uptime() });
}
