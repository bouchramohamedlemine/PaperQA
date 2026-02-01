import { NextRequest, NextResponse } from "next/server";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "";

export async function POST(req: NextRequest) {
  try {
    const { functionName, body } = (await req.json()) as {
      functionName?: string;
      body?: Record<string, unknown>;
    };

    if (!functionName || !body) {
      return NextResponse.json(
        { error: "Missing functionName or body" },
        { status: 400 }
      );
    }

    if (!supabaseUrl || !supabaseAnonKey) {
      return NextResponse.json(
        { error: "Missing NEXT_PUBLIC_SUPABASE_URL or NEXT_PUBLIC_SUPABASE_ANON_KEY" },
        { status: 500 }
      );
    }

    if (functionName === "Generator") {
      console.log("[api/invoke] Payload passed to Generator:", JSON.stringify(body, null, 2));
    }

    const url = `${supabaseUrl.replace(/\/$/, "")}/functions/v1/${functionName}`;
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${supabaseAnonKey}`,
      },
      body: JSON.stringify(body),
    });

    const data = await res.json().catch(() => null);

    if (!res.ok) {
      return NextResponse.json(
        { error: (data && typeof data === "object" && "error" in data ? (data as { error: string }).error : null) ?? res.statusText ?? "Request failed", status: res.status },
        { status: res.status }
      );
    }

    return NextResponse.json(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
