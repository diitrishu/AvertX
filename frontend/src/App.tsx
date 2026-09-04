import { useState, useRef, useCallback, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, PieChart, Pie,
} from "recharts";

// ─── API base ───────────────────────────────────────────────────────
const API = "/api";

// ─── Auth ───────────────────────────────────────────────────────────
// Real accounts, not a demo toggle: JWT issued by POST /auth/login or
// /auth/signup, stored client-side, sent as a Bearer token on every
// request. Every signup is forced to Reporter server-side -- role
// elevation only happens through an Admin (see AdminUsersScreen).
type Role = "Reporter" | "Supervisor" | "HSE" | "Admin";

interface AuthUser {
  id: number;
  name: string;
  email: string;
  role: Role;
  site?: string;
}

const AUTH_TOKEN_KEY = "sif_token";
const AUTH_USER_KEY = "sif_user";

function loadStoredAuth(): { token: string; user: AuthUser } | null {
  try {
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    const userRaw = localStorage.getItem(AUTH_USER_KEY);
    if (!token || !userRaw) return null;
    return { token, user: JSON.parse(userRaw) as AuthUser };
  } catch {
    return null;
  }
}

function storeAuth(token: string, user: AuthUser) {
  try {
    localStorage.setItem(AUTH_TOKEN_KEY, token);
    localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
  } catch { /* localStorage unavailable — session still works in-memory */ }
}

function clearStoredAuth() {
  try {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(AUTH_USER_KEY);
  } catch { /* ignore */ }
}

// Module-level token used by apiFetch. Kept in sync with the AuthProvider's
// state on every login/logout so every screen's fetch calls (most of which
// were written before auth existed) get the header without being rewired
// one-by-one.
let currentToken: string | null = null;
function setCurrentToken(t: string | null) { currentToken = t; }

// Same pattern for the logged-in user — screens deep in the tree (e.g.
// ReportsScreen deciding whether to show HSE status controls) read this
// instead of threading the user through every prop list.
let currentUser: AuthUser | null = null;
function setCurrentUser(u: AuthUser | null) { currentUser = u; }
function getCurrentUser(): AuthUser | null { return currentUser; }

async function apiFetch(path: string, opts: RequestInit = {}): Promise<Response> {
  const headers = new Headers(opts.headers || {});
  if (currentToken) headers.set("Authorization", `Bearer ${currentToken}`);
  const res = await fetch(`${API}${path}`, { ...opts, headers });
  if (res.status === 401) {
    // Session expired or invalid — force back to login rather than let
    // every screen fail silently with a stale token.
    clearStoredAuth();
    setCurrentToken(null);
    window.dispatchEvent(new CustomEvent("navigate", { detail: "login" }));
  }
  return res;
}

// ─── Types ──────────────────────────────────────────────────────────
type Screen = "landing" | "login" | "signup" | "dashboard" | "analyze" | "reports" | "bulk" | "admin";

// Real offshore photography for the cover screen — sourced from Unsplash,
// free to use. A dark navy gradient overlay (matching the palette below)
// unifies it to the app's brand regardless of the photo's native color grade.
const HERO_IMAGE = "https://images.unsplash.com/photo-1609337231803-2adad48ea1d1?auto=format&fit=crop&w=1800&q=80";

interface DashboardStats {
  total_reports: number;
  sif_yes: number;
  sif_no: number;
  sif_density: number;
  critical_open?: number;
  by_status?: Record<string, number>;
  by_rule: { rule: string; count: number }[];
  by_site: { site: string; total: number; sif_count: number; density: number }[];
  by_activity: { activity: string; total: number; sif_count: number; density: number }[];
  top_precursors: { site: string; activity: string; rule: string; count: number }[];
}

interface AnalysisResult {
  sif_potential: string;
  confidence: number;
  risk_score: number;
  life_saving_rule: string;
  rule_confidence: number;
  top_contributing_phrases: [string, number][];
  typical_hazard: string;
  typical_barrier: string;
  risk_level: string;
  extracted_text?: string;
}

interface ReportRow {
  id: number;
  report_id: string;
  report_text: string;
  date: string;
  site: string;
  activity: string;
  report_type: string;
  sif_potential: string;
  life_saving_rule: string;
  confidence: number;
  risk_score: number;
  risk_level: string;
  reporter_id?: number;
  status?: string;
  critical?: number;
}

// ─── Color utils ────────────────────────────────────────────────────
const riskColor = (risk: string) => {
  if (risk === "High") return { bg: "rgba(251,59,92,0.14)", text: "#FB7185", border: "#FB3B5C" };
  if (risk === "Medium") return { bg: "rgba(245,158,11,0.14)", text: "#FBBF24", border: "#F59E0B" };
  return { bg: "rgba(16,185,129,0.14)", text: "#34D399", border: "#10B981" };
};

// Three-way: Yes (red/danger), NEEDS_REVIEW (amber/uncertain -- must never
// look like a safe "No"), No (green/safe). See backend/sif_output_contract.py.
const sifColor = (sif: string) => {
  if (sif === "Yes") return { bg: "rgba(251,59,92,0.14)", text: "#FB7185", border: "#FB3B5C" };
  if (sif === "NEEDS_REVIEW") return { bg: "rgba(245,158,11,0.14)", text: "#FBBF24", border: "#F59E0B" };
  return { bg: "rgba(16,185,129,0.14)", text: "#34D399", border: "#10B981" };
};

// Human-readable label for a raw sif_potential value.
const sifLabel = (sif: string) => {
  if (sif === "Yes") return "YES -- SIF Potential";
  if (sif === "NEEDS_REVIEW") return "NEEDS REVIEW -- Uncertain";
  return "NO -- Not SIF";
};

const densityColor = (d: number) => {
  if (d >= 40) return "#FB3B5C";
  if (d >= 25) return "#F59E0B";
  return "#10B981";
};

// ─── Shared components ──────────────────────────────────────────────
function Badge({ children, colors }: { children: React.ReactNode; colors: { bg: string; text: string; border: string } }) {
  return (
    <span className="inline-flex items-center px-2 py-0.5 text-[11px] font-semibold tracking-wide uppercase rounded"
      style={{ backgroundColor: colors.bg, color: colors.text, border: `1px solid ${colors.border}` }}>
      {children}
    </span>
  );
}

function DensityBar({ value }: { value: number }) {
  const color = densityColor(value);
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full" style={{ backgroundColor: "#141A22" }}>
        <div className="h-full rounded-full transition-all"
          style={{ width: `${Math.min(value, 100)}%`, backgroundColor: color }} />
      </div>
      <span className="mono text-xs w-10 text-right" style={{ color, fontWeight: 600 }}>
        {value.toFixed(1)}%
      </span>
    </div>
  );
}

function KPICard({ label, value, sub, color }: { label: string; value: string; sub: string; color?: string }) {
  return (
    <div className="flex flex-col gap-2 px-5 py-4 rounded-sm"
      style={{ background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)" }}>
      <span className="text-[11px] font-semibold uppercase tracking-widest" style={{ color: "#64748B" }}>{label}</span>
      <span className="mono text-3xl font-semibold leading-none"
        style={{ color: color ?? "#F1F5F9", letterSpacing: "-0.02em" }}>{value}</span>
      <span className="text-[11px]" style={{ color: "#64748B" }}>{sub}</span>
    </div>
  );
}

// Compact secondary stat, used beside the big hero number instead of a
// row of identical boxes — typography carries the hierarchy, not borders.
function StatChip({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="mono text-xl font-semibold leading-none" style={{ color: color ?? "#F1F5F9" }}>{value}</span>
      <span className="text-[10px] uppercase tracking-widest" style={{ color: "#64748B" }}>{label}</span>
    </div>
  );
}

// Semicircular risk gauge — replaces a flat progress bar with something
// that actually reads as "instrument", matching the safety-critical tone.
function RiskGauge({ value, size = 128 }: { value: number; size?: number }) {
  const r = size / 2 - 10;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = Math.PI * r; // half circle
  const pct = Math.max(0, Math.min(100, value)) / 100;
  const color = value >= 70 ? "#FB3B5C" : value >= 40 ? "#F59E0B" : "#10B981";
  const label = value >= 70 ? "HIGH" : value >= 40 ? "MEDIUM" : "LOW";
  // Arc runs from 180deg (left) to 0deg (right) along the top half.
  const describeArc = () => {
    const startAngle = Math.PI; // 180deg
    const endAngle = Math.PI - Math.PI * pct;
    const x1 = cx + r * Math.cos(startAngle);
    const y1 = cy + r * Math.sin(startAngle);
    const x2 = cx + r * Math.cos(endAngle);
    const y2 = cy + r * Math.sin(endAngle);
    const largeArc = pct > 0.5 ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`;
  };
  return (
    <div className="flex flex-col items-center" style={{ width: size }}>
      <svg width={size} height={size / 2 + 12} viewBox={`0 0 ${size} ${size / 2 + 12}`}>
        <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 1 1 ${cx + r} ${cy}`}
          fill="none" stroke="#141A22" strokeWidth={10} strokeLinecap="round" />
        {pct > 0 && (
          <path d={describeArc()} fill="none" stroke={color} strokeWidth={10} strokeLinecap="round"
            className="gauge-arc" style={{ ["--gauge-full" as string]: circumference, ["--gauge-value" as string]: circumference * (1 - pct) }} />
        )}
        <text x={cx} y={cy - 4} textAnchor="middle" className="mono" fontSize={size * 0.22} fontWeight={700} fill={color}>
          {Math.round(value)}
        </text>
      </svg>
      <span className="text-[10px] font-semibold uppercase tracking-widest -mt-1" style={{ color }}>{label} RISK</span>
    </div>
  );
}

// Auto-generated headline insight — the single most urgent finding,
// surfaced above everything else so the dashboard leads with "what matters"
// instead of a wall of equally-weighted numbers.
function HeroInsight({ stats }: { stats: DashboardStats }) {
  const top = stats.top_precursors?.[0];
  if (!top) return null;
  const urgent = top.count >= 10;
  return (
    <div className="fade-in rounded-sm px-6 py-5 relative overflow-hidden"
      style={{
        background: "linear-gradient(135deg, #1A0B0B 0%, #0A0E14 60%)",
        border: `1px solid ${urgent ? "rgba(251,59,92,0.45)" : "rgba(34,211,238,0.16)"}`,
      }}>
      <div className="flex items-start gap-4">
        <div className="flex items-center justify-center w-9 h-9 rounded-sm shrink-0 pulse-glow"
          style={{ background: "rgba(251,59,92,0.14)", border: "1px solid rgba(251,59,92,0.45)" }}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FB7185" strokeWidth="2">
            <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-[11px] font-semibold uppercase tracking-widest mb-1" style={{ color: "#FB7185" }}>
            Top Priority · Recurring SIF Precursor
          </div>
          <div className="text-xl font-semibold leading-snug" style={{ color: "#F1F5F9", letterSpacing: "-0.01em" }}>
            {top.rule} failures at <span style={{ color: "#FB7185" }}>{top.site}</span> during {top.activity.toLowerCase()}
            <span className="mono ml-2 text-base font-normal" style={{ color: "#94A3B8" }}>— {top.count} reports</span>
          </div>
          <div className="text-sm mt-1.5" style={{ color: "#94A3B8" }}>
            Priority intervention: review {top.rule.toLowerCase()} controls for {top.activity.toLowerCase()} at {top.site} before the next work order is issued.
          </div>
        </div>
      </div>
    </div>
  );
}

const TABLE_HEAD = "text-[11px] font-semibold uppercase tracking-widest text-left py-2.5 px-4";
const TABLE_CELL = "py-2.5 px-4 text-sm";

// Compact per-rule glyphs for the bottom category-card row — small, single
// idea each, matching the icon-per-category pattern of dense ops dashboards.
const RULE_ICONS: Record<string, React.ReactNode> = {
  "Energy Isolation": <path d="M7 11V7a5 5 0 0 1 10 0v4M5 11h14v10H5z" />,
  "Line of Fire": <path d="M12 2 1 21h22L12 2zM12 9v5M12 17h.01" />,
  "Confined Space": <path d="M21 8 12 3 3 8v8l9 5 9-5V8zM3 8l9 5 9-5M12 13v8" />,
  "Hot Work": <path d="M12 2c1 3-2 4-2 7a4 4 0 0 0 8 0c0-1-.5-2-1-3 1 1 2 3 2 5a6 6 0 1 1-12 0c0-4 3-5 5-9z" />,
  "Working at Height": <path d="M12 19V5M5 12l7-7 7 7" />,
  "Safe Mechanical Lifting": <path d="M8 7a4 4 0 0 1 8 0v3a4 4 0 0 1-8 0zM8 14a4 4 0 0 0 8 0v3a4 4 0 0 1-8 0z" />,
  "Driving": <path d="M5 17h14M6 17l1.5-5h9L18 17M9 10V6h6v4M5 17a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3zM19 17a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3z" />,
  "Work Authorisation": <path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2M9 5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2M9 5a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2m-6 9 2 2 4-4" />,
  "Bypassing Safety Controls": <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10zM9.5 9.5l5 5m0-5-5 5" />,
};

const RULE_ORDER = [
  "Energy Isolation", "Line of Fire", "Confined Space", "Hot Work", "Working at Height",
  "Safe Mechanical Lifting", "Driving", "Work Authorisation", "Bypassing Safety Controls",
];

const DONUT_COLORS = ["#FB3B5C", "#F59E0B", "#22D3EE", "#8B5CF6", "#34D399", "#FBBF24", "#F472B6", "#38BDF8", "#A3E635"];

// Sites arranged as a colored tile grid, sized/tinted by volume and density —
// stands in for a literal geo-map (no lat/long data exists for these sites)
// while still giving the dashboard a big scannable "where is risk" panel.
function SiteRiskGrid({ sites }: { sites: DashboardStats["by_site"] }) {
  const rows = sites.filter(s => s.site).sort((a, b) => b.total - a.total).slice(0, 12);
  const maxTotal = Math.max(...rows.map(r => r.total), 1);
  return (
    <div className="grid grid-cols-3 sm:grid-cols-4 gap-2.5 h-full content-start">
      {rows.map(row => {
        const color = densityColor(row.density);
        const scale = 0.55 + 0.45 * (row.total / maxTotal);
        return (
          <div key={row.site} className="rounded-sm px-3 flex flex-col justify-center relative overflow-hidden"
            style={{
              background: `color-mix(in srgb, ${color} 10%, #0A0E14)`,
              border: `1px solid color-mix(in srgb, ${color} 40%, transparent)`,
              minHeight: 64 * scale,
            }}>
            <div className="text-[11px] font-semibold leading-tight truncate" style={{ color: "#F1F5F9" }}>{row.site}</div>
            <div className="flex items-baseline gap-1.5 mt-1">
              <span className="mono font-bold" style={{ fontSize: 18, color }}>{row.total}</span>
              <span className="text-[10px]" style={{ color: "#64748B" }}>reports</span>
            </div>
            <div className="mono text-[10px] font-semibold" style={{ color }}>{row.density.toFixed(0)}% density</div>
          </div>
        );
      })}
    </div>
  );
}

// ─── Dashboard Screen ───────────────────────────────────────────────
function DashboardScreen() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedRule, setSelectedRule] = useState<string>("All");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await apiFetch(`/dashboard/stats`);
      if (!res.ok) {
        setError(res.status === 404
          ? "This endpoint isn't available yet — the backend server likely needs a restart to pick up the latest code."
          : `Could not load the dashboard (HTTP ${res.status}).`);
        setLoading(false);
        return;
      }
      setStats(await res.json());
    } catch (e) {
      console.error(e);
      setError("Could not reach the server.");
    }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  if (loading && !stats) {
    return <div className="flex items-center justify-center h-full" style={{ color: "#64748B" }}>Loading dashboard...</div>;
  }
  if (error) {
    return (
      <div className="flex items-center justify-center h-full p-6">
        <div className="max-w-md text-center text-sm px-5 py-4 rounded-sm"
          style={{ background: "rgba(251,59,92,0.08)", border: "1px solid rgba(251,59,92,0.3)", color: "#FB7185" }}>
          {error}
        </div>
      </div>
    );
  }
  if (!stats) return null;

  const ruleData = stats.by_rule.filter(r => r.rule !== "Unmapped" && r.rule !== "N/A" && r.rule !== "None");

  return (
    <div className="flex flex-col gap-5 p-6 overflow-y-auto h-full">
      {/* Hero insight — leads the page, F/Z-pattern top priority */}
      <HeroInsight stats={stats} />

      {/* Rule filter tabs — narrows the Immediate Action list below and
          highlights the matching category everywhere else on the page. */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-1 fade-in-delay-1" style={{ scrollbarWidth: "none" }}>
        {["All", ...RULE_ORDER].map(rule => {
          const active = selectedRule === rule;
          return (
            <button key={rule} onClick={() => setSelectedRule(rule)}
              className="shrink-0 px-3 py-1.5 text-[11px] font-semibold rounded-sm transition-all whitespace-nowrap"
              style={{
                background: active ? "rgba(34,211,238,0.14)" : "#0A0E14",
                color: active ? "#22D3EE" : "#64748B",
                border: `1px solid ${active ? "rgba(34,211,238,0.5)" : "rgba(34,211,238,0.16)"}`,
              }}>
              {rule}
            </button>
          );
        })}
      </div>

      {/* Dense command-center grid: left stat rail + horizontal severity
          chart, center site-risk tile grid (stands in for a geo map — no
          lat/long data exists per site), right donut + top-rule legend. */}
      <div className="fade-in-delay-1 grid gap-4" style={{ gridTemplateColumns: "260px 1fr 280px" }}>
        {/* Left rail */}
        <div className="flex flex-col gap-3">
          <div className="rounded-sm px-4 py-3.5" style={{ background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)" }}>
            <div className="text-[11px] font-semibold uppercase tracking-widest mb-1" style={{ color: "#64748B" }}>Total Reports</div>
            <div className="mono font-bold leading-none" style={{ fontSize: 32, color: "#F1F5F9" }}>{stats.total_reports}</div>
          </div>
          <div className="rounded-sm px-4 py-3.5" style={{ background: "rgba(251,59,92,0.08)", border: "1px solid rgba(251,59,92,0.3)" }}>
            <div className="text-[11px] font-semibold uppercase tracking-widest mb-1" style={{ color: "#FB7185" }}>SIF Potential</div>
            <div className="mono font-bold leading-none" style={{ fontSize: 32, color: "#FB7185" }}>{stats.sif_yes}</div>
          </div>
          <div className="rounded-sm px-4 py-3.5" style={{ background: "rgba(16,185,129,0.06)", border: "1px solid rgba(16,185,129,0.25)" }}>
            <div className="text-[11px] font-semibold uppercase tracking-widest mb-1" style={{ color: "#34D399" }}>Non-SIF</div>
            <div className="mono font-bold leading-none" style={{ fontSize: 32, color: "#34D399" }}>{stats.sif_no}</div>
          </div>
          <div className="rounded-sm px-5 py-4" style={{ background: "#0A0E14", border: "1px solid rgba(245,158,11,0.3)" }}>
            <div className="text-[11px] font-semibold uppercase tracking-widest mb-1" style={{ color: "#FBBF24" }}>SIF Density</div>
            <div className="mono font-bold leading-none" style={{ fontSize: 40, letterSpacing: "-0.02em", color: "#FBBF24" }}>
              {stats.sif_density}<span style={{ fontSize: 20, color: "#94A3B8" }}>%</span>
            </div>
            <div className="text-[10px] mt-1 leading-snug" style={{ color: "#64748B" }}>carry genuine fatal-risk potential</div>
          </div>
        </div>

        {/* Center — site risk grid */}
        <div className="rounded-sm px-4 py-4" style={{ background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)" }}>
          <div className="flex items-center justify-between mb-3">
            <div className="text-sm font-semibold" style={{ color: "#F1F5F9" }}>Site Risk Map</div>
            <div className="text-[11px]" style={{ color: "#64748B" }}>sized by volume, tinted by density</div>
          </div>
          <SiteRiskGrid sites={stats.by_site} />
        </div>

        {/* Right — donut + legend */}
        <div className="rounded-sm px-4 py-4 flex flex-col" style={{ background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)" }}>
          <div className="text-sm font-semibold mb-1" style={{ color: "#F1F5F9" }}>SIF Share by Rule</div>
          <ResponsiveContainer width="100%" height={150}>
            <PieChart>
              <Pie data={ruleData} dataKey="count" nameKey="rule" innerRadius={38} outerRadius={58} paddingAngle={2}>
                {ruleData.map((entry, i) => (
                  <Cell key={i} fill={DONUT_COLORS[i % DONUT_COLORS.length]}
                    opacity={selectedRule === "All" || selectedRule === entry.rule ? 1 : 0.25} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ background: "#07090C", border: "1px solid rgba(34,211,238,0.35)", borderRadius: 4, color: "#F1F5F9", fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-col gap-1.5 mt-2 overflow-y-auto">
            {ruleData.slice(0, 6).map((r, i) => (
              <div key={r.rule} className="flex items-center gap-2 text-[11px]">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ background: DONUT_COLORS[i % DONUT_COLORS.length] }} />
                <span className="truncate flex-1" style={{ color: "#94A3B8" }}>{r.rule}</span>
                <span className="mono font-semibold" style={{ color: "#CBD5E1" }}>{r.count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom category-card row — one per Life-Saving Rule, click to filter
          the Immediate Action list below (same interaction as the tabs). */}
      <div className="fade-in-delay-2 grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))" }}>
        {RULE_ORDER.map(rule => {
          const entry = ruleData.find(r => r.rule === rule);
          const count = entry?.count ?? 0;
          const active = selectedRule === rule;
          return (
            <button key={rule} onClick={() => setSelectedRule(active ? "All" : rule)}
              className="rounded-sm px-3.5 py-3 flex items-start gap-2.5 text-left transition-all"
              style={{
                background: "#0A0E14",
                border: `1px solid ${active ? "rgba(34,211,238,0.5)" : "rgba(34,211,238,0.16)"}`,
              }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22D3EE" strokeWidth="2" className="shrink-0 mt-0.5">
                {RULE_ICONS[rule]}
              </svg>
              <div className="min-w-0">
                <div className="mono font-bold leading-none" style={{ fontSize: 20, color: "#F1F5F9" }}>{count}</div>
                <div className="text-[10px] leading-tight mt-1" style={{ color: "#64748B" }}>{rule}</div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Tables */}
      <div className="grid grid-cols-2 gap-4">
        {/* Site table */}
        <div className="rounded-sm overflow-hidden" style={{ background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)" }}>
          <div className="px-4 py-3 border-b" style={{ borderColor: "rgba(34,211,238,0.16)" }}>
            <div className="text-sm font-semibold" style={{ color: "#F1F5F9" }}>Site-wise Breakdown</div>
          </div>
          <table className="w-full">
            <thead style={{ background: "#07090C" }}>
              <tr>
                <th className={TABLE_HEAD} style={{ color: "#334155" }}>Site</th>
                <th className={`${TABLE_HEAD} text-right`} style={{ color: "#334155" }}>Total</th>
                <th className={`${TABLE_HEAD} text-right`} style={{ color: "#334155" }}>SIF</th>
                <th className={TABLE_HEAD} style={{ color: "#334155", minWidth: 120 }}>Density</th>
              </tr>
            </thead>
            <tbody>
              {stats.by_site.filter(s => s.site).map((row, i) => (
                <tr key={row.site} style={{ background: i % 2 === 1 ? "#0C1015" : "transparent", borderTop: "1px solid #161C24" }}>
                  <td className={TABLE_CELL} style={{ color: "#CBD5E1" }}>{row.site}</td>
                  <td className={`${TABLE_CELL} mono text-right`} style={{ color: "#94A3B8" }}>{row.total}</td>
                  <td className={`${TABLE_CELL} mono text-right font-semibold`} style={{ color: "#FB7185" }}>{row.sif_count}</td>
                  <td className={TABLE_CELL} style={{ minWidth: 120 }}><DensityBar value={row.density} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Activity table */}
        <div className="rounded-sm overflow-hidden" style={{ background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)" }}>
          <div className="px-4 py-3 border-b" style={{ borderColor: "rgba(34,211,238,0.16)" }}>
            <div className="text-sm font-semibold" style={{ color: "#F1F5F9" }}>Activity-wise Breakdown</div>
          </div>
          <table className="w-full">
            <thead style={{ background: "#07090C" }}>
              <tr>
                <th className={TABLE_HEAD} style={{ color: "#334155" }}>Activity</th>
                <th className={`${TABLE_HEAD} text-right`} style={{ color: "#334155" }}>Total</th>
                <th className={`${TABLE_HEAD} text-right`} style={{ color: "#334155" }}>SIF</th>
                <th className={TABLE_HEAD} style={{ color: "#334155", minWidth: 120 }}>Density</th>
              </tr>
            </thead>
            <tbody>
              {stats.by_activity.filter(a => a.activity).map((row, i) => (
                <tr key={row.activity} style={{ background: i % 2 === 1 ? "#0C1015" : "transparent", borderTop: "1px solid #161C24" }}>
                  <td className={TABLE_CELL} style={{ color: "#CBD5E1", maxWidth: 180, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{row.activity}</td>
                  <td className={`${TABLE_CELL} mono text-right`} style={{ color: "#94A3B8" }}>{row.total}</td>
                  <td className={`${TABLE_CELL} mono text-right font-semibold`} style={{ color: "#FB7185" }}>{row.sif_count}</td>
                  <td className={TABLE_CELL} style={{ minWidth: 120 }}><DensityBar value={row.density} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Immediate Action — recurring precursors reframed as a command-center
          action list: severity count badge, stacked cards, explicit next-step
          buttons. This is the section HSE actually acts from. */}
      {stats.top_precursors.length > 0 && (() => {
        const filtered = selectedRule === "All" ? stats.top_precursors : stats.top_precursors.filter(p => p.rule === selectedRule);
        const critical = filtered.filter(p => p.count >= 10).length;
        return (
          <div className="fade-in-delay-2">
            <div className="flex items-center gap-3 mb-3">
              <span className="w-1.5 h-1.5 rounded-full pulse-glow" style={{ background: "#FB3B5C", boxShadow: "0 0 6px #FB3B5C" }} />
              <div className="text-sm font-semibold tracking-wide uppercase" style={{ color: "#F1F5F9" }}>Immediate Action</div>
              {critical > 0 && (
                <span className="mono text-[11px] font-bold px-2 py-0.5 rounded-sm"
                  style={{ background: "rgba(251,59,92,0.16)", color: "#FB7185", border: "1px solid rgba(251,59,92,0.4)" }}>
                  {critical} CRITICAL
                </span>
              )}
              <span className="text-[11px] font-normal ml-auto" style={{ color: "#64748B" }}>
                {selectedRule === "All" ? "AI-identified recurring patterns" : `Filtered · ${selectedRule}`}
              </span>
            </div>
            {filtered.length === 0 ? (
              <div className="text-xs py-6 text-center rounded-sm" style={{ color: "#64748B", background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)" }}>
                No recurring precursors for {selectedRule} yet.
              </div>
            ) : (
            <div className="flex flex-col gap-2.5">
              {filtered.map((p, i) => {
                const sev = p.count >= 10 ? "High" : p.count >= 5 ? "Medium" : "Low";
                const colors = riskColor(sev);
                return (
                  <div key={i} className="rounded-sm px-4 py-3.5 flex items-center gap-4 flex-wrap"
                    style={{ background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)", borderLeft: `3px solid ${colors.border}` }}>
                    <div className="flex-1 min-w-[240px]">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-semibold leading-snug" style={{ color: "#F1F5F9" }}>
                          {p.rule} Failure Pattern
                        </span>
                        <Badge colors={colors}>{sev}</Badge>
                        <span className="mono text-[11px] font-semibold" style={{ color: colors.text }}>{p.count}&times;</span>
                      </div>
                      <div className="flex gap-3 text-[11px] mb-1.5" style={{ color: "#64748B" }}>
                        <span>{p.site}</span>
                        <span>&middot;</span>
                        <span>{p.activity}</span>
                      </div>
                      <div className="text-xs leading-relaxed" style={{ color: "#94A3B8" }}>
                        <span className="font-semibold" style={{ color: "#22D3EE" }}>AI Rec: </span>
                        Review {p.rule.toLowerCase()} controls for {p.activity.toLowerCase()} at {p.site}.
                      </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <button className="flex items-center gap-1.5 text-[11px] font-semibold px-3 py-1.5 rounded-sm transition-all hover:opacity-85"
                        style={{ background: "rgba(34,211,238,0.12)", color: "#22D3EE", border: "1px solid rgba(34,211,238,0.35)" }}
                        onClick={() => window.dispatchEvent(new CustomEvent("navigate", { detail: "reports" }))}>
                        View Reports
                      </button>
                      <button className="text-[11px] font-semibold px-3 py-1.5 rounded-sm transition-all hover:opacity-85"
                        style={{ background: "transparent", color: colors.text, border: `1px solid ${colors.border}` }}>
                        Flag for Audit
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
            )}
          </div>
        );
      })()}
    </div>
  );
}

// ─── Role-scoped dashboards ─────────────────────────────────────────
// The dashboard above (DashboardScreen) is org-wide HSE intelligence --
// SIF density, site risk, precursor analytics. A Reporter should never
// see that: they should see what happened to *their* reports. A
// Supervisor sits in between -- KPI tiles plus a review queue, not the
// full analytics suite. See the role/dashboard scoping discussion.
function greetingForNow(): string {
  const h = new Date().getHours();
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}

interface MyStats {
  total: number;
  under_review: number;
  action_in_progress: number;
  closed: number;
  recent: { report_id: string; report_type: string; status: string; date: string; id: number }[];
}

function ReporterDashboardScreen({ onNavigate }: { onNavigate: (s: Screen) => void }) {
  const user = getCurrentUser();
  const [stats, setStats] = useState<MyStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await apiFetch(`/dashboard/my-stats`);
      if (!res.ok) {
        // A stale/old backend process (missing this route) or a real
        // server error both land here -- never trust the body's shape
        // without checking status first, or a 404's {"detail":...} gets
        // treated as report data and crashes the render below.
        setError(res.status === 404
          ? "This endpoint isn't available yet — the backend server likely needs a restart to pick up the latest code."
          : `Could not load your dashboard (HTTP ${res.status}).`);
        setLoading(false);
        return;
      }
      setStats(await res.json());
    } catch (e) {
      console.error(e);
      setError("Could not reach the server.");
    }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  if (loading && !stats) {
    return <div className="flex items-center justify-center h-full" style={{ color: "#64748B" }}>Loading your reports…</div>;
  }
  if (error) {
    return (
      <div className="flex items-center justify-center h-full p-6">
        <div className="max-w-md text-center text-sm px-5 py-4 rounded-sm"
          style={{ background: "rgba(251,59,92,0.08)", border: "1px solid rgba(251,59,92,0.3)", color: "#FB7185" }}>
          {error}
        </div>
      </div>
    );
  }
  if (!stats) return null;

  const mostRecentOpen = stats.recent.find(r => r.status !== "Closed");

  return (
    <div className="flex flex-col gap-5 p-6 overflow-y-auto h-full max-w-3xl">
      <div className="fade-in">
        <div className="text-xl font-semibold" style={{ color: "#F1F5F9" }}>{greetingForNow()}{user ? `, ${user.name.split(" ")[0]}` : ""}</div>
        <div className="text-sm mt-0.5" style={{ color: "#64748B" }}>Here's what's happening with the safety issues you've reported.</div>
      </div>

      <button onClick={() => onNavigate("analyze")}
        className="self-start px-5 py-3 text-sm font-semibold rounded-sm transition-all hover:opacity-90"
        style={{ background: "linear-gradient(135deg, #0E7490, #0EA5C9)", color: "#F1F5F9" }}>
        + Report a Safety Issue
      </button>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 fade-in-delay-1">
        <KPICard label="Total" value={String(stats.total)} sub="reports submitted" />
        <KPICard label="Under Review" value={String(stats.under_review)} sub="with Supervisor/HSE" color="#FBBF24" />
        <KPICard label="Action in Progress" value={String(stats.action_in_progress)} sub="corrective action assigned" color="#C084FC" />
        <KPICard label="Closed" value={String(stats.closed)} sub="resolved" color="#34D399" />
      </div>

      <div className="rounded-sm overflow-hidden fade-in-delay-1" style={{ background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)" }}>
        <div className="px-4 py-3 text-sm font-semibold" style={{ borderBottom: "1px solid rgba(34,211,238,0.16)", color: "#F1F5F9" }}>Recent Reports</div>
        {stats.recent.length === 0 ? (
          <div className="text-center py-10 text-sm" style={{ color: "#334155" }}>You haven't submitted any reports yet.</div>
        ) : stats.recent.map(r => (
          <div key={r.id} className="flex items-center justify-between px-4 py-3" style={{ borderTop: "1px solid #161C24" }}>
            <div className="flex items-center gap-3">
              <span className="mono text-xs" style={{ color: "#22D3EE" }}>{r.report_id}</span>
              <span className="text-xs" style={{ color: "#94A3B8" }}>{r.report_type}</span>
            </div>
            <StatusBadge status={r.status} />
          </div>
        ))}
      </div>

      {mostRecentOpen && (
        <div className="rounded-sm px-4 py-3.5 text-sm fade-in-delay-2" style={{ background: "rgba(34,211,238,0.06)", border: "1px solid rgba(34,211,238,0.25)", color: "#94A3B8" }}>
          <span style={{ color: "#22D3EE", fontWeight: 600 }}>Important update — </span>
          Your report {mostRecentOpen.report_id} is currently <strong style={{ color: "#CBD5E1" }}>{mostRecentOpen.status || "Submitted"}</strong>.
        </div>
      )}
    </div>
  );
}

function SupervisorDashboardScreen({ onNavigate }: { onNavigate: (s: Screen) => void }) {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [queue, setQueue] = useState<ReportRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [statsRes, queueRes] = await Promise.all([
        apiFetch(`/dashboard/stats`),
        apiFetch(`/reports?per_page=6`),
      ]);
      if (!statsRes.ok || !queueRes.ok) {
        const badStatus = !statsRes.ok ? statsRes.status : queueRes.status;
        setError(badStatus === 404
          ? "This endpoint isn't available yet — the backend server likely needs a restart to pick up the latest code."
          : `Could not load the dashboard (HTTP ${badStatus}).`);
        setLoading(false);
        return;
      }
      setStats(await statsRes.json());
      const q = await queueRes.json();
      setQueue(q.reports || []);
    } catch (e) {
      console.error(e);
      setError("Could not reach the server.");
    }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  if (loading && !stats) {
    return <div className="flex items-center justify-center h-full" style={{ color: "#64748B" }}>Loading review queue…</div>;
  }
  if (error) {
    return (
      <div className="flex items-center justify-center h-full p-6">
        <div className="max-w-md text-center text-sm px-5 py-4 rounded-sm"
          style={{ background: "rgba(251,59,92,0.08)", border: "1px solid rgba(251,59,92,0.3)", color: "#FB7185" }}>
          {error}
        </div>
      </div>
    );
  }
  if (!stats) return null;

  const submitted = stats.by_status?.["Submitted"] || 0;
  const reviewed = stats.by_status?.["Reviewed"] || 0;

  return (
    <div className="flex flex-col gap-5 p-6 overflow-y-auto h-full">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 fade-in">
        <KPICard label="New Reports" value={String(submitted)} sub="not yet reviewed" />
        <KPICard label="Awaiting Review" value={String(submitted + reviewed)} sub="in Submitted or Reviewed" color="#FBBF24" />
        <KPICard label="Critical — Open" value={String(stats.critical_open ?? 0)} sub="SIF-potential, unresolved" color="#FB7185" />
      </div>

      <div className="rounded-sm overflow-hidden fade-in-delay-1" style={{ background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)" }}>
        <div className="flex items-center justify-between px-4 py-3" style={{ borderBottom: "1px solid rgba(34,211,238,0.16)" }}>
          <div className="text-sm font-semibold" style={{ color: "#F1F5F9" }}>Review Queue — critical first</div>
          <button onClick={() => onNavigate("reports")}
            className="text-[11px] font-semibold px-2.5 py-1 rounded-sm transition-all"
            style={{ background: "rgba(34,211,238,0.14)", color: "#22D3EE", border: "1px solid rgba(34,211,238,0.35)" }}>
            View full queue &rarr;
          </button>
        </div>
        {queue.length === 0 ? (
          <div className="text-center py-10 text-sm" style={{ color: "#334155" }}>Queue is empty.</div>
        ) : queue.map(r => (
          <div key={r.id} className="flex items-center justify-between px-4 py-3" style={{ borderTop: "1px solid #161C24" }}>
            <div className="flex items-center gap-3 min-w-0">
              {!!r.critical && <span title="Critical — immediate alert" className="pulse-glow shrink-0" style={{ color: "#FB3B5C" }}>&#9888;</span>}
              <span className="mono text-xs shrink-0" style={{ color: "#22D3EE" }}>{r.report_id}</span>
              <span className="text-xs truncate" style={{ color: "#94A3B8" }}>{r.site} &middot; {r.activity}</span>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <Badge colors={riskColor(r.risk_level || "Low")}>{r.risk_level || "Low"}</Badge>
              <StatusBadge status={r.status} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Analyze Report Screen ──────────────────────────────────────────
function AnalyzeScreen() {
  const [text, setText] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [site, setSite] = useState("");
  const [activity, setActivity] = useState("");
  const [activeTab, setActiveTab] = useState<"text" | "pdf">("text");
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleAnalyzeText = async () => {
    if (!text.trim()) return;
    setAnalyzing(true);
    try {
      const res = await apiFetch(`/analyze/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ report_text: text, site: site || "Unknown", activity: activity || "Unknown" }),
      });
      setResult(await res.json());
    } catch (e) { console.error(e); }
    setAnalyzing(false);
  };

  const handleAnalyzePdf = async () => {
    if (!pdfFile) return;
    setAnalyzing(true);
    try {
      const fd = new FormData();
      fd.append("file", pdfFile);
      fd.append("site", site || "Unknown");
      fd.append("activity", activity || "Unknown");
      const res = await apiFetch(`/analyze/pdf`, { method: "POST", body: fd });
      const data = await res.json();
      setResult(data);
      if (data.extracted_text) setText(data.extracted_text);
    } catch (e) { console.error(e); }
    setAnalyzing(false);
  };

  // Highlight LIME phrases in original text
  const renderHighlightedText = (originalText: string, phrases: [string, number][]) => {
    if (!phrases.length) return <span>{originalText}</span>;
    // Build a regex from the top positive phrases
    const positivePhrases = phrases.filter(([, w]) => w > 0).map(([w]) => w);
    if (!positivePhrases.length) return <span>{originalText}</span>;
    const escaped = positivePhrases.map(p => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const regex = new RegExp(`(${escaped.join("|")})`, "gi");
    const parts = originalText.split(regex);
    return (
      <>
        {parts.map((part, i) => {
          const isMatch = positivePhrases.some(p => p.toLowerCase() === part.toLowerCase());
          return isMatch ? (
            <mark key={i} style={{ backgroundColor: "rgba(248,113,113,0.25)", color: "#FB7185", padding: "1px 3px", borderRadius: 2, fontWeight: 600 }}>
              {part}
            </mark>
          ) : (
            <span key={i}>{part}</span>
          );
        })}
      </>
    );
  };

  return (
    <div className="flex flex-col gap-5 p-6 overflow-y-auto h-full max-w-5xl mx-auto w-full">
      {/* Input panel */}
      <div className="rounded-sm overflow-hidden" style={{ background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)" }}>
        <div className="px-5 py-3.5 border-b flex items-center justify-between" style={{ borderColor: "rgba(34,211,238,0.16)" }}>
          <div>
            <div className="text-sm font-semibold" style={{ color: "#F1F5F9" }}>Incident / Observation Report Input</div>
            <div className="text-[11px] mt-0.5" style={{ color: "#64748B" }}>Paste text or upload a PDF for AI analysis.</div>
          </div>
          {/* Tabs */}
          <div className="flex gap-1 p-0.5 rounded-sm" style={{ background: "#050608" }}>
            {(["text", "pdf"] as const).map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)}
                className="px-3 py-1 text-xs font-medium rounded-sm transition-all"
                style={{
                  background: activeTab === tab ? "rgba(34,211,238,0.16)" : "transparent",
                  color: activeTab === tab ? "#F1F5F9" : "#64748B",
                }}>
                {tab === "text" ? "Paste Text" : "Upload PDF"}
              </button>
            ))}
          </div>
        </div>
        <div className="p-4">
          {/* Site & Activity fields */}
          <div className="flex gap-3 mb-3">
            <input value={site} onChange={e => setSite(e.target.value)} placeholder="Site (e.g. Duliajan)"
              className="flex-1 rounded-sm text-sm px-3 py-2 outline-none"
              style={{ background: "#050608", border: "1px solid rgba(34,211,238,0.16)", color: "#CBD5E1" }} />
            <input value={activity} onChange={e => setActivity(e.target.value)} placeholder="Activity (e.g. Maintenance)"
              className="flex-1 rounded-sm text-sm px-3 py-2 outline-none"
              style={{ background: "#050608", border: "1px solid rgba(34,211,238,0.16)", color: "#CBD5E1" }} />
          </div>

          {activeTab === "text" ? (
            <>
              <textarea className="w-full rounded-sm text-sm resize-none outline-none focus:ring-1 placeholder-[#334155] leading-relaxed p-3"
                style={{ background: "#050608", border: "1px solid rgba(34,211,238,0.16)", color: "#CBD5E1", minHeight: 140, fontFamily: "IBM Plex Sans, sans-serif" }}
                placeholder="Paste incident report text here..."
                value={text} onChange={e => setText(e.target.value)} rows={6} />
              <div className="flex items-center justify-end gap-3 mt-3">
                <span className="mono text-[11px]" style={{ color: "#334155" }}>{text.length} chars</span>
                <button onClick={handleAnalyzeText} disabled={analyzing || !text.trim()}
                  className="flex items-center gap-2 px-5 py-2 text-sm font-semibold rounded-sm transition-all"
                  style={{
                    background: analyzing || !text.trim() ? "rgba(34,211,238,0.16)" : "#0E7490",
                    color: analyzing || !text.trim() ? "#334155" : "#F1F5F9",
                    cursor: analyzing || !text.trim() ? "not-allowed" : "pointer",
                    border: "1px solid", borderColor: analyzing || !text.trim() ? "rgba(34,211,238,0.16)" : "#22D3EE",
                  }}>
                  {analyzing ? "Analyzing..." : "Analyze Report"}
                </button>
              </div>
            </>
          ) : (
            <>
              <div className={`rounded-sm border-2 border-dashed p-8 text-center cursor-pointer transition-all ${dragOver ? "border-[#22D3EE] bg-[rgba(34,211,238,0.08)]" : "border-[rgba(34,211,238,0.16)] bg-[#050608]"}`}
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={e => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files[0]) { setPdfFile(e.dataTransfer.files[0]); setText(""); } }}
                onClick={() => fileRef.current?.click()}>
                <input ref={fileRef} type="file" accept=".pdf" className="hidden"
                  onChange={e => { if (e.target.files?.[0]) { setPdfFile(e.target.files[0]); setText(""); } }} />
                {pdfFile ? (
                  <div className="flex flex-col items-center gap-2">
                    <div className="text-sm font-semibold" style={{ color: "#F1F5F9" }}>{pdfFile.name}</div>
                    <div className="text-[11px]" style={{ color: "#64748B" }}>{(pdfFile.size / 1024).toFixed(0)} KB</div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-2">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#334155" strokeWidth="1.5">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                    <div className="text-sm" style={{ color: "#64748B" }}>Drag & drop a PDF, or click to browse</div>
                  </div>
                )}
              </div>
              <div className="flex items-center justify-end mt-3">
                <button onClick={handleAnalyzePdf} disabled={analyzing || !pdfFile}
                  className="flex items-center gap-2 px-5 py-2 text-sm font-semibold rounded-sm transition-all"
                  style={{
                    background: analyzing || !pdfFile ? "rgba(34,211,238,0.16)" : "#0E7490",
                    color: analyzing || !pdfFile ? "#334155" : "#F1F5F9",
                    cursor: analyzing || !pdfFile ? "not-allowed" : "pointer",
                    border: "1px solid", borderColor: analyzing || !pdfFile ? "rgba(34,211,238,0.16)" : "#22D3EE",
                  }}>
                  {analyzing ? "Analyzing..." : "Analyze PDF"}
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      {/* PDF extracted text preview (editable) */}
      {activeTab === "pdf" && result?.extracted_text && (
        <div className="rounded-sm overflow-hidden" style={{ background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)" }}>
          <div className="px-5 py-3 border-b" style={{ borderColor: "rgba(34,211,238,0.16)" }}>
            <div className="text-sm font-semibold" style={{ color: "#F1F5F9" }}>Extracted Text (editable -- fix OCR errors before re-analyzing)</div>
          </div>
          <div className="p-4">
            <textarea className="w-full rounded-sm text-sm resize-y outline-none p-3 leading-relaxed"
              style={{ background: "#050608", border: "1px solid rgba(34,211,238,0.16)", color: "#CBD5E1", minHeight: 120 }}
              value={text} onChange={e => setText(e.target.value)} rows={5} />
          </div>
        </div>
      )}

      {/* Results panel */}
      {result && (
        <div className="flex flex-col gap-4">
          {/* Top result bar */}
          <div className="fade-in rounded-sm px-5 py-4"
            style={{ background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)", borderLeft: `3px solid ${sifColor(result.sif_potential).border}` }}>
            <div className="flex items-center justify-between flex-wrap gap-6">
              <div className="flex items-center gap-6 flex-wrap">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-widest mb-1.5" style={{ color: "#64748B" }}>SIF Potential</div>
                  <div className="flex items-center gap-2">
                    <Badge colors={sifColor(result.sif_potential)}>
                      {sifLabel(result.sif_potential)}
                    </Badge>
                    <span className="mono text-sm font-semibold" style={{ color: sifColor(result.sif_potential).text }}>
                      {(result.confidence * 100).toFixed(1)}% confidence
                    </span>
                  </div>
                </div>
                <div className="w-px h-10" style={{ background: "rgba(34,211,238,0.16)" }} />
                {result.life_saving_rule !== "N/A" && (
                  <>
                    <div>
                      <div className="text-[11px] font-semibold uppercase tracking-widest mb-1.5" style={{ color: "#64748B" }}>Life-Saving Rule</div>
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-sm text-xs font-semibold"
                        style={result.life_saving_rule === "NEEDS_REVIEW"
                          ? { background: "rgba(245,158,11,0.14)", border: "1px solid #F59E0B", color: "#FBBF24" }
                          : { background: "rgba(34,211,238,0.08)", border: "1px solid #0E7490", color: "#22D3EE" }}>
                        {result.life_saving_rule === "NEEDS_REVIEW" ? "NEEDS REVIEW" : result.life_saving_rule}
                      </span>
                    </div>
                    <div className="w-px h-10" style={{ background: "rgba(34,211,238,0.16)" }} />
                  </>
                )}
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-widest mb-1.5" style={{ color: "#64748B" }}>Typical Hazard / Barrier</div>
                  <div className="text-xs" style={{ color: "#94A3B8" }}>{result.typical_hazard} &middot; {result.typical_barrier}</div>
                </div>
              </div>
              <RiskGauge value={result.risk_score} size={116} />
            </div>
          </div>

          {/* Highlighted report text */}
          {result.top_contributing_phrases.length > 0 && (
            <div className="rounded-sm px-4 py-4" style={{ background: "rgba(34,211,238,0.08)", border: "1px solid #0E7490" }}>
              <div className="text-[11px] font-semibold uppercase tracking-widest mb-3" style={{ color: "#22D3EE" }}>
                Report Text with AI-Highlighted Risk Indicators
              </div>
              <div className="text-sm leading-relaxed whitespace-pre-wrap" style={{ color: "#94A3B8" }}>
                {renderHighlightedText(text, result.top_contributing_phrases)}
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                {result.top_contributing_phrases.map(([word, weight], i) => (
                  <span key={i} className="mono text-[11px] px-2 py-0.5 rounded-sm"
                    style={{
                      background: weight > 0 ? "rgba(248,113,113,0.15)" : "rgba(74,222,128,0.15)",
                      color: weight > 0 ? "#FB7185" : "#34D399",
                      border: `1px solid ${weight > 0 ? "rgba(220,38,38,0.3)" : "rgba(22,163,74,0.3)"}`,
                    }}>
                    {word} ({weight > 0 ? "+" : ""}{(weight * 100).toFixed(1)}%)
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Reports Log Screen ─────────────────────────────────────────────
const REPORT_STATUSES = ["Submitted", "Reviewed", "Investigation", "Action Assigned", "Closed"] as const;

function StatusBadge({ status }: { status?: string }) {
  const s = status || "Submitted";
  const colors: Record<string, { bg: string; text: string; border: string }> = {
    Submitted: { bg: "rgba(148,163,184,0.14)", text: "#94A3B8", border: "#475569" },
    Reviewed: { bg: "rgba(59,130,246,0.14)", text: "#60A5FA", border: "#3B82F6" },
    Investigation: { bg: "rgba(245,158,11,0.14)", text: "#FBBF24", border: "#F59E0B" },
    "Action Assigned": { bg: "rgba(168,85,247,0.14)", text: "#C084FC", border: "#A855F7" },
    Closed: { bg: "rgba(16,185,129,0.14)", text: "#34D399", border: "#10B981" },
  };
  return <Badge colors={colors[s] || colors.Submitted}>{s}</Badge>;
}

function ReportsScreen() {
  const user = getCurrentUser();
  const canReview = !!user && user.role !== "Reporter";

  const [reports, setReports] = useState<ReportRow[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(1);
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState({ site: "", activity: "", sif_potential: "", life_saving_rule: "" });
  const [filterOpts, setFilterOpts] = useState<{ sites: string[]; activities: string[]; rules: string[] }>({ sites: [], activities: [], rules: [] });
  const [updatingId, setUpdatingId] = useState<number | null>(null);

  const loadFilters = useCallback(async () => {
    try {
      const res = await apiFetch(`/reports/filters`);
      setFilterOpts(await res.json());
    } catch (e) { console.error(e); }
  }, []);

  const loadReports = useCallback(async (p = page, f = filters) => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ page: String(p), per_page: "25" });
      if (f.site) params.set("site", f.site);
      if (f.activity) params.set("activity", f.activity);
      if (f.sif_potential) params.set("sif_potential", f.sif_potential);
      if (f.life_saving_rule) params.set("life_saving_rule", f.life_saving_rule);
      const res = await apiFetch(`/reports?${params}`);
      const data = await res.json();
      setReports(data.reports);
      setTotal(data.total);
      setPages(data.pages);
    } catch (e) { console.error(e); }
    setLoading(false);
  }, [page, filters]);

  // Load filter dropdown options once on mount.
  useEffect(() => {
    loadFilters();
  }, [loadFilters]);

  // Re-fetch reports whenever the page or any filter changes (dropdowns,
  // Clear button, and Prev/Next all just update state -- this effect is
  // the single place that actually talks to the API).
  useEffect(() => {
    loadReports(page, filters);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, filters]);

  const changeStatus = async (reportId: number, newStatus: string) => {
    setUpdatingId(reportId);
    try {
      await apiFetch(`/reports/${reportId}/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: newStatus }),
      });
      await loadReports(page, filters);
    } catch (e) { console.error(e); }
    setUpdatingId(null);
  };

  const SelFilter = ({ label, value, opts, onChange }: { label: string; value: string; opts: string[]; onChange: (v: string) => void }) => (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] font-semibold uppercase" style={{ color: "#334155" }}>{label}:</span>
      <select value={value} onChange={e => { onChange(e.target.value); setPage(1); }}
        className="text-xs px-2 py-1.5 rounded-sm outline-none"
        style={{ background: "#050608", border: "1px solid rgba(34,211,238,0.16)", color: value ? "#CBD5E1" : "#64748B", cursor: "pointer" }}>
        <option value="">All</option>
        {opts.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );

  return (
    <div className="flex flex-col gap-4 p-6 overflow-y-auto h-full">
      {/* Filter bar */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-[11px] font-semibold uppercase tracking-widest" style={{ color: "#64748B" }}>Filter:</span>
        <SelFilter label="Site" value={filters.site} opts={filterOpts.sites} onChange={v => setFilters(f => ({ ...f, site: v }))} />
        <SelFilter label="Activity" value={filters.activity} opts={filterOpts.activities} onChange={v => setFilters(f => ({ ...f, activity: v }))} />
        <SelFilter label="SIF" value={filters.sif_potential} opts={["Yes", "NEEDS_REVIEW", "No"]} onChange={v => setFilters(f => ({ ...f, sif_potential: v }))} />
        <SelFilter label="Rule" value={filters.life_saving_rule} opts={filterOpts.rules} onChange={v => setFilters(f => ({ ...f, life_saving_rule: v }))} />
        <button onClick={() => { setFilters({ site: "", activity: "", sif_potential: "", life_saving_rule: "" }); setPage(1); }}
          className="text-[11px] px-2 py-1 rounded-sm"
          style={{ background: "rgba(34,211,238,0.16)", color: "#64748B", border: "1px solid rgba(34,211,238,0.35)" }}>Clear</button>
        <div className="ml-auto mono text-xs" style={{ color: "#64748B" }}>
          {canReview ? "Reviewer inbox — critical first" : "My reports"} &middot; {total} reports
        </div>
      </div>

      {/* Table */}
      <div className="rounded-sm overflow-hidden" style={{ background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)" }}>
        <table className="w-full">
          <thead style={{ background: "#07090C" }}>
            <tr>
              {(["report_id", "date", "site", "activity", "report_type", "sif_potential", "life_saving_rule", "risk_level", "status"] as const).map(col => (
                <th key={col} className={TABLE_HEAD} style={{ color: "#334155", whiteSpace: "nowrap", cursor: "pointer" }}
                  onClick={() => { setPage(1); }}>
                  {col.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase())}
                </th>
              ))}
              {canReview && <th className={TABLE_HEAD} style={{ color: "#334155", whiteSpace: "nowrap" }}>Action</th>}
            </tr>
          </thead>
          <tbody>
            {reports.map((r, i) => {
              const isCritical = !!r.critical;
              return (
                <tr key={r.id}
                  style={{
                    background: isCritical ? "rgba(251,59,92,0.07)" : (i % 2 === 1 ? "#0C1015" : "transparent"),
                    borderTop: isCritical ? "1px solid rgba(251,59,92,0.3)" : "1px solid #161C24",
                    borderLeft: isCritical ? "2px solid #FB3B5C" : "2px solid transparent",
                  }}>
                  <td className={`${TABLE_CELL} mono`} style={{ color: "#22D3EE", fontSize: 12 }}>
                    <div className="flex items-center gap-1.5">
                      {isCritical && (
                        <span title="Critical — immediate reviewer alert" className="pulse-glow" style={{ color: "#FB3B5C", fontSize: 13 }}>&#9888;</span>
                      )}
                      {r.report_id}
                    </div>
                  </td>
                  <td className={`${TABLE_CELL} mono`} style={{ color: "#94A3B8", fontSize: 12 }}>{r.date}</td>
                  <td className={TABLE_CELL} style={{ color: "#CBD5E1", fontSize: 13 }}>{r.site}</td>
                  <td className={TABLE_CELL} style={{ color: "#94A3B8", fontSize: 12, maxWidth: 160, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.activity}</td>
                  <td className={TABLE_CELL}>
                    <span className="mono text-[11px] font-semibold px-2 py-0.5 rounded-sm"
                      style={{ background: "#07090C", border: "1px solid rgba(34,211,238,0.16)", color: "#94A3B8" }}>
                      {r.report_type}
                    </span>
                  </td>
                  <td className={TABLE_CELL}><Badge colors={sifColor(r.sif_potential)}>{r.sif_potential}</Badge></td>
                  <td className={TABLE_CELL} style={{ color: "#94A3B8", fontSize: 12 }}>{r.life_saving_rule}</td>
                  <td className={TABLE_CELL}><Badge colors={riskColor(r.risk_level || "Low")}>{r.risk_level || "Low"}</Badge></td>
                  <td className={TABLE_CELL}><StatusBadge status={r.status} /></td>
                  {canReview && (
                    <td className={TABLE_CELL}>
                      <select
                        value={r.status || "Submitted"}
                        disabled={updatingId === r.id}
                        onChange={e => changeStatus(r.id, e.target.value)}
                        className="text-[11px] px-1.5 py-1 rounded-sm outline-none"
                        style={{ background: "#050608", border: "1px solid rgba(34,211,238,0.16)", color: "#CBD5E1", cursor: "pointer" }}>
                        {REPORT_STATUSES.map(s => <option key={s} value={s}>{s}</option>)}
                      </select>
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
        {reports.length === 0 && !loading && (
          <div className="text-center py-12 text-sm" style={{ color: "#334155" }}>No reports match the selected filters.</div>
        )}
      </div>

      {/* Pagination */}
      {pages > 1 && (
        <div className="flex items-center justify-center gap-2">
          <button disabled={page <= 1} onClick={() => setPage(p => p - 1)}
            className="px-3 py-1 text-xs rounded-sm"
            style={{ background: "rgba(34,211,238,0.16)", color: page <= 1 ? "#334155" : "#94A3B8" }}>Prev</button>
          <span className="mono text-xs" style={{ color: "#64748B" }}>Page {page} / {pages}</span>
          <button disabled={page >= pages} onClick={() => setPage(p => p + 1)}
            className="px-3 py-1 text-xs rounded-sm"
            style={{ background: "rgba(34,211,238,0.16)", color: page >= pages ? "#334155" : "#94A3B8" }}>Next</button>
        </div>
      )}
    </div>
  );
}

// ─── Admin: Users & Roles ───────────────────────────────────────────
// Real-world rule (see design discussion): every signup starts as
// Reporter. Only an Admin can elevate someone to Supervisor/HSE/Admin,
// and every change lands in the audit log — never a self-service toggle.
interface AdminUserRow {
  id: number;
  name: string;
  email: string;
  role: Role;
  site: string;
  status: string;
  created_at: string;
}

function AdminUsersScreen() {
  const [users, setUsers] = useState<AdminUserRow[]>([]);
  const [roles, setRoles] = useState<Role[]>(["Reporter", "Supervisor", "HSE", "Admin"]);
  const [loading, setLoading] = useState(false);
  const [savingId, setSavingId] = useState<number | null>(null);
  const me = getCurrentUser();

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await apiFetch(`/admin/users`);
      const data = await res.json();
      setUsers(data.users || []);
      if (data.roles) setRoles(data.roles);
    } catch (e) { console.error(e); }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const changeRole = async (userId: number, newRole: string) => {
    setSavingId(userId);
    try {
      await apiFetch(`/admin/users/${userId}/role`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role: newRole, reason: "Elevated via Admin console" }),
      });
      await load();
    } catch (e) { console.error(e); }
    setSavingId(null);
  };

  const toggleStatus = async (userId: number, current: string) => {
    const next = current === "active" ? "disabled" : "active";
    setSavingId(userId);
    try {
      await apiFetch(`/admin/users/${userId}/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: next, reason: "Toggled via Admin console" }),
      });
      await load();
    } catch (e) { console.error(e); }
    setSavingId(null);
  };

  return (
    <div className="flex flex-col gap-4 p-6 overflow-y-auto h-full">
      <div className="text-[11px] leading-relaxed px-4 py-3 rounded-sm"
        style={{ background: "rgba(34,211,238,0.06)", border: "1px solid rgba(34,211,238,0.2)", color: "#94A3B8" }}>
        Every new account starts as <strong style={{ color: "#CBD5E1" }}>Reporter</strong>. Elevating someone to
        Supervisor, HSE or Admin is a deliberate act, recorded in the audit log with who changed what and when.
      </div>

      <div className="rounded-sm overflow-hidden" style={{ background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)" }}>
        <table className="w-full">
          <thead style={{ background: "#07090C" }}>
            <tr>
              {["Name", "Email", "Site", "Role", "Status", "Joined"].map(col => (
                <th key={col} className={TABLE_HEAD} style={{ color: "#334155", whiteSpace: "nowrap" }}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {users.map((u, i) => (
              <tr key={u.id} style={{ background: i % 2 === 1 ? "#0C1015" : "transparent", borderTop: "1px solid #161C24" }}>
                <td className={TABLE_CELL} style={{ color: "#CBD5E1", fontSize: 13 }}>{u.name}</td>
                <td className={`${TABLE_CELL} mono`} style={{ color: "#94A3B8", fontSize: 12 }}>{u.email}</td>
                <td className={TABLE_CELL} style={{ color: "#94A3B8", fontSize: 12 }}>{u.site || "—"}</td>
                <td className={TABLE_CELL}>
                  <select
                    value={u.role}
                    disabled={savingId === u.id || u.id === me?.id}
                    onChange={e => changeRole(u.id, e.target.value)}
                    title={u.id === me?.id ? "You cannot change your own role" : "Change role"}
                    className="text-[11px] px-1.5 py-1 rounded-sm outline-none"
                    style={{ background: "#050608", border: "1px solid rgba(34,211,238,0.16)", color: "#CBD5E1", cursor: u.id === me?.id ? "not-allowed" : "pointer" }}>
                    {roles.map(r => <option key={r} value={r}>{r}</option>)}
                  </select>
                </td>
                <td className={TABLE_CELL}>
                  <button onClick={() => toggleStatus(u.id, u.status)} disabled={savingId === u.id || u.id === me?.id}
                    className="text-[11px] px-2 py-1 rounded-sm font-semibold"
                    style={{
                      background: u.status === "active" ? "rgba(16,185,129,0.14)" : "rgba(251,59,92,0.14)",
                      color: u.status === "active" ? "#34D399" : "#FB7185",
                      border: `1px solid ${u.status === "active" ? "#10B981" : "#FB3B5C"}`,
                      cursor: u.id === me?.id ? "not-allowed" : "pointer",
                    }}>
                    {u.status === "active" ? "Active" : "Disabled"}
                  </button>
                </td>
                <td className={`${TABLE_CELL} mono`} style={{ color: "#64748B", fontSize: 11 }}>{(u.created_at || "").slice(0, 10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {users.length === 0 && !loading && (
          <div className="text-center py-12 text-sm" style={{ color: "#334155" }}>No users yet.</div>
        )}
      </div>
    </div>
  );
}

// ─── Bulk Upload Screen ─────────────────────────────────────────────
function BulkScreen() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<{ total_processed: number; sif_flagged: number; non_sif: number } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await apiFetch(`/reports/bulk-upload`, { method: "POST", body: fd });
      setResult(await res.json());
    } catch (e) { console.error(e); }
    setUploading(false);
  };

  return (
    <div className="flex flex-col gap-5 p-6 overflow-y-auto h-full max-w-3xl mx-auto w-full">
      <div className="rounded-sm overflow-hidden" style={{ background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)" }}>
        <div className="px-5 py-3.5 border-b" style={{ borderColor: "rgba(34,211,238,0.16)" }}>
          <div className="text-sm font-semibold" style={{ color: "#F1F5F9" }}>Bulk Upload Historical Reports</div>
          <div className="text-[11px] mt-0.5" style={{ color: "#64748B" }}>Upload a CSV or Excel file with a "report_text" column. Each row will be analyzed by the AI engine.</div>
        </div>
        <div className="p-6">
          <div className="rounded-sm border-2 border-dashed p-8 text-center cursor-pointer transition-all"
            style={{ borderColor: file ? "#22D3EE" : "rgba(34,211,238,0.16)", background: file ? "rgba(34,211,238,0.08)" : "#050608" }}
            onClick={() => fileRef.current?.click()}>
            <input ref={fileRef} type="file" accept=".csv,.xlsx,.xls" className="hidden"
              onChange={e => { if (e.target.files?.[0]) { setFile(e.target.files[0]); setResult(null); } }} />
            {file ? (
              <div className="flex flex-col items-center gap-2">
                <div className="text-sm font-semibold" style={{ color: "#F1F5F9" }}>{file.name}</div>
                <div className="text-[11px]" style={{ color: "#64748B" }}>{(file.size / 1024).toFixed(0)} KB -- Click to change</div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#334155" strokeWidth="1.5">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                <div className="text-sm" style={{ color: "#64748B" }}>Drag & drop CSV/Excel, or click to browse</div>
              </div>
            )}
          </div>
          <div className="flex justify-end mt-4">
            <button onClick={handleUpload} disabled={uploading || !file}
              className="px-6 py-2 text-sm font-semibold rounded-sm transition-all"
              style={{
                background: uploading || !file ? "rgba(34,211,238,0.16)" : "#0E7490",
                color: uploading || !file ? "#334155" : "#F1F5F9",
                cursor: uploading || !file ? "not-allowed" : "pointer",
                border: "1px solid", borderColor: uploading || !file ? "rgba(34,211,238,0.16)" : "#22D3EE",
              }}>
              {uploading ? "Processing..." : "Upload & Analyze"}
            </button>
          </div>
        </div>
      </div>

      {/* Result */}
      {result && (
        <div className="rounded-sm px-5 py-5" style={{ background: "#0A0E14", border: "1px solid rgba(34,211,238,0.16)" }}>
          <div className="text-sm font-semibold mb-3" style={{ color: "#F1F5F9" }}>Upload Complete</div>
          <div className="grid grid-cols-3 gap-4">
            <KPICard label="Processed" value={String(result.total_processed)} sub="reports analyzed" />
            <KPICard label="SIF Flagged" value={String(result.sif_flagged)} sub="potential SIF precursors" color="#FB7185" />
            <KPICard label="Non-SIF" value={String(result.non_sif)} sub="no SIF precursor" color="#34D399" />
          </div>
          <div className="mt-4 text-center">
            <button className="text-xs underline" style={{ color: "#22D3EE" }}
              onClick={() => { window.dispatchEvent(new CustomEvent("navigate", { detail: "dashboard" })); }}>
              View on Dashboard
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Landing / Cover Screen ─────────────────────────────────────────
// Editorial, cinematic cover screen inspired by modern oil & gas corporate
// sites (full-bleed offshore photography, bold typography, dark navy /
// electric-blue palette) — the "front door" before the operational tool.
function LandingScreen({ onEnter }: { onEnter: () => void }) {
  return (
    <div className="relative h-full w-full overflow-hidden" style={{ background: "#050C1F" }}>
      <img src={HERO_IMAGE} alt="Offshore energy platform"
        className="absolute inset-0 w-full h-full object-cover"
        style={{ opacity: 0.55 }} />
      <div className="absolute inset-0"
        style={{ background: "linear-gradient(180deg, #050C1F 0%, rgba(5,12,31,0.55) 35%, rgba(5,12,31,0.75) 70%, #050C1F 100%)" }} />
      <div className="absolute inset-0"
        style={{ background: "linear-gradient(90deg, #050C1F 0%, rgba(5,12,31,0.15) 45%, rgba(5,12,31,0.15) 55%, #050C1F 100%)" }} />

      <div className="relative z-10 flex flex-col h-full">
        <div className="flex items-center justify-between px-10 py-7">
          <div className="flex items-center gap-2.5">
            <div className="flex items-center justify-center w-8 h-8 rounded-sm text-white font-bold text-sm"
              style={{ background: "linear-gradient(135deg, #2C4674, #61A5D4)" }}>OIL</div>
            <span className="text-xs font-semibold tracking-widest uppercase" style={{ color: "#ADC6D6" }}>AI Safety Analyst</span>
          </div>
          <span className="text-[11px] tracking-widest uppercase" style={{ color: "#61A5D4" }}>Oil India Limited &middot; HSSE Intelligence</span>
        </div>

        <div className="flex-1 flex flex-col justify-center px-10 md:px-16 max-w-3xl fade-in">
          <div className="text-[11px] font-semibold uppercase tracking-[0.25em] mb-4" style={{ color: "#61A5D4" }}>
            SIF Precursor Detection &middot; Powered by NLP
          </div>
          <h1 className="font-bold leading-[1.05] mb-5" style={{ fontSize: "clamp(2.5rem, 5vw, 4.5rem)", color: "#F1F5F9", letterSpacing: "-0.02em" }}>
            See the incident<br />before it happens.
          </h1>
          <p className="text-base md:text-lg leading-relaxed mb-9 max-w-xl" style={{ color: "#9ABACF" }}>
            Every unsafe act, unsafe condition, and near-miss report analyzed in real time —
            classified for Serious Injury &amp; Fatality potential, mapped to IOGP Life-Saving
            Rules, and ranked so your HSE team acts on the highest-risk pattern first.
          </p>
          <div className="flex items-center gap-4">
            <button onClick={onEnter}
              className="px-7 py-3.5 text-sm font-semibold rounded-sm transition-all hover:opacity-90"
              style={{ background: "linear-gradient(135deg, #2C4674, #61A5D4)", color: "#F1F5F9" }}>
              Enter Platform &rarr;
            </button>
            <span className="text-xs" style={{ color: "#64748B" }}>Live NLP engine &middot; recall-optimized classifier</span>
          </div>
        </div>

        <div className="px-10 md:px-16 pb-10 flex items-center gap-10 flex-wrap fade-in-delay-2">
          {[
            ["9", "IOGP Life-Saving Rules mapped"],
            ["1,325+", "Reports analyzed in training corpus"],
            ["98.7%", "SIF recall on held-out test set"],
          ].map(([num, label]) => (
            <div key={label as string}>
              <div className="mono font-bold" style={{ fontSize: 28, color: "#F1F5F9" }}>{num}</div>
              <div className="text-[11px] uppercase tracking-widest mt-1" style={{ color: "#64748B" }}>{label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Auth screens ───────────────────────────────────────────────────
// Warm editorial palette (cream / near-black / tan) — deliberately
// distinct from the dark cyan command-center theme used post-login, to
// match the reference UI's calmer, more "workspace" feel for the
// account-entry moment.
const AUTH_BG = "#EDE7DE";
const AUTH_PANEL_DARK = "#1B1A1A";
const AUTH_TEXT_MUTED = "#7F7C77";
const AUTH_ACCENT = "#D4C6A0";

function AuthShell({ children, tagline }: { children: React.ReactNode; tagline: string }) {
  return (
    <div className="flex h-full w-full" style={{ background: AUTH_BG }}>
      <div className="hidden md:flex flex-col justify-between w-[42%] p-10"
        style={{ background: AUTH_PANEL_DARK, color: "#F1EDE7" }}>
        <div className="flex items-center gap-2.5">
          <div className="flex items-center justify-center w-8 h-8 rounded-full text-sm font-bold"
            style={{ background: AUTH_ACCENT, color: AUTH_PANEL_DARK }}>OIL</div>
          <span className="text-xs font-semibold tracking-widest uppercase" style={{ color: "#D8D2C6" }}>
            HSSE Safety Intelligence
          </span>
        </div>
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.25em] mb-4" style={{ color: AUTH_ACCENT }}>
            SIH26165 &middot; Oil India Limited
          </div>
          <h1 className="font-bold leading-[1.15] mb-4" style={{ fontSize: "clamp(1.8rem, 3vw, 2.6rem)", letterSpacing: "-0.02em" }}>
            {tagline}
          </h1>
          <p className="text-sm leading-relaxed max-w-sm" style={{ color: "#B9B2A3" }}>
            Every report moves through one real lifecycle: submitted, analyzed,
            verified, investigated where it matters, and closed out — with a
            named person accountable at every step.
          </p>
        </div>
        <div className="text-[11px]" style={{ color: "#726B5E" }}>
          Role-based access &middot; audit-logged decisions &middot; no self-elevation
        </div>
      </div>
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-sm">{children}</div>
      </div>
    </div>
  );
}

function AuthInput({ label, type = "text", value, onChange, placeholder, autoFocus, name }: {
  label: string; type?: string; value: string; onChange: (v: string) => void; placeholder?: string; autoFocus?: boolean; name?: string;
}) {
  // autoComplete + name matter here: without them some browsers' autofill
  // writes straight into the DOM node without firing React's onChange,
  // which leaves this component's state empty even though the field
  // looks filled -- the field would then fail the submit-time trim()
  // check silently. Giving each field a real name/autoComplete value
  // keeps the browser's autofill in sync with React state.
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-[11px] font-semibold uppercase tracking-widest" style={{ color: AUTH_TEXT_MUTED }}>{label}</span>
      <input type={type} name={name} autoComplete={name} value={value} placeholder={placeholder} autoFocus={autoFocus}
        onChange={e => onChange(e.target.value)}
        className="px-3.5 py-2.5 text-sm rounded-lg outline-none transition-all"
        style={{ background: "#FFFFFF", border: `1px solid #D9D3C6`, color: AUTH_PANEL_DARK }} />
    </label>
  );
}

function LoginScreen({ onAuthed, onGoSignup }: { onAuthed: (token: string, user: AuthUser) => void; onGoSignup: () => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!email.trim() || !password) { setError("Enter your email and password."); return; }
    setBusy(true);
    setError("");
    try {
      const res = await fetch(`${API}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email.trim(), password }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.detail || "Login failed."); setBusy(false); return; }
      onAuthed(data.token, data.user);
    } catch {
      setError("Could not reach the server.");
    }
    setBusy(false);
  };

  return (
    <AuthShell tagline="See the incident before it happens.">
      <div className="mb-7">
        <h2 className="text-2xl font-bold mb-1" style={{ color: AUTH_PANEL_DARK }}>Sign in</h2>
        <p className="text-sm" style={{ color: AUTH_TEXT_MUTED }}>Use your OIL HSSE platform account.</p>
      </div>
      <form className="flex flex-col gap-4" onSubmit={submit}>
        <AuthInput label="Email" type="email" name="email" value={email} onChange={setEmail} placeholder="you@oil-hsse.local" autoFocus />
        <AuthInput label="Password" type="password" name="current-password" value={password} onChange={setPassword} placeholder="••••••••" />
        {error && <div className="text-xs font-medium" style={{ color: "#B3432E" }}>{error}</div>}
        <button type="submit" disabled={busy}
          className="mt-2 px-5 py-3 text-sm font-semibold rounded-lg transition-all hover:opacity-90 disabled:opacity-60"
          style={{ background: AUTH_PANEL_DARK, color: "#F1EDE7" }}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
      <div className="mt-6 text-xs" style={{ color: AUTH_TEXT_MUTED }}>
        New field user?{" "}
        <button type="button" onClick={onGoSignup} className="font-semibold underline" style={{ color: AUTH_PANEL_DARK }}>
          Create a Reporter account
        </button>
      </div>
      <div className="mt-8 pt-5 text-[11px] leading-relaxed" style={{ borderTop: "1px solid #D9D3C6", color: "#9A9384" }}>
        New accounts always start as <strong>Reporter</strong>. Supervisor, HSE and Admin
        access is granted by an Administrator — never self-selected — and every
        grant is recorded in the audit log.
      </div>
    </AuthShell>
  );
}

function SignupScreen({ onAuthed, onGoLogin }: { onAuthed: (token: string, user: AuthUser) => void; onGoLogin: () => void }) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [site, setSite] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!name.trim() || !email.trim() || !password) { setError("Name, email and password are required."); return; }
    if (password.length < 8) { setError("Password must be at least 8 characters."); return; }
    setBusy(true);
    setError("");
    try {
      const res = await fetch(`${API}/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim(), email: email.trim(), password, site: site.trim() }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.detail || "Signup failed."); setBusy(false); return; }
      onAuthed(data.token, data.user);
    } catch {
      setError("Could not reach the server.");
    }
    setBusy(false);
  };

  return (
    <AuthShell tagline="Report what you see. We'll route it to the right person.">
      <div className="mb-7">
        <h2 className="text-2xl font-bold mb-1" style={{ color: AUTH_PANEL_DARK }}>Create your account</h2>
        <p className="text-sm" style={{ color: AUTH_TEXT_MUTED }}>
          Starts as a <strong>Reporter</strong> account — you can submit observations immediately.
        </p>
      </div>
      <form className="flex flex-col gap-4" onSubmit={submit}>
        <AuthInput label="Full name" name="name" value={name} onChange={setName} placeholder="Jane Doe" autoFocus />
        <AuthInput label="Email" type="email" name="email" value={email} onChange={setEmail} placeholder="you@oil-hsse.local" />
        <AuthInput label="Site (optional)" name="organization" value={site} onChange={setSite} placeholder="Assam Region — Facility A" />
        <AuthInput label="Password" type="password" name="new-password" value={password} onChange={setPassword} placeholder="At least 8 characters" />
        {error && <div className="text-xs font-medium" style={{ color: "#B3432E" }}>{error}</div>}
        <button type="submit" disabled={busy}
          className="mt-2 px-5 py-3 text-sm font-semibold rounded-lg transition-all hover:opacity-90 disabled:opacity-60"
          style={{ background: AUTH_PANEL_DARK, color: "#F1EDE7" }}>
          {busy ? "Creating account…" : "Create account"}
        </button>
      </form>
      <div className="mt-6 text-xs" style={{ color: AUTH_TEXT_MUTED }}>
        Already have an account?{" "}
        <button type="button" onClick={onGoLogin} className="font-semibold underline" style={{ color: AUTH_PANEL_DARK }}>
          Sign in
        </button>
      </div>
    </AuthShell>
  );
}

// ─── Navigation ─────────────────────────────────────────────────────
const NAV_ITEMS: { id: Screen; label: string; icon: React.ReactNode; roles?: Role[] }[] = [
  { id: "dashboard", label: "Dashboard", icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
      <rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" />
    </svg>
  )},
  { id: "analyze", label: "Analyze Report", icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
    </svg>
  )},
  { id: "reports", label: "Reports Log", icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" />
    </svg>
  )},
  { id: "bulk", label: "Bulk Upload", roles: ["HSE", "Admin"], icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  )},
  { id: "admin", label: "Users & Roles", roles: ["Admin"], icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" /><circle cx="9" cy="7" r="4" />
      <path d="M22 21v-2a4 4 0 0 0-3-3.87" /><path d="M16 3.13a4 4 0 0 1 0 7.75" />
    </svg>
  )},
];

const SCREEN_TITLES: Record<Screen, string> = {
  landing: "AI Safety Analyst",
  login: "Sign In",
  signup: "Create Account",
  dashboard: "Operations Dashboard",
  analyze: "Analyze Report",
  reports: "Reports Log",
  bulk: "Bulk Upload",
  admin: "Users & Roles",
};

// Per-role copy overrides -- same screens, different framing. A Reporter's
// "New Report" is the same route as everyone else's "Analyze Report", and
// their "My Reports" is the same ReportsScreen a Supervisor sees as a
// "Review Queue" -- the backend already scopes the data per role, this
// just keeps the label honest about what's on screen.
const NAV_LABEL_OVERRIDES: Partial<Record<Role, Partial<Record<Screen, string>>>> = {
  Reporter: { analyze: "New Report", reports: "My Reports" },
  Supervisor: { reports: "Review Queue" },
};
const SCREEN_TITLE_OVERRIDES: Partial<Record<Role, Partial<Record<Screen, string>>>> = {
  Reporter: { dashboard: "My Dashboard", analyze: "New Report", reports: "My Reports" },
  Supervisor: { dashboard: "Supervisor Dashboard", reports: "Review Queue" },
};

const ROLE_LABELS: Record<Role, string> = {
  Reporter: "Reporter",
  Supervisor: "Supervisor",
  HSE: "HSE Officer",
  Admin: "Administrator",
};

export default function App() {
  const stored = loadStoredAuth();
  const [screen, setScreen] = useState<Screen>(stored ? "dashboard" : "landing");
  const [authUser, setAuthUser] = useState<AuthUser | null>(stored ? stored.user : null);

  // Hydrate the module-level token/user from localStorage on first mount
  // so apiFetch has the Bearer token before any screen's effect fires.
  useEffect(() => {
    if (stored) {
      setCurrentToken(stored.token);
      setCurrentUser(stored.user);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Listen for custom navigation events (e.g. from bulk upload, or a 401
  // bouncing the session back to login) — registered once, not per render.
  useEffect(() => {
    const handler = ((e: CustomEvent) => {
      if (e.detail) setScreen(e.detail as Screen);
    }) as EventListener;
    window.addEventListener("navigate", handler);
    return () => window.removeEventListener("navigate", handler);
  }, []);

  const handleAuthed = (token: string, user: AuthUser) => {
    storeAuth(token, user);
    setCurrentToken(token);
    setCurrentUser(user);
    setAuthUser(user);
    setScreen("dashboard");
  };

  const handleLogout = () => {
    clearStoredAuth();
    setCurrentToken(null);
    setCurrentUser(null);
    setAuthUser(null);
    setScreen("landing");
  };

  if (screen === "landing") {
    return <LandingScreen onEnter={() => setScreen(authUser ? "dashboard" : "login")} />;
  }

  if (screen === "login") {
    return <LoginScreen onAuthed={handleAuthed} onGoSignup={() => setScreen("signup")} />;
  }

  if (screen === "signup") {
    return <SignupScreen onAuthed={handleAuthed} onGoLogin={() => setScreen("login")} />;
  }

  if (!authUser) {
    // Guard against a direct deep-link into an app screen with no session.
    return <LoginScreen onAuthed={handleAuthed} onGoSignup={() => setScreen("signup")} />;
  }

  const visibleNav = NAV_ITEMS.filter(item => !item.roles || item.roles.includes(authUser.role));
  const navLabel = (item: typeof NAV_ITEMS[number]) => NAV_LABEL_OVERRIDES[authUser.role]?.[item.id] ?? item.label;
  const pageTitle = SCREEN_TITLE_OVERRIDES[authUser.role]?.[screen] ?? SCREEN_TITLES[screen];

  return (
    <div className="flex h-full" style={{ background: "#050608", color: "#F1F5F9" }}>
      {/* Sidebar */}
      <aside className="flex flex-col shrink-0" style={{ width: 220, background: "#07090C", borderRight: "1px solid rgba(34,211,238,0.16)" }}>
        <div className="px-4 py-4 border-b" style={{ borderColor: "rgba(34,211,238,0.16)" }}>
          <div className="flex items-center gap-2.5">
            <div className="flex items-center justify-center w-8 h-8 rounded-sm text-white font-bold text-sm"
              style={{ background: "linear-gradient(135deg, #0E7490, #0EA5C9)" }}>OIL</div>
            <div>
              <div className="text-xs font-semibold" style={{ color: "#CBD5E1" }}>AI Safety Analyst</div>
              <div className="text-[10px]" style={{ color: "#334155" }}>Oil India Limited</div>
            </div>
          </div>
        </div>
        <nav className="flex flex-col gap-0.5 p-2 flex-1">
          {visibleNav.map(item => {
            const active = screen === item.id;
            return (
              <button key={item.id} onClick={() => setScreen(item.id)}
                className="flex items-center gap-2.5 px-3 py-2 rounded-sm text-left w-full text-sm transition-all"
                style={{
                  background: active ? "rgba(34,211,238,0.16)" : "transparent",
                  color: active ? "#F1F5F9" : "#64748B",
                  fontWeight: active ? 500 : 400,
                  borderLeft: active ? "2px solid #22D3EE" : "2px solid transparent",
                }}>
                <span style={{ color: active ? "#22D3EE" : "#334155" }}>{item.icon}</span>
                {navLabel(item)}
              </button>
            );
          })}
        </nav>
        <div className="px-3 py-3 border-t flex items-center gap-2.5" style={{ borderColor: "rgba(34,211,238,0.16)" }}>
          <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold shrink-0"
            style={{ background: "#0E7490", color: "#F1F5F9" }}>
            {authUser.name.split(" ").map(p => p[0]).slice(0, 2).join("").toUpperCase()}
          </div>
          <div className="flex flex-col min-w-0 flex-1">
            <span className="text-xs font-medium truncate" style={{ color: "#CBD5E1" }}>{authUser.name}</span>
            <span className="text-[10px] truncate" style={{ color: "#334155" }}>{ROLE_LABELS[authUser.role]}</span>
          </div>
          <button onClick={handleLogout} title="Log out"
            className="shrink-0 w-6 h-6 rounded-sm flex items-center justify-center transition-all"
            style={{ color: "#64748B" }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><polyline points="16 17 21 12 16 7" /><line x1="21" y1="12" x2="9" y2="12" />
            </svg>
          </button>
        </div>
      </aside>

      {/* Main */}
      <div className="flex flex-col flex-1 min-w-0">
        <header className="flex items-center justify-between px-6 shrink-0"
          style={{ height: 52, background: "#07090C", borderBottom: "1px solid rgba(34,211,238,0.16)" }}>
          <h1 className="text-sm font-semibold" style={{ color: "#F1F5F9" }}>{pageTitle}</h1>
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-sm"
            style={{ background: "rgba(16,185,129,0.08)", border: "1px solid rgba(16,185,129,0.3)" }}>
            <span className="w-1.5 h-1.5 rounded-full pulse-glow" style={{ background: "#10B981", boxShadow: "0 0 6px #10B981" }} />
            <span className="text-[11px] font-semibold tracking-widest uppercase" style={{ color: "#34D399" }}>NLP Engine Active</span>
          </div>
        </header>
        <main className="flex-1 overflow-hidden command-grid">
          {screen === "dashboard" && authUser.role === "Reporter" && <ReporterDashboardScreen onNavigate={setScreen} />}
          {screen === "dashboard" && authUser.role === "Supervisor" && <SupervisorDashboardScreen onNavigate={setScreen} />}
          {screen === "dashboard" && (authUser.role === "HSE" || authUser.role === "Admin") && <DashboardScreen />}
          {screen === "analyze" && <AnalyzeScreen />}
          {screen === "reports" && <ReportsScreen />}
          {screen === "bulk" && <BulkScreen />}
          {screen === "admin" && <AdminUsersScreen />}
        </main>
      </div>
    </div>
  );
}
