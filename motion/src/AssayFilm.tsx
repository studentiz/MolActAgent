import type { ReactNode } from "react";
import { AbsoluteFill, Sequence, useCurrentFrame } from "remotion";
import { draw, reveal, rise } from "./components/animation";
import {
  ClassTag,
  CompoundCard,
  Eyebrow,
  FooterSource,
  PlateHeader,
  RegistrationMarks,
  RuleLabel,
  SceneTitle,
  Structure,
  sans,
} from "./components/AssayVisuals";
import { Fonts } from "./components/Fonts";
import { useFonts } from "./components/UseFonts";
import { evidence, palette } from "./data/assay";

const fade = (frame: number, _duration: number) => ({ opacity: reveal(frame, 0, 12) });
const openingFade = (_frame: number, _duration: number) => ({ opacity: 1 });

const AssayCard: React.FC<{ label: string; value: string; detail: string; index: number }> = ({ label, value, detail, index }) => (
  <div
    style={{
      flex: 1,
      minWidth: 0,
      padding: "22px 22px 20px",
      background: index === 2 ? palette.greenPale : palette.white,
      border: "1px solid " + (index === 2 ? palette.greenLine : palette.rule),
    }}
  >
    <Eyebrow color={index === 2 ? palette.green : palette.muted}>{label}</Eyebrow>
    <div style={{ marginTop: 15, fontFamily: sans, fontSize: 38, fontWeight: 680, letterSpacing: "-0.04em", lineHeight: 1.02, color: palette.ink }}>{value}</div>
    <div style={{ marginTop: 14, fontFamily: sans, fontSize: 16, lineHeight: 1.35, color: palette.softInk }}>{detail}</div>
  </div>
);

const ReporterSchematic: React.FC = () => (
  <div style={{ padding: "24px 26px 22px", background: palette.white, border: "1px solid " + palette.rule }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
      <Eyebrow color={palette.green}>COS-1 reporter assay</Eyebrow>
      <span style={{ fontFamily: sans, fontSize: 15, color: palette.muted }}>measured endpoint</span>
    </div>
    <svg viewBox="0 0 860 250" style={{ width: "100%", height: 250, marginTop: 10, display: "block" }} role="img" aria-label="Schematic of a PPAR-alpha reporter assay in COS-1 cells, read as EC50">
      <defs>
        <marker id="assay-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill={palette.ruleDark} />
        </marker>
      </defs>
      <rect x="20" y="38" width="294" height="176" rx="18" fill={palette.paper} stroke={palette.ruleDark} strokeWidth="1.5" />
      <text x="40" y="65" fill={palette.softInk} fontFamily="DM Sans" fontSize="14" fontWeight="700" letterSpacing="1.3">COS-1 REPORTER PLATE</text>
      {[0, 1, 2, 3].flatMap((row) =>
        [0, 1, 2, 3, 4].map((column) => {
          const highlighted = row === 1 && column === 2;
          return (
            <circle
              key={`${row}-${column}`}
              cx={52 + column * 51}
              cy={93 + row * 27}
              r="10"
              fill={highlighted ? palette.greenPale : palette.white}
              stroke={highlighted ? palette.green : palette.ruleDark}
              strokeWidth={highlighted ? 2 : 1}
            />
          );
        }),
      )}
      <text x="40" y="203" fill={palette.softInk} fontFamily="DM Sans" fontSize="13">cells in wells · one compound per well</text>
      <line x1="318" y1="126" x2="358" y2="126" stroke={palette.ruleDark} strokeWidth="1.5" markerEnd="url(#assay-arrow)" />
      <rect x="366" y="58" width="244" height="140" rx="12" fill={palette.white} stroke={palette.ruleDark} strokeWidth="1.5" />
      <text x="386" y="82" fill={palette.softInk} fontFamily="DM Sans" fontSize="13" fontWeight="700" letterSpacing="1.2">CELL REPORTER LOGIC</text>
      <circle cx="422" cy="132" r="31" fill={palette.greenPale} stroke={palette.greenLine} strokeWidth="1.5" />
      <text x="422" y="129" textAnchor="middle" fill={palette.ink} fontFamily="DM Sans" fontSize="17" fontWeight="700">PPARα</text>
      <text x="422" y="148" textAnchor="middle" fill={palette.softInk} fontFamily="DM Sans" fontSize="12">receptor</text>
      <line x1="455" y1="132" x2="483" y2="132" stroke={palette.ruleDark} strokeWidth="1.5" markerEnd="url(#assay-arrow)" />
      <rect x="487" y="106" width="103" height="52" rx="7" fill={palette.white} stroke={palette.ruleDark} />
      <text x="538" y="128" textAnchor="middle" fill={palette.ink} fontFamily="DM Sans" fontSize="15" fontWeight="650">reporter</text>
      <text x="538" y="146" textAnchor="middle" fill={palette.softInk} fontFamily="DM Sans" fontSize="12">expression</text>
      <line x1="614" y1="126" x2="648" y2="126" stroke={palette.ruleDark} strokeWidth="1.5" markerEnd="url(#assay-arrow)" />
      <rect x="652" y="72" width="184" height="110" rx="10" fill={palette.greenPale} stroke={palette.green} strokeWidth="1.5" />
      <text x="744" y="101" textAnchor="middle" fill={palette.greenDark} fontFamily="DM Sans" fontSize="14" fontWeight="700">MEASURED ENDPOINT</text>
      <text x="744" y="140" textAnchor="middle" fill={palette.ink} fontFamily="DM Sans" fontSize="30" fontWeight="720">EC50</text>
      <text x="744" y="164" textAnchor="middle" fill={palette.softInk} fontFamily="DM Sans" fontSize="12">≤ 1 µM active · &gt; 1 µM inactive</text>
    </svg>
  </div>
);

const AssayScene: React.FC = () => {
  return (
    <AbsoluteFill style={{ background: palette.paper, ...openingFade(0, 156) }}>
      <PlateHeader section="Measured assay" />
      <div style={{ position: "absolute", inset: "116px 72px 72px", display: "grid", gridTemplateColumns: ".82fr 1.18fr", gap: 64 }}>
        <div style={{ paddingTop: 42 }}>
          <div>
            <SceneTitle
              title="Measure PPARα activity."
              deck="A human PPARα reporter assay in COS-1 cells returns a measured EC50 for each compound."
            />
          </div>
          <div style={{ marginTop: 52 }}>
            <RuleLabel />
            <p style={{ margin: "20px 0 0", fontFamily: sans, fontSize: 18, lineHeight: 1.55, color: palette.softInk }}>
              Reporter expression turns PPARα activation into an experimentally measured response.
            </p>
          </div>
        </div>
        <div style={{ borderLeft: "1px solid " + palette.ruleDark, paddingLeft: 52, paddingTop: 36 }}>
          <div style={{ display: "flex", gap: 14 }}>
            {[
              { label: "Target", value: "PPAR-alpha", detail: "nuclear receptor" },
              { label: "Cell", value: "COS-1", detail: "reporter context" },
              { label: "Readout", value: "EC50", detail: "measured response" },
            ].map((item, index) => (
              <div key={item.label} style={{ flex: 1 }}>
                <AssayCard label={item.label} value={item.value} detail={item.detail} index={index} />
              </div>
            ))}
          </div>
          <div style={{ marginTop: 30 }}><ReporterSchematic /></div>
        </div>
      </div>
      <FooterSource>
        <span>Assay fixture: examples/cell-assay/endpoint.txt</span>
        <span>Active: measured EC50 at or below 1 µM</span>
      </FooterSource>
    </AbsoluteFill>
  );
};

const ReferenceScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ background: palette.paper, ...fade(frame, 138) }}>
      <PlateHeader section="Reference panel" />
      <div style={{ position: "absolute", inset: "116px 72px 72px", display: "grid", gridTemplateColumns: "390px 1fr", gap: 58 }}>
        <div style={{ paddingTop: 42 }}>
          <div style={rise(frame, 8)}>
            <SceneTitle
              title="Start with 7 measured structures."
              deck="The panel keeps the original measured classes and EC50 values."
            />
          </div>
          <div style={{ marginTop: 44, display: "grid", gap: 12, ...rise(frame, 28) }}>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "13px 0", borderTop: "1px solid " + palette.rule, borderBottom: "1px solid " + palette.rule, fontFamily: sans, fontSize: 18, color: palette.softInk }}>
              <span>Active references</span><strong style={{ color: palette.green, fontSize: 23 }}>3</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "13px 0", borderBottom: "1px solid " + palette.rule, fontFamily: sans, fontSize: 18, color: palette.softInk }}>
              <span>Inactive references</span><strong style={{ color: palette.warm, fontSize: 23 }}>4</strong>
            </div>
          </div>
          <p style={{ margin: "38px 0 0", fontFamily: sans, fontSize: 17, lineHeight: 1.55, color: palette.softInk, ...rise(frame, 42) }}>
            R3 is a distinct scaffold. The other six belong to the same aryl-ether thiazolidinedione series.
          </p>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gridTemplateRows: "1fr 1fr", gap: 16, paddingTop: 4 }}>
          {evidence.references.map((compound, index) => (
            <div key={compound.id} style={{ ...rise(frame, 12 + index * 6, 18) }}>
              <CompoundCard compound={compound} status="measured" compact />
            </div>
          ))}
          <div style={{ gridColumn: "3 / 5", alignSelf: "end", justifySelf: "stretch", padding: "12px 18px", border: "1px solid " + palette.rule, fontFamily: sans, fontSize: 16, lineHeight: 1.4, color: palette.softInk, ...rise(frame, 62) }}>
            3 active and 4 inactive. Q2 compares R2 para-Cl, R5 para-OH and R7 para-F + meta-Me.
          </div>
        </div>
      </div>
      <FooterSource>
        <span>Reference table: 7 structures / R1-R7</span>
        <span>Measured records: ChEMBL 37 / activity IDs in asset notes</span>
      </FooterSource>
    </AbsoluteFill>
  );
};

const CandidateScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ background: palette.paper, ...fade(frame, 108) }}>
      <PlateHeader section="Waiting candidates" />
      <div style={{ position: "absolute", inset: "116px 72px 72px", display: "grid", gridTemplateColumns: "350px 1fr", gap: 52 }}>
        <div style={{ paddingTop: 42 }}>
          <div style={rise(frame, 8)}>
            <SceneTitle
              title="Four candidates wait for evidence."
              deck="All four structures start untested. The evidence and MolActAgent's recorded judgment come next."
            />
          </div>
          <div style={{ marginTop: 44, fontFamily: sans, fontSize: 17, lineHeight: 1.55, color: palette.softInk, ...rise(frame, 38) }}>
            <strong style={{ color: palette.ink }}>Q2</strong> proposes an F-only para substitution.<br />
            <strong style={{ color: palette.ink }}>Q4</strong> places chlorine beside the conserved para-Cl.
          </div>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, alignItems: "start", paddingTop: 4 }}>
          {evidence.queries.map((compound, index) => (
            <div key={compound.id} style={{ ...rise(frame, 12 + index * 8, 22) }}>
              <CompoundCard compound={compound} status="untested" neutral compact={false} />
              <div style={{ marginTop: 10, paddingTop: 9, borderTop: "1px solid " + palette.rule, fontFamily: sans, fontSize: 14, fontWeight: 650, letterSpacing: "0.11em", textTransform: "uppercase", color: palette.muted }}>
                Awaiting assay
              </div>
            </div>
          ))}
        </div>
      </div>
      <FooterSource>
        <span>Four untested candidate structures / no measured query result shown</span>
        <span>Models and the chemist evaluate the evidence next</span>
      </FooterSource>
    </AbsoluteFill>
  );
};

const FocusCard: React.FC<{
  id: string;
  label: string;
  value: string;
  active: boolean;
  start: number;
}> = ({ id, label, value, active, start }) => {
  const frame = useCurrentFrame();
  return (
    <div
      style={{
        flex: 1,
        padding: "17px 17px 14px",
        background: active ? palette.greenPale : palette.warmPale,
        border: "1px solid " + (active ? palette.greenLine : palette.warmLine),
        ...rise(frame, start, 18),
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", fontFamily: sans }}>
        <strong style={{ fontSize: 22, color: palette.ink }}>{id}</strong>
        <span style={{ fontSize: 15, fontWeight: 700, color: active ? palette.green : palette.warm }}>{value}</span>
      </div>
      <div style={{ height: 138, margin: "8px 0", overflow: "hidden", background: palette.white, border: "1px solid " + palette.rule }}>
        <Structure id={id} focus style={{ width: "100%", height: "100%", objectFit: "contain" }} />
      </div>
      <div style={{ fontFamily: sans, fontSize: 15, color: palette.softInk }}>{label}</div>
    </div>
  );
};

const ComparisonScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ background: palette.paper, ...fade(frame, 156) }}>
      <PlateHeader section="Q2 decision trace" />
      <div style={{ position: "absolute", inset: "116px 72px 72px" }}>
        <div style={{ ...rise(frame, 6) }}>
          <SceneTitle
            title="How MolActAgent chose Q2."
            deck="ML sample comparison → computed structural change → MolActAgent language-model judgment → chemist decision."
          />
        </div>
        <div style={{ marginTop: 28, display: "grid", gridTemplateColumns: "1fr 1fr 1.12fr", gap: 16 }}>
          <div style={{ minHeight: 410, padding: "22px 22px 18px", border: "1px solid " + palette.ruleDark, background: palette.white, ...rise(frame, 22, 22) }}>
            <Eyebrow color={palette.green}>ML sample comparison</Eyebrow>
            <div style={{ marginTop: 12, fontFamily: sans, fontSize: 24, fontWeight: 680, lineHeight: 1.2, color: palette.ink }}>
              Three measured analogs frame the para substitution.
            </div>
            <div style={{ marginTop: 20, display: "grid", gap: 11 }}>
              {[
                { id: "R2", change: "para-Cl", value: "180 nM", label: "active" as const },
                { id: "R5", change: "para-OH", value: ">3000 nM", label: "inactive" as const },
                { id: "R7", change: "para-F + meta-Me", value: "85 nM", label: "active" as const },
              ].map((item) => (
                <div key={item.id} style={{ display: "grid", gridTemplateColumns: "58px 1fr auto", alignItems: "center", gap: 12, minHeight: 68, padding: "10px 12px", borderTop: "1px solid " + palette.rule }}>
                  <strong style={{ fontFamily: sans, fontSize: 23, color: palette.ink }}>{item.id}</strong>
                  <div>
                    <div style={{ fontFamily: sans, fontSize: 17, color: palette.ink }}>{item.change}</div>
                    <div style={{ marginTop: 4 }}><ClassTag label={item.label} compact /></div>
                  </div>
                  <span style={{ fontFamily: sans, fontSize: 16, fontWeight: 700, color: item.label === "active" ? palette.green : palette.warm }}>{item.value}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ minHeight: 410, padding: "22px 20px 18px", border: "1px solid " + palette.ruleDark, background: palette.white, ...rise(frame, 34, 22) }}>
            <Eyebrow color={palette.green}>Computed structural difference</Eyebrow>
            <div style={{ marginTop: 12, fontFamily: sans, fontSize: 24, fontWeight: 680, lineHeight: 1.2, color: palette.ink }}>
              Cl → OH crosses the cutoff. Cl → F is the candidate.
            </div>
            <div style={{ marginTop: 20, display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
              <FocusCard id="R2" label="measured" value="Cl" active start={46} />
              <FocusCard id="R5" label="measured" value="OH" active={false} start={52} />
              <FocusCard id="Q2" label="candidate" value="F" active start={58} />
            </div>
            <div style={{ marginTop: 16, fontFamily: sans, fontSize: 16, lineHeight: 1.45, color: palette.softInk }}>
              R2 para-Cl is active at 180 nM. R5 para-OH is inactive above 3000 nM. Q2 tests para-F without R7's adjacent methyl.
            </div>
          </div>

          <div style={{ minHeight: 410, padding: "22px 24px 20px", border: "2px solid " + palette.greenLine, background: palette.greenPale, ...rise(frame, 46, 22) }}>
            <Eyebrow color={palette.green}>MolActAgent / language-model judgment</Eyebrow>
            <div style={{ marginTop: 18, fontFamily: sans, fontSize: 31, fontWeight: 650, lineHeight: 1.25, letterSpacing: "-0.03em", color: palette.ink }}>
              “On the shared scaffold, a single para halogen on the terminal phenoxy ring distinguishes the active analogs from the inactive para-hydroxy and para-alkyl analogs.”
            </div>
            <div style={{ marginTop: 24, paddingTop: 18, borderTop: "1px solid " + palette.greenLine, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <div style={{ fontFamily: sans, fontSize: 13, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase", color: palette.softInk }}>Actual recorded judgment</div>
                <div style={{ marginTop: 8 }}><ClassTag label="active" /></div>
              </div>
              <div style={{ fontFamily: sans, fontSize: 15, color: palette.softInk, textAlign: "right" }}>MolActAgent prediction<br />Q2 / active</div>
            </div>
          </div>
        </div>
        <div style={{ marginTop: 18, padding: "18px 24px", borderTop: "1px solid " + palette.ink, borderBottom: "1px solid " + palette.ruleDark, display: "flex", alignItems: "center", justifyContent: "space-between", ...rise(frame, 68) }}>
          <div style={{ fontFamily: sans, fontSize: 24, fontWeight: 690, color: palette.ink }}>Chemist decision: Q2 moves forward to the assay decision.</div>
          <div style={{ display: "flex", alignItems: "center", gap: 12, fontFamily: sans, fontSize: 16, fontWeight: 700, color: palette.green }}>
            <span style={{ width: 34, height: 34, display: "grid", placeItems: "center", border: "1px solid " + palette.green, borderRadius: 99 }}>✓</span>
            Q2 SELECTED
          </div>
        </div>
      </div>
      <FooterSource>
        <span>Measured: R2 180 nM active / R5 &gt;3000 nM inactive / R7 85 nM active</span>
        <span>Language-model judgment and Q2 call copied from the recorded MolActAgent run</span>
      </FooterSource>
    </AbsoluteFill>
  );
};

const Q4Scene: React.FC = () => {
  const frame = useCurrentFrame();
  const q4 = evidence.queries.find((item) => item.id === "Q4")!;
  return (
    <AbsoluteFill style={{ background: palette.paper, ...fade(frame, 108) }}>
      <PlateHeader section="Q4 structural rationale" />
      <div style={{ position: "absolute", inset: "116px 72px 72px", display: "grid", gridTemplateColumns: "690px 1fr", gap: 68 }}>
        <div style={{ paddingTop: 30 }}>
          <div style={rise(frame, 6)}>
            <SceneTitle
              title="Q4 tests the neighboring meta-Cl."
              deck="Measured R4 carries para-Cl plus meta-Me. Q4 keeps para-Cl and replaces meta-Me with chlorine."
            />
          </div>
          <div style={{ marginTop: 30, ...rise(frame, 30) }}>
            <CompoundCard compound={q4} status="predicted" emphasis />
          </div>
        </div>
        <div style={{ alignSelf: "center", padding: "38px 46px 34px", border: "1px solid " + palette.ruleDark, background: palette.white, ...rise(frame, 20, 24) }}>
          <Eyebrow color={palette.green}>Computed structural change</Eyebrow>
          <div style={{ marginTop: 16, fontFamily: sans, fontSize: 42, fontWeight: 700, lineHeight: 1.08, letterSpacing: "-0.045em", color: palette.ink }}>
            R4 para-Cl + meta-Me<br />Q4 para-Cl + meta-Cl
          </div>
          <div style={{ marginTop: 30, padding: "20px 22px", borderTop: "1px solid " + palette.ruleDark, borderBottom: "1px solid " + palette.ruleDark }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 20, fontFamily: sans }}>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase", color: palette.muted }}>Measured anchor</div>
                <div style={{ marginTop: 8, fontSize: 23, fontWeight: 680, color: palette.ink }}>R4 para-Cl + meta-Me · 156 nM</div>
              </div>
              <ClassTag label="active" compact />
            </div>
            <div style={{ marginTop: 18, fontFamily: sans, fontSize: 40, color: palette.ruleDark }}>↓</div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 20, fontFamily: sans }}>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase", color: palette.muted }}>MolActAgent prediction</div>
                <div style={{ marginTop: 8, fontSize: 23, fontWeight: 680, color: palette.ink }}>Q4 para-Cl + meta-Cl · untested</div>
              </div>
              <ClassTag label="active" compact />
            </div>
          </div>
          <div style={{ marginTop: 24, fontFamily: sans, fontSize: 17, lineHeight: 1.5, color: palette.softInk, ...rise(frame, 56) }}>
            The only structural change from measured R4 is meta-Me → meta-Cl. Q4 is untested.
          </div>
        </div>
      </div>
      <FooterSource>
        <span>Q4 predicted active in the recorded MolActAgent run</span>
        <span>Full structure: RDKit coordinates / measured comparator: R4</span>
      </FooterSource>
    </AbsoluteFill>
  );
};

const ScheduleRow: React.FC<{ id: string; change: string; start: number }> = ({ id, change, start }) => {
  const frame = useCurrentFrame();
  return (
    <div style={{ position: "relative", display: "grid", gridTemplateColumns: "122px 1fr 170px", alignItems: "center", minHeight: 132, borderTop: "1px solid " + palette.ruleDark, opacity: reveal(frame, start, 12) }}>
      <div style={{ fontFamily: sans, fontSize: 42, fontWeight: 760, color: palette.ink }}>{id}</div>
      <div>
        <div style={{ fontFamily: sans, fontSize: 23, fontWeight: 700, color: palette.ink }}>{change}</div>
        <div style={{ marginTop: 7 }}><ClassTag label="active" compact /></div>
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 10, fontFamily: sans, fontSize: 14, fontWeight: 700, letterSpacing: "0.1em", color: palette.green }}>
        <span style={{ width: 30, height: 30, display: "grid", placeItems: "center", border: "1px solid " + palette.green, borderRadius: 99, opacity: reveal(frame, start + 14, 8) }}>✓</span>
        ON LIST
      </div>
      <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 2, background: palette.green, transformOrigin: "left", transform: "scaleX(" + draw(frame, start + 4, start + 30) + ")", opacity: 0.72 }} />
    </div>
  );
};

const SignoffScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ background: palette.paper, color: palette.ink, opacity: reveal(frame, 0, 12) }}>
      <RegistrationMarks />
      <PlateHeader section="Chemist decision" />
      <div style={{ position: "absolute", inset: "132px 96px 92px", display: "grid", gridTemplateColumns: "1fr 820px", gap: 78 }}>
        <div style={{ paddingTop: 64, ...rise(frame, 6) }}>
          <Eyebrow color={palette.green}>Chemist decision / final frame</Eyebrow>
          <h2 style={{ margin: "22px 0 0", fontFamily: sans, fontSize: 82, fontWeight: 720, lineHeight: 0.98, letterSpacing: "-0.06em", color: palette.ink }}>
            Q2 and Q4 go into the next assay.
          </h2>
          <p style={{ margin: "30px 0 0", maxWidth: 780, fontFamily: sans, fontSize: 25, lineHeight: 1.5, color: palette.softInk }}>
            The chemist writes the two experimental questions into an illustrative next-assay list.
          </p>
        </div>
        <div style={{ alignSelf: "center", padding: "34px 38px 30px", background: palette.white, border: "1px solid " + palette.ruleDark, ...rise(frame, 15) }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <Eyebrow color={palette.green}>Decision illustration / next assay list</Eyebrow>
            <span style={{ fontFamily: sans, fontSize: 14, fontWeight: 700, color: palette.muted }}>ILLUSTRATIVE, NOT A HISTORICAL SCHEDULE</span>
          </div>
          <div style={{ marginTop: 8, fontFamily: sans, fontSize: 46, fontWeight: 720, letterSpacing: "-0.05em", color: palette.ink }}>Q2 + Q4</div>
          <ScheduleRow id="Q2" change="para-F" start={6} />
          <ScheduleRow id="Q4" change="meta-Cl" start={20} />
          <div style={{ marginTop: 22, fontFamily: sans, fontSize: 17, lineHeight: 1.45, color: palette.softInk }}>
            MolActAgent predicted both active. The illustration places both on the proposed list.
          </div>
        </div>
      </div>
      <FooterSource>
        <span>MolActAgent / AI-agent decision illustration</span>
        <span>7 measured / 4 candidates / illustrative Q2 + Q4 list</span>
      </FooterSource>
    </AbsoluteFill>
  );
};

const SceneLayer: React.FC<{ children: ReactNode }> = ({ children }) => (
  <AbsoluteFill style={{ background: palette.paper, fontFamily: sans, color: palette.ink, overflow: "hidden" }}>{children}</AbsoluteFill>
);

export const AssayFilm: React.FC = () => {
  useFonts();
  return (
    <SceneLayer>
      <Fonts />
      <Sequence from={0} durationInFrames={156}><AssayScene /></Sequence>
      <Sequence from={144} durationInFrames={138}><ReferenceScene /></Sequence>
      <Sequence from={270} durationInFrames={108}><CandidateScene /></Sequence>
      <Sequence from={366} durationInFrames={156}><ComparisonScene /></Sequence>
      <Sequence from={510} durationInFrames={108}><Q4Scene /></Sequence>
      <Sequence from={606} durationInFrames={234}><SignoffScene /></Sequence>
    </SceneLayer>
  );
};
