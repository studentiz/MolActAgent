import { AbsoluteFill } from "remotion";
import { Fonts } from "./components/Fonts";
import { useFonts } from "./components/UseFonts";
import {
  RegistrationMarks,
  Structure,
  sans,
} from "./components/AssayVisuals";
import {
  evidence,
  palette,
  type Compound,
} from "./data/assay";

const ResultTile: React.FC<{ compound: Compound }> = ({ compound }) => {
  const active = compound.label === "active";
  return (
    <div
      style={{
        boxSizing: "border-box",
        height: 94,
        padding: "10px 11px",
        background: active ? palette.greenPale : palette.warmPale,
        border: "1px solid " + (active ? palette.greenLine : palette.warmLine),
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}
    >
      <div
        aria-hidden
        style={{
          width: 25,
          height: 25,
          flex: "0 0 25px",
          borderRadius: 99,
          background: active ? palette.green : palette.warm,
          boxShadow: "inset 0 0 0 6px " + (active ? palette.greenLine : palette.warmLine),
        }}
      />
      <div style={{ minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 7, fontFamily: sans }}>
          <strong style={{ fontSize: 26, color: palette.ink }}>{compound.id}</strong>
          <strong style={{ fontSize: 21, color: active ? palette.green : palette.warm }}>{compound.label}</strong>
        </div>
        <div style={{ marginTop: 5, fontFamily: sans, fontSize: 21, color: palette.softInk }}>{compound.measurement}</div>
      </div>
    </div>
  );
};

const CandidateTile: React.FC<{ compound: Compound }> = ({ compound }) => {
  const active = compound.label === "active";
  return (
    <div
      style={{
        position: "relative",
        boxSizing: "border-box",
        minWidth: 0,
        height: 104,
        padding: "10px 11px",
        background: active ? palette.greenPale : palette.warmPale,
        border: "1px solid " + (active ? palette.greenLine : palette.warmLine),
      }}
    >
      {compound.selected ? <div aria-hidden style={{ position: "absolute", inset: 0, border: "4px solid " + palette.green }} /> : null}
      <div style={{ display: "flex", alignItems: "baseline", gap: 7 }}>
        <strong style={{ fontFamily: sans, fontSize: 34, color: palette.ink }}>{compound.id}</strong>
        <span style={{ width: 10, height: 10, borderRadius: 99, background: active ? palette.green : palette.warm }} />
      </div>
      <div style={{ marginTop: 10, fontFamily: sans, fontSize: 28, color: active ? palette.greenDark : palette.warm }}>
        {compound.label}
      </div>
    </div>
  );
};

const FocusTile: React.FC<{
  id: string;
  title: string;
  measurement: string;
  status: string;
  active: boolean;
}> = ({ id, title, measurement, status, active }) => (
  <div style={{ boxSizing: "border-box", minWidth: 0, padding: "10px 11px 9px", background: active ? palette.greenPale : palette.warmPale, border: "1px solid " + (active ? palette.greenLine : palette.warmLine) }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 7 }}>
      <strong style={{ fontFamily: sans, fontSize: 28, color: palette.ink }}>{id}</strong>
      <strong style={{ fontFamily: sans, fontSize: 22, color: active ? palette.green : palette.warm }}>{measurement}</strong>
    </div>
    <div style={{ height: 154, marginTop: 7, padding: 4, boxSizing: "border-box", background: palette.white, border: "1px solid " + palette.rule, overflow: "hidden" }}>
      <Structure id={id} focus style={{ width: "100%", height: "100%", objectFit: "contain" }} />
    </div>
    <div style={{ marginTop: 7, fontFamily: sans }}>
      <strong style={{ display: "block", fontSize: 25, color: palette.ink }}>{title}</strong>
      <span style={{ display: "block", marginTop: 2, fontSize: 21, color: palette.softInk }}>{status}</span>
    </div>
  </div>
);

export const AssayPoster: React.FC = () => {
  useFonts();
  return (
    <AbsoluteFill style={{ background: palette.white, color: palette.ink, fontFamily: sans, overflow: "hidden" }}>
      <Fonts />
      <RegistrationMarks />

      <div style={{ position: "absolute", top: 28, left: 56, right: 56, display: "flex", justifyContent: "space-between", fontFamily: sans, fontSize: 22, fontWeight: 730, letterSpacing: "0.1em", textTransform: "uppercase", color: palette.softInk }}>
        <span>MolActAgent / AI agent</span>
        <span>Retrospective illustration / ChEMBL 37</span>
      </div>
      <div style={{ position: "absolute", top: 62, left: 56, right: 56, height: 1, background: palette.ink }} />

      <div style={{ position: "absolute", top: 88, left: 56, right: 56, height: 84, display: "grid", gridTemplateColumns: "1fr 510px", gap: 44, alignItems: "center" }}>
        <div>
          <h1 style={{ margin: 0, fontFamily: sans, fontSize: 52, fontWeight: 780, lineHeight: 1.02, letterSpacing: "-0.05em", color: palette.ink }}>
            From assay result to next assay.
          </h1>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "8px 20px", paddingLeft: 24, borderLeft: "1px solid " + palette.ruleDark, fontFamily: sans, fontSize: 24, color: palette.softInk }}>
          <span>7 measured references</span><strong style={{ color: palette.ink }}>3 active / 4 inactive</strong>
          <span>4 candidates</span><strong style={{ color: palette.ink }}>Q2 + Q4 selected</strong>
          <span>Class cutoff</span><strong style={{ color: palette.green }}>1 µM</strong>
        </div>
      </div>

      <div style={{ position: "absolute", top: 188, left: 56, right: 56, height: 540, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <section style={{ boxSizing: "border-box", padding: "16px 18px", border: "1px solid " + palette.ruleDark, background: palette.white }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <h2 style={{ margin: 0, fontFamily: sans, fontSize: 36, fontWeight: 760, letterSpacing: "-0.04em" }}>A / measured assay</h2>
            <span style={{ fontFamily: sans, fontSize: 23, color: palette.softInk }}>COS-1 · PPARα · EC50</span>
          </div>

          <div style={{ marginTop: 12, height: 164, display: "grid", gridTemplateColumns: "1fr 100px 176px", alignItems: "center", gap: 18, padding: "12px 15px", boxSizing: "border-box", background: palette.paper, border: "1px solid " + palette.rule }}>
            <div>
              <div style={{ fontFamily: sans, fontSize: 23, fontWeight: 730, color: palette.ink }}>Cell reporter assay</div>
              <div style={{ marginTop: 4, fontFamily: sans, fontSize: 22, color: palette.softInk }}>PPARα activation drives reporter expression.</div>
              <div style={{ marginTop: 13, display: "flex", alignItems: "center", gap: 8, padding: "10px 12px", border: "1px solid " + palette.ruleDark, borderRadius: 14, background: palette.white }}>
                {evidence.references.map((compound) => (
                  <span key={compound.id} style={{ width: 18, height: 18, flex: "0 0 18px", borderRadius: 99, border: "1px solid " + palette.ruleDark, background: compound.label === "active" ? palette.greenPale : palette.warmPale }} />
                ))}
              </div>
            </div>
            <div style={{ fontFamily: sans, fontSize: 40, color: palette.ruleDark, textAlign: "center" }}>→</div>
            <div style={{ padding: "14px 13px", border: "2px solid " + palette.greenLine, background: palette.greenPale }}>
              <div style={{ fontFamily: sans, fontSize: 20, fontWeight: 760, letterSpacing: "0.07em", textTransform: "uppercase", color: palette.green }}>Measured endpoint</div>
              <div style={{ marginTop: 4, fontFamily: sans, fontSize: 37, fontWeight: 760, color: palette.ink }}>EC50</div>
            </div>
          </div>

          <div style={{ marginTop: 14, display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <strong style={{ fontFamily: sans, fontSize: 28, color: palette.ink }}>7 measured results</strong>
            <span style={{ fontFamily: sans, fontSize: 22, color: palette.softInk }}>≤ 1 µM active · &gt; 1 µM inactive</span>
          </div>
          <div style={{ marginTop: 9, display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 8 }}>
            {evidence.references.map((compound) => <ResultTile key={compound.id} compound={compound} />)}
          </div>
        </section>

        <section style={{ boxSizing: "border-box", padding: "16px 18px", border: "1px solid " + palette.ruleDark, background: palette.white }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <h2 style={{ margin: 0, fontFamily: sans, fontSize: 36, fontWeight: 760, letterSpacing: "-0.04em" }}>B / predicted decision</h2>
            <span style={{ fontFamily: sans, fontSize: 23, color: palette.green }}>Q2 + Q4</span>
          </div>
          <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 8 }}>
            {evidence.queries.map((compound) => <CandidateTile key={compound.id} compound={compound} />)}
          </div>
          <div style={{ marginTop: 15, display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <strong style={{ fontFamily: sans, fontSize: 27, color: palette.ink }}>Same para position</strong>
            <span style={{ fontFamily: sans, fontSize: 22, color: palette.softInk }}>ML comparison + computed change</span>
          </div>
          <div style={{ marginTop: 8, display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 8 }}>
            <FocusTile id="R2" title="para-Cl" measurement="180 nM" status="measured active" active />
            <FocusTile id="R5" title="para-OH" measurement=">3000 nM" status="measured inactive" active={false} />
            <FocusTile id="Q2" title="para-F" measurement="untested" status="predicted active" active />
          </div>
        </section>
      </div>

      <div style={{ position: "absolute", top: 730, left: 56, right: 56, height: 140, boxSizing: "border-box", padding: "14px 18px", border: "3px solid " + palette.greenLine, background: palette.greenPale }}>
        <div style={{ fontFamily: sans, fontSize: 22, fontWeight: 760, letterSpacing: "0.07em", textTransform: "uppercase", color: palette.green }}>Recorded MolActAgent / language-model judgment</div>
        <div style={{ marginTop: 7, fontFamily: sans, fontSize: 28, fontWeight: 650, lineHeight: 1.3, letterSpacing: "-0.015em", color: palette.ink }}>
          “On the shared scaffold, a single para halogen on the terminal phenoxy ring distinguishes the active analogs from the inactive para-hydroxy and para-alkyl analogs.”
        </div>
      </div>

      <div style={{ position: "absolute", top: 882, left: 56, right: 56, height: 126, boxSizing: "border-box", padding: "14px 18px", border: "2px solid " + palette.green, background: palette.white }}>
        <div style={{ display: "grid", gridTemplateColumns: "320px 1fr auto", gap: 24, alignItems: "center", height: "100%" }}>
          <div>
            <div style={{ fontFamily: sans, fontSize: 22, fontWeight: 760, letterSpacing: "0.07em", textTransform: "uppercase", color: palette.green }}>Chemist decision illustration</div>
            <div style={{ marginTop: 4, fontFamily: sans, fontSize: 27, color: palette.softInk }}>next assay list</div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            {[
              { id: "Q2", change: "para-F" },
              { id: "Q4", change: "meta-Cl" },
            ].map((item) => (
              <div key={item.id} style={{ display: "grid", gridTemplateColumns: "52px 1fr auto", alignItems: "center", gap: 10, paddingTop: 8, borderTop: "1px solid " + palette.greenLine }}>
                <strong style={{ fontFamily: sans, fontSize: 34, color: palette.ink }}>{item.id}</strong>
                <span style={{ fontFamily: sans, fontSize: 27, color: palette.ink }}>{item.change}</span>
                <span style={{ display: "grid", placeItems: "center", width: 30, height: 30, border: "1px solid " + palette.green, borderRadius: 99, color: palette.green, fontFamily: sans, fontSize: 19 }}>✓</span>
              </div>
            ))}
          </div>
          <div style={{ fontFamily: sans, fontSize: 29, fontWeight: 780, color: palette.green }}>Q2 + Q4</div>
        </div>
      </div>

      <div style={{ position: "absolute", left: 56, right: 56, bottom: 22, paddingTop: 9, borderTop: "1px solid " + palette.ruleDark, fontFamily: sans, fontSize: 18, lineHeight: 1.3, color: palette.muted }}>
        ChEMBL 37 (EMBL-EBI, CC BY-SA 3.0) · Zdrazil B, et al. Nucleic Acids Res. 2024;52:D1180-D1192 · Structures: RDKit 2025.09.2.
      </div>
    </AbsoluteFill>
  );
};
