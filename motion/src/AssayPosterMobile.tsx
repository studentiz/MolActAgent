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

const MobileResult: React.FC<{ compound: Compound }> = ({ compound }) => {
  const active = compound.label === "active";
  return (
    <div style={{ boxSizing: "border-box", height: 104, padding: "12px 14px", background: active ? palette.greenPale : palette.warmPale, border: "1px solid " + (active ? palette.greenLine : palette.warmLine) }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ width: 28, height: 28, borderRadius: 99, background: active ? palette.green : palette.warm, boxShadow: "inset 0 0 0 7px " + (active ? palette.greenLine : palette.warmLine) }} />
          <strong style={{ fontFamily: sans, fontSize: 32, color: palette.ink }}>{compound.id}</strong>
        </div>
        <strong style={{ fontFamily: sans, fontSize: 28, color: active ? palette.green : palette.warm }}>{compound.label}</strong>
      </div>
      <div style={{ marginTop: 8, fontFamily: sans, fontSize: 28, color: palette.softInk }}>{compound.measurement}</div>
    </div>
  );
};

const MobileCandidate: React.FC<{ compound: Compound }> = ({ compound }) => {
  const active = compound.label === "active";
  return (
    <div style={{ position: "relative", boxSizing: "border-box", height: 128, padding: "12px 10px", background: active ? palette.greenPale : palette.warmPale, border: "1px solid " + (active ? palette.greenLine : palette.warmLine) }}>
      {compound.selected ? <div aria-hidden style={{ position: "absolute", inset: 0, border: "4px solid " + palette.green }} /> : null}
      <strong style={{ display: "block", fontFamily: sans, fontSize: 36, color: palette.ink }}>{compound.id}</strong>
      <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 7, fontFamily: sans, fontSize: 28, color: active ? palette.greenDark : palette.warm }}>
        <span style={{ width: 10, height: 10, borderRadius: 99, background: active ? palette.green : palette.warm }} />
        {compound.label}
      </div>
    </div>
  );
};

const MobileFocus: React.FC<{
  id: string;
  title: string;
  measurement: string;
  status: string;
  active: boolean;
}> = ({ id, title, measurement, status, active }) => (
  <div style={{ boxSizing: "border-box", minWidth: 0, padding: "12px 11px 10px", background: active ? palette.greenPale : palette.warmPale, border: "1px solid " + (active ? palette.greenLine : palette.warmLine) }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 4 }}>
      <strong style={{ fontFamily: sans, fontSize: 34, color: palette.ink }}>{id}</strong>
      <strong style={{ fontFamily: sans, fontSize: 28, color: active ? palette.green : palette.warm, textAlign: "right" }}>{measurement}</strong>
    </div>
    <div style={{ height: 140, marginTop: 8, padding: 4, boxSizing: "border-box", background: palette.white, border: "1px solid " + palette.rule, overflow: "hidden" }}>
      <Structure id={id} focus style={{ width: "100%", height: "100%", objectFit: "contain" }} />
    </div>
    <div style={{ marginTop: 8, fontFamily: sans, fontSize: 30, fontWeight: 750, color: palette.ink }}>{title}</div>
    <div style={{ marginTop: 3, fontFamily: sans, fontSize: 28, lineHeight: 1.18, color: palette.softInk }}>{status}</div>
  </div>
);

export const AssayPosterMobile: React.FC = () => {
  useFonts();
  return (
    <AbsoluteFill style={{ background: palette.white, color: palette.ink, fontFamily: sans, overflow: "hidden" }}>
      <Fonts />
      <RegistrationMarks />

      <div style={{ position: "absolute", top: 22, left: 40, right: 40, display: "flex", justifyContent: "space-between", fontFamily: sans, fontSize: 19, fontWeight: 730, letterSpacing: "0.09em", textTransform: "uppercase", color: palette.softInk }}>
        <span>MolActAgent / AI agent</span>
        <span>Retrospective</span>
      </div>
      <div style={{ position: "absolute", top: 58, left: 40, right: 40, height: 1, background: palette.ink }} />

      <div style={{ position: "absolute", top: 82, left: 40, right: 40 }}>
        <h1 style={{ margin: 0, fontFamily: sans, fontSize: 61, fontWeight: 780, lineHeight: 1.02, letterSpacing: "-0.05em", color: palette.ink }}>
          From assay result<br />to next assay.
        </h1>
        <p style={{ margin: "14px 0 0", fontFamily: sans, fontSize: 31, lineHeight: 1.32, color: palette.softInk }}>
          MolActAgent predicted Q2 and Q4 active.<br />This plate illustrates that decision.
        </p>
      </div>

      <div style={{ position: "absolute", top: 308, left: 40, right: 40, height: 66, display: "grid", gridTemplateColumns: "1fr 1fr", border: "1px solid " + palette.ruleDark, fontFamily: sans, fontSize: 28, fontWeight: 760 }}>
        <div style={{ display: "flex", alignItems: "center", padding: "0 14px", background: palette.greenPale, color: palette.greenDark, borderRight: "1px solid " + palette.ruleDark }}>ACTIVE ≤ 1 µM</div>
        <div style={{ display: "flex", alignItems: "center", padding: "0 14px", background: palette.warmPale, color: palette.warm }}>INACTIVE &gt; 1 µM</div>
      </div>

      <section style={{ position: "absolute", top: 404, left: 40, right: 40 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 16 }}>
          <h2 style={{ margin: 0, fontFamily: sans, fontSize: 44, fontWeight: 760, letterSpacing: "-0.04em" }}>A / measured assay</h2>
          <span style={{ fontFamily: sans, fontSize: 28, color: palette.softInk }}>COS-1 · PPARα</span>
        </div>

        <div style={{ marginTop: 14, height: 220, padding: "15px 16px", boxSizing: "border-box", background: palette.paper, border: "1px solid " + palette.rule }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 190px", gap: 18, alignItems: "center", height: "100%" }}>
            <div>
              <div style={{ fontFamily: sans, fontSize: 31, fontWeight: 740, color: palette.ink }}>Cell reporter assay</div>
              <div style={{ marginTop: 4, fontFamily: sans, fontSize: 28, lineHeight: 1.25, color: palette.softInk }}>PPARα activation drives reporter expression.</div>
              <div style={{ marginTop: 12, display: "flex", gap: 8, padding: "10px 11px", border: "1px solid " + palette.ruleDark, borderRadius: 14, background: palette.white }}>
                {evidence.references.map((compound) => (
                  <span key={compound.id} style={{ width: 22, height: 22, flex: "0 0 22px", borderRadius: 99, border: "1px solid " + palette.ruleDark, background: compound.label === "active" ? palette.greenPale : palette.warmPale }} />
                ))}
              </div>
            </div>
            <div style={{ padding: "15px 12px", border: "2px solid " + palette.greenLine, background: palette.greenPale }}>
              <div style={{ fontFamily: sans, fontSize: 24, fontWeight: 760, letterSpacing: "0.05em", textTransform: "uppercase", color: palette.green }}>Measured</div>
              <div style={{ marginTop: 4, fontFamily: sans, fontSize: 46, fontWeight: 780, color: palette.ink }}>EC50</div>
            </div>
          </div>
        </div>

        <div style={{ marginTop: 16, display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <strong style={{ fontFamily: sans, fontSize: 34, color: palette.ink }}>7 measured</strong>
          <span style={{ fontFamily: sans, fontSize: 28, color: palette.softInk }}>3 active / 4 inactive</span>
        </div>
        <div style={{ marginTop: 10, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          {evidence.references.map((compound) => <MobileResult key={compound.id} compound={compound} />)}
        </div>
      </section>

      <section style={{ position: "absolute", top: 1224, left: 40, right: 40 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 16 }}>
          <h2 style={{ margin: 0, fontFamily: sans, fontSize: 44, fontWeight: 760, letterSpacing: "-0.04em" }}>B / predicted decision</h2>
          <span style={{ fontFamily: sans, fontSize: 30, fontWeight: 750, color: palette.green }}>Q2 + Q4</span>
        </div>
        <div style={{ marginTop: 14, display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 8 }}>
          {evidence.queries.map((compound) => <MobileCandidate key={compound.id} compound={compound} />)}
        </div>

        <div style={{ marginTop: 20, display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
          <strong style={{ fontFamily: sans, fontSize: 32, color: palette.ink }}>Same para position</strong>
          <span style={{ fontFamily: sans, fontSize: 28, color: palette.softInk }}>zoomed</span>
        </div>
        <div style={{ marginTop: 10, display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 9 }}>
          <MobileFocus id="R2" title="para-Cl" measurement="180 nM" status="active" active />
          <MobileFocus id="R5" title="para-OH" measurement=">3000 nM" status="inactive" active={false} />
          <MobileFocus id="Q2" title="para-F" measurement="untested" status="predicted active" active />
        </div>
      </section>

      <section style={{ position: "absolute", top: 1858, left: 40, right: 40, height: 290, boxSizing: "border-box", padding: "16px 18px", border: "3px solid " + palette.greenLine, background: palette.greenPale }}>
        <div style={{ fontFamily: sans, fontSize: 28, fontWeight: 760, letterSpacing: "0.05em", textTransform: "uppercase", color: palette.green }}>Recorded MolActAgent judgment</div>
        <div style={{ marginTop: 9, fontFamily: sans, fontSize: 31, fontWeight: 650, lineHeight: 1.28, letterSpacing: "-0.015em", color: palette.ink }}>
          “On the shared scaffold, a single para halogen on the terminal phenoxy ring distinguishes the active analogs from the inactive para-hydroxy and para-alkyl analogs.”
        </div>
      </section>

      <section style={{ position: "absolute", top: 2180, left: 40, right: 40, boxSizing: "border-box", height: 242, padding: "16px 18px", border: "3px solid " + palette.green, background: palette.white }}>
        <div style={{ fontFamily: sans, fontSize: 28, fontWeight: 760, letterSpacing: "0.05em", textTransform: "uppercase", color: palette.green }}>Chemist decision illustration / next assay list</div>
        {[
          { id: "Q2", change: "para-F" },
          { id: "Q4", change: "meta-Cl" },
        ].map((item) => (
          <div key={item.id} style={{ display: "grid", gridTemplateColumns: "64px 1fr auto", alignItems: "center", gap: 12, padding: "13px 0 10px", borderBottom: "1px solid " + palette.greenLine }}>
            <strong style={{ fontFamily: sans, fontSize: 38, color: palette.ink }}>{item.id}</strong>
            <span style={{ fontFamily: sans, fontSize: 32, color: palette.ink }}>{item.change}</span>
            <span style={{ display: "grid", placeItems: "center", width: 34, height: 34, border: "1px solid " + palette.green, borderRadius: 99, color: palette.green, fontFamily: sans, fontSize: 20 }}>✓</span>
          </div>
        ))}
      </section>

      <div style={{ position: "absolute", left: 40, right: 40, bottom: 20, paddingTop: 10, borderTop: "1px solid " + palette.ruleDark, fontFamily: sans, fontSize: 24, lineHeight: 1.2, color: palette.muted, whiteSpace: "nowrap" }}>
        ChEMBL 37 · CC BY-SA 3.0 · RDKit · Retrospective.
      </div>
    </AbsoluteFill>
  );
};
