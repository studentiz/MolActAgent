import type { CSSProperties, ReactNode } from "react";
import { Img, staticFile } from "remotion";
import { palette, type ActivityLabel, type Compound } from "../data/assay";

export const sans = '"DM Sans", system-ui, sans-serif';

export const Structure: React.FC<{
  id: string;
  focus?: boolean;
  style?: CSSProperties;
}> = ({ id, focus = false, style }) => {
  return (
    <Img
      src={staticFile(`molecules/${id}${focus ? "-focus" : ""}.svg`)}
      style={{
        display: "block",
        width: "100%",
        height: "auto",
        objectFit: "contain",
        ...style,
      }}
    />
  );
};

export const ClassTag: React.FC<{
  label: ActivityLabel;
  compact?: boolean;
}> = ({ label, compact = false }) => {
  const active = label === "active";
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: compact ? 7 : 9,
        padding: compact ? "5px 9px 4px" : "7px 12px 6px",
        borderRadius: 999,
        color: active ? palette.greenDark : palette.warm,
        background: active ? palette.greenPale : palette.warmPale,
        border: `1px solid ${active ? palette.greenLine : palette.warmLine}`,
        fontFamily: sans,
        fontSize: compact ? 16 : 18,
        fontWeight: 650,
        lineHeight: 1,
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        whiteSpace: "nowrap",
      }}
    >
      <span
        style={{
          width: compact ? 7 : 8,
          height: compact ? 7 : 8,
          borderRadius: 99,
          background: active ? palette.green : palette.warm,
        }}
      />
      {label}
    </span>
  );
};

export const Eyebrow: React.FC<{ children: ReactNode; color?: string }> = ({
  children,
  color = palette.softInk,
}) => (
  <div
    style={{
      fontFamily: sans,
      fontSize: 17,
      fontWeight: 700,
      letterSpacing: "0.16em",
      lineHeight: 1.2,
      textTransform: "uppercase",
      color,
    }}
  >
    {children}
  </div>
);

export const SceneTitle: React.FC<{
  title: string;
  deck?: string;
  dark?: boolean;
}> = ({ title, deck, dark = false }) => {
  return (
    <div>
      <h2
        style={{
          margin: 0,
          fontFamily: sans,
          fontSize: 62,
          fontWeight: 680,
          letterSpacing: "-0.045em",
          lineHeight: 1.02,
          color: dark ? palette.paper : palette.ink,
          textWrap: "balance",
        }}
      >
        {title}
      </h2>
      {deck ? (
        <p
          style={{
            margin: "20px 0 0",
            maxWidth: 610,
            fontFamily: sans,
            fontSize: 23,
            fontWeight: 430,
            lineHeight: 1.45,
            color: dark ? "#DCE7E1" : palette.softInk,
          }}
        >
          {deck}
        </p>
      ) : null}
    </div>
  );
};

export const PlateHeader: React.FC<{ section: string; note?: string }> = ({
  section,
  note = "PPARα reporter / COS-1 cells",
}) => (
  <div
    style={{
      position: "absolute",
      top: 26,
      left: 72,
      right: 72,
      height: 46,
      borderTop: `1px solid ${palette.ink}`,
      borderBottom: `1px solid ${palette.rule}`,
      display: "grid",
      gridTemplateColumns: "1fr auto 1fr",
      alignItems: "center",
      fontFamily: sans,
      fontSize: 13,
      fontWeight: 650,
      letterSpacing: "0.13em",
      textTransform: "uppercase",
      color: palette.softInk,
    }}
  >
    <span>MolActAgent / AI agent</span>
    <span style={{ color: palette.ink }}>{note}</span>
    <span style={{ justifySelf: "end" }}>{section}</span>
  </div>
);

export const CompoundCard: React.FC<{
  compound: Compound;
  status?: "measured" | "predicted" | "outcome" | "untested";
  emphasis?: boolean;
  compact?: boolean;
  neutral?: boolean;
  style?: CSSProperties;
}> = ({
  compound,
  status = "measured",
  emphasis = false,
  compact = false,
  neutral = false,
  style,
}) => {
  const active = compound.label === "active";
  const cardBackground = neutral ? palette.white : active ? palette.greenPale : palette.warmPale;
  const cardBorder = neutral ? palette.rule : active ? palette.greenLine : palette.warmLine;
  const displayedStatus = neutral ? "candidate" : status;
  return (
    <div
      style={{
        position: "relative",
        overflow: "hidden",
        minWidth: 0,
        padding: compact ? "15px 16px 12px" : "20px 20px 16px",
        background: cardBackground,
        border: `1px solid ${cardBorder}`,
        boxShadow: emphasis ? `inset 0 0 0 3px ${palette.paper}` : undefined,
        ...style,
      }}
    >
      {emphasis ? (
        <div
          aria-hidden
          style={{
            position: "absolute",
            inset: 0,
            border: `3px solid ${active ? palette.green : palette.warm}`,
            pointerEvents: "none",
          }}
        />
      ) : null}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 12,
          fontFamily: sans,
          color: palette.ink,
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <span style={{ fontSize: compact ? 21 : 25, fontWeight: 760, letterSpacing: "-0.03em" }}>
            {compound.id}
          </span>
          <span
            style={{
              fontSize: compact ? 11 : 12,
              fontWeight: 680,
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: palette.softInk,
            }}
          >
            {displayedStatus}
          </span>
        </div>
        {neutral ? (
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: compact ? 7 : 9,
              padding: compact ? "5px 9px 4px" : "7px 12px 6px",
              border: `1px solid ${palette.ruleDark}`,
              borderRadius: 999,
              color: palette.softInk,
              background: palette.white,
              fontSize: compact ? 16 : 18,
              fontWeight: 650,
              lineHeight: 1,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              whiteSpace: "nowrap",
            }}
          >
            <span
              style={{
                width: compact ? 7 : 8,
                height: compact ? 7 : 8,
                borderRadius: 99,
                border: `1px solid ${palette.ruleDark}`,
              }}
            />
            untested
          </span>
        ) : (
          <ClassTag label={compound.label} compact={compact} />
        )}
      </div>
      <div
        style={{
          margin: compact ? "7px 0 4px" : "9px 0 5px",
          display: "flex",
          justifyContent: "space-between",
          gap: 12,
          minHeight: compact ? 21 : 25,
          fontFamily: sans,
          fontSize: compact ? 15 : 17,
          color: palette.softInk,
        }}
      >
        <span>{compound.change}</span>
        {compound.measurement ? (
          <strong style={{ color: palette.ink, fontWeight: 670 }}>{compound.measurement}</strong>
        ) : null}
      </div>
      <Structure
        id={compound.id}
        style={{
          width: "100%",
          height: compact ? 118 : 164,
          objectFit: "contain",
          objectPosition: "center",
        }}
      />
    </div>
  );
};

export const RuleLabel: React.FC<{
  activeText?: string;
  inactiveText?: string;
  compact?: boolean;
}> = ({ activeText = "ACTIVE  ≤ 1 µM", inactiveText = "INACTIVE  > 1 µM", compact = false }) => (
  <div
    style={{
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      border: `1px solid ${palette.ruleDark}`,
      fontFamily: sans,
      fontSize: compact ? 17 : 20,
      fontWeight: 690,
      letterSpacing: "0.07em",
    }}
  >
    <div
      style={{
        padding: compact ? "12px 16px" : "15px 20px",
        color: palette.greenDark,
        background: palette.greenPale,
        borderRight: `1px solid ${palette.ruleDark}`,
      }}
    >
      {activeText}
    </div>
    <div
      style={{
        padding: compact ? "12px 16px" : "15px 20px",
        color: palette.warm,
        background: palette.warmPale,
      }}
    >
      {inactiveText}
    </div>
  </div>
);

export const RegistrationMarks: React.FC = () => (
  <div aria-hidden style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
    {[
      { left: 40, top: 40 },
      { right: 40, top: 40 },
      { left: 40, bottom: 40 },
      { right: 40, bottom: 40 },
    ].map((position, index) => (
      <div
        key={index}
        style={{
          position: "absolute",
          width: 13,
          height: 13,
          ...position,
          borderTop: `1px solid ${palette.ruleDark}`,
          borderLeft: `1px solid ${palette.ruleDark}`,
          transform:
            index === 1 || index === 3
              ? "rotate(90deg)"
              : index === 2
                ? "rotate(-90deg)"
                : undefined,
        }}
      />
    ))}
  </div>
);

export const FooterSource: React.FC<{ children: ReactNode; light?: boolean }> = ({
  children,
  light = false,
}) => (
  <div
    style={{
      position: "absolute",
      left: 72,
      right: 72,
      bottom: 28,
      paddingTop: 12,
      borderTop: `1px solid ${light ? "#6E8B82" : palette.rule}`,
      display: "flex",
      justifyContent: "space-between",
      gap: 28,
      fontFamily: sans,
      fontSize: 14,
      lineHeight: 1.35,
      color: light ? "#C9D9D1" : palette.muted,
    }}
  >
    {children}
  </div>
);
