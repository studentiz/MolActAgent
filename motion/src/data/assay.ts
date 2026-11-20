import evidenceJson from "../../data/evidence.json";

export type ActivityLabel = "active" | "inactive";

export type Compound = {
  id: string;
  smiles: string;
  label: ActivityLabel;
  change: string;
  name?: string;
  measurement?: string;
  selected?: boolean;
  sourceUrl?: string;
};

export type Outcome = {
  id: string;
  accession: string;
  activityId: number;
  measurement: string;
  label: ActivityLabel;
  sourceUrl: string;
};

type EvidenceData = {
  assay: {
    target: string;
    host: string;
    readout: string;
    endpoint: string;
    cutoffNanoMolar: number;
    rule: string;
    predictionRecordSha256: string;
    referenceFileSha256: string;
    queryFileSha256: string;
    outcomeFileSha256: string;
  };
  references: Compound[];
  queries: Compound[];
  outcomes: Outcome[];
};

export const evidence = evidenceJson as EvidenceData;

export const palette = {
  paper: "#FBFAF6",
  white: "#FFFFFF",
  ink: "#152F2D",
  softInk: "#4B5F5C",
  muted: "#7A8985",
  rule: "#C9D2CE",
  ruleDark: "#96A8A2",
  green: "#22684F",
  greenDark: "#174C3A",
  greenPale: "#EAF2EC",
  greenLine: "#A9C5B5",
  warm: "#A7653F",
  warmPale: "#F4ECE4",
  warmLine: "#D6B49D",
  blue: "#527879",
} as const;

export const outcomeById = Object.fromEntries(
  evidence.outcomes.map((outcome) => [outcome.id, outcome]),
) as Record<string, Outcome>;
