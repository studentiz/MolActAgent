# MolActAgent assay film and figures

This directory contains the reproducible Remotion source for three release assets:

- `docs/assets/assay-film.mp4`: 28-second silent film, 1920 x 1080, 30 fps, H.264/yuv420p, no audio track.
- `docs/assets/assay-poster.png`: two-panel decision figure, 1600 x 1100 px.
- `docs/assets/assay-poster-mobile.png`: single-column portrait figure, 780 x 2560 px. At a 390 CSS-pixel display width this is a 2x asset, with body copy rendered at 28 source px (14 CSS px).

The assets use the checked-in COS-1 PPAR-alpha assay fixture and the published recorded
run in `examples/cell-assay/predictions.json`. They do not call an API or create new
predictions.

## Scientific content

- Host cells: COS-1.
- Target: human PPAR-alpha.
- Measured endpoint: EC50.
- Classification rule: active at or below 1 µM; inactive above 1 µM.
- The film shows all seven measured structures and all four candidate structures.
- The posters summarize the seven measured results as three active and four inactive markers,
  then show the four candidate IDs in one row.
- Q2 comparison: R2 `para-Cl`, 180 nM active; R5 `para-OH`, >3000 nM inactive;
  Q2 `para-F`, untested and predicted active.
- Q2's exact fluorine-only analogue is an extrapolation because R7 also carries a meta-methyl.
- Q4 retains `para-Cl` and changes the neighboring `meta-Me` to `meta-Cl`, using measured
  R4 (`para-Cl` + `meta-Me`, 156 nM active) as its matched comparator.
- The film does not display the held-out outcomes. The final frame is an explicitly
  illustrative chemist decision placing Q2 and Q4 on a proposed next-assay list.

The assay opening is a restrained schematic of the reporter-assay logic. It contains no
fabricated microscopy, fitted response curve, synthetic measurement trace, or error bar.

## Reproduce

Pinned versions:

- Node.js 24.14.1
- npm 11.11.0
- Python 3.9.6
- RDKit 2025.09.2
- Remotion 4.0.545
- React 19.3.1
- TypeScript 5.9.3

From this directory:

    npm ci
    python3 scripts/generate_molecules.py
    python3 scripts/verify_evidence.py
    npm run typecheck

`verify_evidence.py` reads `../examples/cell-assay/predictions.json` by default. It checks
the published prediction-file hash, all four predicted classes, the exact recorded Q2
judgment, the Q2 comparator classes, and the Cl/OH/F focus-crop margins. To verify a
different copy explicitly:

    SOURCE_PREDICTIONS="/path/to/predictions.json" python3 scripts/verify_evidence.py

Render the release assets with Google Chrome:

    npx remotion render src/index.ts AssayFilm ../docs/assets/assay-film.mp4 \
      --codec=h264 --crf=17 --pixel-format=yuv420p \
      --browser-executable="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
      --concurrency=4 --overwrite

    npx remotion still src/index.ts AssayPoster ../docs/assets/assay-poster.png \
      --browser-executable="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --overwrite

    npx remotion still src/index.ts AssayPosterMobile ../docs/assets/assay-poster-mobile.png \
      --browser-executable="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --overwrite

To render representative film frames:

    npm run render:frames -- --browser-executable="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

## Structure workflow

`scripts/generate_molecules.py` reads the exact SMILES strings in `data/evidence.json`,
generates deterministic RDKit 2D coordinates aligned to R2, and writes full structures plus
tight focus crops from those same drawings. `public/molecules/manifest.json` records the
highlighted atom indices, symbols, draw coordinates, and crop view boxes. The verifier
rejects focus crops that place Cl, OH, or F too close to an edge.

## Visual system

- White field with fine scientific-plate rules.
- DM Sans only, under the SIL Open Font License 1.1.
- Restrained green for active and warm brown for inactive.
- The 28-second film starts on a COS-1 reporter plate and measured EC50 endpoint, then shows
  the seven measured references, four neutral untested candidates, the recorded MolActAgent
  Q2 evidence judgment, the Q4 structural rationale, and the final chemist decision.
- No audio, no interaction, no HTML report ending, and no separate held-out-results screen.
- The desktop poster is composed for the GitHub README at reduced width. The mobile poster is
  a separate 780 x 2560 asset intended for a 390 CSS-pixel single-column display.

See `SOURCES.md` for provenance, required ChEMBL attribution, licenses, and limitations.
