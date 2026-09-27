# Thesis figure review

Historical review: the figure numbers below refer to the earlier eight-chapter
layout. The 27 September submission revision consolidates the report into five
chapters and corrects the preprocessing illustrations; see `SUBMISSION_REVIEW.md`.

Reviewed 23 September 2026. The six requested figures were recreated as vector
PDFs with consistent typography, restrained colours, readable labels and explicit
arrow meanings. Their numbering is unchanged.

| Figure | Purpose of the revised figure | Content checked or corrected |
| --- | --- | --- |
| 3.2 | MLII polarity, followed by the channel used by each model | The positive electrode is LL-equivalent, rather than the previous LA label. The torso is schematic; the reference electrode describes the project sensor connection. NSRDB ECG1 is not relabelled as MLII. |
| 3.4 | Two preparation paths that meet at a 242-value feature vector | NSRDB fitting and validation are subject-disjoint. MITDB filtering and z-scoring occur within each 512-sample beat window. Measured previous/next RR intervals and earlier history contribute timing features. Labels stay separate from features. |
| 5.3 | Offline training, saved components, and two uses of the frozen models | Autoencoder, binary gate and subtype classifier have distinct roles. The corrected threshold is 0.5708333333. Temporal scoring and session prediction apply fixed components, without an arrow suggesting test results update training. |
| 5.4 | Read down the encoder, through the bottleneck, then up the decoder | A forward pass verified every displayed tensor shape and 2,680,641 parameters. Skip connections are scaled by 0.5; up blocks resize to their skip length, and the final 1 x 256 output is interpolated to 1 x 512. |
| 6.1 | Compare candidate selection with final refitting and testing | Cuts use accepted-beat order. Selection excludes 16 beats before both 64% and 80%; refitting restores the 64% exclusion and retains the 80% exclusion. The retrospective, known-patient interpretation is explicit. |
| 7.1 | Three parallel evaluation questions with overlapping populations | Binary: 20,969 beats; conditional subtype: 7,958; complete system: 20,043. The subtype-only evaluation bypasses the gate. The 926 other abnormal beats contribute only to binary evaluation. |

## Evidence used

- `experiments/corrected_temporal_holdout.json`: authoritative corrected protocol,
  candidate results, threshold, counts and supports.
- `experiments/nsrdb_autoencoder_training.json`: 18 subjects, split 15/3;
  288,673 fitting and 57,177 validation windows.
- `tools/final_temporal_holdout.py`: actual selection, refitting, boundary exclusions,
  subject grouping and scoring masks.
- `tools/corrected_dataset.py` and `../backend/app/ml/beat_preparation.py`:
  beat extraction, local filtering, local scaling and observed RR context.
- `../backend/app/ml/model.py`: executable architecture. Forward hooks independently
  confirmed encoder shapes 32 x 256, 64 x 128, 128 x 64, 256 x 32; decoder shapes
  128 x 32, 64 x 64, 32 x 128, 32 x 256; and final reconstruction 1 x 512.
- [PhysioNet lead nomenclature](https://archive.physionet.org/faq.shtml) and
  [MITDB signal description](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm):
  modified Lead II and channel exceptions. The lead reference is added to the bibliography.

## Related consistency repairs

The method and results generators previously read `final_temporal_holdout.json`,
the historical run, although the text reports the corrected experiment. They now
read `corrected_temporal_holdout.json`. Figures 3.1, 6.2, 6.3, 6.4 and 7.2 were
refreshed accordingly. Figure 6.4 was also changed from an overlapping sequential
diagram to a population-size chart, so it no longer implies that three separate
evaluations are successive filtering stages. Captions and list-of-figures entries
were updated. No new numbered figure was needed.

## Verification and limits

- Checked test-population arithmetic and summed per-record test counts.
- Executed the autoencoder with forward hooks to confirm dimensions and parameter count.
- The new diagram renderer rejects overlapping text, text outside cards and text
  outside the canvas. Rendered diagrams and compiled report pages were also inspected.
- Rebuilt the bibliography and PDF with BibTeX and pdfLaTeX; the source and figure PDFs
  are retained for further editing.
- This is a review of figure accuracy against the saved artifacts and implementation;
  it does not rerun training or establish new clinical or unseen-patient validity.
  The earlier test exposure remains disclosed. Existing appendix line-overflow
  warnings are outside these figure changes.

## Regeneration

Run from `Final_Report_Research`:

```powershell
python tools/thesis_diagrams.py
python tools/generate_method_chapter_figures.py
python tools/generate_results_chapter_figures.py
..\backend\venv\Scripts\python.exe -c "from tools.generate_data_chapter_figures import dataset_overview_figure; dataset_overview_figure()"
pdflatex -synctex=1 -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -synctex=1 -interaction=nonstopmode -halt-on-error main.tex
pdflatex -synctex=1 -interaction=nonstopmode -halt-on-error main.tex
```

If LaTeX requests another pass for references, run the last command again.
The existing chapter generators delegate their revised diagrams to
`tools/thesis_diagrams.py`, so they cannot silently recreate the previous layouts.
