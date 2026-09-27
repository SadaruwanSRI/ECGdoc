# Submission review — 27 September 2026

The report was reviewed against `Research project guidelines.pdf`, particularly
the five-chapter requirement on page 2. The revised source is `main.tex`.

## Required structure

| Guideline | Revised report |
| --- | --- |
| Introduction | Chapter 1: Introduction |
| Literature review | Chapter 2: Literature Review |
| Methods | Chapter 3: Methodology, retaining the author's requested title |
| Results | Chapter 4: Results and Statistical Validation |
| Discussion and conclusions with recommendations | Chapter 5: Discussion, Conclusions and Recommendations |

The supplied three-page guideline does not prescribe margins, typeface, page
limit, declaration text, or an approval page. The existing academic page design
and author/supervisor details were retained. No declaration or signature was
invented.

## Changes

- Consolidated the former data, theory, implementation, and experimental chapters
  into the Methods chapter, with an explicit reading map and updated references.
- Preserved the full mathematical training/inference explanation, small worked
  examples, and real ECG case study. The examples remain clearly distinguished
  from the complete experimental results.
- Removed repeated fitting loops and definitions from the theory and evaluation
  sections; moved the live implementation description beside the model design.
- Moved external-test selection and procedure into Methodology, leaving measured
  external performance in Results.
- Corrected the normal-reference preprocessing figure to show the actual archived
  NSRDB 16272 input. Corrected the supervised illustration to filter and scale
  its own 512-sample MLII window. Regenerated the data-allocation diagram.
- Carried the archived NSRDB scaling limitation into the discussion and clarified
  conditional-subtype bootstrap handling and undefined metric conventions.
- Corrected list-of-figures/table links, chapter/section cross-references, heading
  hierarchy, and page breaks. Renumbered equations, figures and tables automatically.

## Verification

`tools/verify_final_results.py` reproduced the saved test predictions, six-class
confusion matrix, macro F1, and whole-system subject-clustered confidence interval.
It also confirmed the autoencoder's implemented tensor dimensions and 2,680,641
parameters. Report edits did not retrain or replace the research models.

The bibliography and PDF were rebuilt with BibTeX and pdfLaTeX, with all citation
and cross-reference targets resolved. Rendered pages were reviewed for layout.
The review preserves the scientific limitations: retrospective known-patient
testing, prior test exposure, weak PAC recall, no autoencoder ablation, archived
normal-data scaling, and an external sample representing only one patient.

## Submission logistics from the supplied guideline

The guideline requires a hard copy at the department office and a soft copy on
the LMS by 4 p.m., one month after the Level IV Semester I examinations. It does
not supply the final calendar date. The defence is scheduled one week after report
submission. This editing review does not substitute for the department's
plagiarism check or supervisor's academic assessment.
