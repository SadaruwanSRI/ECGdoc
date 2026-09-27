from pathlib import Path
import re

ROOT = Path('Final_Report_Research/content')
changes = []

def edit(name, pairs):
    p = ROOT / name
    s = (Path('tmp/report_text_review_20260925/content') / name).read_text(encoding='utf-8')
    for old, new in pairs:
        assert old in s, f'Missing text in {name}: {old[:100]}'
        s = s.replace(old, new)
    p.write_text(s, encoding='utf-8', newline='\n')
    changes.append((name, len(pairs)))

edit('0_pre/4_abstract.tex', [(Path('tmp/report_text_review_20260925/content/0_pre/4_abstract.tex').read_text(encoding='utf-8'), r'''\chapter*{Abstract}\label{abstract}
\phantomsection
\pdfbookmark[1]{Abstract}{abstract}
\prevspace
Long ECG recordings contain many beats, making manual review difficult.
This study developed a single-lead electrocardiogram (ECG) system to detect
abnormal beats and classify them into five supported groups. A binary Extra
Trees model first separates normal and abnormal beats. A second Extra Trees
model assigns an abnormal subtype. The six output codes are N, PVC, PAC,
LBBB, RBBB, and AFib. These are study-specific annotation groups; in
particular, AFib includes atrial flutter, and some groups include escape beats.

A convolutional autoencoder was trained on the first stored ECG channel of
the MIT--BIH Normal Sinus Rhythm Database, using 15 subjects for fitting and
three for validation. Its weights were then fixed. The supervised models
used modified Lead II (MLII) data from 46 MIT--BIH Arrhythmia Database records,
representing 45 subject clusters. Each beat was described by 223 waveform,
spectral, and statistical features, 15 RR-interval features, and four
autoencoder residual features. Supervised windows were filtered and
standardised separately.

The final 20\% of accepted beats in each record formed the temporal test.
Earlier beats were used for model selection and fitting, with a 16-beat gap
before each evaluation boundary. Because the test segment had been viewed
during earlier development, this was a retrospective evaluation of known
patients, with a risk of optimistic performance estimates.

Binary accuracy was 98.81\% on 20,969 beats, and conditional subtype accuracy
was 97.63\% on 7,958 supported abnormal beats. Whole-system accuracy was
98.07\% and macro F1 was 0.9524 on 20,043 supported beats. Mean accuracy
across subjects was 98.03\%, with a subject-clustered 95\% confidence interval
of 96.29--99.25\%. However, PAC recall was only 70.63\%. A limited external
check used 5,416 beats from two INCART records belonging to one patient.
Whole-system accuracy was 92.85\%, while binary precision fell to 64.22\%.

The system met the aggregate accuracy and interval-width targets in the
retrospective test. It did not meet the recall target for every class.
Independent patient evaluation and real-sensor testing are still needed.
''')])

edit('final/1_introduction.tex', [
(r'\section{Introduction}', r'\section{Background}'),
('Electrodes on the body see this\nactivity from different directions.', 'Electrodes on the body record this\nactivity from different directions.'),
('An arrhythmia is a problem in where the heart\'s electrical activity starts,\nwhen it happens, how it is ordered, or how it travels. It may affect one beat,\nmany repeated beats, or a longer rhythm. This study uses one normal output and\nfive arrhythmia outputs.', 'Arrhythmias affect the timing or sequence of the heart\'s electrical activity.\nConduction disorders can also change the ECG waveform. This study uses one\nnormal-like output group and five abnormal output groups.'),
('These descriptions\ndefine the model task only. A clinical diagnosis still needs the full ECG and\npatient context', 'These descriptions explain the underlying ECG patterns. The study\'s annotation\ngroups are broader than these clinical descriptions and are defined in\nChapter~\\ref{ch:data}. A clinical diagnosis requires the full ECG and patient\ncontext'),
('Normal rhythm and the five arrhythmia types considered in this study.', 'ECG patterns underlying the six output groups considered in this study.'),
('Type & Main reason & Typical ECG behaviour and importance', 'Pattern & Main cause & Typical ECG features'),
('Normal sinus rhythm (N)', 'Normal sinus rhythm'),
('PVC & A premature', 'Premature ventricular contraction (PVC) & A premature'),
('PAC & A premature', 'Premature atrial contraction (PAC) & A premature'),
('LBBB & Electrical', 'Left bundle branch block (LBBB) & Electrical'),
('RBBB & Electrical', 'Right bundle branch block (RBBB) & Electrical'),
('AFib & Atrial', 'Atrial fibrillation (AFib) & Atrial'),
('Each panel comes from the\nfixed final 20\\% of its record.', 'Each panel comes from the\nfinal temporal-test portion of its record.'),
("that held-out group's mean waveform", "that record-and-class group's mean waveform"),
('This rule gives a representative example without choosing the most\ndramatic-looking beat.', 'This rule selects an example close to the group mean.'),
('How can an explainable two-stage ECG system use\nnormal-only reconstruction, beat morphology, and heartbeat timing to detect\nabnormal beats and correctly identify PVC, PAC, LBBB, RBBB, and AFib with more\nthan 90\\% whole-system accuracy on a fixed temporal test, while providing\nstatistically valid evidence across patients?', 'How accurately can a two-stage ECG system combine\nnormal-reference reconstruction, beat shape, and heartbeat timing to detect\nabnormal beats and classify five supported abnormal groups in later recordings\nfrom known patients? Can it exceed 90\\% whole-system accuracy while meeting\nthe study\'s confidence-interval and per-class recall targets?'),
('Testing must keep the final signal unseen and must calculate uncertainty\n    using complete patient clusters rather than treating beats as independent\n    people.', 'Evaluation must separate model fitting from testing and estimate uncertainty\n    using complete patient clusters. Any earlier exposure to the test data\n    must also be reported.'),
('and the final 20\\% is reserved for temporal testing.', 'and the final 20\\% is used for temporal testing.'),
('identifies five supported\narrhythmia types in later unseen signal from known patients.', 'classifies five supported\nabnormal groups in later ECG from known patients.'),
('with the complete first ECG channel\n    from all 18 MIT--BIH Normal Sinus Rhythm Database subjects and use its\n    reconstruction errors as evidence.', 'using the first ECG channel from all\n    18 MIT--BIH Normal Sinus Rhythm Database subjects, with separate fitting\n    and validation subjects, and use reconstruction errors as features.'),
('while keeping every final 20\\% portion unavailable.', 'and exclude the final 20\\% from fitting and selection in the rerun.'),
('features;\n    \\item Build', 'features.\n    \\item Build'),
('with a next-beat delay;','with a delay until the required signal and next R peak are available.'),
('\\item measure', '\\item Measure'),
('separately;\n    \\item report', 'separately.\n    \\item Report'),
('for every arrhythmia;\n    and\n    \\item validate uncertainty', 'for every supported abnormal group.\n    \\item Estimate uncertainty'),
('whether every supported class\n    reaches the accuracy target', 'whether every supported abnormal class\n    exceeds 90\\% recall'),
('an honest statement of the data, statistical, hardware, and clinical', 'a clear account of the data, statistical, hardware, and clinical'),
('has its independent benefit been measured?', 'what evidence supports its independent benefit?'),
('what evidence exists for external use?', 'what does the external check show?'),
('A successful software\nimplementation is distinct from meeting every statistical target.', '''A working prototype does not by itself show that every performance target
has been met.

\\section{Organisation of the report}
Chapter~\\ref{ch:literature} reviews related research and identifies the gap
addressed by this study. Chapter~\\ref{ch:data} describes the datasets,
labels, and preparation steps. Chapter~\\ref{ch:theory} explains the main
mathematical ideas, and Chapter~\\ref{ch:architecture} describes the system
architecture. Chapter~\\ref{ch:method} presents the experimental procedure.
Chapter~\\ref{ch:results} reports the results, and
Chapter~\\ref{ch:conclusion} discusses their meaning, limitations, and
implications for future research.''')])

edit('final/2_literature_review.tex', [
('This chapter reviews the work most relevant to the thesis questions: can a\nsystem detect and name ECG abnormalities, and how does performance change when\nearlier ECG from the same patient is available?', 'This chapter reviews research on ECG abnormality detection, heartbeat\nclassification, and evaluation when earlier ECG from the same patient is\navailable.'),
('PhysioNet provides open physiological databases', '\\section{ECG datasets and evaluation protocols}\n\nPhysioNet provides open physiological databases'),
('Its results therefore justify using morphology and ensemble learning, but they\ncannot be copied as an expected accuracy for this thesis.', 'Its findings support considering morphology and ensemble learning, but its\naccuracy does not provide a direct benchmark for the broader task in this study.'),
('Early automatic ECG systems used', '\\section{Supervised heartbeat classification}\n\nEarly automatic ECG systems used'),
('they can be controlled when only a few\npatients contain a class.', 'their settings can be adjusted when few\npatients represent a class.'),
('Autoencoders learn a compressed representation', '\\section{Normal-reference reconstruction}\n\nAutoencoders learn a learned representation'),
('a learned representation', 'an internal representation'),
('The literature does not mean that reconstruction error is a diagnosis.', 'Reconstruction error alone does not establish a diagnosis.'),
('one reconstruction threshold as the final\nanswer.', 'a single reconstruction threshold for its final\nclassification.'),
('ECG datasets contain unequal class counts', '\\section{Class imbalance and statistical validation}\n\nECG datasets contain unequal class counts'),
('If individual beats are resampled, the confidence\nintervals become too narrow.', 'Resampling individual beats can underestimate uncertainty and produce\nconfidence intervals that are too narrow.'),
('The literature provides strong individual ideas:', '\\section{Research gap and contribution}\n\nThe reviewed studies provide several relevant methods:'),
('However, reported results usually address only part of the present\nproblem.', 'However, the reviewed studies address different parts of the present\nproblem.'),
('The gap addressed by this thesis is therefore not simply the need for another\nclassifier. The need is to evaluate a complete two-stage system while making\nthe generalisation claim explicit:', 'This study brings these methods together in a two-stage system and evaluates\nboth its component models and its complete output. Its contribution is the\nintegration and evaluation of the following elements:'),
('keep each final test portion unavailable during selection;', 'exclude the final test portion from selection in the rerun and report\n    its earlier use during development;'),
('trains\non the first 80\\% of every record, and tests on the final 20\\%. The boundaries count accepted beats.', 'uses\nthe earlier 80\\% of each record for development and the final 20\\% for testing.\nBoundary gaps separate the regions, and percentages count accepted beats.'),
('Their accuracy\nvalues cannot be compared directly.', 'Their accuracy\nvalues cannot be compared directly without accounting for these differences.'),
('\\end{longtable}\n\\endgroup', '\\end{longtable}\n\\endgroup\n\nThe next chapter defines the data sources and annotation groups used to put\nthese design choices into practice. The study evaluates their combination;\nit does not establish that any single component improves performance on its own.\n')])

edit('final/2_data.tex', [
('\\chapter{Data}', '\\chapter{Data and Preparation}'),
('explains how the real signals are\ncleaned', 'explains how the signals are\nprepared'),
('supplies only healthy ECG for the autoencoder.', 'supplies normal-reference ECG for the autoencoder.'),
('Figure~\\ref{fig:data-source-overview} gives the same information as a flow\ndiagram. In short, NSRDB contributes only healthy windows for the frozen\nautoencoder, while MITDB contributes MLII beats for supervised development and\nthe final temporal test.', 'Figure~\\ref{fig:data-source-overview} summarises these roles. NSRDB supplies\nnormal-reference windows for the autoencoder, while MITDB supplies labelled\nMLII beats for supervised development and temporal testing.'),
('The two model stages are one-channel systems, but the channel identities are\nnot falsely treated as identical:', 'Both data streams contain one channel, but their lead identities differ:'),
('The MLII cache contains 104,755 accepted beats: 63,073\nnormal and 41,682 abnormal.', 'The prepared MLII dataset contains 104,755 accepted beats: 63,073\nin group N and 41,682 abnormal.'),
('Data cleaning converts two different raw ECG sources into the same numerical\ninput:', 'Data preparation converts the two ECG sources into a common input shape:'),
('The cleaning rules are fixed\nbefore evaluation and are reused after installation.', 'The supervised preparation rules were fixed for the rerun and are also used\nduring session prediction. NSRDB retains its original preparation procedure.'),
('The cleaning pipeline is easier to check as a list. Each item has one purpose\nand one failure rule.', 'Table~\\ref{tab:data-cleaning-checklist} lists the preparation steps and their\npurpose. Rejected windows do not enter model fitting or evaluation.'),
('Reduces baseline\nmovement and high-frequency noise without shifting the QRS complex.', 'Reduces baseline drift and high-frequency noise while avoiding filter-induced\nphase delay.'),
('centred on a beat: 200 samples before the R peak\nand 312 samples after it.', 'aligned to a beat: 200 samples before the R peak\nand 312 samples starting at the R peak.'),
('corrected\nresults in this thesis replace that preprocessing.', 'results reported here use the local\npreparation procedure.'),
('beat-centered MLII examples.', 'beat-aligned MLII examples.'),
('The accepted MLII cache\ncontains 104,755 beats: 63,073 normal and 41,682 abnormal.', 'The prepared MLII dataset\ncontains 104,755 beats: 63,073 in group N and 41,682 abnormal.'),
('\\subsection{Installed Use}', '\\subsection{Preparation during prediction}'),
('The unseen-use example is MLII from MITDB record', 'The supervised-input example is MLII from MITDB record'),
('a healthy $[1,512]$ window', 'a normal-reference $[1,512]$ window'),
('    \\item the next 16\\% selects the model structures and abnormal-probability\n    threshold;\n    \\item a 16-beat gap is removed before the test boundary;', '    \\item the next 16\\% selects the model structures and abnormal-score\n    threshold;\n    \\item a 16-beat gap is removed before both the validation and test\n    boundaries during selection;'),
('The final run removes a 16-beat gap before both selection-validation and\ntest boundaries and checks sample separation.', 'Final refitting uses the earlier 80\\% while retaining the gap before the test.\nThe procedure checks that signal windows do not overlap across each boundary.'),
('During prediction, one MLII window and its causal RR\nhistory are converted', 'During prediction, one MLII window, its observed previous and next RR\nintervals, and its recent RR history are converted'),
('A normal decision stops. An abnormal decision reuses that vector\nfor PVC, PAC, LBBB, RBBB, or AFib classification.', 'A decision of N ends classification for that beat. An abnormal decision passes\nthe same vector to the subtype model. The next chapter explains the theory\nbehind these features and the two-stage decision.')])

edit('final/2_theory.tex', [
('It follows the\nactual implementation: a one-lead MLII beat classifier that uses waveform\nshape, RR timing, autoencoder residuals, and two supervised Extra Trees\nensembles. The chapter does not assume a second ECG lead or an offline\nrecord-summary input, because the deployed model does not use those inputs.', 'The system combines\nMLII waveform shape, RR timing, and autoencoder residuals in two supervised\nExtra Trees ensembles. These components provide complementary information\nabout beat shape and rhythm.'),
('For every mathematical operation, it states what the symbols mean, what the\noperation changes, and why that change is useful.', 'The equations define the main operations and explain how they contribute to\nclassification.'),
('a fourth-order Butterworth band-pass\nwith 0.5--50~Hz cutoffs', 'a Butterworth band-pass filter with prototype order four\nand 0.5--50~Hz cutoffs'),
('The logarithm compresses large ratios.', 'The logarithm reduces differences between the relative-power values.'),
('This explicit calculation makes the feature vector\nauditable against the code.', 'This count matches the implemented feature vector.'),
('It creates a small beat-level delay during\nstreaming, but it also makes the live RR feature vector match the experiment.', 'It delays the streaming decision until the following peak is observed and\nallows the same RR features to be used in offline and live processing.'),
('where $P_b$ is the power inside the band', 'where $P_b$ is the power inside the band'),
('The final problem is not solved with one direct classifier. The system first\nasks whether the beat is abnormal. Only when this binary gate is true does it\nask which supported abnormal subtype is most likely.', 'The hierarchy divides classification into two stages. The first model assigns\na normal or abnormal label. Beats assigned to the abnormal group then enter\nthe subtype model.'),
('For beat $i$, the binary detector estimates $p_i=P(\\mathrm{abnormal}\\mid x_i)$.', 'For beat $i$, the binary detector returns an abnormal score $p_i$ from its\n242-value feature vector. This score estimates abnormal-class membership\nbut has not been independently calibrated as a clinical probability.'),
('This is the\ncore difference between a 240-tree model and 240 training epochs. The code\nfits 240 separate trees for the binary stage and 260 for the subtype stage;\nthere is no supervised epoch-loss trace to plot.', 'The binary model contains 240 trees and the subtype model contains 260 trees.\nTree count describes ensemble size, whereas an epoch describes one complete\npass through training data in neural-network fitting.'),
('class-$c$ beats in record\n$r$ (records 201 and 202 are combined).', 'class-$c$ beats in subject cluster\n$r$ (records 201 and 202 are combined).'),
('Later ECG from the same patient\nis then easier than ECG from a completely unseen patient.', 'This shared patient information can make classification easier than it would\nbe for a patient whose ECG was absent from development.'),
('The available lag is intentional: the system needs 312 post-peak samples\n(about 2.44 seconds) for the window and must also observe the next R peak.', 'The delay follows from the required input: the window includes 312 samples\nstarting at the R peak (about 2.44 seconds), and the next R peak must also\nbe observed.'),
('For binary detection, sensitivity is $TP/(TP+FN)$ and specificity is', 'For binary detection, $TP$, $TN$, $FP$, and $FN$ denote true positives, true\nnegatives, false positives, and false negatives. Sensitivity is $TP/(TP+FN)$\nand specificity is'),
('\\section{Theory-to-system alignment}', '\\section{Summary of the theoretical framework}'),
('\\end{table}\n', '\\end{table}\n'),
])

edit('final/3_architecture.tex', [
('\\chapter{Methodology}', '\\chapter{System Architecture and Implementation}'),
('This chapter explains the final system saved and used by the application. The\ninstalled model is a one-lead MLII beat classifier. It does not use two ECG\nleads, does not retrain during live use, and does not make an arrhythmia\ndecision directly from the autoencoder threshold. The final decision is made by\ntwo supervised Extra Trees models. Both receive the same 242-value feature\nvector.', 'This chapter describes how the theoretical components are implemented in the\nECG prototype. The system uses one MLII channel and three trained components:\na normal-reference autoencoder, a binary abnormality detector, and an abnormal\nsubtype classifier. The two supervised models use the same 242-value feature\nvector. All model parameters remain fixed during prediction.'),
('\\section{Architecture in one sentence}', '\\section{System overview}'),
('This rule prevents the model from silently scoring a lead that\ndoes not match the supervised training signal.', 'This rule checks the declared lead against the supervised training signal.\nIt cannot verify the physical electrode placement.'),
('\\item 312 samples after the R peak;', '\\item 312 samples starting at the R peak;'),
('The classifier is beat based, not arbitrary-window based.', 'The classifier assigns one output to each eligible beat.'),
('wave shape.', 'waveform shape.'),
('The autoencoder is a normal-reconstruction branch. It learns how normal ECG\nwindows can be reconstructed. The final system then uses reconstruction error\nas extra evidence. The autoencoder does not decide ``normal\'\' or\n``arrhythmia\'\' by itself in the final classifier.', 'The autoencoder learns to reconstruct normal-reference ECG windows. Its\nreconstruction errors provide four input features for the supervised models.\nThe final classification depends on the combined feature vector.'),
('For an input window $x\\in\\mathbb{R}^{512}$', 'For an input window $x\\in\\mathbb{R}^{512}$'),
(' f_3&=\\frac{1}{240}\\sum_{t=104}^{343}|e_t|', ' f_3&=\\frac{1}{240}\\sum_{t=105}^{344}|e_t|'),
('The central feature focuses on the QRS and nearby morphology.', 'These equations use one-based sample indices. The central range corresponds\nto the zero-based code slice \\texttt{104:344} and covers the QRS region and\nnearby waveform.'),
('The first supervised model is the abnormality detector. It receives all 242\nfeatures and estimates $p_i=P(\\mathrm{abnormal}\\mid x_i)$ for beat $i$.', 'The first supervised model is the abnormality detector. It receives all 242\nfeatures and returns an abnormal-class score $p_i$ for beat $i$.'),
('considers about $\\sqrt{242}\\approx15.6$ of the 242 feature positions', 'uses $\\lfloor\\sqrt{242}\\rfloor=15$ candidate feature positions'),
('\\item one-third feature sampling at each split;', '\\item feature-sampling fraction 0.33 at each split;'),
('The feature-sampling fraction 0.33 means roughly\n80 of the 242 feature positions are proposed at a node.', 'The feature-sampling fraction 0.33 gives a nominal candidate count of\n$\\lfloor0.33(242)\\rfloor=79$ feature positions at a node.'),
('the subtype with the largest subtype probability.', 'the subtype with the largest ensemble score.'),
('\\item Estimate $p_i=P(\\mathrm{abnormal}\\mid x_i)$ using the 240-tree binary\ndetector.', '\\item Calculate the abnormal score $p_i$ using the 240-tree binary detector.'),
('output N with confidence $1-p_i$.', 'output N with displayed score $1-p_i$.'),
('highest-probability supported subtype.', 'supported subtype with the highest score.'),
('Two-stage hierarchy & allows unsupported abnormal beats to count in binary\ntesting without forcing them into one of the five supported subtype labels', 'Two-stage hierarchy & separates binary detection from subtype evaluation;\nunsupported labels enter binary scoring only, although runtime predictions\nstill assign a supported subtype'),
('The architecture is therefore best described as a known-patient temporal\ncontinuation system. Its final performance numbers describe later MLII signal\nfrom patients whose earlier MLII signal was available during model development.\nThey should not be presented as unseen-patient generalisation without a\nseparate patient-disjoint evaluation.', 'The architecture supports both retrospective evaluation and streaming\nprediction. The evaluation reported here uses later MLII beats from patients\nwhose earlier data were available during development. Chapter~\\ref{ch:method}\ndescribes how the models were selected and how this performance was measured.')])

edit('final/4_method.tex', [
('The final experiment measures one specific use case: the system has already\nseen earlier MLII ECG from a patient, and it must classify later ECG from that\nsame patient. This is a known-patient temporal holdout. It is not a\nnew-patient external validation and it is not a shuffled random beat split.', 'The main experiment evaluated later MLII beats from patients whose earlier\nlabelled ECG was available during development. Beats were split in\nchronological order within each record. This design measures performance on\nknown patients. The final segment had been viewed during earlier development,\nso the reported results form a retrospective evaluation.'),
('pooled whole-system accuracy at least 90\\%;', 'pooled whole-system accuracy above 90\\%;'),
('Per-class recall and macro F1 were also checked so that a high overall score\ncould not hide failure on a smaller arrhythmia class.', 'The secondary target was recall above 90\\% for each supported abnormal\ngroup. Macro F1 and per-class results were also examined because overall\naccuracy can hide poor performance on a smaller class.'),
('measures clean-input MSE on the three untouched validation subjects.', 'measures clean-input MSE on the three subjects excluded from fitting.'),
('three different\nsubjects', 'three separate\nsubjects'),
('The two kinds of decrease should not be conflated.', 'Node entropy and validation error therefore measure different aspects of\nmodel fitting.'),
('one-third feature sampling,', 'a feature-sampling fraction of 0.33,'),
('For a binary outcome,', 'For binary detection, an abnormal beat is the positive class. $TP$ and $TN$\nare correctly classified abnormal and N beats; $FP$ is an N beat predicted\nas abnormal, and $FN$ is an abnormal beat predicted as N. The metrics are'),
('For class $c$,', 'For each output class $c$, the remaining classes are treated as negative\nwhen calculating'),
('Subject-macro accuracy first calculates accuracy inside each of', 'Equal-subject accuracy first calculates accuracy within each of'),
('The subject-macro value is primary for uncertainty because a long record should\nnot behave like many independent patients.', 'The equal-subject mean is used for uncertainty analysis so that subjects\nwith more beats do not receive more weight. For conditional subtype accuracy,\nonly subjects with supported abnormal test beats have a defined score; the\nmean uses that smaller set of subjects.'),
('Resampling individual beats would create an artificially narrow interval', 'Resampling individual beats can create an artificially narrow interval'),
('Tests against the 90\\% target', 'Exploratory tests against the 90\\% target'),
('Two\nnon-parametric tests provide sensitivity analyses.', 'Two\nnon-parametric tests provide supplementary, exploratory analyses.'),
('patient-independent generalisation claim.\n', 'patient-independent generalisation claim.\n\nThe next chapter reports each evaluation population separately and examines\nerrors at both stages of the hierarchy.\n')])

edit('final/5_results.tex', [
('This chapter reports the corrected retrospective temporal experiment in\n\\nolinkurl{experiments/corrected_temporal_holdout.json}. The rerun replaces\nrecord-wide supervised normalisation with local window preparation, removes\nunobserved neighbouring-RR substitutions, and saves beat-level predictions.\nThe original August artifact is retained for provenance. No new candidate\nfamily or threshold was selected from the temporal-test results.', 'This chapter presents the retrospective temporal results, their uncertainty,\nand the limited external INCART check. Binary detection, conditional subtype\nclassification, and whole-system classification are reported separately\nbecause they use different sets of beats. All results use the local\nwindow-preparation procedure described in Chapter~\\ref{ch:method}.\nThe saved experiment and prediction files are listed in the appendix.'),
('Evaluation population and supports', 'Evaluation populations and sample sizes'),
('Headline temporal results', 'Main temporal results'),
('The three aggregate checks pass: pooled accuracy exceeds 90\\%, the\nequal-subject lower confidence bound exceeds 90\\%, and the interval width is\nat most three percentage points.', 'For the whole system, all three primary targets were met: pooled accuracy\nwas above 90\\%, the lower confidence bound for equal-subject accuracy was\nabove 90\\%, and interval width was at most three percentage points.'),
('unbiased confirmatory success.', 'independent confirmation of these targets.'),
('The independent subtype model correctly names', 'When evaluated without the gate, the subtype model correctly classifies'),
('Gate errors can still send correctly\nclassifiable abnormalities to N.', 'Gate errors can assign N to an abnormal beat even when the subtype model\nwould have classified it correctly.'),
('The gate detects 8,698 of 8,884 abnormal beats and correctly rejects\n12,021 of 12,085 N beats.', 'The gate detects 8,698 of 8,884 abnormal beats and correctly assigns N to\n12,021 of 12,085 N beats.'),
('No controlled ablation establishes its independent benefit.', 'No controlled comparison with and without these features was conducted,\nso their independent contribution is not established.'),
('on their stated supports.', 'on their respective evaluation sets.'),
('The three aggregate accuracy/precision checks pass, but PAC', 'The whole-system accuracy and confidence-interval targets were met, but PAC'),
('broad external validation\nremains incomplete.', 'broad external validation\nremains incomplete.'),
])

edit('final/5_results_external_incart.tex', [
('After fitting the corrected hierarchy, its frozen threshold was applied to', 'After model fitting, the fixed hierarchy and threshold were applied to'),
('at the start of the audit.', 'at the start of the external evaluation.'),
('Only N and PVC have reference support.', 'Only N and PVC occur in the reference labels.'),
('Only N and PVC have reference support', 'Only N and PVC occur in the reference labels'),
] if False else [
('After fitting the corrected hierarchy, its frozen threshold was applied to', 'After model fitting, the fixed hierarchy and threshold were applied to'),
('at the start of the audit.', 'at the start of the external evaluation.'),
('Only N and PVC have reference support.', 'Only N and PVC occur in the reference labels.'),
('The stored six-label macro F1 is 0.3039 because it includes four zero-support\nclasses with F1 set to zero.', 'Macro F1 over the six predefined labels is 0.3039 when the four classes\nabsent from the reference labels are assigned F1 values of zero.'),
('reference-supported\nclasses', 'observed reference\nclasses'),
('test with support in every class.', 'test containing reference examples of every class.'),
('No exploratory threshold\noptimum from external labels is installed or reported as validation performance.', 'The external labels were not used to change the deployed threshold or to\nreport an optimised validation score.'),
])

edit('final/6_conclusion.tex', [
('The corrected one-lead hierarchy achieves', 'The single-lead hierarchy achieved'),
('The 2.95-point interval width meets the three-point\nprecision target.', 'The interval width was 2.95 percentage points, meeting the target of at most\nthree percentage points. The width was calculated from unrounded bounds.'),
('These results support further external\ntesting, not a clinical generalisation claim.', 'The lower precision shows that false alarms remain a major issue in this\nexternal sample. The one-patient design limits the scope of this finding.'),
('\\section{Achievement of the objectives}', '''\\section{Interpretation of the findings}
The high pooled accuracy shows that the system classified most supported
beats correctly in this temporal evaluation. It does not show uniform
performance across classes or patients. PAC recall was 70.63\\%, compared
with more than 96\\% for each of the other supported abnormal groups. The
113 PAC-to-PVC errors show that subtype confusion was a major source of
failure. Both the binary gate and subtype model therefore need improvement
for this group.

Conditional subtype accuracy measures how well the second model classifies
true supported abnormal beats when the gate is ignored. Whole-system accuracy
also includes the gate and the larger N group. The two percentages have
different denominators, so their difference is not a direct measure of the
cost of the gate. The class-level counts provide a clearer view of those
errors. Likewise, the wide conditional-subtype confidence interval shows
substantial variation across subjects despite high pooled accuracy.

The use of waveform and RR features follows earlier heartbeat-classification
research \\cite{dechazal2004heartbeat}. Normal-reference reconstruction adds
another source of information, but this experiment does not isolate its
benefit. Differences in labels, leads, and patient splits also prevent a
direct accuracy ranking against the studies reviewed in
Chapter~\\ref{ch:literature}.

The INCART check shows a further limitation. High sensitivity was accompanied
by 317 false alarms and binary precision of 64.22\\%. Thus, detecting most
abnormal beats did not ensure that most alerts were correct. Changes in
recording conditions, lead configuration, and the proportion of abnormal
beats may contribute to this difference, but the one-patient sample cannot
separate their effects.

\\section{Achievement of the objectives}'''),
('Interpretation of the corrections', 'Effect of the preprocessing revision'),
('The corrected architecture description reports those actual\ndimensions.', 'Chapter~\\ref{ch:architecture} reports these implemented dimensions.'),
('Scientific limitations', 'Limitations'),
('The result cannot establish cold-start accuracy.', 'The result does not establish accuracy for patients whose data were absent\nfrom model development.'),
('Selection optimism may\nremain', 'Earlier model-development choices may still make the results optimistic'),
('A matched\nablation should compare', 'A controlled\ncomparison should evaluate'),
('The already viewed temporal\nand INCART records should be treated as development evidence in future work.', 'Future work should treat the already viewed temporal and INCART records as\ndevelopment data and reserve separate patients for final evaluation.'),
('\\section{Final statement}', '\\section{Conclusion}'),
('This thesis delivers a corrected and reproducible single-lead research\nprototype with transparent retrospective results. It satisfies the measured\naggregate targets within the stated evaluation, while exposing persistent\nPAC and external-domain weaknesses.', 'This study developed a reproducible single-lead ECG prototype and evaluated\nits binary, subtype, and complete classification stages. The retrospective\nresults met the aggregate targets, but PAC recall and external false alarms\nremain important weaknesses.'),
])

edit('final/appendix.tex', [
('Final experiment artifacts', 'Experiment files'),
('Healthy fitting/validation subjects', 'Autoencoder fitting/validation subjects'),
('Healthy fitting/validation windows', 'Autoencoder fitting/validation windows'),
('Boundary gap & 16 beats before test', 'Boundary gap & 16 beats before validation and test'),
('Saved success checks', 'Recorded evaluation targets'),
('The JSON artifact records three Boolean checks. Pooled whole-system accuracy\nabove 90\\% is true. A whole-system subject-cluster lower 95\\% bound above 90\\%\nis true. Interval width at most three percentage points is true for the corrected rerun.', 'The saved JSON file records that all three whole-system targets were met:\npooled accuracy above 90\\%, a lower 95\\% confidence bound above 90\\%, and\nan interval width of at most three percentage points. These checks describe\nthe retrospective evaluation; they do not establish independent validation.'),
('The fixed temporal-holdout experiment is stored in:', 'Paths in this appendix are relative to \\nolinkurl{Final_Report_Research/}\nunless they begin with \\nolinkurl{backend/} or\n\\nolinkurl{Final_Report_Research/}; those paths are relative to the repository\nroot. The main temporal experiment is stored in:'),
])

# Keep all figures, including embedded TikZ drawings, unchanged.
for p in ROOT.rglob('*.tex'):
    s = p.read_text(encoding='utf-8')
    protected = []
    def save_drawing(m):
        protected.append(m.group(0))
        return f'@@DRAWING{len(protected)-1}@@'
    s = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', save_drawing, s, flags=re.S)
    for a,b in [('this thesis','this report'),('the thesis','the report'),
                ('labeled','labelled'),('beat-centered','beat-centred'),
                ('analyzer','analyser')]:
        s = s.replace(a,b)
    for i,drawing in enumerate(protected):
        s = s.replace(f'@@DRAWING{i}@@',drawing)
    p.write_text(s, encoding='utf-8', newline='\n')

print(changes)
