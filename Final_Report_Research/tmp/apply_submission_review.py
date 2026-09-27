from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
def read(name): return (root/name).read_text(encoding='utf-8')
def write(name, value): (root/name).write_text(value, encoding='utf-8')
def lower(s):
    return re.sub(r'\\(chapter|section|subsection|subsubsection)(\*?)\{',
                  lambda m: '\\'+{'chapter':'section','section':'subsection','subsection':'subsubsection','subsubsection':'paragraph'}[m[1]]+m[2]+'{', s)
def replace(s, old, new):
    assert old in s, old[:100]
    return s.replace(old, new)

# Match the five substantive chapters required by the supplied guideline.
s=read('main.tex')
s=replace(s, '\\include{content/final/2_data.tex}\n\\include{content/final/2_theory.tex}\n\\include{content/final/3_architecture.tex}\n\\include{content/final/4_method.tex}', '\\include{content/final/3_methodology.tex}')
s=s.replace('\\addcontentsline{toc}{chapter}{List of Figures}\n','').replace('\\addcontentsline{toc}{chapter}{List of Tables}\n','').replace('\\balance\n','')
write('main.tex',s)
write('content/final/3_methodology.tex', r'''\chapter{Methodology}\label{ch:methodology}

This chapter describes how the research was conducted, from ECG data preparation
to evaluation of the frozen classification system. Section~\ref{ch:data}
defines the datasets, annotation groups, and allocation of examples.
Section~\ref{ch:theory} explains the theoretical basis of the signal features
and learning algorithms. Section~\ref{ch:architecture} specifies the implemented
models and their computational operations. Section~\ref{sec:method-training-inference}
then follows training and inference mathematically, using small worked
calculations and an end-to-end case study with real ECG segments.
Finally, Section~\ref{ch:method} defines the experimental design, evaluation
populations, metrics, and uncertainty analysis used for the results in
Chapter~\ref{ch:results}.

\input{content/final/2_data.tex}
\input{content/final/2_theory.tex}
\input{content/final/3_architecture.tex}
\input{content/final/4_method.tex}
''')

s=lower(read('content/final/2_data.tex'))
s=s.replace('This chapter identifies','This section identifies')
for a,b in {'Data and Preparation':'Data and preparation','Data Sources and Roles':'Data sources and roles','Data Cleaning':'Signal cleaning and window construction','Cleaning Checklist':'Cleaning checks','Window Construction':'Window construction','Cleaning Yield':'Cleaning yield','Data Preparation':'Model inputs and data allocation','Prepared Objects':'Prepared inputs','Exact Model Allocation':'Exact model allocation','Feeding the Frozen System':'Inputs to the frozen system'}.items(): s=s.replace('{'+a+'}', '{'+b+'}')
s=replace(s, 'The next chapter explains the theory\nbehind these features and the two-stage decision.', 'Section~\\ref{ch:theory} explains the theoretical basis of these features and\nthe two-stage decision. The allocation is evaluated under the protocol in\nSection~\\ref{ch:method}.')
write('content/final/2_data.tex',s)

s=read('content/final/2_theory.tex')
s=replace(s, 'This chapter explains the theory behind the final ECG system.', 'This section explains the theoretical basis of the ECG system.')
start=s.index('The concrete run used 288,673')
end=s.index('\\section{Hierarchical classification theory}',start)
s=s[:start]+r'''The implemented training settings and checkpoint-selection rule are specified
in Section~\ref{ch:architecture}. Section~\ref{sec:method-training-inference}
demonstrates the updates numerically and distinguishes the original training
record from the separate real-data teaching demonstration.

'''+s[end:]
start=s.index('\\subsection{How the two supervised models are fitted}')
end=s.index('\\section{Class and patient imbalance theory}',start)
s=s[:start]+r'''The binary model learns a normal/abnormal target from all eligible labelled
beats. The subtype model learns only from supported abnormal groups. Both
receive the same feature representation, while the reference labels remain
outside that representation. Section~\ref{sec:method-training-inference}
gives the fitting, weighting, validation, and refitting procedure for each
model. Selection criteria are defined in Section~\ref{ch:method}.

'''+s[end:]
start=s.index('\\section{From saved models to a live ECG decision}')
end=s.index('\\section{Validation theory}',start)
live=s[start:end]
s=s[:start]+s[end:]
# The statistical definitions belong to the evaluation protocol; retain their rationale here.
start=s.index('For binary detection, $TP$, $TN$, $FP$, and $FN$')
end=s.index('\\section{Summary of the theoretical framework}',start)
s=s[:start]+r'''A confusion matrix counts every true/predicted class pair, making aggregate
accuracy easier to interpret. Section~\ref{ch:method} defines the metrics and
bootstrap procedure. Subject-clustered uncertainty does not correct for prior
test exposure or establish generalisation to patients absent from development.

'''+s[end:]
s=s.replace(' a_i=\\mathbb{1}[p_i>0.5708333].',' a_i=\\mathbb{1}[p_i>\\tau].')
s=replace(s, 'If $a_i=0$, the output is N.', 'Here $\\tau$ is the operating threshold selected using validation data.\nIf $a_i=0$, the output is N.')
s=lower(s)
s=s.replace('{Theoretical Foundations}','{Theoretical foundations}')
write('content/final/2_theory.tex',s)

s=read('content/final/3_architecture.tex')
start=s.index('\\section{Mathematical training and inference procedure}')
end=s.index('\\section{Why this architecture was selected}')
components=s[:start]
training=s[start:end]
rationale=s[end:]
rationale=rationale.replace('Chapter~\\ref{ch:method} describes model selection and performance evaluation.', '')
components=replace(components, '\\chapter{Methodology}', '\\chapter{System design and model implementation}')
components=replace(components, 'This chapter presents the computational methodology, model training, and\nprediction procedure implemented in the ECG prototype.', 'This section specifies the model architecture and computations implemented\nin the ECG prototype.')
components=lower(components+live+rationale)
components+=r'''The following section connects these components into a complete training
and inference procedure, first with small calculations and then with recorded
ECG signals and the saved models.

'''
training+=r'''
The examples explain how the fixed hierarchy produces a label. Its overall
performance must be assessed on the complete evaluation populations rather
than on selected examples. Section~\ref{ch:method} defines that assessment.
'''
write('content/final/3_architecture.tex', components+training)

s=read('content/final/4_method.tex')
s=replace(s, '\\chapter{Experimental Method}', '\\chapter{Experimental design and evaluation}')
s=s.replace('\\section{Method overview}', '\\section{Evaluation workflow}')
start=s.index('\\section{Input data and feature construction}')
end=s.index('\\section{Temporal split}',start)
s=s[:start]+r'''\section{Separation of normal-reference and supervised development}

The data allocation in Section~\ref{ch:data} uses 15 NSRDB subjects for
autoencoder fitting and three separate subjects for checkpoint validation.
The normal-reference checkpoint is frozen before supervised development.
NSRDB windows do not enter the final MITDB temporal test, and MITDB class
labels do not update the autoencoder. The archived NSRDB scaling is retained;
it differs from the beat-local standardisation of supervised MLII windows,
as documented in Section~\ref{sec:method-real-case}.

Both supervised models use the 242-value representation specified in
Section~\ref{ch:architecture}. The fitting equations and numerical examples
are given in Section~\ref{sec:method-training-inference}; the remainder of
this section defines how that procedure was selected and evaluated.

'''+s[end:]
start=s.index('The fitting sequence for each candidate is explicit:')
end=s.index('\\begin{figure}[H]',start)
s=s[:start]+r'''The candidate-fitting rows, validation rows, sample weights, and final refit
are defined mathematically in Section~\ref{sec:method-training-inference}.
After selection, the binary model was refitted on all 83,050 eligible
development beats and the subtype model on 28,758 supported abnormal
development beats. The selected threshold was retained. Thus validation
measures model-selection performance, while the final temporal segment
measures the frozen refits under the stated retrospective protocol.

'''+s[end:]
start=s.index('The deployed decision is hierarchical.')
end=s.index('\\begin{figure}[H]',start)
s=s[:start]+'The three evaluation populations answer different questions about the\nhierarchy defined in Section~\\ref{ch:architecture}.\n\n'+s[end:]
external=read('content/final/5_results_external_incart.tex')
start=external.index('After model fitting,')
end=external.index('\\begin{table}',start)
external_protocol=external[start:end]
external=external[:start]+r'''The frozen hierarchy was evaluated on the limited INCART sample defined in
Section~\ref{sec:external-protocol}. The two records represent one source
patient and contain only N and PVC reference labels. Table~\ref{tab:external-incart-summary}
reports the measured performance on this sample.

'''+external[end:]
write('content/final/5_results_external_incart.tex',external)
start=s.index('\\section{Metrics}')
s=s[:start]+r'''\section{Limited external evaluation protocol}\label{sec:external-protocol}

'''+external_protocol+r'''No external labels were used to refit the models or adjust the threshold.
The result is an exploratory transfer check; one source patient cannot
support a patient-cluster population confidence interval. Performance is
reported in Section~\ref{sec:external-incart}.

'''+s[start:]
s=lower(s)
write('content/final/4_method.tex',s)

s=read('content/final/1_introduction.tex')
start=s.index('Chapters~\\ref{ch:literature} and~\\ref{ch:data}')
s=s[:start]+r'''The report follows five main chapters. Chapter~\ref{ch:literature} reviews
related work and identifies the research gap. Chapter~\ref{ch:methodology}
describes the methods: data preparation, theoretical foundations, system
implementation, mathematical training and inference, and experimental
evaluation. Chapter~\ref{ch:results} presents the results and their statistical
uncertainty. Chapter~\ref{ch:conclusion} discusses the findings, limitations,
conclusions, and recommendations. The appendix records the files and settings
needed to trace the reported experiments.
'''
s=s.replace('{Research Problem}', '{Research problem}')
write('content/final/1_introduction.tex',s)

s=read('content/final/2_literature_review.tex')
s=s.replace('learn normal reconstruction from one fixed channel of all 18 real\n    MIT--BIH normal-sinus recordings;', 'develop a normal-reference reconstruction model using one stored channel\n    from 18 MIT--BIH normal-sinus subjects, with 15 for fitting and three\n    for checkpoint validation;')
s=s.replace('The next chapter defines the data sources and annotation groups used to put\nthese design choices into practice.', 'Chapter~\\ref{ch:methodology} defines the data, computational methods, and\nevaluation protocol used to put these design choices into practice.')
write('content/final/2_literature_review.tex',s)

s=read('content/final/5_results.tex')
s=s.replace('window-preparation procedure described in Chapter~\\ref{ch:method}.', 'window-preparation procedure in Section~\\ref{ch:data} and the evaluation\nprotocol in Section~\\ref{ch:method}.')
write('content/final/5_results.tex',s)

s=read('content/final/6_conclusion.tex')
s=s.replace('\\chapter{Discussion and Conclusion}', '\\chapter[head=Discussion and conclusions]{Discussion, Conclusions and Recommendations}')
s=s.replace('MLII ensembles, with the same local model-window preparation at training and use.', 'MLII ensembles. The supervised models use the same local window preparation\nduring fitting and prediction.')
s=s.replace('The two-epoch training record does not establish optimal convergence.', 'The two-epoch training record does not establish optimal convergence.\nThe archived NSRDB windows also retain a historical amplitude scaling that\ndiffers from the current per-window standardisation path. Reusing those\ncaches reproduces the documented experiment, but a fresh reconstruction of\nthe normal-reference training data requires resolving that preprocessing\nhistory. This limits reproducibility from raw NSRDB signals alone.')
s=s.replace('\\section{Further research}', '\\section{Recommendations for further research}')
write('content/final/6_conclusion.tex',s)

# Cross-references now point to sections within the methods chapter.
for p in (root/'content/final').glob('*.tex'):
    s=p.read_text(encoding='utf-8')
    for label in ('ch:data','ch:theory','ch:architecture','ch:method'):
        s=s.replace('Chapter~\\ref{'+label+'}', 'Section~\\ref{'+label+'}')
    p.write_text(s,encoding='utf-8')

s=read('tex/config.tex')
s=s.replace('\\usepackage{balance}\n','')
s=s.replace('%% mathematics, and Source Sans Pro for navigation and headings.', '%% mathematics, and Helvetica for navigation and headings.')
s+='\n% KOMA-Script creates list entries and links at each list heading.\n\\KOMAoptions{listof=totoc}\n'
write('tex/config.tex',s)
s=read('content/0_pre/1_title.tex')
s=s.replace('\\pdfbookmark[0]{Prologue}{prologue}%\n\\pdfbookmark[1]{Title Page}{title}%', '\\pdfbookmark[0]{Title Page}{title}%')
write('content/0_pre/1_title.tex',s)
print('Applied submission structure and flow review.')
