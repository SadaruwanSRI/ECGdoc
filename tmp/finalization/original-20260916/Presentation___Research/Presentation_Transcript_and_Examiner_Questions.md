# Presentation Transcript and Examiner Questions

Presentation: **MLII ECG Anomaly Detection and Arrhythmia Classification**  
Deck file: `Final_Research_Presentation.pdf`  
Purpose: practice a clear final research defense with the same theory, results, and limitations used in the report and presentation.

## How To Use This Script

Aim for a calm technical delivery. Do not read every sentence mechanically. Use each script as the core message for that slide, then speak naturally.

Recommended timing:

- Title and problem framing: 2 minutes
- Data, theory, and architecture: 7 minutes
- Method and validation: 5 minutes
- Results and limitations: 6 minutes
- Conclusion: 1 minute
- Questions: 10 minutes or more

## Slide-By-Slide Speaking Script

### Slide 1: MLII ECG Anomaly Detection and Arrhythmia Classification

Good morning. My research is about ECG anomaly detection and arrhythmia classification using one ECG lead, specifically MLII.

The final system is a one-lead temporal-holdout system. It first decides whether a beat is normal or abnormal. If the beat is abnormal, it then classifies the abnormal beat into one of five supported arrhythmia classes: PVC, PAC, LBBB, RBBB, or AFib.

The important boundary of this work is that the final result is not an unseen-patient clinical validation. It measures later ECG from known MIT-BIH patients, where earlier ECG from the same patients was available during development.

### Slide 2: The study makes two decisions from one ECG lead

This slide shows the main decision sequence. The system has two stages.

First, it receives a beat from the MLII lead and predicts whether that beat is normal or abnormal. Second, only if the beat is abnormal, the subtype classifier predicts the arrhythmia type.

The target was to exceed 90 percent whole-system accuracy. I also evaluated whether the subject-clustered 95 percent confidence interval had a lower bound above 90 percent.

The claim boundary is important. The result supports known-patient temporal continuation, not performance on completely unseen patients or an external hospital dataset.

### Slide 3: The problem is not a simple ECG-threshold task

This figure explains why a simple threshold on ECG amplitude is not enough.

Different arrhythmias affect the ECG in different ways. PVC, LBBB, and RBBB often change beat morphology. PAC and AFib can depend strongly on rhythm timing and RR interval patterns.

Because of this, the final model does not use only one signal value or one reconstruction error. It combines three kinds of evidence: MLII beat shape, rhythm timing from RR intervals, and mismatch from a normal-reference autoencoder.

### Slide 4: Normal-reference and arrhythmia data have separate roles

This slide separates the two datasets and their roles.

The MIT-BIH Normal Sinus Rhythm Database is used to train the normal-only autoencoder. Its job is to learn normal ECG reconstruction behavior.

The MIT-BIH Arrhythmia Database is used for the supervised part of the study. From MITDB, I use MLII beats to build the feature table, train the binary detector and subtype classifier, and perform the final temporal test.

This separation is important because the normal database is not used as labeled arrhythmia test data.

### Slide 5: The supervised system is strictly MLII-only

This slide clarifies the lead rule.

The supervised and deployed system uses only MLII. That means MLII is selected by signal name, not simply by taking the first channel. This matters because some MIT-BIH records do not contain MLII, and record 114 has MLII as the second stored channel.

Records 102 and 104 are excluded because they do not contain MLII. The autoencoder uses the first stored NSRDB ECG channel, but the supervised detector, subtype classifier, final test, and installed replay are MLII-only.

### Slide 6: Cleaning keeps the two signal roles explicit

Both signal pipelines produce one 512-sample window, but the two roles are not confused.

For the normal reference pathway, the NSRDB signal is used to train the autoencoder. For the arrhythmia pathway, MITDB MLII is cleaned, aligned, and converted into supervised features.

The key point is that preprocessing makes the inputs usable, but it does not change the fact that the NSRDB reference channel and MITDB MLII deployment channel are different signal sources.

### Slide 7: Training and use follow one fixed preparation route

This slide shows the preparation route used during development and installed use.

The signal is resampled, filtered, standardized, aligned around R peaks, and then converted into RR features and morphology features. The feature order is fixed.

This is important for deployment. The live or replay system must produce the same type of evidence that the saved classifier saw during training. Otherwise, the model would receive a different problem at runtime.

### Slide 8: The model uses morphology and rhythm evidence together

The theory behind the system is that ECG classification needs evidence at more than one scale.

Morphology describes the local shape of the beat: the QRS complex, local slopes, and waveform structure. Rhythm evidence describes how the beat occurs in time, especially through RR intervals.

This combination is especially important because some arrhythmias are mostly shape-based, while others are more rhythm-dependent.

### Slide 9: Raw ECG becomes compact numerical evidence

The model does not directly pass the raw waveform into the final Extra Trees classifiers.

Instead, each beat is summarized into compact numerical evidence. This includes pooled waveform values, derivatives, frequency information, and statistical summaries.

This design reduces sensitivity to small shifts and noise while keeping clinically meaningful information about beat shape and signal behavior.

### Slide 10: Normal reconstruction is evidence, not the diagnosis

The autoencoder is used as a source of evidence, not as the final decision-maker.

The autoencoder reconstructs a beat using knowledge learned from normal ECG. If the reconstruction error is high, that can suggest the beat is unusual.

However, the final diagnosis is not made by reconstruction error alone. The final supervised tree models also see MLII morphology and RR timing. This avoids treating every reconstruction mismatch as a direct arrhythmia label.

### Slide 11: The hierarchy separates abnormality from subtype

This slide shows the hierarchical decision design.

The first model is the binary gate. It estimates abnormal probability and compares it with the selected threshold.

If the beat is below the abnormal threshold, the output is normal. If it is above the threshold, the same feature vector is passed to the subtype classifier.

This separation is useful because the system first solves the broader normal-versus-abnormal problem, then solves the more specific abnormal subtype problem.

### Slide 12: Validation must respect patient clusters

ECG data contains many beats from the same person, so individual beats are not fully independent.

If I treated every beat as an independent patient, the uncertainty would look too optimistic. A record with many beats could dominate the result.

Therefore, the final uncertainty analysis uses subject-clustered resampling. This gives each subject cluster a more appropriate role in the confidence interval.

### Slide 13: The final runtime system is a one-lead hierarchy

This is the final system flow used by the installed software.

For each detected MLII heartbeat, the system builds one 242-value feature vector. That vector is passed to the abnormality detector. If abnormal, it is passed to the subtype classifier.

The key point is that the runtime system follows the same one-lead hierarchy described in the report. It is not a separate or simplified demo model.

### Slide 14: Each beat is scored from an R-aligned window

Each beat is represented by a 512-sample window.

The R peak is fixed at sample index 200. This gives the model signal information before the QRS complex and after it. That context is useful because arrhythmia evidence may appear before, during, or after the main QRS peak.

In live use, the system waits until enough post-peak signal and RR context are available before scoring the beat.

### Slide 15: Training, testing, and installed use are separated

This slide shows that training, testing, and installed use are separate stages.

The autoencoder, binary detector, and subtype classifier are frozen before installed use. During a live or replay session, the software applies the saved models. It does not update model weights using the live signal.

This matters because the final performance claim must refer to a fixed model, not a model that keeps changing during evaluation.

### Slide 16: The autoencoder checkpoint is frozen before supervised fitting

The autoencoder was trained only on normal sinus rhythm data.

It used 288,673 fitting windows from 15 NSRDB subjects and 57,177 validation windows from 3 different NSRDB subjects.

After this training, the checkpoint was frozen. The supervised MITDB models then used residual features produced by this frozen autoencoder. This prevents the autoencoder from adapting to the final test beats.

### Slide 17: Autoencoder mismatch becomes four residual features

The reconstruction mismatch is summarized using four residual features.

These are whole-window MAE, whole-window MSE, central morphology MAE, and residual-change MAE.

These features capture different aspects of reconstruction error. The whole-window values measure general mismatch, the central morphology value focuses on the beat center, and the residual-change value captures how sharply the error changes.

These four values are added to the supervised feature vector.

### Slide 18: Both supervised stages receive the same 242 values

The final feature vector has 242 values.

It contains 223 MLII waveform and statistical values, 15 RR timing values, and 4 autoencoder residual values.

Both supervised stages receive the same 242-value vector. The difference is in the task: the first model predicts abnormality, and the second model predicts the abnormal subtype.

### Slide 19: The final experiment keeps selection away from the test

The final experiment is designed to keep model selection away from the final test.

Candidate models are selected using earlier temporal regions. Then the selected approach is refit before the final test boundary. The final 20 percent timeline is used only for scoring.

This is important because using the test region for selection would inflate the final result.

### Slide 20: Every eligible record follows the same temporal split

This figure audits the split record by record.

For each eligible MITDB MLII record, the earlier section is used for development, a 16-beat boundary gap is inserted, and the final 20 percent is held out for temporal testing.

This makes the evaluation chronological within each record. It tests later signal from the same patients, rather than random beats mixed across time.

### Slide 21: Validation selected the final Extra Trees pair

The validation process selected an Extra Trees pair.

The binary detector uses 240 trees and a threshold of 0.6208333. The subtype model uses 260 trees and one-third feature sampling.

Extra Trees fits this design because the input is a structured tabular feature vector. The trees can learn nonlinear interactions between morphology, RR timing, and autoencoder residual features.

### Slide 22: Three evaluations answer different questions

The final study reports three related evaluations.

Binary detection asks whether the system separates normal and abnormal beats. This uses all accepted held-out beats.

Conditional subtype scoring asks whether the subtype model can classify true supported abnormal beats after the abnormal problem is isolated.

Whole-system scoring asks what the complete deployed hierarchy outputs across normal plus the five supported abnormal classes.

### Slide 23: The fixed test is later ECG from known patients

This slide states the final test scope.

The fixed test is the final 20 percent timeline of each eligible MITDB MLII record. Earlier ECG from the same records was available for development.

Therefore, this is a known-patient temporal test. It is useful because it tests later unseen signal, but it does not prove generalization to a completely new patient.

### Slide 24: The whole-system test contains six supported outputs

The whole-system test contains six outputs: normal, PVC, PAC, LBBB, RBBB, and AFib.

The whole-system test has 20,062 supported beats. Unsupported abnormal beats are not forced into one of these five subtype labels. They remain part of the binary abnormality evaluation only.

This avoids pretending the subtype classifier can identify classes it was not designed to support.

### Slide 25: Temporal whole-system accuracy is 96.86 percent

These are the main pooled temporal results.

The binary detector reached 98.47 percent accuracy. The conditional subtype classifier reached 95.20 percent accuracy. The complete six-class hierarchy reached 96.86 percent whole-system accuracy, with a macro F1 score of 0.9303.

The strongest single headline is that the complete system exceeded the 90 percent point-accuracy target in the stated temporal setting.

### Slide 26: The 95 percent interval clears 90 percent, but is still wide

The subject-clustered confidence interval for whole-system accuracy is 94.02 to 99.05 percent.

The lower bound is above 90 percent, so that planned success condition is met.

However, the interval width is 5.03 percentage points. The planned precision target was at most 3 percentage points, so this part was not met. That means the result is strong, but the uncertainty is still wider than desired.

### Slide 27: Most subject clusters exceed the target

This slide shows performance across subject clusters.

Forty-two out of 45 subject clusters are above 90 percent whole-system accuracy. The equal-subject whole-system accuracy is 96.90 percent.

This supports the conclusion that the result is not caused only by one large record. But the few lower-performing subject clusters are still important because they show where robustness can improve.

### Slide 28: PAC is the main unsupported recall goal

This slide shows end-to-end recall by arrhythmia class.

PVC recall is 94.63 percent, LBBB is 90.15 percent, RBBB is 97.76 percent, and AFib is 94.13 percent. PAC recall is lower at 70.05 percent.

So the main class-level weakness is PAC. This is clinically and technically important because PAC can be subtle and may depend strongly on rhythm context.

### Slide 29: The confusion matrix shows the remaining failure modes

The confusion matrix explains where the remaining errors occur.

The largest subtype confusions include LBBB being predicted as PVC, PAC being predicted as PVC, and PVC being predicted as LBBB. Some AFib and PAC beats are also sent to normal by the binary gate.

This tells us that future work should focus not only on overall accuracy, but also on boundary cases between similar abnormal morphologies and rhythm-sensitive classes.

### Slide 30: The goals were reached only for the stated temporal scope

This slide summarizes what was reached and what was not reached.

The system reached pooled whole-system accuracy above 90 percent. It also reached a subject-clustered lower confidence bound above 90 percent.

But the confidence interval was wider than planned, and PAC recall was below 90 percent. Also, completely unseen patients and external hospital data were not tested.

The safety conclusion is that this is a research prototype, not a clinical diagnostic device.

### Slide 31: The next study must add independent patients

The next study should keep the MLII input rule fixed, but add independent patients.

It should include more PAC, LBBB, RBBB, and AFib patients, and reserve an external patient-level database for one final test.

It should also pre-register the lower confidence bound, interval width, and per-class recall targets. Finally, it should compare the full model with and without autoencoder residual features to measure the exact contribution of the autoencoder evidence.

### Slide 32: Final conclusion

The final conclusion is that a single MLII lead can achieve 96.86 percent whole-system accuracy on unseen later signal from known patients.

The subject-clustered 95 percent confidence interval is 94.02 to 99.05 percent, so the lower bound exceeds 90 percent.

But the result must be interpreted carefully. It does not prove the same accuracy for a completely new patient. PAC recall and confidence interval width remain important weaknesses.

The final software, experiment artifact, report, and presentation now describe the same one-channel architecture.

### Slide 33: References

This slide lists the main sources used in the study.

If I am asked about the databases, I will refer to MIT-BIH Arrhythmia Database and MIT-BIH Normal Sinus Rhythm Database. If I am asked about ECG interpretation and validation theory, I will refer to the ECG standards, heartbeat classification literature, and clustered validation references listed here.

I do not need to present every reference in detail unless the examiner asks.

### Slide 34: References, continued

This is the continuation of the reference list.

The main point is that the work is grounded in existing ECG databases, ECG preprocessing and interpretation standards, anomaly detection theory, autoencoder theory, and patient-clustered validation methods.

I will keep this slide available for questions about sources and methodology.

## Examiner Questions And Suggested Answers

### 1. What is the main contribution of your research?

The main contribution is a one-lead MLII ECG system that combines morphology, RR timing, and frozen normal-autoencoder residual features in a supervised hierarchical classifier. It evaluates the complete system on a fixed temporal holdout and reports subject-clustered uncertainty.

### 2. Why did you use only MLII?

MLII is commonly available in the MIT-BIH Arrhythmia Database and is a useful lead for rhythm and beat morphology analysis. Using only MLII also makes the system simpler and keeps the training, testing, and deployment input rule consistent.

### 3. Why were records 102 and 104 excluded?

They were excluded because they do not contain MLII. The final supervised system is MLII-only, so including records without MLII would break the input rule.

### 4. Why is record 114 important?

Record 114 is important because MLII is not the first stored channel. The system selects MLII by signal name, so record 114 can be included correctly instead of being wrongly excluded or using the wrong channel.

### 5. Is this an unseen-patient validation?

No. This is a known-patient temporal holdout. Earlier ECG from each patient was used during development, and later ECG from the same patients was used for final testing. The result should not be claimed as unseen-patient or external clinical validation.

### 6. Why did you use a temporal split instead of a random beat split?

A random beat split can mix nearby beats from the same record into both training and testing, which can overestimate performance. A temporal split is stricter because the model is tested on later signal after earlier signal was used for development.

### 7. What does the 16-beat boundary gap do?

The boundary gap reduces leakage from very nearby beats around the development-test split. It helps separate the final test region from the training region in time.

### 8. Why use an autoencoder if the final model is supervised?

The autoencoder provides normal-reference residual features. It learns reconstruction behavior from normal ECG, and its reconstruction mismatch becomes additional evidence for the supervised models. It is not the final classifier by itself.

### 9. Why is reconstruction error not enough for diagnosis?

A high reconstruction error can indicate that a beat is unusual, but it does not directly identify the arrhythmia type. Some normal noise or morphology variation can also increase error. Therefore, the final decision combines reconstruction residuals with morphology and RR timing in supervised classifiers.

### 10. What are the 242 features?

The 242 features are 223 MLII waveform and statistical features, 15 RR timing features, and 4 autoencoder residual features. The same vector is used by both the binary detector and subtype classifier.

### 11. Why did you use Extra Trees?

Extra Trees works well for structured tabular features and can model nonlinear interactions. It also requires less deep-learning tuning than a large neural classifier and is suitable for the engineered morphology, RR, and residual feature vector.

### 12. What is the binary threshold?

The final binary threshold is 0.6208333. A beat with abnormal probability above this threshold is passed to the abnormal subtype classifier.

### 13. What exactly is whole-system accuracy?

Whole-system accuracy measures the final output of the complete hierarchy across six supported outputs: normal, PVC, PAC, LBBB, RBBB, and AFib. It includes the effect of both the binary gate and the subtype classifier.

### 14. Why report macro F1?

Macro F1 gives each class equal importance when averaging, so it is useful when classes are imbalanced. Accuracy can look high if the majority class dominates, while macro F1 better shows whether minority classes are also handled.

### 15. What was the best result?

The complete whole-system accuracy was 96.86 percent, with whole-system macro F1 of 0.9303. The equal-subject whole-system accuracy was 96.90 percent, with a subject-clustered 95 percent confidence interval of 94.02 to 99.05 percent.

### 16. Did the system meet all targets?

No. It met the pooled whole-system accuracy target and the lower confidence bound target. It did not meet the confidence interval width target, because the interval width was 5.03 percentage points instead of at most 3. It also did not meet a 90 percent recall goal for PAC.

### 17. Why is PAC performance lower?

PAC can be subtle and rhythm-dependent. Some PAC beats may look similar to normal or to other abnormal beats in a short window. The lower PAC recall suggests the system needs more independent PAC examples and possibly stronger rhythm-context modeling.

### 18. What are the main failure modes?

The main subtype confusions are LBBB to PVC, PAC to PVC, and PVC to LBBB. Some AFib and PAC beats are also missed by the binary gate and sent to normal.

### 19. Why use subject-clustered confidence intervals?

Because many beats come from the same subject or record. Beat-level confidence intervals would treat correlated beats as independent and could underestimate uncertainty. Subject-clustered resampling is more appropriate for patient-level evidence.

### 20. What does it mean that 42 of 45 subject clusters exceeded 90 percent?

It means most subject clusters performed above the target. This supports the stability of the result across subjects, but the three lower-performing clusters still show that the system is not uniformly reliable.

### 21. Can this system be used clinically now?

No. It is a research prototype. It has not been validated on independent patients, external hospital data, or real clinical deployment conditions. It should not be used as a clinical diagnostic device.

### 22. How does the live system align with the report experiment?

The live system enforces MLII, uses R-aligned 512-sample windows, calculates RR context, computes the four autoencoder residual features, and applies the saved final hierarchy. The deployed model artifact matches the report experiment artifact.

### 23. What is the role of AFib in a beat-level system?

AFib is a rhythm condition, so beat-level classification is limited. In this system, AFib evidence comes from RR timing and beat context. A future system should evaluate rhythm episodes more directly, not only individual beats.

### 24. What would you improve first?

I would first add independent patient-level validation, especially with more PAC and AFib examples. Then I would run ablation studies to measure the contribution of morphology, RR timing, and autoencoder residual features separately.

### 25. What is the biggest limitation of the study?

The biggest limitation is that the final test contains later signal from known patients, not completely unseen patients. Therefore, the result is meaningful for temporal continuation but not enough for broad clinical generalization.

### 26. Why are unsupported abnormal beats excluded from subtype scoring?

The subtype classifier was designed for five supported abnormal classes. Unsupported abnormal beats cannot be fairly assigned to one of those labels. They are included in binary abnormality testing, but not in the six-class supported subtype evaluation.

### 27. How do you know the presentation, report, and software are aligned?

They use the same final experiment artifact and the same deployed model bundle. The report and presentation state the same MLII-only input rule, temporal-holdout design, feature vector size, model choices, result values, and known-patient limitation.

### 28. Why not use a deep neural network classifier for the final stage?

A deep classifier could be tested in future work, but this study used engineered features plus Extra Trees for a transparent and stable tabular design. The aim was to combine interpretable evidence types and evaluate the complete hierarchy rigorously.

### 29. What is meant by frozen model?

Frozen means the model parameters are not updated during testing or live use. The autoencoder, binary detector, and subtype classifier are trained before evaluation and then kept fixed.

### 30. What would be required before clinical use?

The system would need independent-patient validation, external database validation, prospective testing, robustness checks with real acquisition hardware, clinical safety analysis, and regulatory review.

## Short Defense Summary To Memorize

My research developed a one-lead MLII ECG anomaly detection and arrhythmia classification system. The final system uses a frozen normal autoencoder only to provide residual features, while supervised Extra Trees models make the abnormality and subtype decisions. Each beat is represented by 242 values: morphology and statistics, RR timing, and autoencoder residual evidence.

The final temporal test used the last 20 percent of each eligible MIT-BIH MLII record. The complete six-class hierarchy reached 96.86 percent whole-system accuracy and 0.9303 macro F1. The subject-clustered 95 percent confidence interval was 94.02 to 99.05 percent.

The result is valid for later ECG from known patients. It is not an unseen-patient or clinical validation. The main weaknesses are PAC recall, the 5.03 percentage-point confidence interval width, and the need for independent external testing.
