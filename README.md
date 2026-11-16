# CAM-CPCRN: Cross-Modal Alignment with Predictive Coding Recursive Network

CAM-CPCRN is a multimodal affective-state recognition framework for human-computer interaction (HCI). It combines self-supervised cross-modal representation alignment with recurrent predictive coding to learn from synchronized physiological and behavioral signals.

The framework is designed to process five modalities:

- **EEG** — electroencephalography
- **Video** — visual facial/behavioral information
- **Audio** — acoustic information
- **Gaze** — eye-tracking features
- **Physiology** — peripheral physiological measurements

The model predicts five study-defined analytical affective states: **Calm, Stressed, Cognitively Engaged, Fatigued, and Emotionally Dysregulated**. These categories are research labels, not clinical diagnoses.

## 1. Architecture

CAM-CPCRN contains the following components:

1. **Modality-specific encoders:** Separate neural encoders extract modality-specific representations from EEG, video, audio, gaze, and physiological inputs.
2. **Self-Supervised Cross-Modal Alignment Module (SCAM):** Projection heads map modality representations into a shared 128-dimensional latent space. A bidirectional InfoNCE objective encourages corresponding representations from synchronized modalities to become more compatible.
3. **Multimodal fusion:** The aligned representations are combined into a shared representation. Adaptive modality weighting can be used to account for unavailable modalities.
4. **Cross-Modal Predictive Coding Recursive Network (CPCRN):** A GRU-based recurrent module models temporal information. MLP predictors estimate one modality representation from another, and prediction errors provide feedback to the recurrent representation.
5. **Classification head:** A feed-forward classifier produces probabilities for the five study-defined affective-state categories.

### High-level pipeline

```text
EEG ───────────────┐
Video ─────────────┤
Audio ─────────────┼──> Modality-specific encoders
Gaze ──────────────┤             |
Physiology ────────┘             v
                      Cross-modal projection heads
                                  |
                                  v
                         SCAM / InfoNCE alignment
                                  |
                                  v
                       Adaptive multimodal fusion
                                  |
                                  v
                       GRU-based CPCRN module
                         /                 \
            Cross-modal prediction     Temporal state
                  errors                 representation
                         \                 /
                                  v
                           Classification
                                  |
                                  v
                     Five study-defined states
```

## 2. Datasets

The research evaluates the framework using publicly available affective-computing datasets, subject to each dataset's access conditions:

| Dataset | Official resource | Annotation considerations |
|---|---|---|
| MAHNOB-HCI | https://mahnob-db.eu/hci-tagging/ | Multimodal affective recordings and annotations |
| DEAP | https://www.eecs.qmul.ac.uk/mmv/datasets/deap/ | Continuous valence, arousal, and dominance annotations; derived categories must be defined explicitly |
| SEED | https://bcmi.sjtu.edu.cn/home/seed/ | Native Positive, Neutral, and Negative labels |
| AMIGOS | https://eecs.qmul.ac.uk/mmv/datasets/amigos/ | Affective annotations and physiological/multimodal recordings |

Dataset access may require registration, an application, a signed agreement, or compliance with the relevant terms of use. This repository does not redistribute dataset files.

### Label handling

The five CAM-CPCRN categories are study-defined analytical labels. They should not be assumed to be native labels in every dataset.

- **MAHNOB-HCI:** Apply the study's documented annotation-to-category procedure and report the exact thresholds and precedence rules used.
- **DEAP:** Treat valence and arousal as continuous dimensions. If deriving categories, document the thresholding procedure and label mapping.
- **SEED:** Prefer evaluation using the native three-class labels (Positive, Neutral, Negative), unless a separately justified harmonization protocol is explicitly implemented.
- **AMIGOS:** Document the selected annotation variables, thresholding procedure, and any mapping used for each experiment.

Results from datasets with different label spaces should not be presented as directly equivalent five-class evaluations.

## 3. Data preparation

Before training, prepare synchronized samples for each available modality. The preprocessing pipeline should document:

- participant identifiers and trial/clip identifiers;
- sampling rates, channel definitions, and signal filtering;
- video frame selection and face/behavior feature extraction, if applicable;
- audio segmentation and feature extraction;
- gaze and physiological signal cleaning;
- temporal synchronization and window segmentation;
- normalization parameters fitted using training data only;
- missing-modality representation and masking;
- label-generation rules and class distribution.

**Important:** Windows extracted from the same participant, trial, or stimulus clip are correlated. Use participant-grouped splits for participant-independent evaluation. If the objective is generalization to unseen stimuli, use a separate clip-grouped or leave-clips-out protocol so that no windows from a held-out clip occur in training. Do not randomly split overlapping windows across training and test sets.

## 4. Model configuration

The following values describe the configuration reported for the study. Confirm them against the released training configuration before reproducing results.

| Component | Reported configuration |
|---|---|
| Shared embedding dimension | 128 |
| EEG encoder output | 128 |
| Video encoder output | 256 |
| Audio encoder output | 128 |
| Gaze encoder output | 64 |
| Physiology encoder output | 64 |
| Fusion representation | 128 |
| CPCRN recurrent unit | GRU |
| CPCRN hidden dimension | 256 |
| Classifier | 128 → 64 → 5 |
| Dropout | 0.30 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Batch size | 64 |
| Maximum epochs | 100 |
| Early-stopping patience | 10 |
| Weight decay | 0.00001 |
| Adam β values | 0.9, 0.999 |

The final experimental configuration should also record the random seeds, software versions, hardware, preprocessing parameters, checkpoint-selection rule, and exact data partitions.

## 5. Training objectives

The training objective combines cross-modal alignment, cross-modal prediction, and supervised classification:

\[
\mathcal{L}_{total}
=
\lambda_{align}\mathcal{L}_{SCAM}
+
\lambda_{pred}\mathcal{L}_{pred}
+
\lambda_{cls}\mathcal{L}_{CE}
\]

where:

- \(\mathcal{L}_{SCAM}\) is the cross-modal contrastive alignment loss;
- \(\mathcal{L}_{pred}\) is the cross-modal prediction-error loss;
- \(\mathcal{L}_{CE}\) is the classification cross-entropy loss;
- \(\lambda_{align}\), \(\lambda_{pred}\), and \(\lambda_{cls}\) are loss weights.

Report the exact loss weights, temperature, negative-pair construction, and modality-pair selection used in each experiment. Do not assume that these values are identical across datasets unless the implementation enforces that condition.

## 6. Evaluation

Recommended evaluation outputs include:

- accuracy;
- macro-precision, macro-recall, and macro-F1;
- per-class precision, recall, and F1;
- confusion matrix;
- participant-level results;
- confidence intervals with the resampling unit clearly specified;
- inference latency and throughput, measured separately and with batch size stated;
- ablation and missing-modality results.

For participant-independent evaluation, group samples by participant before splitting. For unseen-stimulus evaluation, group by clip or stimulus identity. Report the number of participants, clips, trials, and windows in every partition. A large number of windows must not be treated as an equivalent number of independent participants.

## 7. Privacy and model compression

The study also describes a DP-SGD configuration and a knowledge-distillation experiment. The reported DP-SGD settings are:

| Parameter | Reported value |
|---|---:|
| Per-sample gradient clipping norm | 1.0 |
| Gaussian noise multiplier | 1.1 |
| Privacy parameter \(\epsilon\) | 2.1 |
| Privacy parameter \(\delta\) | \(1 \times 10^{-5}\) |
| Privacy accountant | Rényi Differential Privacy (RDP) |

These values constitute a valid privacy claim only when the accountant, sampling scheme, number of optimization steps, sampling rate, and implementation are verified and reported. Do not claim formal differential privacy based solely on adding noise to embeddings or gradients without a complete accounting procedure.

For the reported distillation experiment, the full model acts as the teacher and a smaller student is trained using classification and teacher-distribution matching. The reported temperature is 4.0 and the distillation-loss weight is 0.5. The reported model size changes from 62 MB to 25 MB. Verify the exact student architecture, loss definition, checkpoint sizes, and evaluation protocol before treating these as reproduced results.

## 8. Reproducibility checklist

Before publishing or comparing results, ensure that the repository provides:

- [ ] exact dataset access instructions and preprocessing scripts;
- [ ] participant-, trial-, and clip-level split files;
- [ ] label-generation code and documented thresholds;
- [ ] model definitions for all encoders, SCAM, fusion, and CPCRN;
- [ ] complete training and evaluation configuration files;
- [ ] fixed random seeds and repeated-run results;
- [ ] participant-level metrics and confidence intervals;
- [ ] clip-grouped evaluation and clip-ID-only control, where applicable;
- [ ] majority-class baseline and class-distribution report;
- [ ] modality ablations and missing-modality experiments;
- [ ] hardware, software versions, latency, and throughput measurement details;
- [ ] DP-SGD accountant output and distillation implementation, if those claims are included.

## 9. Limitations

The five output categories are study-defined affective-state labels and should not be interpreted as clinical mental-health diagnoses. Performance can be influenced by stimulus identity, repeated clips, correlated windows, participant overlap, and differences in annotation schemes across datasets. Participant-grouped evaluation does not by itself establish generalization to unseen clips; clip-grouped evaluation is needed for that claim. Cross-dataset experiments using different native label spaces should be interpreted as dataset-specific evaluations unless a validated common-label protocol is established.

## 10. Citation

If you use this implementation or reproduce the experiments, cite the associated CAM-CPCRN paper. Add the final bibliographic reference here once the manuscript's publication details are confirmed:

```bibtex
@article{camcpcrn,
  title   = {CAM-CPCRN: Cross-Modal Alignment with Predictive Coding Recursive Network},
  author  = {Add authors},
  journal = {Add journal or conference},
  year    = {Add year},
  doi     = {Add DOI}
}
```

## 11. License

Add the license governing this repository before public release. Dataset licenses and access agreements remain separate and must be followed independently.
