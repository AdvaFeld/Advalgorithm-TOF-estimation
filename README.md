# Advalgorithm-TOF-estimation

DEEP UNFOLDED WEIGHTED SPICE FOR RAPID AND ROBUST TOF ESTIMATION

This repository contains the research implementation of AI-adided TOF estimation framework based on deep unfolding of the weighted-SPICE fmaily of sparse covariance based estimators.
The objective is to combine the interpretability and model-based structure of classical methods such as SPICE, LIKES, and SLIM with data-driven learning, while maintaining a fixed and pre-controlled inference depth.

The proposed system uses two specialized deep-unfolded Weighted-SPICE estimators together with a lightweight neural classifier that performs differentiable soft routing between them.

## 1. Project Overview

This repository implements a deep-unfolded Weighted-SPICE framework for Time-of-Flight (ToF) estimation in multipath wireless channels.

The proposed approach combines the model-based structure of the SPICE family with learned parameters and a soft mixture-of-experts architecture. The goal is to improve robustness across different multipath conditions while maintaining a fixed inference depth.

---

## 2. Signal Model

The received frequency-domain signal is modeled as

```math
y_m=\sum_{c=1}^{C}\gamma_c e^{-j2\pi f_m\tau_c}+\epsilon_m,
```

where $\tau_c$ is the delay of the $c$-th propagation path, $\gamma_c$ is its complex coefficient, and $\epsilon_m$ represents noise.

A discretized delay dictionary is constructed and augmented with an identity matrix,

```math
\bar{\mathbf A}=[\mathbf A,\mathbf I_M],
```

and the algorithms estimate an augmented power vector $\mathbf p$, whose first $K$ entries correspond to the delay grid.

---

## 3. Deep-Unfolded Weighted-SPICE

The classical SPICE-family update is based on

```math
p_k^{(i+1)}
=
p_k^{(i)}
\frac{
|\bar{\mathbf a}_k^H\mathbf R^{-1}\mathbf y|
}{
\sqrt{w_k^{(i)}}
},
```

with

```math
\mathbf R
=
\bar{\mathbf A}
\operatorname{diag}(\mathbf p)
\bar{\mathbf A}^{H}.
```

The implemented model learns a combination of three weighting rules:

* **SPICE:** $w_k=\|\bar{\mathbf a}_k\|_2^2$
* **LIKES:** $w_k=\bar{\mathbf a}_k^H\mathbf R^{-1}\bar{\mathbf a}_k$
* **SLIM:** $w_k=1/p_k$

At every unfolded layer,

```math
w_{\mathrm{mix}}
=
q_1w_{\mathrm{SPICE}}
+
q_2w_{\mathrm{LIKES}}
+
q_3w_{\mathrm{SLIM}},
```

where the coefficients $q_1,q_2,q_3$ are learned and normalized using softmax.

The current implementation uses **10 unfolded layers**.

---

## 4. Model Architecture

The system contains two specialized deep-unfolded estimators:

* **Baseline DU expert** — uses one learned set of SPICE/LIKES/SLIM weights per unfolded layer.
* **Piecewise DU expert** — uses component-dependent learned weights and is intended for more challenging closely spaced multipath conditions.

A small MLP classifier receives the real and imaginary parts of the observed signal and produces one logit.

A sigmoid converts this logit into

```math
\pi=P(\text{non-baseline}\mid\mathbf y).
```

The final estimate is obtained through soft routing:

```math
\hat{\mathbf p}
=
(1-\pi)\hat{\mathbf p}_{\mathrm{baseline}}
+
\pi\hat{\mathbf p}_{\mathrm{piecewise}}.
```

Both experts are therefore evaluated and combined differentiably.

---

## 5. Training

Training is performed in two phases.

### Phase 1

The three components are trained separately:

1. Baseline DU model
2. Piecewise DU model
3. Binary classifier

The classifier distinguishes between

```text
0 -> baseline
1 -> non-baseline
```

### Phase 2

The pretrained models are loaded into the unified architecture and jointly fine-tuned.

The Phase-2 objective combines:

* a differentiable ToF-related power-profile loss;
* binary classifier supervision using `BCEWithLogitsLoss`.

Because the routing is soft and differentiable, gradients from the estimation loss can also propagate through the classifier probability.

---

## 6. Dataset and Scenarios

Synthetic multipath data are generated using:

* 64 frequency samples
* 40 MHz bandwidth
* 200 ns maximum delay
* 0.5 ns delay-grid spacing
* 3 propagation paths
* SNR values from 0 to 30 dB

The non-baseline dataset contains three path-separation regimes:

| Scenario   | Adjacent path spacing |
| ---------- | --------------------: |
| Moderate   |              10–20 ns |
| Close      |                3–6 ns |
| Very close |                1–2 ns |

The baseline samples are stored separately in `dataset_baseline/`.

---

## 7. Benchmarks

The proposed model is compared against classical Weighted-SPICE-family algorithms implemented in `Adva_algo_implementation.py`:

* SPICE
* LIKES
* LIKES with weight updates every 30 iterations
* SLIM
* SLIM limited to 5 iterations

The iterative benchmarks are evaluated with a maximum of 200 iterations and convergence tolerance of $10^{-3}$.

---

## 8. Evaluation

The estimated power vector is converted to ToF estimates by detecting the strongest peaks over the delay grid.

The main evaluation metrics are:

* all-path ToF RMSE;
* first-path ToF RMSE;
* classifier accuracy;
* classical algorithm runtime;
* number of iterations and convergence status.

Results are evaluated separately across SNR values and propagation scenarios.

---

## 9. Repository Structure

```text
model.py
    Model architectures and unified model

model_train.py
    Phase-1 training

model_train_phase2.py
    Joint Phase-2 training

loss_func.py
    Training losses

model_config.py
    Experiment configuration

ToA_DataGenerator.py
    Raw multipath signal generation

data_generator.py
    Dataset generation

tof_dataset.py
    PyTorch dataset interface

Adva_algo_implementation.py
    SPICE / LIKES / SLIM baselines

tof_eval_utils.py
    Peak detection and ToF extraction

result_analysis.py
    Individual-model evaluation

unified_result_analysis.py
    Unified-model and benchmark evaluation

classifier_validation.py
    Classifier checkpoint evaluation
```

---

## 10. Running the Code

### Generate the dataset

```bash
python data_generator.py
```

### Train the Phase-1 models

```bash
python model_train.py
```

or call the desired training function directly.

### Run Phase-2 joint training

```bash
python model_train_phase2.py
```

### Evaluate the final system

```bash
python unified_result_analysis.py
```

Checkpoint names and selected epochs should be configured inside the corresponding training and evaluation scripts.

---

## References

[1] P. Stoica, P. Babu, and J. Li, “New method of sparse parameter estimation in separable models and its use for spectral analysis of irregularly sampled data,” *IEEE Transactions on Signal Processing*, vol. 59, no. 1, pp. 35–47, Jan. 2011.

[2] P. Stoica, D. Zachariah, and J. Li, “Weighted SPICE: A unifying approach for hyperparameter-free sparse estimation,” *Digital Signal Processing*, vol. 33, pp. 1–12, 2014.

[3] İ. Güvenç and C.-C. Chong, “A survey on TOA based wireless localization and NLOS mitigation techniques,” *IEEE Communications Surveys & Tutorials*, vol. 11, no. 3, pp. 107–124, 2009.
