# URL Variational Autoencoder (VAE) for QR Phishing (Quishing)
## Introduction

Quishing is phishing through QR codes. The code hides its link, so a person cannot inspect the URL before scanning it. This project asks whether a generative model can flag malicious QR code URLs after seeing only benign ones. We train a character-level Variational Autoencoder (VAE), with a GRU encoder and an autoregressive GRU decoder, on benign URLs. A URL the model finds hard to reproduce, measured by its per-character negative log-likelihood or negative ELBO, gets a higher score and is treated as more likely malicious. No malicious URL is used to train or tune the VAE. We judge the VAE against two simple floors that use only properties such as URL length and scheme, and against a supervised reference that has the same encoder but learns from labels. Splits are grouped by domain, results use 3 seeds, and confidence intervals resample whole domains. This is a class project for a graduate generative modeling course.

## Executive Summary

A character-level VAE trained only on benign QR code URLs separates malicious from benign test URLs better than both simple baselines (floors), but not well enough to work alone. Test results on 150,856 held-out URLs (mean of 3 seeds, 95% intervals from resampling whole domains):

| Model | AUROC | Recall at 1% false alarms |
|---|---|---|
| Floor A (URL length) | 0.587 [0.584, 0.590] | 0.014 |
| Floor B (6 simple URL features) | 0.764 [0.761, 0.767] | 0.096 |
| URL VAE (benign only) | 0.817 [0.815, 0.820] | 0.210 |
| Supervised reference (uses labels) | 0.975 [0.974, 0.975] | 0.584 |

The VAE beats floor A by 0.231 AUROC and floor B by 0.053, in every stratum we could test, and the weakest of its 3 seeds still beats floor A. It closes about 59% of the gap between floor A and the supervised reference.

Final Report: https://github.com/AnhJoe/url_vae_quishing/releases/tag/docs

Presentation: https://github.com/AnhJoe/url_vae_quishing/releases/tag/presentation

## Project Motivation

A QR code hides its destination, so a person cannot judge the link before scanning it, and attackers use that for phishing (quishing; see the FTC and Unit 42 references). Labeled attack data goes stale as attackers change tactics. A detector that learns only what benign URLs look like needs no labeled attacks, and flags a URL when the model finds it unlikely. Whether this works is an open question, autoencoders' reliability for anomaly detection is often challenged (Bouman & Heskes, 2025).

## Key Contributions

1. A character-level GRU VAE on URLs, trained on benign URLs only and scored by per-character negative log-likelihood and negative ELBO.
2. An evaluation framework: domain-grouped splits, two floors and a supervised reference, domain-bootstrap confidence intervals, paired comparisons, strata by URL length and scheme, and success criteria fixed on validation before the test set was used.
3. Evidence that the VAE learns more than the simple URL properties the floors use, with the caveats below.

## Data Summary

Data: https://drive.google.com/drive/folders/1i5dQyllOAbZAM4XvXrJ0d1MuUuJMGVeb?usp=sharing

The data is the URL lists of CIC-Trap4Phish (Canadian Institute for Cybersecurity, 2025), one benign and one malicious. We drop 38 rows (19 URLs) that appear in both classes, which leaves 1,005,700 URLs, about 57% malicious. All URLs of a domain go to the same split, so the model is tested on sites it has not seen.

| Split | Benign | Malicious |
|---|---|---|
| Train (70%) | 300,969 | 403,020 |
| Validation (15%) | 64,494 | 86,361 |
| Test (15%) | 64,494 | 86,362 |

Training uses only the benign train URLs. The malicious URLs are used for evaluation, and by the supervised reference for training.

## Methodology

- **Data and splits.** Each URL is one item, read as characters (up to 256). Items are split 70/15/15 into train, validation, and test, with all URLs of a domain in the same split.
- **Floors.** Two benign-only baselines, each a Gaussian fitted to the benign train URLs and scored by distance from typical. Floor A uses URL length only. Floor B uses six simple properties (length, scheme, IP host, host labels, path length, digit share).
- **URL VAE.** A one-direction GRU encoder (embedding 32, hidden 64) maps a URL to a latent code, and a GRU decoder rebuilds it character by character. It trains on benign train URLs for 8 epochs with the KL weight raised to 1 over the first 3 epochs and 10% word dropout. The epoch kept has the lowest negative ELBO per character on benign validation URLs. A sweep over latent size (8, 16, 32) and KL weight (0.01, 0.1, 1) on seed 0 picks the setting, using no malicious URL.
- **Scores.** Per-character negative log-likelihood (using the posterior mean) and negative ELBO (4 samples). A higher score means more likely malicious. The one with the higher validation AUROC is the headline score.
- **Supervised reference.** The same encoder plus a linear layer, trained with labels. It shows how much labels add and is not a contribution.
- **Evaluation.** AUROC, AUPRC, and recall at 1% and 5% false-positive rate, where the alarm threshold comes from benign validation URLs. Intervals resample whole test domains, and models are compared on the same resampled data. Results are also split by URL length and scheme. Success criteria were fixed on validation before the test set was used: the VAE beats floor A (paired AUROC difference with a 95% interval above 0) overall and in most strata, and the weakest of 3 seeds still beats it. The share of the floor-to-reference gap closed is reported without a threshold.

## Limitations and Caveats

- **Not deployable alone.** At 1% false alarms the VAE flags about 21% of malicious URLs. The supervised reference flags 58%, and it uses labels and shortcuts directly.
- **Small margin over floor B.** The VAE's gain over floor B is 0.053 AUROC. Malicious and benign URLs differ in length, scheme, and host, so part of the signal may be the same shortcuts the floors use. Winning in every stratum makes this less likely but does not rule it out.
- **The latent code is barely used.** Only 3 of 32 latent dimensions are active and the KL term is about 1 nat per URL, against roughly 100 nats to rebuild one. The model acts mostly like a character language model. We did not run a plain language model baseline to measure what the latent adds.
- **The tuning sweep is weak evidence.** Latent size 32 won over the default by 0.002 nats, which is within noise. The sweep ranks by negative ELBO, so it favors beta = 1 by construction and says nothing about whether a smaller beta helps detection. It uses seed 0 only.
- **Intervals describe the test data, not everything.** They resample domains and do not include training randomness. Seeds are reported separately (VAE AUROC 0.814 to 0.821), and 3 seeds is a small sample.
- **One dataset.** Everything comes from CIC-Trap4Phish URLs. Results may not carry over to other QR sources or newer attacks, and the malicious URLs may carry generation artifacts. The split is by domain, but similar sites may still appear on both sides.
- **URL text only.** The model sees no page content, QR image, or context. An attacker can craft URLs that look benign, and the VAE misses clean-looking `https` URLs.
- **Selection uses a few labels.** The choice between the two VAE scores and the alarm thresholds use the validation split. Training and the sweep use no malicious URL.
- **Truncation.** URLs are cut at 256 characters. This affects 0.19% of malicious and 0.05% of benign test URLs.

## Repository Structure

```
url_vae_quishing/
├── data/
│   ├── raw/                 QR code image folders (downloaded by 1_eda.ipynb, git-ignored)
│   ├── processed/           manifest and the shared items table (git-ignored)
│   └── artifacts/           trained model weights and training histories (git-ignored)
├── notebooks/
│   ├── 1_eda.ipynb
│   ├── 2_data_processing.ipynb
│   ├── 3_floors.ipynb
│   ├── 4_url_vae.ipynb
│   ├── 5_supervised_reference.ipynb
│   └── 6_results.ipynb
├── results/                 saved scores and run records, one folder per model, score, and seed (git-ignored)
├── src/
│   ├── prepare_data.py      builds the items table
│   ├── splits.py            domain-grouped train, validation, and test split
│   ├── loaders.py           URL cleaning and character encoding
│   ├── url_features.py      URL features used by the floors and the strata
│   ├── floors.py            floor A and floor B
│   ├── nn_common.py         shared batching, tensors, and device handling
│   ├── url_vae.py           URL VAE model, training, and scoring
│   ├── supervised.py        supervised reference model
│   ├── runner.py            run functions, tuning sweep, run records, and command line
│   ├── evaluate.py          metrics for one saved run
│   ├── analysis.py          confidence intervals, strata, paired comparisons, and success criteria
│   ├── inspect_vae.py       numbers behind the surprise, latent, and prior-sample figures
│   ├── plots.py             report figures
│   └── checks.py            data, model, and analysis checks
├── requirements.txt
└── README.md
```

## Notebooks and Run Order

Install the dependencies with `pip install -r requirements.txt`, then run the notebooks in this order. Each notebook is written to be run top to bottom.

1. `1_eda.ipynb` downloads the dataset into `data/raw`, cleans it, and saves the manifest (every URL with its cleaning decision) plus a summary of the data. Run once.
2. `2_data_processing.ipynb` assigns every URL to train, validation, or test with a domain-grouped split and writes the items table that all later notebooks read. Run once.
3. `3_floors.ipynb` fits two deliberately simple detectors (floor A on URL length and floor B on a few URL features) that a real model has to beat. No training and no randomness.
4. `4_url_vae.ipynb` is the main model. It runs the checks, sweeps the latent size and beta on validation data, and trains the character-level GRU VAE on benign URLs only for seeds 0, 1, and 2.
5. `5_supervised_reference.ipynb` trains a classifier with the same encoder as the VAE on labeled URLs for seeds 0, 1, and 2. It is context for how much labels add and not a contribution. Run it after notebook 4, because it reads the VAE's encoder sizes.
6. `6_results.ipynb` trains nothing. It reads the saved scores and produces the final tables, confidence intervals, strata, success criteria, and report figures.

Notebooks 3 to 5 skip a seed whose saved scores match the current settings and data, so an interrupted run can be restarted. Setting `QUICK = True` in notebooks 4 and 5 runs a short trial on a small slice of the data to test that everything works. Notebooks 3 to 6 need only `data/processed/items.csv.gz` and `data/processed/items_summary.json`, and their first cells set them up on Google Colab.

## References
Canadian Institute for Cybersecurity. (2025). CIC-Trap4Phish 2025 [Data set]. University of New Brunswick. https://www.unb.ca/cic/datasets/trap4phish2025.html

Nejati, F., Rabbani, M., Eskandarian, M., Mirani, M., Piya, G., Opushnyev, I., Ghorbani, A. A., & Dadkhah, S. (2026). CIC-Trap4Phish: A unified multi-format dataset for phishing and quishing attachment detection. arXiv. https://arxiv.org/abs/2602.09015

Marty, D.-O., Farooqi, S., & Starov, A. (2026, February 13). Phishing on the edge of the web and mobile using QR codes. Unit 42, Palo Alto Networks. https://unit42.paloaltonetworks.com/qr-codes-as-attack-vector/

Puig, A. (2023, December 6). Scammers hide harmful links in QR codes to steal your information. Federal Trade Commission, Consumer Advice. https://consumer.ftc.gov/consumer-alerts/2023/12/scammers-hide-harmful-links-qr-codes-steal-your-information

Almousa, H., & Alsuhibany, S. A. (2026). QR code security: An adaptive retraining approach for dynamic URL-based threat detection. Scientific Reports, 16, Article 13143. https://doi.org/10.1038/s41598-026-43002-z

Bouman, R., & Heskes, T. (2025). Autoencoders for anomaly detection are unreliable. arXiv. https://arxiv.org/abs/2501.13864

Trad, F., & Chehab, A. (2025). Detecting quishing attacks with machine learning techniques through QR code analysis. arXiv. https://arxiv.org/abs/2505.03451

## Team Information:
Joe Nguyen

Adithya Rajendra

Anishwar Chakraborty

Atharva Deshpande

Swaroop Veerla


