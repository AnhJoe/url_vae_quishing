# URL Variational Autoencoder (VAE) for QR Phishing (Quishing)
## Introduction
This project trains a character-level Variational Autoencoder (VAE) on the URLs of benign QR codes only, then detects malicious URLs as strings the model explains poorly, without ever training on a malicious/benign label.

## Executive Summary

## Project Motivation

## Key Contributions

## Data Summary

Data: https://drive.google.com/drive/folders/1i5dQyllOAbZAM4XvXrJ0d1MuUuJMGVeb?usp=sharing

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


