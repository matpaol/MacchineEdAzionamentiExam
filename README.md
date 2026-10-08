# Re-testing a bearing-fault classifier on motor-current data

Reconstruction and evaluation of the deep-autoencoder + CNN method of Toma, Piltan & Kim, *A Deep Autoencoder-Based Convolution Neural Network Framework for Bearing Fault Classification in Induction Motors*, Sensors 21, 8453 (2021), on the real-damage bearings of the Paderborn KAt dataset.

Write-up with figures: **[matpaol.github.io/projects/bearing-diagnosis](https://matpaol.github.io/projects/bearing-diagnosis/)**

## Findings

- My reconstruction of the method reached **46.9%** test accuracy, against the **99.6%** reported in the paper.
- The SELU output layer cannot produce values below −1.758, while the unnormalized current reaches about −3 A: **25.4%** of samples are outside the decoder's output range, and 89.5% of the residual energy sits on those samples.
- Changing one thing at a time: bottleneck size, early stopping, weight decay, batch size and learning rate leave the AUC at about 0.60; a linear output layer gives 0.72 and scaling the input into the SELU range gives 0.77.

## Notebooks

Designed for Google Colab with data and results on Google Drive. Run them in order:

1. `01_analisi_dataset.ipynb` — dataset inventory and signal checks.
2. `02_replica_paper.ipynb` — reconstruction of the published method.
3. `03_indagine_replica.ipynb` — investigation of the residual, one change at a time.
4. `04_costruzione_dataset.ipynb` — extended dataset split by whole bearings.
5. `05_dataset_totale.ipynb` — training and evaluation on the extended dataset.

`config.py` holds shared paths and parameters; `funzioni.py` the functions used by the notebooks. The dataset and generated files are not included. Notebook text is in Italian.
