# macchine ed azionamenti exam-> readme


Codice sviluppato per la replica e la valutazione di un metodo basato su autoencoder e CNN applicato al dataset di Paderborn.

I notebook sono predisposti per Google Colab, lavorano in integrazione con Google Drive per leggere dati e salvare risultati e devono essere eseguiti nell’ordine indicato:

1. `01_analisi_dataset.ipynb`: analisi preliminare del dataset.
2. `02_replica_paper.ipynb`: replica del metodo descritto nel paper.
3. `03_indagine_replica.ipynb`: analisi dei risultati della replica e del residuo.
4. `04_costruzione_dataset.ipynb`: costruzione del dataset esteso suddiviso per cuscinetto.
5. `05_dataset_totale.ipynb`: addestramento e valutazione sul dataset esteso.

`config.py` contiene percorsi e parametri condivisi.  
`funzioni.py` contiene le funzioni utilizzate dai notebook.

Il dataset e i file generati non sono inclusi nel repository.
