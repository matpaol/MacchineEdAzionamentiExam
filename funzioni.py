"""
Funzioni condivise dai notebook del progetto.

Diagnosi di guasto ai cuscinetti dalla corrente di statore, dataset KAt-DataCenter
dell'Università di Paderborn. Approfondimento su Toma, Piltan, Kim (2021).

Uso su Colab:

    from google.colab import drive
    drive.mount('/content/drive')

    import sys
    sys.path.insert(0, '/content/drive/MyDrive/bearings_detection/codice')

    import config
    import funzioni as fn
    P = config.percorsi(sottocartella='01_analisi_dataset')

Le librerie pesanti (scipy, torch, scikit-learn) sono importate dentro le
funzioni che le usano, così il file resta importabile anche senza di esse.
"""

import os
import re
import glob
import hashlib
import math
import time
import subprocess

import numpy as np
import pandas as pd

import config


# download ed estrazione

def elenco_remoto(url=config.URL_REPOSITORY):
    """Nomi dei cuscinetti disponibili sul repository di Paderborn."""
    import requests
    html = requests.get(url).text
    return [l[:-4] for l in re.findall(r'href="([^"]+)"', html) if l.endswith('.rar')]


def scarica_archivi(cuscinetti, cartella_raw, url=config.URL_REPOSITORY, verbose=True):
    """Scarica i .rar mancanti. Rieseguibile: salta quelli già presenti."""
    import requests
    for nome in cuscinetti:
        destinazione = os.path.join(cartella_raw, nome + '.rar')
        if os.path.exists(destinazione):
            continue
        if verbose:
            print('scarico', nome)
        with open(destinazione, 'wb') as f:
            f.write(requests.get(url + nome + '.rar').content)
    archivi = sorted(glob.glob(os.path.join(cartella_raw, '*.rar')))
    return archivi, sum(os.path.getsize(a) for a in archivi)


def estrai_schede(cuscinetti, cartella_raw, cartella_documenti):
    """
    Estrae i soli PDF (scheda del danno e measuring log) senza scompattare i .mat.
    Sono la fonte dell'anagrafica in config.ANAGRAFICA.
    """
    for nome in cuscinetti:
        destinazione = os.path.join(cartella_documenti, nome)
        if glob.glob(os.path.join(destinazione, '*.pdf')):
            continue
        os.makedirs(destinazione, exist_ok=True)
        subprocess.run(['unrar', 'e', '-o+', '-inul',
                        os.path.join(cartella_raw, nome + '.rar'),
                        '*.pdf', destinazione + '/'], check=True)
    return {n: sorted(os.listdir(os.path.join(cartella_documenti, n)))
            for n in cuscinetti if os.path.isdir(os.path.join(cartella_documenti, n))}


def estrai_misure(cuscinetti, cartella_raw, cartella_estratti):
    """Scompatta i .mat dei cuscinetti indicati. Salta quelli già estratti."""
    for nome in cuscinetti:
        if glob.glob(os.path.join(cartella_estratti, nome, '**', '*.mat'), recursive=True):
            continue
        archivio = os.path.join(cartella_raw, nome + '.rar')
        if not os.path.exists(archivio):
            raise FileNotFoundError('archivio mancante: ' + archivio)
        subprocess.run(['unrar', 'x', '-o+', '-inul', archivio,
                        os.path.join(cartella_estratti, nome) + '/'], check=True)


# lettura dei file .mat

def apri(mat_file):
    """Restituisce la struct del file, qualunque sia il nome della variabile."""
    from scipy.io import loadmat
    mat = loadmat(mat_file, simplify_cells=True)
    for chiave in mat:
        if not chiave.startswith('__'):
            return mat[chiave]
    raise KeyError('nessuna variabile utile in ' + str(mat_file))


def leggi(mat_file, nome_canale, struct=None):
    """Campioni di un canale, come array float."""
    struct = struct if struct is not None else apri(mat_file)
    for segnale in struct['Y']:
        if segnale['Name'] == nome_canale:
            return np.asarray(segnale['Data'], dtype=float)
    raise KeyError(nome_canale + ' non trovato in ' + str(mat_file))


def ricampiona(x, fs_originale, fs_bersaglio=config.FS_ATTESO,
               tolleranza=config.TOLLERANZA):
    """
    Riporta un segnale alla frequenza di campionamento voluta.

    Se la frequenza misurata gia coincide con quella voluta entro la tolleranza
    il segnale torna intatto, senza interpolazioni inutili. Altrimenti si
    ricampiona con un filtro polifase.

    Serve perche il DAE ha un ingresso di dimensione fissa: un giro meccanico
    deve durare lo stesso numero di campioni in tutte le registrazioni, e nel
    dataset la frequenza non e uniforme.
    """
    from fractions import Fraction
    from scipy.signal import resample_poly

    x = np.asarray(x, dtype=float)
    if abs(fs_originale - fs_bersaglio) / fs_bersaglio <= tolleranza:
        return x

    rapporto = Fraction(fs_bersaglio / fs_originale).limit_denominator(1000)
    return resample_poly(x, rapporto.numerator, rapporto.denominator)


def leggi_ricampionato(mat_file, nome_canale=config.CANALE_CORRENTE,
                       fs_bersaglio=config.FS_ATTESO, struct=None):
    """
    Legge un canale e lo riporta alla frequenza voluta.

    Restituisce (segnale, campionamento), dove campionamento contiene le
    grandezze misurate prima del ricampionamento: servono per sapere quali
    registrazioni sono state toccate e di quanto.
    """
    struct = struct if struct is not None else apri(mat_file)
    campionamento = misura_campionamento(mat_file, nome_canale, struct)
    x = leggi(mat_file, nome_canale, struct)
    return ricampiona(x, campionamento['frequenza_Hz'], fs_bersaglio), campionamento


def assi_dei_tempi(mat_file, struct=None):
    """
    Frequenza di campionamento di ciascun asse dei tempi, misurata come
    (numero di passi) / (intervallo coperto). Sul singolo passo pesa
    l'arrotondamento della memorizzazione, sull'intero intervallo no.
    """
    struct = struct if struct is not None else apri(mat_file)
    righe = []
    for asse in struct['X']:
        valori = np.asarray(asse['Data'], dtype=float)
        intervallo = valori[-1] - valori[0]
        righe.append({'raster': asse['Raster'], 'campioni': valori.size,
                      'intervallo_s': intervallo,
                      'frequenza_Hz': (valori.size - 1) / intervallo})
    return pd.DataFrame(righe)


def canali(mat_file, struct=None):
    """Tabella dei segnali, ciascuno con la frequenza del proprio raster."""
    struct = struct if struct is not None else apri(mat_file)
    frequenze = {r['raster']: r['frequenza_Hz']
                 for _, r in assi_dei_tempi(mat_file, struct).iterrows()}
    righe = []
    for segnale in struct['Y']:
        n = len(segnale['Data'])
        f = frequenze[segnale['Raster']]
        righe.append({'segnale': segnale['Name'], 'raster': segnale['Raster'],
                      'campioni': n, 'frequenza_Hz': f, 'durata_s': n / f})
    return pd.DataFrame(righe).sort_values('frequenza_Hz', ascending=False)


def misura_campionamento(mat_file, nome_canale=config.CANALE_CORRENTE, struct=None):
    """
    Frequenza di campionamento e durata effettive di un canale, ricavate
    dall'asse dei tempi che il canale stesso dichiara.

    Non solleva eccezioni. Nel dataset la frequenza non e uniforme: alcune
    registrazioni coprono gli stessi quattro secondi con piu campioni, quindi a
    una frequenza piu alta. Non e un errore da cui fermarsi, e un dato di cui
    tenere conto quando si calcola la lunghezza di un giro. Chi chiama decide
    cosa farne guardando 'scarto_pct'.
    """
    struct = struct if struct is not None else apri(mat_file)
    raster = next(s['Raster'] for s in struct['Y'] if s['Name'] == nome_canale)
    asse = next(np.asarray(x['Data'], dtype=float)
                for x in struct['X'] if x['Raster'] == raster)

    campioni = asse.size
    intervallo = asse[-1] - asse[0]
    fs = (campioni - 1) / intervallo

    return {'raster': raster,
            'campioni': campioni,
            'intervallo_s': intervallo,
            'frequenza_Hz': fs,
            'durata_s': campioni / fs,
            'scarto_pct': 100 * (fs - config.FS_ATTESO) / config.FS_ATTESO}


# inventario e controlli sui dati

def metadati_nome(percorso):
    """Ricava regime e cuscinetto dal nome del file, es. N15_M07_F10_K001_3.mat."""
    parti = os.path.basename(percorso).split('_')
    return {'file': percorso, 'registrazione': os.path.basename(percorso),
            'regime': '_'.join(parti[:3]), 'cuscinetto': parti[3]}


def elenco_registrazioni(cuscinetti, cartella_estratti, regimi=None):
    """Percorsi dei .mat dei cuscinetti indicati, filtrati per regime."""
    trovati = []
    for nome in cuscinetti:
        for f in sorted(glob.glob(os.path.join(cartella_estratti, nome, '**', '*.mat'),
                                  recursive=True)):
            if regimi is None or metadati_nome(f)['regime'] in regimi:
                trovati.append(f)
    return trovati


def inventario(cuscinetti, cartella_estratti, regimi=None,
               canale=config.CANALE_CORRENTE, classe_di=None):
    """
    Censimento delle registrazioni disponibili.

    Serve a verificare che l'estrazione sia completa: un'estrazione parziale non
    produce errori, produce silenziosamente meno dati.
    """
    classe_di = classe_di if classe_di is not None else config.CLASSE_DI
    righe = []
    for percorso in elenco_registrazioni(cuscinetti, cartella_estratti, regimi):
        m = metadati_nome(percorso)
        x = leggi(percorso, canale)
        m['classe'] = classe_di.get(m['cuscinetto'])
        m['campioni'] = x.size
        m['secondi_interi'] = x.size // config.FS_ATTESO
        righe.append(m)
    return pd.DataFrame(righe)


def riepilogo_inventario(inv):
    """Registrazioni, cuscinetti e secondi disponibili per regime e classe."""
    return (inv.groupby(['regime', 'classe'])
              .agg(registrazioni=('registrazione', 'count'),
                   cuscinetti=('cuscinetto', 'nunique'),
                   secondi=('secondi_interi', 'sum'))
              .reset_index())


def trova_duplicati(inv, canale=config.CANALE_CORRENTE):
    """
    Cerca registrazioni con contenuto identico confrontando l'hash del segnale.

    Il measuring log di KA04 documenta la sostituzione di un file con una copia
    di un altro; questo controllo verifica in modo indipendente se esistano
    duplicati, documentati o meno.
    """
    righe = []
    for _, r in inv.iterrows():
        x = leggi(r['file'], canale)
        firma = hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=8).hexdigest()
        righe.append({'cuscinetto': r['cuscinetto'], 'regime': r['regime'],
                      'registrazione': r['registrazione'], 'firma': firma})
    tabella = pd.DataFrame(righe)
    doppie = tabella[tabella.duplicated('firma', keep=False)]
    return tabella, doppie.sort_values(['cuscinetto', 'regime', 'firma'])


def anagrafica_cuscinetti():
    """config.ANAGRAFICA come DataFrame, comodo per tabelle e raggruppamenti."""
    righe = []
    for nome, info in config.ANAGRAFICA.items():
        riga = {'cuscinetto': nome}
        riga.update(info)
        riga['classe_nome'] = config.NOMI_CLASSI[info['classe']]
        riga['diametro_primitivo_mm'] = config.DIAMETRO_PRIMITIVO_MM[info['produttore']]
        righe.append(riga)
    return pd.DataFrame(righe)


# statistiche dei segnali (Parte 20 del corso, "Features Extraction")

def caratteristiche(x, fs=config.FS_ATTESO):
    """Grandezze descrittive di un segnale, usate per il controllo di qualità."""
    finiti = np.isfinite(x)
    v = np.asarray(x, dtype=float)[finiti]
    rms = np.sqrt(np.mean(v * v))
    picco = np.max(np.abs(v))
    spettro_amp = np.abs(np.fft.rfft(v))
    frequenze = np.fft.rfftfreq(v.size, 1 / fs)
    return {
        'campioni': int(v.size),
        'non_finiti': int(np.sum(~finiti)),
        'media': float(np.mean(v)),
        'std': float(np.std(v)),
        'picco': float(picco),
        'rms': float(rms),
        'crest': float(picco / rms),
        'zcr': float(int(np.sum(v[:-1] * v[1:] < 0)) / (v.size / fs)),
        'f_dominante': float(frequenze[np.argmax(spettro_amp)]),
        'al_fondo_scala': int(np.sum(np.abs(v) > 0.999 * picco)),
    }


def stazionarieta(velocita):
    """Variabilità della velocità dentro una registrazione, in percentuale."""
    v = np.asarray(velocita, dtype=float)
    return {'media': float(np.mean(v)), 'std': float(np.std(v)),
            'minimo': float(np.min(v)), 'massimo': float(np.max(v)),
            'variazione_pct': float(100 * np.std(v) / np.mean(v))}


# segmentazione (equazioni 7-9 del paper)

def lunghezza_frame(rpm, fs=config.FS_ATTESO):
    """
    Campioni contenuti in un giro meccanico.
        RPS = rpm / 60 ;  TOR = 1 / RPS ;  F = fs * TOR
    A 1500 rpm e 64 kHz restituisce 2560, il valore usato dal paper.
    """
    return int(round(fs / (rpm / 60.0)))


def frame_di_un_giro(mat_file, nome_canale=config.CANALE_CORRENTE, struct=None, rpm=None):
    """
    Quanti campioni dura un giro meccanico in questa specifica registrazione.

    Frequenza e velocita vengono misurate sul file invece di essere assunte, cosi
    il frame corrisponde davvero a un giro anche quando la registrazione e stata
    campionata a una frequenza diversa da quella nominale. E la differenza fra
    assumere 2560 per tutti e adattarsi al singolo dato.
    """
    struct = struct if struct is not None else apri(mat_file)
    campionamento = misura_campionamento(mat_file, nome_canale, struct)
    if rpm is None:
        rpm = float(np.mean(leggi(mat_file, 'speed', struct)))
    return lunghezza_frame(rpm, campionamento['frequenza_Hz'])


def segmenta(x, lunghezza, massimo=None, passo=None):
    """
    Divide un segnale in blocchi di lunghezza fissa.

    passo   : distanza fra inizi consecutivi; se None i blocchi non si sovrappongono
    massimo : numero massimo di blocchi da estrarre
    """
    x = np.asarray(x)
    passo = passo if passo is not None else lunghezza
    blocchi = []
    for inizio in range(0, len(x) - lunghezza + 1, passo):
        blocchi.append(x[inizio:inizio + lunghezza])
        if massimo is not None and len(blocchi) >= massimo:
            break
    return np.asarray(blocchi, dtype=np.float32)


def costruisci_segmenti(inv, canale=config.CANALE_CORRENTE,
                        lunghezza_segmento=config.FS_ATTESO,
                        segmenti_per_registrazione=4, passo_secondi=None,
                        fs_bersaglio=config.FS_ATTESO):
    """
    Costruisce la matrice dei segmenti a partire da un inventario.

    Ogni registrazione viene prima riportata a fs_bersaglio, cosi un giro
    meccanico dura lo stesso numero di campioni ovunque e i frame hanno la
    dimensione fissa che il DAE si aspetta.

    Restituisce (X, anagrafica) dove ogni riga di X e un segmento e la
    corrispondente riga dell'anagrafica ne conserva cuscinetto, registrazione e
    posizione: sono le informazioni necessarie per suddivisioni non banali.
    """
    passo = int(round(passo_secondi * lunghezza_segmento)) if passo_secondi else lunghezza_segmento
    segmenti, righe = [], []
    for _, r in inv.iterrows():
        x, campionamento = leggi_ricampionato(r['file'], canale, fs_bersaglio)
        limite = min(len(x), segmenti_per_registrazione * lunghezza_segmento)
        for inizio in range(0, limite - lunghezza_segmento + 1, passo):
            segmenti.append(x[inizio:inizio + lunghezza_segmento])
            righe.append({'segmento': len(segmenti) - 1,
                          'cuscinetto': r['cuscinetto'], 'classe': r['classe'],
                          'regime': r['regime'], 'registrazione': r['registrazione'],
                          'inizio_campione': inizio,
                          'fs_originale_Hz': campionamento['frequenza_Hz'],
                          'ricampionata': abs(campionamento['scarto_pct']) > 100 * config.TOLLERANZA})
    return np.asarray(segmenti, dtype=np.float32), pd.DataFrame(righe)


# frequenze di guasto e analisi spettrale

def frequenze_guasto(f_rotazione, n=config.N_SFERE, d=config.DIAMETRO_SFERA_MM,
                     D=29.05, beta_deg=config.ANGOLO_CONTATTO_DEG):
    """
    BPFI, BPFO, BSF e FTF in Hz, dalle equazioni (1)-(4) del paper.

    f_rotazione : frequenza di rotazione dell'albero, in Hz
    D           : diametro primitivo, che nel dataset dipende dal costruttore
    """
    c = (d / D) * np.cos(np.deg2rad(beta_deg))
    return {'BPFI': n / 2 * f_rotazione * (1 + c),
            'BPFO': n / 2 * f_rotazione * (1 - c),
            'BSF':  D / (2 * d) * f_rotazione * (1 - c ** 2),
            'FTF':  f_rotazione / 2 * (1 - c)}


def frequenze_guasto_per_cuscinetto(cuscinetto, rpm):
    """Frequenze caratteristiche usando il diametro primitivo del costruttore."""
    produttore = config.ANAGRAFICA[cuscinetto]['produttore']
    return frequenze_guasto(rpm / 60.0, D=config.DIAMETRO_PRIMITIVO_MM[produttore])


def spettro(x, fs=config.FS_ATTESO):
    """Spettro di ampiezza a un lato."""
    x = np.asarray(x, dtype=float)
    return np.fft.rfftfreq(x.size, 1 / fs), np.abs(np.fft.rfft(x)) * 2 / x.size


def spettro_inviluppo(x, fs=config.FS_ATTESO, banda=None):
    """
    Spettro dell'inviluppo, dal modulo del segnale analitico.

    banda : (basso, alto) in Hz per filtrare prima di calcolare l'inviluppo.
            L'analisi d'inviluppo è il metodo standard per far emergere le
            modulazioni prodotte dagli impulsi di guasto.
    """
    from scipy.signal import hilbert, butter, filtfilt
    x = np.asarray(x, dtype=float)
    if banda is not None:
        b, a = butter(4, [banda[0] / (fs / 2), banda[1] / (fs / 2)], btype='band')
        x = filtfilt(b, a, x)
    env = np.abs(hilbert(x))
    env = env - env.mean()
    return np.fft.rfftfreq(env.size, 1 / fs), np.abs(np.fft.rfft(env)) * 2 / env.size


def ampiezza_alla_frequenza(frequenze, ampiezze, bersaglio, larghezza=2.0):
    """Ampiezza massima in una finestra stretta attorno a una frequenza."""
    dentro = np.abs(frequenze - bersaglio) <= larghezza
    return float(np.max(ampiezze[dentro])) if dentro.any() else float('nan')


# modelli: DAE e CNN (Tabelle 3 e 4 del paper)

DIMENSIONI_DAE = [2560, 1280, 640, 320, 128, 32, 128, 320, 640, 1280, 2560]


def dispositivo():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def conta_parametri(modello):
    return sum(p.numel() for p in modello.parameters())


def crea_dae(dimensioni=None, uscita_selu=True):
    """
    Autoencoder simmetrico completamente connesso (Tabella 3).

    uscita_selu : se True l'ultimo strato ha attivazione SELU, come dichiarato
                  dal paper; se False è lineare (braccio di controllo usato per
                  isolare l'effetto del limite inferiore della SELU).

    L'inizializzazione è LeCun normal, l'unica per cui vale l'auto-normalizzazione
    della SELU dimostrata da Klambauer et al. Il paper non la dichiara.
    """
    import torch.nn as nn
    dimensioni = dimensioni if dimensioni is not None else DIMENSIONI_DAE
    strati = []
    for i in range(len(dimensioni) - 1):
        strati.append(nn.Linear(dimensioni[i], dimensioni[i + 1]))
        if i < len(dimensioni) - 2:
            strati.append(nn.SELU())
    if uscita_selu:
        strati.append(nn.SELU())
    rete = nn.Sequential(*strati)
    for modulo in rete:
        if isinstance(modulo, nn.Linear):
            nn.init.normal_(modulo.weight, mean=0.0, std=1 / math.sqrt(modulo.in_features))
            nn.init.zeros_(modulo.bias)
    return rete


def crea_cnn(lunghezza, n_classi=3, nucleo=3):
    """
    Rete convolutiva 1-D (Tabella 4): due blocchi convoluzione + max pooling,
    poi flatten e strato denso.

    Il nucleo non è dichiarato dal paper: vale 3, ricavato in modo univoco dai
    conteggi di parametri riportati (256 e 6176).
    """
    import torch
    import torch.nn as nn

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 64, kernel_size=nucleo)
            self.conv2 = nn.Conv1d(64, 32, kernel_size=nucleo)
            self.riduzione = nn.MaxPool1d(2)
            dopo_primo = (lunghezza - (nucleo - 1)) // 2
            dopo_secondo = (dopo_primo - (nucleo - 1)) // 2
            self.denso = nn.Linear(dopo_secondo * 32, n_classi)

        def forward(self, x):
            x = self.riduzione(torch.relu(self.conv1(x.unsqueeze(1))))
            x = self.riduzione(torch.relu(self.conv2(x)))
            return self.denso(x.flatten(1))

    return CNN()


def addestra_dae(frame_train, frame_val, uscita_selu=True, epoche=500, lotto=256,
                 passo=3e-4, seme=0, dev=None, stampa_ogni=50, etichetta=''):
    """
    Addestra il DAE sui soli frame sani, per un numero fisso di epoche e senza
    arresto anticipato, come descritto nel paper.

    Restituisce (modello, curva_addestramento, curva_validazione).
    """
    import torch
    import torch.nn as nn
    dev = dev if dev is not None else dispositivo()
    torch.manual_seed(seme)
    modello = crea_dae(uscita_selu=uscita_selu).to(dev)
    ottimizzatore = torch.optim.Adam(modello.parameters(), lr=passo)
    costo = nn.MSELoss()

    train = torch.as_tensor(np.asarray(frame_train, dtype=np.float32))
    val = torch.as_tensor(np.asarray(frame_val, dtype=np.float32)).to(dev)

    curva_train, curva_val = [], []
    inizio = time.time()
    for epoca in range(1, epoche + 1):
        modello.train()
        ordine = torch.randperm(len(train))
        somma = 0.0
        for i in range(0, len(ordine), lotto):
            batch = train[ordine[i:i + lotto]].to(dev)
            errore = costo(modello(batch), batch)
            ottimizzatore.zero_grad()
            errore.backward()
            ottimizzatore.step()
            somma += errore.item() * len(batch)
        curva_train.append(somma / len(train))
        modello.eval()
        with torch.no_grad():
            curva_val.append(float(costo(modello(val), val)))
        if epoca == 1 or epoca % stampa_ogni == 0:
            print(f'   {etichetta} epoca {epoca:4d} | addestramento {curva_train[-1]:.6f}'
                  f' | validazione {curva_val[-1]:.6f}')

    print(f'   parametri {conta_parametri(modello):,d} | durata {time.time() - inizio:.1f} s')
    return modello, curva_train, curva_val


def calcola_residui(modello, frame, lotto=256, dev=None):
    """Residuo r = x - x_ricostruito, campione per campione, su tutti i frame."""
    import torch
    dev = dev if dev is not None else dispositivo()
    frame = np.asarray(frame, dtype=np.float32)
    residui = np.empty_like(frame)
    modello.eval()
    with torch.no_grad():
        for i in range(0, len(frame), lotto):
            pezzo = torch.from_numpy(frame[i:i + lotto]).to(dev)
            residui[i:i + len(pezzo)] = (pezzo - modello(pezzo)).cpu().numpy()
    return residui


def mse_per_frame(residui):
    return np.mean(np.asarray(residui, dtype=np.float64) ** 2, axis=1)


def addestra_cnn(dati, etichette, indici_train, indici_test, epoche=500, lotto=64,
                 passo=3e-4, seme=0, dev=None, stampa_ogni=50, etichetta=''):
    """
    Addestra la CNN sui segmenti indicati e valuta su quelli di verifica.

    Restituisce (modello, previsioni, vere, storia); storia contiene costo e
    accuratezza di addestramento per epoca, utili per i grafici.
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score

    dev = dev if dev is not None else dispositivo()
    torch.manual_seed(seme)
    modello = crea_cnn(dati.shape[1]).to(dev)
    ottimizzatore = torch.optim.Adam(modello.parameters(), lr=passo)
    costo = nn.CrossEntropyLoss()
    print(f'{etichetta} | parametri {conta_parametri(modello):,d}')

    X = torch.as_tensor(np.asarray(dati, dtype=np.float32))
    y = torch.as_tensor(np.asarray(etichette, dtype=np.int64))
    indici_train = np.asarray(indici_train)
    indici_test = np.asarray(indici_test)

    storia = {'costo': [], 'accuratezza': []}
    inizio = time.time()
    for epoca in range(1, epoche + 1):
        modello.train()
        ordine = torch.randperm(len(indici_train))
        somma, giusti = 0.0, 0
        for i in range(0, len(ordine), lotto):
            scelta = indici_train[ordine[i:i + lotto].numpy()]
            xb, yb = X[scelta].to(dev), y[scelta].to(dev)
            uscita = modello(xb)
            errore = costo(uscita, yb)
            ottimizzatore.zero_grad()
            errore.backward()
            ottimizzatore.step()
            somma += errore.item() * len(scelta)
            giusti += int((uscita.argmax(1) == yb).sum())
        storia['costo'].append(somma / len(indici_train))
        storia['accuratezza'].append(giusti / len(indici_train))
        if epoca == 1 or epoca % stampa_ogni == 0:
            print(f'   epoca {epoca:4d} | costo {storia["costo"][-1]:.5f}'
                  f' | accuratezza {storia["accuratezza"][-1]:.4f}')

    modello.eval()
    previsioni = []
    with torch.no_grad():
        for i in range(0, len(indici_test), lotto):
            previsioni.append(modello(X[indici_test[i:i + lotto]].to(dev))
                              .argmax(1).cpu().numpy())
    previsioni = np.concatenate(previsioni)
    vere = y[indici_test].numpy()
    print(f'   accuratezza sulla verifica: {100 * accuracy_score(vere, previsioni):.2f} %'
          f' | durata {time.time() - inizio:.1f} s')
    return modello, previsioni, vere, storia


# suddivisione, metriche e feature della Tabella 5

NOMI_FEATURE_TABELLA5 = ['rms', 'energia', 'deviazione', 'curtosi', 'varianza',
                         'asimmetria', 'fattore_cresta', 'entropia_shannon',
                         'fattore_forma', 'entropia_log']


def feature_tabella5(dati):
    """
    Le dieci grandezze statistiche elencate nella Tabella 5 del paper:
    valore efficace, energia, deviazione standard, curtosi, varianza, asimmetria,
    fattore di cresta, entropia di Shannon, fattore di forma, entropia log-energetica.

    dati : matrice (n_segmenti, n_campioni)
    """
    from scipy.stats import kurtosis, skew
    v = np.asarray(dati, dtype=np.float64)

    rms = np.sqrt(np.mean(v ** 2, axis=1))
    energia = np.sum(v ** 2, axis=1)
    deviazione = np.std(v, axis=1, ddof=1)
    varianza = deviazione ** 2
    curtosi = kurtosis(v, axis=1, fisher=False)
    asimmetria = skew(v, axis=1)

    # Il paper definisce il fattore di cresta come massimo su minimo.
    minimo = np.min(v, axis=1)
    fattore_cresta = np.max(v, axis=1) / np.where(np.abs(minimo) < 1e-12, np.nan, minimo)
    fattore_forma = np.mean(v, axis=1) / np.where(rms < 1e-12, np.nan, rms)

    quadrati = np.clip(v ** 2, 1e-30, None)
    entropia_shannon = -np.sum(quadrati * np.log(quadrati), axis=1)
    entropia_log = -np.sum(np.log(quadrati), axis=1)

    return np.column_stack([rms, energia, deviazione, curtosi, varianza, asimmetria,
                            fattore_cresta, entropia_shannon, fattore_forma, entropia_log])


def suddividi(anagrafica, livello='segmento', frazione_test=0.2, seme=0):
    """
    Suddivide i segmenti stratificando per classe, a tre livelli di severità
    crescente:

      'segmento'      suddivisione casuale sui segmenti. È il protocollo dichiarato
                      dal paper: segmenti della stessa registrazione possono finire
                      sia in addestramento sia in verifica.
      'registrazione' registrazioni intere separate.
      'cuscinetto'    cuscinetti fisici interi separati: misura la generalizzazione
                      a esemplari mai visti.

    Restituisce (indici_train, indici_test).
    """
    rng = np.random.default_rng(seme)
    train, test = [], []
    for classe in sorted(anagrafica['classe'].unique()):
        parte = anagrafica[anagrafica['classe'] == classe]
        if livello == 'segmento':
            indici = parte.index.to_numpy().copy()
            rng.shuffle(indici)
            n = max(1, int(round(len(indici) * frazione_test)))
            test.extend(indici[:n]); train.extend(indici[n:])
        else:
            gruppi = np.array(sorted(parte[livello].unique()))
            rng.shuffle(gruppi)
            n = max(1, int(round(len(gruppi) * frazione_test)))
            in_test = set(gruppi[:n])
            test.extend(parte.index[parte[livello].isin(in_test)])
            train.extend(parte.index[~parte[livello].isin(in_test)])
    return np.asarray(train), np.asarray(test)


def sovrapposizione(anagrafica, indici_train, indici_test):
    """Quanti cuscinetti e registrazioni sono condivisi fra i due insiemi."""
    a, b = anagrafica.loc[indici_train], anagrafica.loc[indici_test]
    return {'cuscinetti_condivisi': len(set(a['cuscinetto']) & set(b['cuscinetto'])),
            'registrazioni_condivise': len(set(a['registrazione']) & set(b['registrazione'])),
            'n_train': len(indici_train), 'n_test': len(indici_test)}


def baseline_degenere(etichette):
    """
    Accuratezza del classificatore che predice sempre la classe più numerosa.
    È il riferimento minimo: un modello che non la supera non ha imparato nulla.
    """
    valori, conteggi = np.unique(etichette, return_counts=True)
    return {'classe': int(valori[np.argmax(conteggi)]),
            'accuratezza': float(np.max(conteggi) / len(etichette))}


def metriche(vere, previste, nomi=None):
    """Accuratezza, macro-F1, report e matrice di confusione."""
    from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                                 confusion_matrix)
    nomi = nomi if nomi is not None else config.NOMI_CLASSI
    return {'accuratezza': float(accuracy_score(vere, previste)),
            'macro_f1': float(f1_score(vere, previste, average='macro', zero_division=0)),
            'report': classification_report(vere, previste, target_names=nomi,
                                            digits=3, zero_division=0),
            'confusione': confusion_matrix(vere, previste, labels=range(len(nomi))),
            'degenere': baseline_degenere(vere)}


def rapporti_residuo(mse, classi, nomi=None):
    """
    Residuo medio per classe e rapporto rispetto alla classe normale,
    affiancati ai valori dichiarati dal paper.
    """
    nomi = nomi if nomi is not None else config.NOMI_CLASSI
    mse, classi = np.asarray(mse, dtype=np.float64), np.asarray(classi)
    righe = [{'classe': nome,
              'residuo_medio': float(np.mean(mse[classi == c])),
              'paper': config.PAPER_RESIDUO[nome]} for c, nome in enumerate(nomi)]
    t = pd.DataFrame(righe)
    t['rapporto'] = t['residuo_medio'] / t['residuo_medio'].iloc[0]
    t['rapporto_paper'] = t['paper'] / t['paper'].iloc[0]
    return t


def auc_residuo(mse_segmento, classi):
    """
    AUC ottenuta usando il solo MSE del residuo come punteggio, per ciascuna
    classe di guasto contro la normale. Misura quanta informazione contenga il
    residuo di per sé, prima di qualunque classificatore.
    """
    from sklearn.metrics import roc_auc_score
    mse_segmento, classi = np.asarray(mse_segmento), np.asarray(classi)
    risultati = {}
    for c, nome in [(1, 'esterno'), (2, 'interno')]:
        sel = classi != (2 if c == 1 else 1)
        risultati[nome] = float(roc_auc_score((classi[sel] == c).astype(int),
                                              mse_segmento[sel]))
    risultati['media'] = float(np.mean(list(risultati.values())))
    return risultati


# utilità per figure e tabelle

def stile_grafici():
    """Impostazioni comuni alle figure. Da chiamare una volta per notebook."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 170, 'font.size': 10,
        'axes.titlesize': 10.5, 'axes.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'legend.frameon': False, 'figure.autolayout': True,
    })


# Palette usata in tutte le figure della relazione
COLORI = {'normale': '#8aa8cc', 'esterno': '#e2a45f', 'interno': '#b08bbf',
          'paper': '#8aa8cc', 'nostro': '#e2a45f', 'accento': '#c96a4f',
          'scuro': '#33475b', 'neutro': '#9aa7b4'}


def salva_figura(fig, nome, cartella_figure):
    """
    Salva la figura in PNG senza canale alpha: il PDF/A della tesi non ammette
    trasparenza, e le immagini RGBA rallentano molto la compilazione.
    """
    os.makedirs(cartella_figure, exist_ok=True)
    percorso = os.path.join(cartella_figure, nome + '.png')
    fig.savefig(percorso, bbox_inches='tight', facecolor='white', transparent=False)
    print('figura salvata:', percorso)
    return percorso


def salva_tabella(df, nome, cartella_tabelle, indice=False):
    """Salva un DataFrame come CSV e ne stampa il percorso."""
    os.makedirs(cartella_tabelle, exist_ok=True)
    percorso = os.path.join(cartella_tabelle, nome + '.csv')
    df.to_csv(percorso, index=indice)
    print('tabella salvata:', percorso)
    return percorso
