"""
Costanti, percorsi e anagrafica del progetto.

Tutto ciò che è "dato di fatto" del dataset o del paper sta qui, in un posto solo:
i notebook non ripetono numeri magici e una correzione si fa una volta sola.
"""

import os

# sorgente dei dati

URL_REPOSITORY = 'https://groups.uni-paderborn.de/kat/BearingDataCenter/'

# parametri di acquisizione dichiarati dal banco prova (da verificare sui dati)

FS_ATTESO = 64000        # Hz, canali di corrente e vibrazione
DURATA_ATTESA = 4.0      # s per registrazione
TOLLERANZA = 0.001       # scarto relativo massimo ammesso nelle verifiche

CANALE_CORRENTE = 'phase_current_1'
CANALE_CORRENTE_ALT = 'phase_current_2'

# condizioni operative

REGIMI = {
    'N15_M07_F10': {'rpm': 1500, 'coppia_Nm': 0.7, 'forza_N': 1000},
    'N09_M07_F10': {'rpm':  900, 'coppia_Nm': 0.7, 'forza_N': 1000},
    'N15_M01_F10': {'rpm': 1500, 'coppia_Nm': 0.1, 'forza_N': 1000},
    'N15_M07_F04': {'rpm': 1500, 'coppia_Nm': 0.7, 'forza_N':  400},
}

REGIME_PRINCIPALE = 'N15_M07_F10'

# cuscinetti della Tabella 2 del paper (6 sani + 11 a danno reale)

CUSCINETTI_PER_CLASSE = {
    0: ['K001', 'K002', 'K003', 'K004', 'K005', 'K006'],
    1: ['KA04', 'KA15', 'KA16', 'KA22', 'KA30'],
    2: ['KI04', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21'],
}

NOMI_CLASSI = ['normale', 'esterno', 'interno']

CLASSE_DI = {b: c for c, lista in CUSCINETTI_PER_CLASSE.items() for b in lista}
CUSCINETTI_PAPER = list(CLASSE_DI)

# geometria del cuscinetto 6203

N_SFERE = 8
DIAMETRO_SFERA_MM = 6.75
ANGOLO_CONTATTO_DEG = 0.0

# Il diametro primitivo non è unico: dipende dal costruttore (vedi ANAGRAFICA).
DIAMETRO_PRIMITIVO_MM = {'IBU': 29.05, 'MTK': 29.05, 'IBU/IBB': 29.05, 'FAG': 28.55}

# Anagrafica dei 17 cuscinetti, trascritta dalle schede PDF allegate agli archivi
# ("Profile of rolling bearing damage", C. Lessmeier, Universität Paderborn).
#
#   ore        : rodaggio (sani) o durata della prova di vita (guasti), in ore
#   estensione : scala 1-3 riportata nella scheda
#   secondario : eventuale secondo danno documentato

ANAGRAFICA = {
    # sani
    'K001': dict(classe=0, produttore='IBU', ore=50.0, modo=None, componente=None,
                 caratteristica=None, estensione=0, n_danni=0, secondario=None),
    'K002': dict(classe=0, produttore='IBU', ore=19.0, modo=None, componente=None,
                 caratteristica=None, estensione=0, n_danni=0, secondario=None),
    'K003': dict(classe=0, produttore='IBU', ore=1.0,  modo=None, componente=None,
                 caratteristica=None, estensione=0, n_danni=0, secondario=None),
    'K004': dict(classe=0, produttore='IBU', ore=5.0,  modo=None, componente=None,
                 caratteristica=None, estensione=0, n_danni=0, secondario=None),
    'K005': dict(classe=0, produttore='IBU', ore=10.0, modo=None, componente=None,
                 caratteristica=None, estensione=0, n_danni=0, secondario=None),
    'K006': dict(classe=0, produttore='IBU', ore=16.0, modo=None, componente=None,
                 caratteristica=None, estensione=0, n_danni=0, secondario=None),
    # pista esterna, danno reale
    'KA04': dict(classe=1, produttore='FAG', ore=12.02, modo='fatica', componente='OR',
                 caratteristica='puntuale', estensione=1, n_danni=1, secondario=None),
    'KA15': dict(classe=1, produttore='FAG', ore=6.02, modo='deformazione plastica',
                 componente='OR', caratteristica='puntuale', estensione=1, n_danni=1,
                 secondario=None),
    'KA16': dict(classe=1, produttore='MTK', ore=9.02, modo='fatica', componente='OR',
                 caratteristica='puntuale', estensione=2, n_danni=2, secondario=None),
    'KA22': dict(classe=1, produttore='IBU/IBB', ore=14.02, modo='fatica', componente='OR',
                 caratteristica='puntuale', estensione=1, n_danni=1, secondario=None),
    'KA30': dict(classe=1, produttore='MTK', ore=21.02, modo='deformazione plastica',
                 componente='OR', caratteristica='distribuito', estensione=1, n_danni=1,
                 secondario=None),
    # pista interna, danno reale
    'KI04': dict(classe=2, produttore='MTK', ore=12.02, modo='fatica', componente='IR',
                 caratteristica='puntuale', estensione=1, n_danni=2,
                 secondario='deformazione plastica su OR'),
    'KI14': dict(classe=2, produttore='MTK', ore=21.02, modo='fatica', componente='IR',
                 caratteristica='puntuale', estensione=1, n_danni=2,
                 secondario='deformazione plastica su OR'),
    'KI16': dict(classe=2, produttore='FAG', ore=3.02, modo='fatica', componente='IR',
                 caratteristica='puntuale', estensione=3, n_danni=1, secondario=None),
    'KI17': dict(classe=2, produttore='MTK', ore=19.02, modo='fatica', componente='IR',
                 caratteristica='puntuale', estensione=1, n_danni=2, secondario=None),
    'KI18': dict(classe=2, produttore='MTK', ore=7.02, modo='fatica', componente='IR',
                 caratteristica='puntuale', estensione=2, n_danni=1, secondario=None),
    'KI21': dict(classe=2, produttore='FAG', ore=20.02, modo='fatica', componente='IR',
                 caratteristica='puntuale', estensione=1, n_danni=1, secondario=None),
}

# valori dichiarati dal paper, usati come riferimento nei confronti

PAPER_RESIDUO = {'normale': 0.104, 'esterno': 0.386, 'interno': 0.479}

# Tabella 6 del paper, per intero: richiamo, precisione, F1 e accuratezza.
PAPER_TABELLA6 = {
    'DAE + residuo + CNN': {'richiamo': 0.99, 'precisione': 0.99, 'f1': 0.99, 'accuratezza': 99.60},
    'segnale grezzo + CNN': {'richiamo': 0.60, 'precisione': 0.59, 'f1': 0.60, 'accuratezza': 61.06},
    'residuo + feature + SVM': {'richiamo': 0.42, 'precisione': 0.41, 'f1': 0.41, 'accuratezza': 41.67},
    'residuo + feature + RF': {'richiamo': 0.53, 'precisione': 0.52, 'f1': 0.52, 'accuratezza': 53.06},
    'residuo + feature + KNN': {'richiamo': 0.47, 'precisione': 0.49, 'f1': 0.44, 'accuratezza': 47.08},
}

PAPER_ACCURATEZZA = {m: v['accuratezza'] for m, v in PAPER_TABELLA6.items()}

# Matrice di confusione della Figura 9b: 238 segmenti di verifica e un solo errore,
# da cui l'accuratezza dichiarata di 0,9958.
PAPER_CONFUSIONE = [[78, 0, 0], [0, 87, 1], [0, 0, 72]]

PAPER_SEGMENTI_DICHIARATI = 1320

# Estremo inferiore dell'immagine della SELU, dai coefficienti di Klambauer et al.
LAMBDA_SELU = 1.0507009873554805
ALPHA_SELU = 1.6732632423543772
PAVIMENTO_SELU = -LAMBDA_SELU * ALPHA_SELU     # ~ -1.7581

# ---------------------------------------------------------------------------
# Dataset esteso (fase 2): tutti i cuscinetti, tutti i regimi
# ---------------------------------------------------------------------------

# I tre KB restano fuori: hanno danni su entrambi gli anelli e non ricadono
# in modo univoco in nessuna delle tre classi.
CUSCINETTI_KB = ['KB23', 'KB24', 'KB27']

ARTIFICIALI = {
    1: ['KA01', 'KA03', 'KA05', 'KA06', 'KA07', 'KA08', 'KA09'],   # pista esterna
    2: ['KI01', 'KI03', 'KI05', 'KI07', 'KI08'],                   # pista interna
}

REALI = {1: CUSCINETTI_PER_CLASSE[1], 2: CUSCINETTI_PER_CLASSE[2]}

CUSCINETTI_ESTESO = (CUSCINETTI_PER_CLASSE[0]
                     + ARTIFICIALI[1] + REALI[1]
                     + ARTIFICIALI[2] + REALI[2])

CLASSE_DI_ESTESO = {b: c for c in (0, 1, 2)
                    for b in (CUSCINETTI_PER_CLASSE[0] if c == 0
                              else ARTIFICIALI[c] + REALI[c])}

NATURA_DI = {b: ('sano' if b in CUSCINETTI_PER_CLASSE[0]
                 else 'artificiale' if b in ARTIFICIALI[1] + ARTIFICIALI[2]
                 else 'reale')
             for b in CUSCINETTI_ESTESO}

# Estensione del danno secondo la norma VDI 3832, dalle schede: il livello 1 ha
# area fra 0,81 e 2 mm quadri, il livello 2 fra 2,2 e 9. I due gruppi non si
# sovrappongono e la ripartizione fra i dodici artificiali e di sei e sei.
ESTENSIONE = {
    1: ['KA01', 'KA05', 'KA07', 'KI01', 'KI03', 'KI05'],
    2: ['KA03', 'KA06', 'KA08', 'KA09', 'KI07', 'KI08'],
}

# Segmentazione angolare del dataset esteso: ogni giro viene portato alla
# lunghezza del giro piu lungo presente (quello a 900 rpm), cosi si sovracampiona
# soltanto e non si introduce aliasing. Il blocco dato alla CNN e di 14 giri.
LUNGHEZZA_GIRO = 4273
GIRI_PER_BLOCCO = 14

# Assegnazione dei cuscinetti agli insiemi, per i sei esperimenti della fase 2.
# Nessun cuscinetto compare in due insiemi dello stesso esperimento.
INSIEMI = {
    'soli_reali': {
        'train': ['K001', 'K002', 'K003', 'K004', 'KA04', 'KA15', 'KA16',
                  'KI04', 'KI14', 'KI16', 'KI17'],
        'val': ['K005', 'KA22', 'KI18'],
        'test': ['K006', 'KA30', 'KI21'],
    },
    'artificiale_verso_reale': {
        'train': ['K001', 'K002', 'K003', 'K004',
                  'KA01', 'KA03', 'KA06', 'KA07', 'KA08', 'KA09',
                  'KI01', 'KI03', 'KI07', 'KI08'],
        'val': ['K005', 'KA05', 'KI05'],
        'test': ['K006'] + REALI[1] + REALI[2],
    },
    'severita': {
        # addestramento sui danni estesi, verifica su quelli incipienti.
        # I soli due interni di livello 2 restano entrambi in addestramento:
        # toglierne uno per la validazione ne lascerebbe uno solo, quindi le
        # epoche vengono fissate a quelle trovate nell'esperimento precedente.
        'train': ['K001', 'K002', 'K003', 'K004'] + ESTENSIONE[2],
        'val': [],
        'test': ['K006'] + ESTENSIONE[1],
    },
}

# Ore di funzionamento precedenti alle misure, per i sei sani (dalle schede).
# Servono all'esperimento sul rodaggio: due definizioni di "normale" a confronto.
SANI_ETEROGENEI = ['K003', 'K002', 'K001']      # 1, 19, oltre 50 ore
SANI_POCO_RODATI = ['K003', 'K004', 'K005']     # 1, 5, 10 ore

# percorsi

RADICE = '/content/drive/MyDrive/bearings_detection'


def percorsi(radice=RADICE, sottocartella=None):
    """
    Costruisce (e crea) l'albero delle cartelle del progetto.

    sottocartella : cartella dei risultati di questo notebook, per non mescolare
                    gli output di esperimenti diversi.
    """
    p = {
        'radice': radice,
        'raw': os.path.join(radice, 'raw'),                 # archivi .rar originali
        'documenti': os.path.join(radice, 'documenti'),     # schede PDF dei cuscinetti
        'dataset': os.path.join(radice, 'dataset_esteso'),  # dataset costruito una volta sola
        'estratti': '/content/estratti',                    # disco locale di Colab
    }
    base = os.path.join(radice, 'risultati')
    if sottocartella:
        base = os.path.join(base, sottocartella)
    p['risultati'] = base
    p['figure'] = os.path.join(base, 'figure')
    p['tabelle'] = os.path.join(base, 'tabelle')
    p['modelli'] = os.path.join(base, 'modelli')

    for chiave in ['raw', 'documenti', 'dataset', 'estratti',
                   'risultati', 'figure', 'tabelle', 'modelli']:
        os.makedirs(p[chiave], exist_ok=True)
    return p
