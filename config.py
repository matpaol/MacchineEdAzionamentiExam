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

PAPER_ACCURATEZZA = {
    'DAE + residuo + CNN': 99.60,
    'segnale grezzo + CNN': 61.06,
    'residuo + feature + SVM': 41.67,
    'residuo + feature + RF': 53.06,
    'residuo + feature + KNN': 47.08,
}

PAPER_SEGMENTI_DICHIARATI = 1320

# Estremo inferiore dell'immagine della SELU, dai coefficienti di Klambauer et al.
LAMBDA_SELU = 1.0507009873554805
ALPHA_SELU = 1.6732632423543772
PAVIMENTO_SELU = -LAMBDA_SELU * ALPHA_SELU     # ~ -1.7581

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
        'estratti': '/content/estratti',                    # disco locale di Colab
    }
    base = os.path.join(radice, 'risultati')
    if sottocartella:
        base = os.path.join(base, sottocartella)
    p['risultati'] = base
    p['figure'] = os.path.join(base, 'figure')
    p['tabelle'] = os.path.join(base, 'tabelle')

    for chiave in ['raw', 'documenti', 'estratti', 'risultati', 'figure', 'tabelle']:
        os.makedirs(p[chiave], exist_ok=True)
    return p
