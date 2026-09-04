# -*- coding: utf-8 -*-
"""
Verifica degli identificatori delle registrazioni e dei blocchi.

Risponde al rilievo sul raggruppamento per nome della registrazione dentro
costruisci_blocchi: se due cuscinetti diversi potessero produrre lo stesso nome
di file, un blocco conterrebbe giri di esemplari diversi sotto un'unica
etichetta.

Lo script non importa funzioni.py e non modifica niente. Le due varianti di
costruzione dei blocchi sono ridefinite qui dentro, cosi il confronto non
dipende da quale versione del progetto e installata.

Sei controlli, in ordine di quanto informano.

  1  COMPLETEZZA      dai soli nomi: ogni cuscinetto ha 4 regimi, ogni regime
                      ha le 20 ripetizioni, nessun numero ripetuto o mancante.
  2  PROVENIENZA      confronta l'archivio di appartenenza con il codice scritto
                      nel nome del file: due fonti indipendenti.
  3  ANAGRAFICA       coerenza interna della tabella salvata dal notebook 04.
  4  BLOCCHI          la chiave di raggruppamento consegnata contro una piu
                      stretta, sugli stessi dati, indice per indice.
  5  INSIEMI          train, val e test dei tre esperimenti non condividono
                      giri, cuscinetti o registrazioni.
  6  INIEZIONE        su dati simulati: si introducono anomalie note e si
                      verifica quale controllo le prende. Serve a dimostrare che
                      i controlli 1-5, quando dicono "a posto", stanno davvero
                      guardando qualcosa.

Uso su Colab, dopo aver montato Drive:

    !python verifica_identificatori.py

Uso in locale, indicando dove sono i dati:

    python verifica_identificatori.py --anagrafica /percorso/anagrafica.csv
    python verifica_identificatori.py --archivi /percorso/raw

Ogni controllo si salta da solo, dichiarandolo, se non trova i suoi dati. Il
controllo 6 gira sempre: non ha bisogno di niente.
"""

import argparse
import glob
import itertools
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# lo schema dei nomi
# --------------------------------------------------------------------------

# N15_M07_F10_KA04_13.mat
#  |   |   |    |    +--- ripetizione, da 1 a 20
#  |   |   |    +-------- codice del cuscinetto
#  |   |   +------------- forza radiale:  F10 -> 1000 N
#  |   +----------------- coppia:         M07 -> 0,7 Nm
#  +--------------------- velocita:       N15 -> 1500 giri/min
# il codice del cuscinetto ha una o due lettere e due o tre cifre:
# K001..K006 per i sani, KA04 / KI16 / KB23 per i danneggiati
SCHEMA_NOME = re.compile(r'^N(\d{2})_M(\d{2})_F(\d{2})_([A-Z]{1,2}\d{2,3})_(\d+)\.mat$')

RIPETIZIONI_ATTESE = 20


def spacchetta(nome):
    """
    Scompone il nome di un file nei suoi cinque campi.

    Restituisce None se il nome non rispetta lo schema. Non solleva eccezioni:
    un nome malformato e un risultato del controllo, non un errore dello script.
    """
    trovato = SCHEMA_NOME.match(os.path.basename(nome))
    if trovato is None:
        return None
    n, m, forza, cuscinetto, ripetizione = trovato.groups()
    return {'nome': os.path.basename(nome),
            'regime': 'N%s_M%s_F%s' % (n, m, forza),
            'cuscinetto': cuscinetto,
            'ripetizione': int(ripetizione),
            'rpm': int(n) * 100,
            'coppia_Nm': int(m) / 10.0,
            'forza_N': int(forza) * 100}


# --------------------------------------------------------------------------
# raccolta dei nomi: dagli archivi .rar oppure dalla cartella estratta
# --------------------------------------------------------------------------

def nomi_dagli_archivi(cartella_raw):
    """
    Elenca il contenuto di ogni .rar senza scompattarlo, con `unrar lb`.

    E la fonte piu diretta: il pacchetto come e stato scaricato dal
    KAt-DataCenter, un archivio per cuscinetto.
    """
    righe = []
    for archivio in sorted(glob.glob(os.path.join(cartella_raw, '*.rar'))):
        nome_archivio = os.path.splitext(os.path.basename(archivio))[0]
        esito = subprocess.run(['unrar', 'lb', archivio],
                               capture_output=True, text=True)
        if esito.returncode != 0:
            raise RuntimeError('unrar fallito su %s: %s' % (archivio, esito.stderr))
        for voce in esito.stdout.splitlines():
            voce = voce.strip()
            if voce.lower().endswith('.mat'):
                righe.append({'archivio': nome_archivio, 'nome': os.path.basename(voce)})
    return pd.DataFrame(righe)


def nomi_dagli_estratti(cartella_estratti):
    """Elenca i .mat gia scompattati, usando come archivio la cartella di primo livello."""
    righe = []
    for voce in sorted(os.listdir(cartella_estratti)):
        radice = os.path.join(cartella_estratti, voce)
        if not os.path.isdir(radice):
            continue
        for percorso in sorted(glob.glob(os.path.join(radice, '**', '*.mat'),
                                         recursive=True)):
            righe.append({'archivio': voce, 'nome': os.path.basename(percorso)})
    return pd.DataFrame(righe)


# --------------------------------------------------------------------------
# controllo 1 - completezza, dai soli nomi
# --------------------------------------------------------------------------

def controllo_completezza(tabella, regimi_attesi, ripetizioni=RIPETIZIONI_ATTESE):
    """
    Ogni cuscinetto deve avere tutti i regimi, e ogni regime le ripetizioni da 1
    a N, tutte, una volta sola.

    E un controllo forte anche se guarda solo i nomi. Se un file finisse
    attribuito al cuscinetto sbagliato, da una parte avanzerebbe un numero di
    ripetizione e dall'altra ne mancherebbe uno: due segnali indipendenti per lo
    stesso errore.
    """
    schema = [spacchetta(n) for n in tabella['nome']]
    malformati = [n for n, s in zip(tabella['nome'], schema) if s is None]
    campi = pd.DataFrame([s for s in schema if s is not None])

    anomalie = []
    if malformati:
        anomalie.append({'tipo': 'nome non conforme allo schema',
                         'dettaglio': malformati[:10],
                         'quanti': len(malformati)})

    if not campi.empty:
        attesi = set(regimi_attesi)
        for cuscinetto, gruppo in campi.groupby('cuscinetto'):
            mancanti = attesi - set(gruppo['regime'])
            if mancanti:
                anomalie.append({'tipo': 'regime mancante', 'cuscinetto': cuscinetto,
                                 'dettaglio': sorted(mancanti)})
            estranei = set(gruppo['regime']) - attesi
            if estranei:
                anomalie.append({'tipo': 'regime non previsto', 'cuscinetto': cuscinetto,
                                 'dettaglio': sorted(estranei)})

            for regime, parte in gruppo.groupby('regime'):
                numeri = sorted(parte['ripetizione'])
                doppi = sorted({x for x in numeri if numeri.count(x) > 1})
                assenti = sorted(set(range(1, ripetizioni + 1)) - set(numeri))
                if doppi:
                    anomalie.append({'tipo': 'ripetizione duplicata',
                                     'cuscinetto': cuscinetto, 'regime': regime,
                                     'dettaglio': doppi})
                if assenti:
                    anomalie.append({'tipo': 'ripetizione mancante',
                                     'cuscinetto': cuscinetto, 'regime': regime,
                                     'dettaglio': assenti})
                if len(numeri) != ripetizioni:
                    anomalie.append({'tipo': 'numero di file diverso dall atteso',
                                     'cuscinetto': cuscinetto, 'regime': regime,
                                     'dettaglio': '%d invece di %d'
                                                  % (len(numeri), ripetizioni)})

    return {'file_esaminati': int(len(tabella)),
            'nomi_conformi': int(len(campi)),
            'nomi_malformati': int(len(malformati)),
            'cuscinetti': int(campi['cuscinetto'].nunique()) if not campi.empty else 0,
            'anomalie': anomalie,
            'superato': not anomalie}


# --------------------------------------------------------------------------
# controllo 2 - provenienza: archivio contro nome
# --------------------------------------------------------------------------

def controllo_provenienza(tabella):
    """
    Confronta due fonti indipendenti per l'identita del cuscinetto.

      archivio  nome del .rar scaricato dal KAt-DataCenter, un file per
                cuscinetto: non dipende da come si chiamano i file dentro;
      nome      quarto campo del nome del .mat.

    E l'unico controllo del progetto che non sia coerenza interna. Prende il
    caso che sfugge al controllo 1: due file che si scambiano il nome, dove i
    conteggi restano intatti da entrambe le parti.
    """
    schema = [spacchetta(n) for n in tabella['nome']]
    utili = pd.DataFrame([{'archivio': a, **s}
                          for a, s in zip(tabella['archivio'], schema) if s is not None])
    if utili.empty:
        return {'confrontabili': 0, 'discordanti': 0, 'superato': False,
                'nota': 'nessun nome conforme allo schema'}

    discordi = utili[utili['archivio'] != utili['cuscinetto']]
    per_nome = utili.groupby('nome')['archivio'].nunique()

    return {'confrontabili': int(len(utili)),
            'discordanti': int(len(discordi)),
            'elenco_discordanti': discordi[['archivio', 'cuscinetto', 'nome']]
                                  .to_dict('records')[:20],
            'nomi_su_piu_archivi': int((per_nome > 1).sum()),
            'nomi_ripetuti': int(utili['nome'].duplicated().sum()),
            'superato': len(discordi) == 0 and int((per_nome > 1).sum()) == 0}


# --------------------------------------------------------------------------
# controllo 3 - coerenza dell'anagrafica dei giri
# --------------------------------------------------------------------------

def controllo_anagrafica(anagrafica, regimi_attesi):
    """
    Coerenza interna della tabella salvata dal notebook 04.

    Va detto con chiarezza: chiedere se ogni registrazione contiene un solo
    cuscinetto e una tautologia, perche nel notebook 04 la colonna cuscinetto e
    ricavata dal nome della registrazione. Il valore di questo controllo non sta
    li: sta nel verificare che il CSV non si sia corrotto o duplicato nel giro
    salvataggio-rilettura, e che i conteggi dei giri stiano nei limiti fisici.
    """
    attesi = {'cuscinetto', 'regime', 'registrazione', 'giro', 'classe'}
    mancanti = attesi - set(anagrafica.columns)
    if mancanti:
        return {'superato': False, 'colonne_mancanti': sorted(mancanti)}

    per_nome = anagrafica.groupby('registrazione')[['cuscinetto', 'regime', 'classe']].nunique()
    conteggio = anagrafica.groupby('registrazione').size()

    # i giri di una registrazione devono essere numerati 0..n-1, senza buchi
    def numerazione_corretta(gruppo):
        numeri = np.sort(gruppo.to_numpy())
        return numeri[0] == 0 and np.array_equal(numeri, np.arange(numeri.size))

    numerazione = anagrafica.groupby('registrazione')['giro'].apply(numerazione_corretta)

    esito = {
        'giri': int(len(anagrafica)),
        'registrazioni': int(anagrafica['registrazione'].nunique()),
        'cuscinetti': int(anagrafica['cuscinetto'].nunique()),
        'regimi': sorted(anagrafica['regime'].unique()),
        'righe_duplicate': int(anagrafica.duplicated().sum()),
        'nomi_su_piu_cuscinetti': int((per_nome['cuscinetto'] > 1).sum()),
        'nomi_su_piu_regimi': int((per_nome['regime'] > 1).sum()),
        'nomi_su_piu_classi': int((per_nome['classe'] > 1).sum()),
        'numerazione_giri_irregolare': int((~numerazione).sum()),
        'giri_per_registrazione_min': int(conteggio.min()),
        'giri_per_registrazione_max': int(conteggio.max()),
        'regimi_non_previsti': sorted(set(anagrafica['regime']) - set(regimi_attesi)),
    }
    esito['superato'] = (esito['righe_duplicate'] == 0
                         and esito['nomi_su_piu_cuscinetti'] == 0
                         and esito['nomi_su_piu_regimi'] == 0
                         and esito['nomi_su_piu_classi'] == 0
                         and esito['numerazione_giri_irregolare'] == 0
                         and not esito['regimi_non_previsti'])
    return esito


# --------------------------------------------------------------------------
# controllo 4 - le due chiavi di raggruppamento
# --------------------------------------------------------------------------

def blocchi_dal_progetto(anagrafica, giri_per_blocco, funzioni):
    """
    Chiama la costruisci_blocchi vera del progetto, non una copia.

    E la funzione sotto esame: riscriverla qui dentro renderebbe il confronto
    privo di valore, perche proverebbe una mia trascrizione invece del codice
    consegnato.
    """
    indici, _ = funzioni.costruisci_blocchi(anagrafica, giri_per_blocco)
    return np.asarray(indici)


def identifica_versione(funzioni):
    """
    Dice quale variante di costruisci_blocchi e installata, leggendone il codice.

    Serve a sapere sempre cosa si sta confrontando: se il progetto fosse gia
    stato aggiornato, il confronto sarebbe fra due versioni della stessa cosa.
    """
    import inspect
    try:
        sorgente = inspect.getsource(funzioni.costruisci_blocchi)
    except (OSError, TypeError):
        return {'variante': 'non ispezionabile', 'file': '?'}

    solo_registrazione = bool(re.search(r"groupby\(\s*'registrazione'", sorgente))
    chiave_multipla = bool(re.search(r"groupby\(\s*\[", sorgente)
                           or re.search(r"groupby\(\s*chiavi", sorgente))
    if solo_registrazione and not chiave_multipla:
        variante = 'consegnata (raggruppa per la sola registrazione)'
    elif chiave_multipla:
        variante = 'gia aggiornata (chiave multipla)'
    else:
        variante = 'non riconosciuta'
    return {'variante': variante,
            'file': getattr(funzioni, '__file__', '?'),
            'righe_sorgente': len(sorgente.splitlines())}


def blocchi_chiave_consegnata(anagrafica, giri_per_blocco):
    """
    Riproduzione della logica consegnata, usata solo sui dati simulati.

    Sui dati veri si usa blocchi_dal_progetto, che chiama il codice originale.
    Qui serve perche sui dati finti vogliamo confrontare le due chiavi anche
    quando il progetto e gia stato aggiornato.
    """
    indici = []
    for _, gruppo in anagrafica.groupby('registrazione', sort=False):
        posizioni = gruppo.sort_values('giro').index.to_numpy()
        for k in range(len(posizioni) // giri_per_blocco):
            indici.append(posizioni[k * giri_per_blocco:(k + 1) * giri_per_blocco])
    return np.asarray(indici)


def blocchi_chiave_completa(anagrafica, giri_per_blocco):
    """
    Variante piu stretta: raggruppa per cuscinetto, regime e registrazione.

    Sul dataset di Paderborn produce gli stessi gruppi, perche il nome del file
    contiene gia cuscinetto e regime. Cambia il caso patologico: con nomi
    ambigui, i giri di cuscinetti diversi finiscono comunque in gruppi separati
    invece che nello stesso blocco.
    """
    indici = []
    for _, gruppo in anagrafica.groupby(['cuscinetto', 'regime', 'registrazione'],
                                        sort=False):
        posizioni = gruppo.sort_values('giro').index.to_numpy()
        for k in range(len(posizioni) // giri_per_blocco):
            indici.append(posizioni[k * giri_per_blocco:(k + 1) * giri_per_blocco])
    return np.asarray(indici)


def blocchi_misti(anagrafica, indici, colonna='cuscinetto'):
    """Quanti blocchi contengono piu di un valore nella colonna indicata."""
    if len(indici) == 0:
        return 0
    valori = anagrafica[colonna].to_numpy()[indici]
    return int(np.sum((valori != valori[:, [0]]).any(axis=1)))


def controllo_blocchi(anagrafica, giri_per_blocco, funzioni):
    """
    Confronta i blocchi prodotti dal codice del progetto con quelli prodotti
    dalla chiave piu stretta, sugli stessi dati, indice per indice.

    La prima viene dal progetto vero (import di funzioni.py), la seconda e
    definita qui. Se le due matrici coincidono, adottare la chiave piu stretta
    non cambierebbe nessun risultato: i blocchi dati alla rete sarebbero
    esattamente gli stessi.
    """
    consegnata = blocchi_dal_progetto(anagrafica, giri_per_blocco, funzioni)
    completa = blocchi_chiave_completa(anagrafica, giri_per_blocco)

    stessa_forma = consegnata.shape == completa.shape
    esito = {
        'versione_in_uso': identifica_versione(funzioni),
        'blocchi_dal_progetto': int(len(consegnata)),
        'blocchi_chiave_completa': int(len(completa)),
        'matrici_identiche': bool(stessa_forma and np.array_equal(consegnata, completa)),
        'giri_coperti_identici': bool(
            set(consegnata.ravel().tolist()) == set(completa.ravel().tolist())),
    }
    for colonna in ('cuscinetto', 'regime', 'classe'):
        if colonna in anagrafica.columns:
            esito['misti_progetto_' + colonna] = blocchi_misti(anagrafica, consegnata, colonna)
            esito['misti_completa_' + colonna] = blocchi_misti(anagrafica, completa, colonna)

    if 'giro' in anagrafica.columns and len(completa):
        giri = anagrafica['giro'].to_numpy()[completa]
        esito['blocchi_non_consecutivi'] = int(np.sum((np.diff(giri, axis=1) != 1).any(axis=1)))

    esito['superato'] = (esito['matrici_identiche']
                         and all(v == 0 for k, v in esito.items() if k.startswith('misti_'))
                         and esito.get('blocchi_non_consecutivi', 0) == 0)
    return esito


# --------------------------------------------------------------------------
# controllo 5 - disgiunzione degli insiemi
# --------------------------------------------------------------------------

def controllo_insiemi(anagrafica, insiemi, giri_per_blocco):
    """
    Verifica che train, val e test non condividano niente.

    E il controllo che risponde direttamente alla parola "leakage". Riproduce
    quello che fa blocchi_di nel notebook 05: filtra l'anagrafica per cuscinetto
    e poi costruisce i blocchi. Se il filtro precede la costruzione, un blocco
    non puo stare a cavallo di due insiemi.
    """
    esito = {}
    for nome, definizione in insiemi.items():
        parti = {p: definizione[p] for p in ('train', 'val', 'test')
                 if definizione.get(p)}
        blocchi_di_parte, giri_di_parte = {}, {}
        for parte, cuscinetti in parti.items():
            maschera = anagrafica['cuscinetto'].isin(cuscinetti).to_numpy()
            indici = blocchi_chiave_completa(anagrafica[maschera], giri_per_blocco)
            blocchi_di_parte[parte] = indici
            giri_di_parte[parte] = set(indici.ravel().tolist()) if len(indici) else set()

        condivisioni, problemi = [], []
        for a, b in itertools.combinations(parti, 2):
            comuni_giri = giri_di_parte[a] & giri_di_parte[b]
            comuni_cusc = set(parti[a]) & set(parti[b])
            comuni_reg = (set(anagrafica.loc[anagrafica['cuscinetto'].isin(parti[a]),
                                             'registrazione'])
                          & set(anagrafica.loc[anagrafica['cuscinetto'].isin(parti[b]),
                                               'registrazione']))
            voce = {'fra': '%s / %s' % (a, b),
                    'giri_condivisi': len(comuni_giri),
                    'cuscinetti_condivisi': len(comuni_cusc),
                    'registrazioni_condivise': len(comuni_reg)}
            condivisioni.append(voce)
            if comuni_giri or comuni_cusc or comuni_reg:
                problemi.append({**voce, 'quali_cuscinetti': sorted(comuni_cusc)})

        esito[nome] = {
            'blocchi': {p: int(len(v)) for p, v in blocchi_di_parte.items()},
            'cuscinetti': {p: len(v) for p, v in parti.items()},
            'condivisioni': condivisioni,
            'sovrapposizioni': problemi,
            'superato': not problemi,
        }
    esito['superato'] = all(v['superato'] for k, v in esito.items() if k != 'superato')
    return esito


# --------------------------------------------------------------------------
# controllo 6 - iniezione di anomalie su dati simulati
# --------------------------------------------------------------------------

def elenco_simulato(cuscinetti, regimi, ripetizioni=RIPETIZIONI_ATTESE):
    """Un dataset finto integro: ogni cuscinetto con tutti i regimi e le ripetizioni."""
    righe = []
    for cuscinetto in cuscinetti:
        for regime in regimi:
            for i in range(1, ripetizioni + 1):
                righe.append({'archivio': cuscinetto,
                              'nome': '%s_%s_%d.mat' % (regime, cuscinetto, i)})
    return pd.DataFrame(righe)


def inietta(tabella, anomalia, cuscinetti, regimi):
    """Introduce un'anomalia nota in una copia della tabella."""
    t = tabella.copy().reset_index(drop=True)
    primo, secondo = cuscinetti[0], cuscinetti[1]
    regime = regimi[0]
    nome_primo = '%s_%s_13.mat' % (regime, primo)
    nome_secondo = '%s_%s_13.mat' % (regime, secondo)

    if anomalia == 'rinominato':
        # un file del secondo cuscinetto porta il nome del primo
        t.loc[t['nome'] == nome_secondo, 'nome'] = nome_primo
    elif anomalia == 'scambio':
        # due file si scambiano il nome: i conteggi restano intatti
        a = t['nome'] == nome_primo
        b = t['nome'] == nome_secondo
        t.loc[a, 'nome'] = '__temp__'
        t.loc[b, 'nome'] = nome_primo
        t.loc[t['nome'] == '__temp__', 'nome'] = nome_secondo
    elif anomalia == 'regime_mancante':
        t = t[~((t['archivio'] == secondo) & (t['nome'].str.startswith(regimi[-1])))]
    elif anomalia == 'nome_non_conforme':
        t.loc[t['nome'] == nome_secondo, 'nome'] = '13.mat'
    elif anomalia == 'cuscinetto_sconosciuto':
        t.loc[t['nome'] == nome_secondo, 'nome'] = '%s_ZZ99_13.mat' % regime
    elif anomalia is not None:
        raise ValueError('anomalia sconosciuta: ' + str(anomalia))
    return t.reset_index(drop=True)


def anagrafica_simulata(cuscinetti, regime, classe_di, natura_di,
                        nome_file, giri_per_registrazione=28):
    """Anagrafica finta dei giri, con i nomi di file decisi dal chiamante."""
    righe = []
    for cuscinetto in cuscinetti:
        for ripetizione in (1, 2):
            for giro in range(giri_per_registrazione):
                righe.append({'cuscinetto': cuscinetto,
                              'natura': natura_di.get(cuscinetto, 'ignota'),
                              'classe': classe_di[cuscinetto],
                              'regime': regime,
                              'registrazione': nome_file(cuscinetto, ripetizione),
                              'giro': giro})
    return pd.DataFrame(righe)


def controllo_iniezione(regimi, cuscinetti, classe_di, natura_di, giri_per_blocco):
    """
    Introduce anomalie note e registra quale controllo le prende.

    Senza questa parte, un esito positivo dei controlli 1-5 sarebbe ambiguo: non
    si saprebbe se il dataset e pulito o se i controlli non guardano niente.
    """
    integro = elenco_simulato(cuscinetti[:3], regimi)
    casi = [None, 'rinominato', 'scambio', 'regime_mancante',
            'nome_non_conforme', 'cuscinetto_sconosciuto']

    righe = []
    for anomalia in casi:
        tabella = inietta(integro, anomalia, cuscinetti[:3], regimi)
        completezza = controllo_completezza(tabella, regimi)
        provenienza = controllo_provenienza(tabella)
        righe.append({
            'caso': anomalia if anomalia else 'nessuna anomalia',
            'completezza': 'passa' if completezza['superato'] else 'RILEVA',
            'provenienza': 'passa' if provenienza['superato'] else 'RILEVA',
        })
    tabella_nomi = pd.DataFrame(righe)

    # la parte sui blocchi: nomi regolari contro nomi che collidono
    righe = []
    for etichetta, nome_file in [
            ('nomi come in Paderborn', lambda c, r: '%s_%s_%d.mat' % (regimi[0], c, r)),
            ('nomi che collidono', lambda c, r: '%d.mat' % r)]:
        finta = anagrafica_simulata(cuscinetti[:2], regimi[0], classe_di, natura_di,
                                    nome_file)
        consegnata = blocchi_chiave_consegnata(finta, giri_per_blocco)
        completa = blocchi_chiave_completa(finta, giri_per_blocco)
        righe.append({
            'caso': etichetta,
            'blocchi_consegnata': int(len(consegnata)),
            'misti_consegnata': blocchi_misti(finta, consegnata),
            'blocchi_completa': int(len(completa)),
            'misti_completa': blocchi_misti(finta, completa),
        })
    tabella_blocchi = pd.DataFrame(righe)

    # esito atteso: il caso senza anomalia passa entrambi i controlli, tutti gli
    # altri vengono rilevati da almeno uno dei due
    riga_pulita = tabella_nomi.iloc[0]
    pulito_passa = (riga_pulita['completezza'] == 'passa'
                    and riga_pulita['provenienza'] == 'passa')
    sporchi = tabella_nomi.iloc[1:]
    tutti_rilevati = bool(((sporchi['completezza'] == 'RILEVA')
                           | (sporchi['provenienza'] == 'RILEVA')).all())
    chiave_regge = bool((tabella_blocchi['misti_completa'] == 0).all())
    chiave_vecchia_cede = bool(
        tabella_blocchi.loc[tabella_blocchi['caso'] == 'nomi che collidono',
                            'misti_consegnata'].iloc[0] > 0)

    return {'tabella_nomi': tabella_nomi,
            'tabella_blocchi': tabella_blocchi,
            'caso_pulito_passa': pulito_passa,
            'anomalie_tutte_rilevate': tutti_rilevati,
            'chiave_completa_regge': chiave_regge,
            'chiave_consegnata_cede_sui_nomi_collidenti': chiave_vecchia_cede,
            'superato': pulito_passa and tutti_rilevati and chiave_regge}


# --------------------------------------------------------------------------
# esecuzione
# --------------------------------------------------------------------------

def trova_progetto():
    """
    Importa config.py e funzioni.py dal progetto, cercandoli nei posti soliti.

    funzioni.py serve perche il confronto del controllo 4 deve usare la
    costruisci_blocchi vera, non una trascrizione.
    """
    candidati = ['.', '..', 'codice', '../codice',
                 '/content/drive/MyDrive/MacchineEdAzionamentiExam',
                 '/content/MacchineEdAzionamentiExam']
    for cartella in candidati:
        if os.path.exists(os.path.join(cartella, 'config.py')):
            sys.path.insert(0, os.path.abspath(cartella))
            import config
            try:
                import funzioni
            except Exception as errore:
                print('funzioni.py non importabile (%s): il controllo 4 sara saltato'
                      % type(errore).__name__)
                funzioni = None
            return config, funzioni, os.path.abspath(cartella)
    return None, None, None


def titolo(numero, testo):
    print()
    print('=' * 78)
    print('CONTROLLO %s  %s' % (numero, testo))
    print('=' * 78)


def stampa(dizionario, salta=()):
    for chiave, valore in dizionario.items():
        if chiave in salta or isinstance(valore, pd.DataFrame):
            continue
        if isinstance(valore, list) and len(valore) > 6:
            print('   %-38s %d voci' % (chiave, len(valore)))
        else:
            print('   %-38s %s' % (chiave, valore))


def main():
    lettore = argparse.ArgumentParser(description=__doc__)
    lettore.add_argument('--anagrafica', help='percorso di anagrafica.csv')
    lettore.add_argument('--archivi', help='cartella con i file .rar')
    lettore.add_argument('--estratti', help='cartella con i .mat gia scompattati')
    lettore.add_argument('--uscita', default='verifica_identificatori.json',
                         help='dove salvare il riepilogo')
    argomenti = lettore.parse_args()

    config, funzioni, dove = trova_progetto()
    if config is None:
        print('config.py non trovato: uso i valori di riferimento del progetto')
        regimi = ['N15_M07_F10', 'N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04']
        giri_per_blocco = 14
        insiemi, classe_di, natura_di, cuscinetti = None, None, None, None
    else:
        print('progetto da', dove)
        if funzioni is not None:
            print('costruisci_blocchi in uso:',
                  identifica_versione(funzioni)['variante'])
        regimi = list(config.REGIMI)
        giri_per_blocco = config.GIRI_PER_BLOCCO
        insiemi = config.INSIEMI
        classe_di = config.CLASSE_DI_ESTESO
        natura_di = config.NATURA_DI
        cuscinetti = list(config.CUSCINETTI_ESTESO)

    percorsi = {}
    if config is not None:
        try:
            P = config.percorsi()
            percorsi = {'archivi': P['raw'], 'estratti': P['estratti'],
                        'anagrafica': os.path.join(P['dataset'], 'anagrafica.csv')}
        except Exception as errore:
            print('config.percorsi() non utilizzabile:', errore)
    for chiave, valore in [('anagrafica', argomenti.anagrafica),
                           ('archivi', argomenti.archivi),
                           ('estratti', argomenti.estratti)]:
        if valore:
            percorsi[chiave] = valore

    riepilogo = {}

    # ---------------------------------------------------------------- 1 e 2
    tabella_nomi, fonte = None, None
    if percorsi.get('archivi') and glob.glob(os.path.join(percorsi['archivi'], '*.rar')):
        try:
            tabella_nomi = nomi_dagli_archivi(percorsi['archivi'])
            fonte = 'archivi .rar in ' + percorsi['archivi']
        except Exception as errore:
            print('lettura degli archivi non riuscita:', errore)
    if tabella_nomi is None and percorsi.get('estratti') and os.path.isdir(percorsi['estratti']):
        tabella_nomi = nomi_dagli_estratti(percorsi['estratti'])
        if len(tabella_nomi):
            fonte = 'cartella estratta ' + percorsi['estratti']
        else:
            tabella_nomi = None

    titolo('1', 'COMPLETEZZA - ogni cuscinetto ha 4 regimi e 20 ripetizioni')
    if tabella_nomi is None:
        print('   saltato: non trovo ne gli archivi .rar ne i file estratti.')
        print('   indicali con --archivi oppure --estratti.')
    else:
        print('   fonte:', fonte)
        esito = controllo_completezza(tabella_nomi, regimi)
        stampa(esito, salta=('anomalie',))
        for anomalia in esito['anomalie'][:20]:
            print('      ->', anomalia)
        print('   ESITO:', 'superato' if esito['superato'] else 'NON superato')
        riepilogo['completezza'] = {k: v for k, v in esito.items() if k != 'anomalie'}
        riepilogo['completezza']['anomalie'] = esito['anomalie'][:20]

    titolo('2', 'PROVENIENZA - l archivio concorda con il nome del file')
    if tabella_nomi is None:
        print('   saltato: stessa ragione del controllo 1.')
    else:
        esito = controllo_provenienza(tabella_nomi)
        stampa(esito, salta=('elenco_discordanti',))
        for voce in esito.get('elenco_discordanti', []):
            print('      ->', voce)
        print('   ESITO:', 'superato' if esito['superato'] else 'NON superato')
        riepilogo['provenienza'] = esito

    # ---------------------------------------------------------------- 3, 4, 5
    anagrafica = None
    if percorsi.get('anagrafica') and os.path.exists(percorsi['anagrafica']):
        anagrafica = pd.read_csv(percorsi['anagrafica'])

    titolo('3', 'ANAGRAFICA - coerenza della tabella salvata dal notebook 04')
    if anagrafica is None:
        print('   saltato: anagrafica.csv non trovata. Indicala con --anagrafica.')
    else:
        print('   fonte:', percorsi['anagrafica'])
        print('   nota: cuscinetto, regime e classe sono ricavati dal nome della')
        print('   registrazione, quindi questo controllo verifica la coerenza del')
        print('   salvataggio, non l identita dei cuscinetti. Quella la verifica il 2.')
        esito = controllo_anagrafica(anagrafica, regimi)
        stampa(esito)
        print('   ESITO:', 'superato' if esito['superato'] else 'NON superato')
        riepilogo['anagrafica'] = esito

    titolo('4', 'BLOCCHI - la funzione del progetto contro la chiave completa')
    if anagrafica is None or funzioni is None:
        print('   saltato: servono anagrafica.csv e funzioni.py.')
    else:
        esito = controllo_blocchi(anagrafica, giri_per_blocco, funzioni)
        stampa(esito)
        print('   ESITO:', 'superato' if esito['superato'] else 'NON superato')
        if esito['matrici_identiche']:
            print('   -> le due chiavi producono blocchi identici: cambiare la chiave')
            print('      non modificherebbe nessun risultato della relazione.')
        riepilogo['blocchi'] = esito

    titolo('5', 'INSIEMI - train, val e test non condividono niente')
    if anagrafica is None or insiemi is None:
        print('   saltato: servono anagrafica.csv e config.INSIEMI.')
    else:
        esito = controllo_insiemi(anagrafica, insiemi, giri_per_blocco)
        for nome, parte in esito.items():
            if nome == 'superato':
                continue
            print('   %s' % nome)
            print('      blocchi     ', parte['blocchi'])
            print('      cuscinetti  ', parte['cuscinetti'])
            print('      %-16s %8s %12s %16s'
                  % ('fra', 'giri', 'cuscinetti', 'registrazioni'))
            for voce in parte['condivisioni']:
                print('      %-16s %8d %12d %16d'
                      % (voce['fra'], voce['giri_condivisi'],
                         voce['cuscinetti_condivisi'], voce['registrazioni_condivise']))
        print('   ESITO:', 'superato' if esito['superato'] else 'NON superato')
        if esito['superato']:
            print('   -> nessun giro, cuscinetto o registrazione e condiviso fra')
            print('      addestramento, validazione e verifica: nessun data leakage.')
        riepilogo['insiemi'] = esito

    # ---------------------------------------------------------------- 6
    titolo('6', 'INIEZIONE - anomalie note su dati simulati')
    if classe_di is None:
        print('   saltato: serve config.py per le tabelle delle classi.')
    else:
        esito = controllo_iniezione(regimi, cuscinetti, classe_di, natura_di,
                                    giri_per_blocco)
        print('   quale controllo rileva quale anomalia:')
        print(esito['tabella_nomi'].to_string(index=False))
        print()
        print('   comportamento delle due chiavi di raggruppamento:')
        print(esito['tabella_blocchi'].to_string(index=False))
        print()
        stampa(esito, salta=('tabella_nomi', 'tabella_blocchi'))
        print('   ESITO:', 'superato' if esito['superato'] else 'NON superato')
        riepilogo['iniezione'] = {k: v for k, v in esito.items()
                                  if not isinstance(v, pd.DataFrame)}
        riepilogo['iniezione']['tabella_nomi'] = esito['tabella_nomi'].to_dict('records')
        riepilogo['iniezione']['tabella_blocchi'] = esito['tabella_blocchi'].to_dict('records')

    # ---------------------------------------------------------------- esito
    print()
    print('=' * 78)
    print('RIEPILOGO')
    print('=' * 78)
    eseguiti = {k: v.get('superato') for k, v in riepilogo.items()}
    for nome, ok in eseguiti.items():
        print('   %-16s %s' % (nome, 'superato' if ok else 'NON superato'))
    saltati = {'completezza', 'provenienza', 'anagrafica', 'blocchi',
               'insiemi', 'iniezione'} - set(riepilogo)
    if saltati:
        print('   saltati:', ', '.join(sorted(saltati)))
    riepilogo['superato'] = all(eseguiti.values()) if eseguiti else False
    print()
    print('esito complessivo:', 'SUPERATO' if riepilogo['superato'] else 'NON SUPERATO')

    with open(argomenti.uscita, 'w') as fh:
        json.dump(riepilogo, fh, indent=2, default=str)
    print('riepilogo salvato in', os.path.abspath(argomenti.uscita))
    return 0 if riepilogo['superato'] else 1


if __name__ == '__main__':
    sys.exit(main())
