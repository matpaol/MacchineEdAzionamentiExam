"""Verifica il raggruppamento usato per costruire i blocchi della CNN.

Uso:
    python verifica_identificatori.py percorso/anagrafica.csv

Lo script deve essere eseguito dalla cartella che contiene ``funzioni.py`` e
``config.py``. Non modifica i dati e non addestra i modelli.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


COLONNE = [
    "cuscinetto", "natura", "regime", "registrazione", "giro", "classe"
]


def costruisci_blocchi_chiave_completa(anagrafica, giri_per_blocco):
    """Costruisce i blocchi separando cuscinetto, regime e registrazione."""
    blocchi = []

    chiavi = ["cuscinetto", "regime", "registrazione"]
    for _, gruppo in anagrafica.groupby(chiavi, sort=False):
        posizioni = gruppo.sort_values("giro").index.to_numpy()
        n_blocchi = len(posizioni) // giri_per_blocco

        for i in range(n_blocchi):
            inizio = i * giri_per_blocco
            blocchi.append(posizioni[inizio:inizio + giri_per_blocco])

    return np.asarray(blocchi)


def conta_blocchi_misti(anagrafica, blocchi, colonna):
    if len(blocchi) == 0:
        return 0

    valori = anagrafica.loc[blocchi.ravel(), colonna].to_numpy()
    valori = valori.reshape(blocchi.shape)
    return int(np.sum((valori != valori[:, [0]]).any(axis=1)))


def verifica(anagrafica, funzioni, giri_per_blocco):
    mancanti = sorted(set(COLONNE) - set(anagrafica.columns))
    if mancanti:
        raise ValueError("Colonne mancanti: " + ", ".join(mancanti))

    if anagrafica[COLONNE].isna().any().any():
        raise ValueError("L'anagrafica contiene valori mancanti")

    associazioni = (
        anagrafica.groupby("registrazione")
        [["cuscinetto", "regime", "classe"]]
        .nunique()
    )
    collisioni = associazioni[(associazioni > 1).any(axis=1)]

    duplicati = int(
        anagrafica.duplicated(
            ["cuscinetto", "regime", "registrazione", "giro"]
        ).sum()
    )

    blocchi_originali, _ = funzioni.costruisci_blocchi(
        anagrafica, giri_per_blocco
    )
    blocchi_originali = np.asarray(blocchi_originali)
    blocchi_completi = costruisci_blocchi_chiave_completa(
        anagrafica, giri_per_blocco
    )

    stessa_forma = blocchi_originali.shape == blocchi_completi.shape
    blocchi_identici = bool(
        stessa_forma and np.array_equal(blocchi_originali, blocchi_completi)
    )

    misti = {
        colonna: conta_blocchi_misti(anagrafica, blocchi_originali, colonna)
        for colonna in ["cuscinetto", "regime", "classe"]
    }

    giri = anagrafica.loc[blocchi_originali.ravel(), "giro"].to_numpy()
    giri = giri.reshape(blocchi_originali.shape)
    non_consecutivi = int(
        np.sum((np.diff(giri, axis=1) != 1).any(axis=1))
    ) if len(giri) else 0

    superato = (
        collisioni.empty
        and duplicati == 0
        and blocchi_identici
        and all(valore == 0 for valore in misti.values())
        and non_consecutivi == 0
    )

    return {
        "righe": int(len(anagrafica)),
        "registrazioni": int(anagrafica["registrazione"].nunique()),
        "collisioni": int(len(collisioni)),
        "duplicati": duplicati,
        "blocchi_originali": int(len(blocchi_originali)),
        "blocchi_chiave_completa": int(len(blocchi_completi)),
        "blocchi_identici": blocchi_identici,
        "blocchi_misti": misti,
        "blocchi_non_consecutivi": non_consecutivi,
        "superato": superato,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("anagrafica", help="percorso di anagrafica.csv")
    parser.add_argument(
        "--progetto",
        default=".",
        help="cartella che contiene config.py e funzioni.py",
    )
    parser.add_argument("--json", help="salva anche un riepilogo JSON")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.progetto))
    try:
        import config
        import funzioni
    except ImportError as errore:
        raise SystemExit(
            "Impossibile importare config.py e funzioni.py: " + str(errore)
        )

    anagrafica = pd.read_csv(args.anagrafica)
    esito = verifica(anagrafica, funzioni, config.GIRI_PER_BLOCCO)

    print(f"Righe: {esito['righe']}")
    print(f"Registrazioni: {esito['registrazioni']}")
    print(f"Collisioni tra registrazioni: {esito['collisioni']}")
    print(f"Giri duplicati: {esito['duplicati']}")
    print(f"Blocchi prodotti: {esito['blocchi_originali']}")
    print(f"Blocchi identici con la chiave completa: {esito['blocchi_identici']}")
    print(f"Blocchi misti: {esito['blocchi_misti']}")
    print(f"Blocchi con giri non consecutivi: {esito['blocchi_non_consecutivi']}")
    print("ESITO:", "SUPERATO" if esito["superato"] else "NON SUPERATO")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump(esito, file, indent=2)

    return 0 if esito["superato"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
