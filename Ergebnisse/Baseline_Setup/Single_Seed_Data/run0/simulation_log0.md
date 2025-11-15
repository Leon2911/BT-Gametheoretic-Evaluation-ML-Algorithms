## Simulationslauf vom: 2025-11-13 23:21:54

### Globale Parameter
- **Begegnungsschema:** `SpatialGridScheme`
- **Gittergröße:** `(10, 20)`
- **Anzahl Matches/Duelle:** `200000`
- **Runden pro Episode:** `200`
- **Zufalls-Seed:** `0`

### Agenten-Population
- **QL:** `200`
- **Gesamt:** `200` Agenten

### Lern-Hyperparameter
- **alpha:** `0.05`
- **gamma:** `0.95`
- **epsilon:** `1.0`
- **epsilon_decay:** `0.9995`
- **min_epsilon:** `0.001`

---

### Finale Ergebnisse (des obigen Laufs)

**Globale Metriken (Gesamtsystem):**
- **System-Effizienz (% des Max):** `59.36%`
- **Durchschnittliche Kooperationsrate:** `26.45%`

**Metriken pro Agententyp (Durchschnitt über den gesamten Zeitraum):**
- **Avg. Reward/Match:**
  - `QLearningAgent`: 355.09

**Finale Cluster-Analyse:**
- **Anzahl gefundener Cluster:** `3`
- **Gesamtfläche pro Strategietyp:**
  - `Defector`: 41.0%
  - `Polarized`: 16.5%
  - `Cautious Coop.`: 16.5%
  - `Cooperator (ALLC)`: 11.5%
  - `TFT-like`: 10.5%
  - `WSLS`: 4.0%
- **Anzahl Cluster pro Strategietyp:**
  - `Defector`: 1
  - `Cooperator (ALLC)`: 1
  - `TFT-like`: 1

---

