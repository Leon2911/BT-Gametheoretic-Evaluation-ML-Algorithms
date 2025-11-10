## Simulationslauf vom: 2025-11-10 20:47:07

### Globale Parameter
- **Begegnungsschema:** `SpatialGridScheme`
- **Gittergröße:** `(10, 20)`
- **Anzahl Matches/Duelle:** `60000`
- **Runden pro Episode:** `200`
- **Zufalls-Seed:** `5`

### Agenten-Population
- **QL:** `200`
- **Gesamt:** `200` Agenten

### Lern-Hyperparameter
- **alpha:** `0.05`
- **gamma:** `0.95`
- **epsilon:** `1.0`
- **epsilon_decay:** `0.9995`
- **min_epsilon:** `0.01`

---

### Finale Ergebnisse (des obigen Laufs)

**Globale Metriken (Gesamtsystem):**
- **System-Effizienz (% des Max):** `46.31%`
- **Durchschnittliche Kooperationsrate:** `15.76%`

**Metriken pro Agententyp (Durchschnitt über den gesamten Zeitraum):**
- **Avg. Reward/Match:**
  - `QLearningAgent`: 278.76
- **Avg. Kooperationsrate:**
  - `QLearningAgent`: 15.76%

**Finale Cluster-Analyse:**
- **Anzahl gefundener Cluster:** `1`
- **Gesamtfläche pro Strategietyp:**
  - `Defector`: 82.5%
  - `Cautious Coop.`: 5.5%
  - `Cooperator (ALLC)`: 4.5%
  - `Polarized`: 4.5%
  - `TFT-like`: 2.0%
  - `WSLS`: 1.0%
- **Anzahl Cluster pro Strategietyp:**
  - `Defector`: 1

---

