## Simulationslauf vom: 2025-11-10 20:23:02

### Globale Parameter
- **Begegnungsschema:** `SpatialGridScheme`
- **Gittergröße:** `(10, 20)`
- **Anzahl Matches/Duelle:** `60000`
- **Runden pro Episode:** `200`
- **Zufalls-Seed:** `4`

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
- **System-Effizienz (% des Max):** `45.67%`
- **Durchschnittliche Kooperationsrate:** `15.30%`

**Metriken pro Agententyp (Durchschnitt über den gesamten Zeitraum):**
- **Avg. Reward/Match:**
  - `QLearningAgent`: 274.59
- **Avg. Kooperationsrate:**
  - `QLearningAgent`: 15.30%

**Finale Cluster-Analyse:**
- **Anzahl gefundener Cluster:** `1`
- **Gesamtfläche pro Strategietyp:**
  - `Defector`: 80.5%
  - `Cautious Coop.`: 7.0%
  - `Polarized`: 7.0%
  - `TFT-like`: 3.0%
  - `WSLS`: 2.0%
  - `Cooperator (ALLC)`: 0.5%
- **Anzahl Cluster pro Strategietyp:**
  - `Defector`: 1

---

