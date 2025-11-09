## Simulationslauf vom: 2025-11-09 23:00:59

### Globale Parameter
- **Begegnungsschema:** `SpatialGridScheme`
- **Gittergröße:** `(10, 20)`
- **Anzahl Matches/Duelle:** `60000`
- **Runden pro Episode:** `200`
- **Zufalls-Seed:** `1`

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
- **System-Effizienz (% des Max):** `45.89%`
- **Durchschnittliche Kooperationsrate:** `15.75%`

**Metriken pro Agententyp (Durchschnitt über den gesamten Zeitraum):**
- **Avg. Reward/Match:**
  - `QLearningAgent`: 275.57
- **Avg. Kooperationsrate:**
  - `QLearningAgent`: 15.75%

**Finale Cluster-Analyse:**
- **Anzahl gefundener Cluster:** `1`
- **Gesamtfläche pro Strategietyp:**
  - `Defector`: 87.0%
  - `Cautious Coop.`: 7.0%
  - `Cooperator (ALLC)`: 3.0%
  - `Polarized`: 2.5%
  - `WSLS`: 0.5%
- **Anzahl Cluster pro Strategietyp:**
  - `Defector`: 1

---

