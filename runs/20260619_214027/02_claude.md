Ich schaue mir zuerst die Struktur deines Projekts an.
Der aktuelle Run hat nur die Gemini-Antwort — der Lauf wurde offenbar bei Claude (also bei mir) abgebrochen, weil du die Frage über mich an mich selbst gestellt hast. Das ist erwartbar. Hier meine Bewertung:

---

## Bewertung: Fusion-Workflow

### ✅ Was gut ist

| Aspekt | Bewertung |
|---|---|
| **Grundidee** | Solides „Council"-Pattern: 3 unabhängige Antworten → 1 Synthese-Judge |
| **Persistenz** | Jeder Run in `runs/<timestamp>/` mit Einzelantworten + Metadata — sauber nachvollziehbar |
| **Clean-Funktion** | `clean_cli_output()` filtert CLI-Noise sinnvoll raus |
| **Judge-Prompt** | Gut strukturiert: Konsens → Widersprüche → Verwerfen → Synthese |
| **Timeout** | 30 Min. ist großzügig aber angemessen für lange Reasoning-Antworten |

### ⚠️ Probleme & Verbesserungspotenzial

**1. Sequenziell statt parallel** — Die drei Modelle laufen nacheinander. Bei ~2–5 Min. pro Antwort dauert ein Run unnötig lang.
```python
# Besser: concurrent.futures.ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
```

**2. Codex ist Judge über sich selbst** — GPT-5.5 Codex liefert Antwort C *und* macht die finale Synthese. Das erzeugt einen systematischen Bias zugunsten der eigenen initialen Antwort. Besser: ein *anderes* Modell als Judge, oder zumindest rotieren.

**3. Kein Error-Handling bei Einzelantworten** — Wenn ein Modell `[ERROR]` zurückgibt, wird der Fehlerstring als „Antwort" an den Judge übergeben. Der Judge verarbeitet dann Fehlertext als fachlichen Input.
```python
# Fehlende Prüfung:
if gemini.startswith("[ERROR]"):
    print("Gemini failed, skipping or retrying...")
```

**4. Hardcoded Pfad** — `WORKDIR = Path("/Users/jaenix/...")` macht das Skript nicht portabel. Besser: `Path(__file__).parent`.

**5. Kein Diff/Vergleich** — Du speicherst alle Antworten, aber es gibt keinen automatischen Vergleich (z.B. Übereinstimmungsgrad, Widerspruchsprotokoll). Das wäre für die Nachanalyse wertvoll.

**6. Prompt-Injection-Risiko** — Die Nutzerfrage wird unescaped in die Prompts eingebettet. Bei adversarial Input kann das den Judge-Prompt manipulieren.

### 📋 Konkrete nächste Schritte

1. **Parallelisierung** mit `ThreadPoolExecutor` — sofort machbar, halbiert die Laufzeit
2. **Judge-Modell rotieren** — z.B. Claude als Judge, wenn Codex eine der Antworten liefert
3. **Error-Guards** einbauen — fehlerhafte Antworten erkennen und aus dem Judge-Input ausschließen
4. **`WORKDIR` relativ machen** — `Path(__file__).parent`

### Fazit

**Sinnvoll aufgebaut, ja** — die Architektur (unabhängige Antworten → Synthese) ist das richtige Pattern für Multi-Model-Fusion. Die Hauptschwächen sind der Codex-Self-Judge-Bias und die fehlende Parallelisierung. Beides ist mit wenig Aufwand behebbar.