Dieser Fusion-Workflow (`fusion.py`) verfolgt einen soliden „Mixture of Experts“-Ansatz. Die Trennung in unabhängige Modell-Anfragen mit anschließender Synthese durch ein Judge-Modell ist methodisch sinnvoll, ebenso wie die saubere Speicherung in einem dedizierten `runs/`-Verzeichnis.

Für einen produktiven Einsatz auf deinem Mac gibt es jedoch architektonische Schwächen.

### Fachliche Bewertung & Schwachstellen
1. **Sequenzielle Ausführung (Flaschenhals):** Gemini, Claude und der initiale Codex-Aufruf werden nacheinander (synchron) ausgeführt. Da die drei Prompts unabhängig voneinander sind, verschenkst du hier massiv Zeit.
2. **CLI vs. API:** Die Nutzung von Kommandozeilen-Tools (`agy`, `codex`) via `subprocess` erzeugt Overhead und ist fehleranfällig gegenüber Änderungen im Output-Format. Python-SDKs oder direkte API-Calls wären robuster und performanter als das Parsen von CLI-Outputs (`clean_cli_output`).
3. **Umgang mit Fehlern im Judge-Prompt:** Wenn `run_cmd` einen Fehler oder Timeout liefert, wird der String `[ERROR]...` als reguläre "Antwort" in den Judge-Prompt injiziert. Das Synthese-Modell wird versuchen, diese Fehlermeldung inhaltlich zu bewerten.

### Unsicherheiten
* **Modell-IDs:** Es ist unklar, ob die `agy`-CLI Strings wie `"Gemini 3.1 Pro (High)"` exakt so als Parameter (`--model`) akzeptiert, oder ob dort formelle Bezeichner wie `gemini-3.1-pro` erwartet werden.
* **Blockierende CLIs:** Die CLI-Tools könnten interaktive Bestätigungen verlangen (z. B. Sandbox-Warnings). Der Parameter `--skip-git-repo-check` bei Codex deutet bereits auf Workarounds hin.

### Konkrete nächste Schritte
1. **Parallelisierung einbauen:** Nutze das Modul `concurrent.futures.ThreadPoolExecutor`, um die Funktionen `agy_answer` und `codex_answer` für die ersten drei Modelle gleichzeitig zu starten.
2. **Fehler abfangen:** Ändere das Skript so ab, dass das Judge-Modell darauf hingewiesen wird, wenn ein Modell ausgefallen ist, anstatt den Ausfall als inhaltlichen Beitrag zu werten (z. B. _"Modell A: <Ausfall, bitte ignorieren>"_).
3. **Modell-Namen validieren:** Teste im Terminal manuell, ob `agy -m "Gemini 3.1 Pro (High)" -p "Test"` reibungslos durchläuft.