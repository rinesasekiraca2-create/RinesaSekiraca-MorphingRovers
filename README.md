# Rinesa Sekiraca — Morphing Rovers

Projekt studentor për sfidën **SpOC 2: Morphing Rovers** të ESA-s. Qëllimi është të optimizohet një rover që ndryshon formën sipas terrenit dhe arrin mostrat në 30 skenarë të ndryshëm.

Programi përdor hartat e terrenit, optimizon katër maskat e roverit, kërkon rrugë të drejta dhe të lakuara për secilin skenar dhe trajnon kontrolluesin për drejtimin dhe ndërrimin e formës. Rezultati vlerësohet me fitness-in mesatar në 30 skenarët; **vlera më e ulët është më e mirë**. Vlera 1.0 është kufiri ideal teorik dhe nuk garantohet nga trajnimi.

## Kërkesat

- Python 3.10 ose më i ri
- [uv](https://docs.astral.sh/uv/)
- Varësitë Python përcaktohen te `pyproject.toml` dhe fiksohen te `uv.lock`

## Instalimi

Nga dosja kryesore e projektit:

```bash
uv sync
```

## Ekzekutimi

Për optimizimin e plotë:

```bash
uv run morphing-rovers run -c configs/default.yml
```

Konfigurimi parazgjedhur bën 100 përsëritje trajnimi, secila me horizonte nga 1 deri në 500 hapa. Ky ekzekutim mund të zgjasë shumë. Parametrat mund të ndryshohen nga komanda, pa redaktuar konfigurimin. Për një provë më të shkurtër:

```bash
uv run morphing-rovers run -c configs/default.yml \
  -s output_dir=runs/trial \
  -s schedule.rounds=0 \
  -s masks.dataset.samples_per_scenario=20 \
  -s masks.optimization.n_iter=10 \
  -s imitation.n_runs=1 \
  -s imitation.horizon_stop=10 \
  -s imitation.n_iter=10
```

Prova e shkurtër kontrollon që projekti niset; nuk është optimizim i plotë dhe mund të japë fitness më të lartë.

## Rezultatet

Rezultatet ruhen te `output_dir` i konfigurimit, parazgjedhur `runs/default/`:

- `submission.json` — kromozomi në format për dorëzim
- `best_chromosome.npy` — kromozomi më i mirë i gjetur
- `history.csv` — fitness-i dhe përparimi përgjatë trajnimit
- `masks.npy` — maskat e optimizuara të roverit
- `route_choices.json` — rruga dhe korrigjimi i drejtimit të zgjedhur për çdo skenar
- `config.yml` — konfigurimi i përdorur për ekzekutimin

Gjatë trajnimit, rreshtat e terminalit tregojnë `run`, `horizon`, `fitness`, rezultatin më të mirë deri atëherë dhe numrin e objektivave të arritura. Në përfundim shfaqet tabela për 30 skenarët dhe `Avg. fitness`.

## Vlerësimi dhe dorëzimi

Për të vlerësuar një kromozom dhe për të shfaqur trajektoret:

```bash
uv run morphing-rovers evaluate runs/default/best_chromosome.npy --plot
```

Për të krijuar skedarin e dorëzimit nga një kromozom ekzistues:

```bash
uv run morphing-rovers submit runs/default/best_chromosome.npy \
  -o submission.json \
  --name "Rinesa Sekiraca"
```

## Ekzekutimi në Kaggle

Ngarko projektin si Dataset dhe shtoje te Notebook-u me **Add Input**. Kopjoje në hapësirën e shkrueshme të Notebook-ut, kalo te dosja e projektit dhe nise modulin:

```bash
PYTHONPATH=src python -m morphing_rovers.cli run \
  -c configs/default.yml \
  -s output_dir=/kaggle/working/runs/kaggle
```

Nëse ZIP-i nxirret në një dosje të ndërmjetme, përdor dosjen që përmban `pyproject.toml` si dosjen kryesore të projektit. Rezultatet ruhen te `/kaggle/working/runs/kaggle/`.

## Testet

Për të ekzekutuar testet:

```bash
uv run pytest
```
