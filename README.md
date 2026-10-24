# When is intraday information worth using? Simulation and analysis code

Code, parameters, and elicitation records for the manuscript *"When is intraday information worth using? Arrivals versus abandonments for afternoon staffing of a service counter with customer reneging"* (submitted to *Pesquisa Operacional*).

The model decides whether to keep a third server at a single-queue service counter in the afternoon block after observing either morning arrivals or morning abandonments. Morning and afternoon demand states are uncertain and partially persistent; customers renege when their wait exceeds their patience. A discrete-event simulation estimates the consequences of each staffing strategy, and backward induction gives the optimal policy, the expected value of sample information (EVSI) of each signal, and the expected value of perfect information (EVPI).

## Repository structure

| Path | Content |
|---|---|
| `scenario_parameters.json` | All scenario parameters (company information and expert elicitation) and simulation settings |
| `core2.py` | Discrete-event simulation of one day: FCFS service, reneging, staffing change at 16:00 with transfer delay, service after closing |
| `sims.py` | Simulates the base case, the reduced-volume scenario, and the sensitivity sets (4 000 replications × 9 demand-state pairs × 4 staffing combinations) |
| `ana.py` | Cost model; policy selection on replications 0–1 999 and evaluation on 2 000–3 999; EVSI and EVPI; threshold, per-count, and three-interval rules |
| `curves.py`, `all.py`, `more.py`, `grid.py`, `uncertainty_and_statecost.py` | Results reported in the manuscript and supplementary material |
| `core3.py`, `sens2.py` | Robustness of the conclusions to modeling assumptions (patience and occupancy distributions, within-state demand variation, hourly profile, transfer delay, allocation of the unspecified prior mass) |
| `support.py` | Supporting estimates: signal distributions by state, simulated consequences by state pair and strategy, prior derivation |
| `figures.py`, `influence_diagram.dot` | Figures 1–3 |
| `elicitation_records.csv`, `elicitation_records.xlsx`, `ELICITATION.md` | Anonymized records of the three elicitation sessions and their description |

## Requirements

- Python 3.12.3
- Packages listed in `requirements.txt` (NumPy 2.4.4, Numba 0.68.0, SciPy 1.17.1, pandas 3.0.2, Matplotlib 3.10.8)
- Graphviz (`dot`) for Figure 1

## How to reproduce

```bash
pip install -r requirements.txt
./run_all.sh
```

The full run takes several minutes on a single core. All customer attributes and the transfer delay are drawn before event processing from seeded streams (morning customers by replication and morning state, afternoon customers by replication and afternoon state, transfer delay by replication), so results are reproducible with the package versions above.

## Mapping to the manuscript

| Output | Script |
|---|---|
| Table 3; valuation comparison (Supplementary S5) | `ana.py`, `all.py` |
| Figure 2 | `grid.py`, `figures.py` |
| Figure 3; persistence tables (S4) | `all.py`, `figures.py` |
| Table 4; sensitivity (S6) | `all.py`, `uncertainty_and_statecost.py` |
| Abandonment contrast, posteriors, Monte Carlo intervals, verification (S2, S3, S9) | `more.py` |
| Joint parameter uncertainty (S7) | `uncertainty_and_statecost.py` |

## Data

The scenario is illustrative. Operational and cost data were provided by the company, and behavioral, demand, and value parameters were elicited from its sales manager in three structured sessions. No customer-level records are included. The company and the manager are anonymized.

## License

Code: MIT License (see `LICENSE`). Data and elicitation records: CC BY 4.0.

## Citation

If you use this code, please cite the article and this repository:

> [Authors]. *When is intraday information worth using? Arrivals versus abandonments for afternoon staffing of a service counter with customer reneging*. Pesquisa Operacional, [year]. Code: https://doi.org/[Zenodo DOI]
