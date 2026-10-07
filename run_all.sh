#!/usr/bin/env bash
# Reproduces all results. Run from the repository root. Requires Graphviz (dot) for Figure 1.
set -e
cd src
python sims.py                       # simulation of all scenarios -> sims.pkl
python ana.py                        # base-case policies (prints)
python curves.py                     # EVSI vs lost-customer value -> curves.pkl
python all.py                        # valuations, persistence, sensitivity -> all.pkl
python more.py                       # block contrast, posteriors, MC intervals, verification -> more.pkl
python grid.py                       # value-cost map for Figure 2 -> grid.pkl
python uncertainty_and_statecost.py  # state-dependent costs and joint uncertainty -> statecost.pkl, unc_raw.pkl
python figures.py                    # fig1.png, fig2.png, fig3.png
