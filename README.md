================================================================================
TASK 2 — COMPLETE STEPS USING DOCKER (in order)
================================================================================

Docker is now working (Hyper-V backend). Use this as your checklist.
All commands run in PowerShell from:  C:\Users\hp\Desktop\neural-subgraph-matcher-miner

Use  -v "${PWD}:/app  so files (pkl, results, ckpt) are saved on your PC.

--------------------------------------------------------------------------------
BEFORE YOU START
--------------------------------------------------------------------------------

[ ] Docker Desktop is running (whale icon in taskbar).
[ ] You are in the project folder in PowerShell.
[ ] The downloaded file is in data/:  email-Eu-core.txt.gz  OR  email-Eu-core (1).txt.gz

--------------------------------------------------------------------------------
STEP 1 — Build the Docker image (once)
--------------------------------------------------------------------------------

  docker build -t neural-subgraph-miner .

  Wait until it finishes. You should see "Successfully built" and "Successfully tagged".
  Then run:  docker images   — you should see neural-subgraph-miner in the list.

--------------------------------------------------------------------------------
STEP 2 — Train the encoder once (creates ckpt/model.pt)
--------------------------------------------------------------------------------

  docker run --rm -v "${PWD}:/app" neural-subgraph-miner python -m subgraph_matching.train --node_anchored

  This uses synthetic data. When it finishes, you should have ckpt/model.pt in your project folder.
  Training can take a long time (e.g. 30+ minutes).

--------------------------------------------------------------------------------
STEP 3 — Convert the dataset to pkl
--------------------------------------------------------------------------------

  docker run --rm -v "${PWD}:/app" neural-subgraph-miner python scripts/convert_email_eu_core_to_pkl.py

  Input:  data/email-Eu-core.txt.gz  or  data/email-Eu-core (1).txt.gz
  Output: data/email-Eu-core.pkl  (in your data/ folder)

  The script reads the .txt.gz and writes the .pkl; the original file is not changed.

--------------------------------------------------------------------------------
STEP 4 — Run the decoder for all three strategies (Task 2: Experiment)
--------------------------------------------------------------------------------

  Run these three commands. From each run, note "Total time: X.XXs" and "Size K: N unique pattern types" for your table.

  Greedy:
  docker run --rm -v "${PWD}:/app" neural-subgraph-miner python -m subgraph_mining.decoder --dataset data/email-Eu-core.pkl --n_trials 1000 --n_neighborhoods 10000 --min_neighborhood_size 3 --max_neighborhood_size 29 --radius 3 --search_strategy greedy --out_path results/patterns_greedy.p

  MCTS:
  docker run --rm -v "${PWD}:/app" neural-subgraph-miner python -m subgraph_mining.decoder --dataset data/email-Eu-core.pkl --n_trials 1000 --n_neighborhoods 10000 --min_neighborhood_size 3 --max_neighborhood_size 29 --radius 3 --search_strategy mcts --out_path results/patterns_mcts.p

  Beam:
  docker run --rm -v "${PWD}:/app" neural-subgraph-miner python -m subgraph_mining.decoder --dataset data/email-Eu-core.pkl --n_trials 1000 --n_neighborhoods 10000 --min_neighborhood_size 3 --max_neighborhood_size 29 --radius 3 --search_strategy beam --out_path results/patterns_beam.p

  Note for each: Runtime (seconds) and total number of patterns (sum of "Size K: N unique pattern types").

--------------------------------------------------------------------------------
STEP 5 — (Optional) Configuration tuning
--------------------------------------------------------------------------------

  Run decoder with different --radius (e.g. 2 and 4) for "Config vs Number of Patterns":

  docker run --rm -v "${PWD}:/app" neural-subgraph-miner python -m subgraph_mining.decoder --dataset data/email-Eu-core.pkl --n_trials 1000 --n_neighborhoods 10000 --min_neighborhood_size 3 --max_neighborhood_size 29 --radius 2 --search_strategy greedy --out_path results/patterns_greedy_r2.p

  docker run --rm -v "${PWD}:/app" neural-subgraph-miner python -m subgraph_mining.decoder --dataset data/email-Eu-core.pkl --n_trials 1000 --n_neighborhoods 10000 --min_neighborhood_size 3 --max_neighborhood_size 29 --radius 4 --search_strategy greedy --out_path results/patterns_greedy_r4.p

  Note runtime and pattern count for each.

--------------------------------------------------------------------------------
STEP 6 — Visualize
--------------------------------------------------------------------------------

  Build from the numbers you wrote down:
  1. Table: Search Strategy vs Runtime (Greedy | MCTS | Beam and their runtimes).
  2. Table or plot: Configuration (radius) vs Number of Patterns Found.
  Use Excel, Google Sheets, or Matplotlib/Seaborn as required.

--------------------------------------------------------------------------------
STEP 7 — Justify (for the report)
--------------------------------------------------------------------------------

  Identify "Best Algorithm" (e.g. which of Greedy/MCTS/Beam) and "Best Config" (e.g. which radius).
  Write a short justification in your report based on your table and plots.

--------------------------------------------------------------------------------
TASK 2 REQUIREMENTS — CHECKLIST
--------------------------------------------------------------------------------

  [ ] Run: SPMiner executed on email-Eu-core (Steps 3 + 4).
  [ ] Experiment: All three strategies run (Greedy, MCTS, Beam); optional: radius varied (Step 5).
  [ ] Visualize: Table or plot — Search Strategy vs Runtime.
  [ ] Visualize: Table or plot — Configuration vs Number of Patterns Found.
  [ ] Justify: Best Config and Best Algorithm identified in the report.

--------------------------------------------------------------------------------
POWERSHELL NOTE
--------------------------------------------------------------------------------

  If  -v "${PWD}:/app"  does not work, try  -v "%CD%:/app"  (in cmd) or ensure you are in the project folder.

================================================================================
