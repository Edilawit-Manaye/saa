================================================================================
TASK 2 — COMPLETE STEPS USING DOCKER (in order)
================================================================================

Use this as your checklist. Do the steps in order. All commands are run in
PowerShell from the project folder:  C:\Users\hp\Desktop\neural-subgraph-matcher-miner

We use  -v "${PWD}:/app"  so that files (pkl, results, ckpt) are saved on your PC.

--------------------------------------------------------------------------------
STEP 0 — DOWNLOAD AND INSTALL DOCKER (do this first, once)
--------------------------------------------------------------------------------

  1. Open a browser and go to:
     https://www.docker.com/products/docker-desktop/

  2. Click "Download for Windows" (or the button that matches your PC).

  3. Run the installer (Docker Desktop Installer.exe).
     - If asked, leave "Use WSL 2 instead of Hyper-V" enabled.
     - Finish the setup and restart the PC if it asks.

  4. After restart, open "Docker Desktop" from the Start menu.
     Wait until it says "Docker Desktop is running" (whale icon in the system tray).

  5. Check that Docker works: open PowerShell and run:
     docker --version
     You should see something like "Docker version 24.x.x".

  6. Open your project folder in PowerShell:
     cd C:\Users\hp\Desktop\neural-subgraph-matcher-miner

  After this, you never need to "download Docker" again. You only run the
  commands in Step 1 and below.

--------------------------------------------------------------------------------
BEFORE YOU START (after Docker is installed)
--------------------------------------------------------------------------------

[ ] Docker Desktop is installed and running.
[ ] You are in the project folder in PowerShell (cd to neural-subgraph-matcher-miner).
[ ] The downloaded file is in the data/ folder:
      data/email-Eu-core.txt.gz   OR   data/email-Eu-core (1).txt.gz

--------------------------------------------------------------------------------
STEP 1 — Build the Docker image (once)
--------------------------------------------------------------------------------

  docker build -t neural-subgraph-miner .

  Wait until it finishes. This installs Python and all dependencies.

--------------------------------------------------------------------------------
STEP 2 — Train the encoder once (creates ckpt/model.pt)
--------------------------------------------------------------------------------

  docker run --rm -v "${PWD}:/app" neural-subgraph-miner python -m subgraph_matching.train --node_anchored

  This uses synthetic data (no extra download). When it finishes, you should have
  ckpt/model.pt in your project folder. Training can take a while (e.g. 30+ min).

--------------------------------------------------------------------------------
STEP 3 — Convert the dataset to pkl
--------------------------------------------------------------------------------

  docker run --rm -v "${PWD}:/app" neural-subgraph-miner python scripts/convert_email_eu_core_to_pkl.py

  Input:  data/email-Eu-core.txt.gz  or  data/email-Eu-core (1).txt.gz
  Output: data/email-Eu-core.pkl  (saved in your data/ folder)

  You are not “changing” the dataset in place — the script reads the .txt.gz and
  writes a new file email-Eu-core.pkl. The .txt.gz stays as is.

--------------------------------------------------------------------------------
STEP 4 — Run the decoder for all three strategies (Task 2: Experiment)
--------------------------------------------------------------------------------

  Run these three commands one by one. Each will print "Total time: X.XXs" and
  "Size K: N unique pattern types". Write those down for your table.

  Greedy:
  docker run --rm -v "${PWD}:/app" neural-subgraph-miner python -m subgraph_mining.decoder --dataset data/email-Eu-core.pkl --n_trials 1000 --n_neighborhoods 10000 --min_neighborhood_size 3 --max_neighborhood_size 29 --radius 3 --search_strategy greedy --out_path results/patterns_greedy.p

  MCTS:
  docker run --rm -v "${PWD}:/app" neural-subgraph-miner python -m subgraph_mining.decoder --dataset data/email-Eu-core.pkl --n_trials 1000 --n_neighborhoods 10000 --min_neighborhood_size 3 --max_neighborhood_size 29 --radius 3 --search_strategy mcts --out_path results/patterns_mcts.p

  Beam:
  docker run --rm -v "${PWD}:/app" neural-subgraph-miner python -m subgraph_mining.decoder --dataset data/email-Eu-core.pkl --n_trials 1000 --n_neighborhoods 10000 --min_neighborhood_size 3 --max_neighborhood_size 29 --radius 3 --search_strategy beam --out_path results/patterns_beam.p

  From each run, note:
    - Runtime (seconds): the line "Total time: X.XXs"
    - Number of patterns: sum of "Size K: N unique pattern types" for all K

--------------------------------------------------------------------------------
STEP 5 — (Optional) Configuration tuning for “Config vs Number of Patterns”
--------------------------------------------------------------------------------

  Run the decoder again with different --radius (e.g. 2 and 4) to compare.
  Example (greedy with radius 2):

  docker run --rm -v "${PWD}:/app" neural-subgraph-miner python -m subgraph_mining.decoder --dataset data/email-Eu-core.pkl --n_trials 1000 --n_neighborhoods 10000 --min_neighborhood_size 3 --max_neighborhood_size 29 --radius 2 --search_strategy greedy --out_path results/patterns_greedy_r2.p

  Example (greedy with radius 4):

  docker run --rm -v "${PWD}:/app" neural-subgraph-miner python -m subgraph_mining.decoder --dataset data/email-Eu-core.pkl --n_trials 1000 --n_neighborhoods 10000 --min_neighborhood_size 3 --max_neighborhood_size 29 --radius 4 --search_strategy greedy --out_path results/patterns_greedy_r4.p

  Note runtime and pattern count for each. You now have data for a “Configuration
  (radius) vs Number of Patterns” table or plot.

--------------------------------------------------------------------------------
STEP 6 — Visualize (table and plots)
--------------------------------------------------------------------------------

  Option A — Run the plot script (if you have Python + matplotlib on your PC):

    python scripts/plot_task2_results.py

  That script expects results/task2_results.csv. Since we ran the decoder by
  hand, that CSV does not exist unless you ran scripts/run_task2_experiments.py
  (which runs the decoder for you). So either:

  Option B — Build the table and plots yourself from the numbers you wrote down:

    1. Table: Search Strategy vs Runtime
           Strategy   |  Runtime (s)
           -----------+-------------
           Greedy     |  ___
           MCTS       |  ___
           Beam       |  ___

    2. Table or plot: Configuration (radius) vs Number of Patterns
           Radius     |  N patterns
           -----------+-------------
           2          |  ___
           3          |  ___
           4          |  ___

    3. Use Excel, Google Sheets, or Matplotlib/Seaborn to draw the plots
       required by your instructor (Search Strategy vs Runtime; Config vs
       Number of Patterns).

--------------------------------------------------------------------------------
STEP 7 — Justify (for the report)
--------------------------------------------------------------------------------

  From your table and plots:
  - Identify the “Best Algorithm”: e.g. which of Greedy / MCTS / Beam had best
    runtime or most patterns (depending on how you define “best”).
  - Identify the “Best Config”: e.g. which radius (or n_trials / n_neighborhoods)
    gave the best trade-off.
  - Write a short justification in your report based on the numbers.

--------------------------------------------------------------------------------
TASK 2 REQUIREMENTS — CHECKLIST
--------------------------------------------------------------------------------

  [ ] Run: SPMiner executed on email-Eu-core (Steps 3 + 4).
  [ ] Experiment: All three strategies run (Greedy, MCTS, Beam); at least one
      hyperparameter (e.g. radius) varied for config tuning (Step 5).
  [ ] Visualize: Metrics table or plot for Search Strategy vs Runtime.
  [ ] Visualize: Metrics table or plot for Configuration Tuning vs Number of
      Patterns Found.
  [ ] Justify: Best Config and Best Algorithm identified and explained in the
      report.

--------------------------------------------------------------------------------
QUICK REFERENCE — “Convert to pkl”?
--------------------------------------------------------------------------------

  Converting the dataset to pkl is Step 3. You run the conversion script once.
  It reads  data/email-Eu-core.txt.gz  (or  email-Eu-core (1).txt.gz)  and
  writes  data/email-Eu-core.pkl.  The decoder then uses the .pkl file; the
  original .txt.gz is not modified. You do not “change” the dataset file
  itself — you create a new .pkl file from it.

================================================================================
