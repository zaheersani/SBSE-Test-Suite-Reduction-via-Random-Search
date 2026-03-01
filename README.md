# SBSE Test Cases Assignment (Node.js)

This project runs **search-based test suite reduction** experiments using two random-search algorithms:

- **A) Pure Random Sampling**
- **C) Elitist Random Search**

The script evaluates candidate test subsets for coverage, execution time, and subset size, then logs per-seed trajectories and a cross-seed summary.

## Requirements

- Node.js 18+ (Node.js 20 tested)

## Run

From the project root:

```bash
node app.js
```

With custom options:

```bash
node app.js --data ./dataset.json --budget 10000 --seeds 30 --out ./results --lambda 0.35 --p 0.35
```

### CLI options

- `--data` Path to input dataset JSON (default: `./dataset.json`)
- `--budget` Number of evaluations per seed (default: `10000`)
- `--seeds` Number of seeds/runs per algorithm (default: `30`)
- `--out` Output directory (default: `./results`)
- `--lambda` Time penalty weight for feasible solutions (default: `0.35`)
- `--p` Inclusion probability for random subset initialization (default: `0.35`)

## Output files

For each run, the script writes:

- Per-seed logs in `results/logs/`
  - `A_pure_random_sampling_seedX.csv`
  - `C_elitist_random_search_seedX.csv`
- Overall summary in `results/summary.csv`

### Log CSV columns (per seed)

- `eval`
- `best_fitness`
- `best_cov`
- `best_time`
- `best_size`
- `is_feasible`
- `best_selected_tests`
- `best_included_elements`

### Summary CSV columns

- `algorithm`
- `seed`
- `final_best_fitness`
- `final_best_cov`
- `final_best_time`
- `final_best_size`
- `final_best_feasible`
- `selected_tests`
- `included_elements`

## Results

![Results Infographic](Infographics.png)
