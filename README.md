# ppcx-domains — Internal README

This repository contains scripts and utilities for batch and single-date clusterization of DIC (Digital Image Correlation) data, using MCMC-based clustering and spatial priors. The main scripts are designed for both interactive and high-throughput batch processing.

---

## Clusterization Scripts Overview

### 1. `ppcx_mcmc_clustering.py`

- **Purpose:** Runs MCMC-based clustering for a single reference date.
- **Inputs:** Reference date, config file (optional), and CLI overrides.
- **Outputs:** Clustered sectors, summary plots, and statistics for the specified date.

#### **Basic Usage (Single Date):**
```bash
python ppcx_mcmc_clustering.py --date 2023-07-01 --config config.yaml
```
You can override config parameters directly from the CLI:
```bash
python ppcx_mcmc_clustering.py --date 2023-07-01 data.dt_min=12 mcmc.sample_options.draws=500
```

---

### 2. `clusterize_batch.py`

- **Purpose:** Batch launcher for running `ppcx_mcmc_clustering.py` over many dates, with parallelization and robust logging.
- **Modes:**
	- **Direct execution:** Python handles parallelism.
	- **Dry-run:** Generates command lines for external tools (e.g., GNU Parallel).

#### **Run a Range of Dates (Direct Execution):**
```bash
python clusterize_batch.py --date-range 2023-06-01 2023-06-05 --jobs 4 data.dt_min=12
```

#### **Run Multiple Date Ranges:**

```bash
python clusterize_batch.py \
	--date-range 2022-06-01 2022-10-30 \
	--date-range 2023-06-01 2023-10-30
```

#### **Dry-Run Mode (Recommended for Production):**
Generate a list of commands for external batch tools:
```bash
python clusterize_batch.py --date-range 2023-06-01 2023-08-01 --dry-run > jobs.txt
```


## Batch Processing with GNU Parallel

**Step 1: Generate the job list**
```bash
python clusterize_batch.py --date-range 2023-06-01 2023-08-01 --dry-run > jobs.txt
```

**Step 2: Run with GNU Parallel**
```bash
parallel -j 4 --bar --joblog run.log --resume < jobs.txt
```
- `-j 4`: Number of parallel jobs (adjust to your CPU/GPU resources).
- `--bar`: Shows a progress bar.
- `--joblog run.log`: Logs job status.
- `--resume`: Skips already completed jobs if re-run.

To keep jobs running after disconnecting from SSH, use:
```bash
nohup parallel -j 4 --bar --joblog run.log --resume < jobs.txt > parallel.out 2>&1 &
```
### Log Output: Inspecting and Retrying Failed Jobs

When running batch jobs with GNU Parallel, all job statuses are recorded in a tab-separated log file (e.g., `run.log`). This file is essential for monitoring progress and troubleshooting failures.

Each line in the joblog corresponds to a job and contains 9 columns:

| Column | Name      | Description                                                                 |
|--------|-----------|-----------------------------------------------------------------------------|
| 1      | Seq       | Job number from jobs.txt (submission order)                                 |
| 2      | Host      | Machine that ran the job (`:` means localhost)                              |
| 3      | Starttime | Unix epoch timestamp when the job started                                   |
| 4      | JobRuntime| Wall-clock seconds the job took to complete                                 |
| 5      | Send      | Bytes sent to the job's stdin (usually 0)                                  |
| 6      | Receive   | Bytes received from the job's stdout                                        |
| 7      | Exitval   | **0 = success, anything else = failure**                                    |
| 8      | Signal    | If the process was killed by a signal (e.g., 9 = SIGKILL); 0 = normal exit |
| 9      | Command   | The exact command that was executed                                         |

Example log line:
```
8    :     1771524402.556  4.042       0     0        1        0       python3 ppcx_mcmc_clustering.py --date 2015-06-08
```

Useful commands to analyze the log:
- See all failed jobs with full details:
  ```bash
  awk 'NR>1 && $7 != 0' run.log
  ```
-  Extract just the failed dates:
  ```bash
  awk 'NR>1 && $7 != 0' run.log | grep -oP '\-\-date \K[0-9-]+'
  ```
- Show jobs killed by a signal (e.g., OOM killer):
  ```bash
  awk 'NR>1 && $8 != 0' run.log
  ```
- Show slowest successful jobs:
  ```bash
  awk 'NR>1 && $7 == 0' run.log | sort -k4 -rn | head -10
  ```
- Show success/failure counts:
  ```bash
  awk 'NR>1 {if ($7==0) ok++; else fail++} END {print "OK:", ok, "FAILED:", fail}' run.log
  ```

- `Exitval` (column 7) is the most important: `0` means success, any other value means failure.
- The log is written in completion order, not submission order.

#### Retrying Failed Jobs

GNU Parallel can automatically retry failed jobs using the `--retries` option. For example, to retry each failed job up to 3 times:

```bash
parallel  -j 4 --bar --joblog run.log --resume-failed < jobs.txt
```

This will only retry jobs that previously failed (non-zero exit code), making it easy to recover from transient errors or missing dependencies.



### Processing Multiple Years in batch
cc
To keep the processing running on a remote server even after the client disconnects via ssh, use `nohup`:

```bash
nohup parallel -j 8 --bar --joblog run.log --resume < jobs.txt > parallel.out 2>&1 &
```

This will run the job in the background and save all output to `parallel.out`, allowing you to safely disconnect from your session.

### Tips

* **Prevent Out-of-Memory Crashes:** Use `--memfree` to prevent starting new jobs if RAM is low:
	```bash
	# Only start a new job if at least 4GB RAM is free
	parallel --memfree 4G ...
	```
* **Handle Hanging Jobs:** Kill jobs automatically if they take too long:
	```bash
	# Kill job if it takes 30 minutes
	parallel --timeout 30m ...
	```
* **Retry Failed Jobs:**
	```bash
	# Retry failed commands up to 3 times
	parallel --retries 3 ...
	```
* **Keep System Responsive:** Limit new jobs if CPU load is too high:
	```bash
	# Don't start new jobs if Load Average > 8
	parallel --load 8 ...
	```

---

## Environment Setup

### Using Conda/Mamba

```bash
conda create -n ppcx python=3.11 -y
conda activate ppcx
pip install -e ../pylamma
pip install .
```

### Using uv

```bash
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/ppcx-dic
uv sync
source $HOME/.venvs/ppcx-dic/bin/activate
```

---

## Notes

- All scripts accept config files and CLI overrides.
- For large-scale processing, always use `--dry-run` with `clusterize_batch.py` and GNU Parallel for robustness and monitoring.
- Logs for each subprocess are saved in the `logs/` directory by default.
- For GPU runs, set:
	```bash
	export XLA_PYTHON_CLIENT_PREALLOCATE=false
	export XLA_PYTHON_CLIENT_MEM_FRACTION=.45
	```

---

**For further details, see docstrings in each script.**
