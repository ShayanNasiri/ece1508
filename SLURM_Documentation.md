# UofT CSLab SLURM Cluster Documentation

A practical guide to using SLURM on the University of Toronto CS Lab cluster. Written from hands-on experience — covers the things that actually trip you up, not just the man pages.

---

## Table of Contents

1. [Cluster Architecture](#1-cluster-architecture)
2. [Connecting to the Cluster](#2-connecting-to-the-cluster)
3. [Core Concepts](#3-core-concepts)
4. [Partitions and GPU Nodes](#4-partitions-and-gpu-nodes)
5. [Submitting Jobs](#5-submitting-jobs)
6. [Resource Requests](#6-resource-requests)
7. [Batch Scripts](#7-batch-scripts)
8. [Interactive Sessions](#8-interactive-sessions)
9. [Monitoring Jobs](#9-monitoring-jobs)
10. [Cancelling Jobs](#10-cancelling-jobs)
11. [Output and Logging](#11-output-and-logging)
12. [GPU Jobs](#12-gpu-jobs)
13. [Environment Setup](#13-environment-setup)
14. [Disk Quota and Storage](#14-disk-quota-and-storage)
15. [Time Limits](#15-time-limits)
16. [Fair-Share Scheduling and Billing](#16-fair-share-scheduling-and-billing)
17. [Common Pitfalls](#17-common-pitfalls)
18. [Useful Command Reference](#18-useful-command-reference)
19. [Batch Script Templates](#19-batch-script-templates)

---

## 1. Cluster Architecture

The CSLab cluster has three types of nodes:

| Node | Hostname(s) | Purpose |
|------|-------------|---------|
| **Login node** | `apps0` | SSH entry point from outside; editing, git, file transfers. **No SLURM commands here.** |
| **Submission nodes** | `comps0.cs`, `comps1.cs`, `comps2.cs`, `comps3.cs` | Where SLURM commands (`sbatch`, `srun`, `squeue`, `scancel`) work. You must SSH here to submit jobs. |
| **Compute nodes** | `gpunode1`–`gpunode34`, `cpunode*` | Where jobs actually run. You never SSH to these directly — SLURM allocates them. |

**Critical**: `apps0` does **not** have SLURM installed. If you try `sbatch` on `apps0` you'll get "command not found". Always SSH to `comps0.cs` (or `comps1-3.cs`) first.

---

## 2. Connecting to the Cluster

### From Your Local Machine

```bash
# Step 1: SSH to the login node
ssh <utorid>@cs.toronto.edu

# Step 2: SSH to a submission node (required for SLURM commands)
ssh comps0.cs
```

### Copying Files To/From the Cluster

From your local machine:
```bash
# Upload to cluster
scp local_file.py <utorid>@cs.toronto.edu:~/project/

# Download from cluster
scp <utorid>@cs.toronto.edu:~/project/results/output.json results/
```

### Typical Workflow

```
Local machine
  → ssh <utorid>@cs.toronto.edu       (lands on apps0)
    → ssh comps0.cs                    (now SLURM works)
      → cd ~/my_project
      → sbatch scripts/my_job.sh       (submit job)
      → squeue -u $USER                (monitor job)
```

---

## 3. Core Concepts

### Partitions
A **partition** is a logical grouping of nodes. You must specify one when submitting a job.

### Jobs
A **job** is a unit of work submitted to SLURM. Two types:
- **Batch job** (`sbatch`): runs a script non-interactively, goes into a queue
- **Interactive job** (`srun --pty`): gives you a live shell on a compute node

### Job States
| State | Meaning |
|-------|---------|
| `PD` (Pending) | Waiting for resources |
| `R` (Running) | Executing on a compute node |
| `CG` (Completing) | Finishing up |
| `CD` (Completed) | Finished successfully |
| `F` (Failed) | Exited with non-zero status |
| `TO` (Timeout) | Killed because it exceeded time limit |
| `OOM` (Out of Memory) | Killed because it exceeded memory limit |
| `CA` (Cancelled) | User or admin cancelled it |

---

## 4. Partitions and GPU Nodes

### Available Partitions

| Partition | Purpose |
|-----------|---------|
| `cpunodes` | Default partition for CPU-only work |
| `bigmemnodes` | Nodes with large memory capacities |
| `gpunodes` | All GPU-equipped nodes |

### GPU Node Inventory

All GPU nodes are in the `gpunodes` partition. Each node has a **single GPU**.

| GPU Type | VRAM | Nodes | `--gres` Flag |
|----------|------|-------|---------------|
| **RTX 4090** | 24 GB | gpunode4, gpunode5, gpunode32, gpunode33, gpunode34 | `gpu:rtx_4090:1` |
| **RTX A6000** | 48 GB | gpunode2, gpunode3 | `gpu:rtx_a6000:1` |
| **RTX A4500** | 20 GB | gpunode15–27, gpunode29, gpunode30 | `gpu:rtx_a4500:1` |
| **RTX A4000** | 16 GB | gpunode6, gpunode11 | `gpu:rtx_a4000:1` |
| **RTX A2000** | 6 GB | gpunode1, gpunode28 | `gpu:rtx_a2000:1` |
| **RTX 2080** | 8 GB | gpunode16, gpunode17 | `gpu:rtx_2080:1` |
| **RTX 2070** | 8 GB | gpunode18–23, gpunode25 | `gpu:rtx_2070:1` |
| **GTX 1080 Ti** | 11 GB | gpunode13 | `gpu:gtx_1080_ti:1` |

**Recommendation**: For deep learning workloads, target the RTX 4090 (best performance, 24 GB VRAM) or RTX A6000 (most VRAM at 48 GB). The RTX A4500 nodes are the most plentiful and easiest to get scheduled on.

### Checking GPU Availability

```bash
slurm_report -g     # CSLab custom wrapper — shows available GPU resources
sinfo --partition=gpunodes -o "%N %G %T"   # Standard SLURM alternative
```

---

## 5. Submitting Jobs

### Batch Jobs (most common)

Write a shell script with `#SBATCH` directives, then submit:

```bash
sbatch my_job.sh
```

SLURM returns a job ID:
```
Submitted batch job 12345
```

The job enters the queue and runs when resources are available. Output goes to a log file.

### One-Liner Jobs

For quick tests, pass the command directly:

```bash
sbatch --partition=gpunodes --gres=gpu:rtx_4090:1 -c 2 --mem=8G -t 0:15:00 --wrap="python my_script.py"
```

### Interactive Jobs

Get a live shell on a compute node:

```bash
srun --partition=gpunodes --gres=gpu:rtx_4090:1 -c 2 --mem=8G -t 1:00:00 --pty bash
```

You are billed for the entire session, even if you're just staring at the terminal.

---

## 6. Resource Requests

Every job must declare what resources it needs. SLURM enforces these limits — exceed them and your job gets killed.

### CPU Cores

```bash
#SBATCH -c 4          # Request 4 CPU cores (shorthand for --cpus-per-task)
```

Use `-c` (not `-n`) for single-task jobs. `-n` is for multi-task MPI jobs and means something different.

### Memory (RAM)

```bash
#SBATCH --mem=16G     # Request 16 GB of RAM for the entire job
```

- If your job exceeds this, SLURM kills it immediately with an OOM error
- CSLab default is ~4 GB per CPU core if not specified — but always specify explicitly
- When in doubt, request more than you think you need, then optimize later based on actual usage

**Common memory requirements by task type:**
| Task | Typical RAM |
|------|-------------|
| Small inference / smoke tests | 4–8G |
| Model inference with batch loading | 8–16G |
| Training with large datasets | 16–32G |
| Large data preprocessing (e.g. tokenizing 1M+ records) | 24–64G |

### GPUs

```bash
#SBATCH --gres=gpu:1              # 1 GPU, any type available in partition
#SBATCH --gres=gpu:rtx_4090:1    # 1 RTX 4090 specifically
```

See [GPU Node Inventory](#gpu-node-inventory) for all available types.

### Combining Resources

```bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 3:00:00
```

---

## 7. Batch Scripts

A batch script is a regular shell script with `#SBATCH` directives at the top. SLURM reads these directives before executing the script.

### Basic Structure

```bash
#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 3:00:00
#SBATCH --output=results/job_%j.log

# Redirect caches away from home dir (prevents quota issues)
export HF_HOME=/tmp/hf_cache_$USER
export TORCH_HOME=/tmp/torch_cache_$USER

# Activate environment
source .venv/bin/activate

# Run your code
python train.py --epochs 10 --batch_size 64
```

### Key Rules

- `#SBATCH` lines must come before any non-comment lines (after the shebang)
- `%j` in filenames gets replaced with the job ID
- The script runs from the directory where you called `sbatch`
- Environment variables set before `sbatch` are NOT inherited — set them inside the script
- Always redirect library caches to `/tmp` (see [Environment Setup](#13-environment-setup))

### Multi-Step Scripts

```bash
#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 5:00:00
#SBATCH --output=results/pipeline_%j.log

export HF_HOME=/tmp/hf_cache_$USER
source .venv/bin/activate

echo "=== Step 1: Preprocessing ==="
python preprocess.py

echo "=== Step 2: Training ==="
python train.py --epochs 10

echo "=== Step 3: Evaluation ==="
python evaluate.py
```

---

## 8. Interactive Sessions

Interactive sessions give you a shell on a compute node. Useful for debugging, testing, and development.

### Basic Interactive GPU Session

```bash
srun --partition=gpunodes --gres=gpu:rtx_4090:1 -c 2 --mem=8G -t 1:00:00 --pty bash
```

### Adding `--login` for Full Shell Setup

```bash
srun --partition=gpunodes --gres=gpu:rtx_4090:1 -c 2 --mem=8G -t 1:00:00 --pty bash --login
```

### Running a Specific Command Interactively

```bash
srun --partition=gpunodes --gres=gpu:rtx_4090:1 -c 2 --mem=8G -t 0:15:00 --pty bash -c "
    source .venv/bin/activate && python -m pytest tests/ -v
"
```

### Running a Script Interactively

```bash
srun --partition=gpunodes --gres=gpu:rtx_4090:1 -c 2 --mem=8G -t 0:30:00 --pty bash my_script.sh
```

### Warnings

- You are billed for the entire interactive session, even when idle
- CSLab has a **10-hour idle timeout** — interactive sessions idle for more than 10 hours are automatically killed
- Interactive sessions are harder to reproduce — prefer batch scripts for anything you might run twice
- GPU billing is 16x CPU rate — don't leave GPU interactive sessions open when you're not using them

---

## 9. Monitoring Jobs

### Check Your Jobs

```bash
squeue -u $USER
```

Output:
```
JOBID  PARTITION  NAME      ST  TIME     NODES  NODELIST
12345  gpunodes   my_job.sh  R  0:05:23  1      gpunode4
12346  gpunodes   train.sh   PD 0:00:00  1      (Resources)
```

- `ST` = state: `R` = running, `PD` = pending
- `(Resources)` in NODELIST means it's waiting for resources to free up

### CSLab Custom Resource Report

```bash
slurm_report        # Your active/waiting jobs and remaining job slots
slurm_report -c     # Available CPU node resources
slurm_report -g     # Available GPU node resources
slurm_report -h     # Help / all options
```

### Check All Jobs on GPU Partition

```bash
squeue --partition=gpunodes
```

### Detailed Job Info

```bash
scontrol show job 12345
```

### Tail a Running Job's Output

```bash
tail -f results/job_12345.log
```

Or to find the most recent log:
```bash
tail -f $(ls -t results/*.log | head -1)
```

### Check Past Job Details (After Completion)

```bash
sacct -j 12345 --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
```

`MaxRSS` shows peak memory usage — compare it to what you requested to right-size future jobs.

---

## 10. Cancelling Jobs

```bash
scancel 12345              # Cancel a specific job
scancel -u $USER           # Cancel ALL your jobs
scancel -u $USER -t PD     # Cancel only pending jobs
```

---

## 11. Output and Logging

### Default Behavior

Without `--output`, SLURM writes stdout and stderr to `slurm-<jobid>.out` in the submission directory.

### Custom Output Files

```bash
#SBATCH --output=results/job_%j.log      # stdout + stderr to one file
#SBATCH --error=results/job_%j.err       # stderr to separate file
```

### Special Filename Patterns

| Pattern | Expands To |
|---------|-----------|
| `%j` | Job ID |
| `%x` | Job name |
| `%N` | Node name |
| `%A` | Array job ID |
| `%a` | Array task ID |

### Tip: Create the Output Directory First

SLURM will not create directories for you. If the output directory doesn't exist, the job fails silently:

```bash
mkdir -p results
sbatch my_job.sh
```

---

## 12. GPU Jobs

### Requesting GPUs

```bash
#SBATCH --gres=gpu:1                  # 1 GPU, any type in the partition
#SBATCH --gres=gpu:rtx_4090:1        # 1 RTX 4090
#SBATCH --gres=gpu:rtx_a6000:1       # 1 RTX A6000
#SBATCH --gres=gpu:rtx_a4500:1       # 1 RTX A4500
```

Each CSLab GPU node has a single GPU, so you can request at most `gpu:<type>:1`.

### Quick GPU Smoke Test

Verify you can access a GPU before running long jobs:

```bash
srun --partition=gpunodes --gres=gpu:rtx_4090:1 -c 1 --mem=2G -t 0:05:00 nvidia-smi -L
```

### CUDA Visibility

Inside a SLURM job, `CUDA_VISIBLE_DEVICES` is automatically set to the GPU allocated to your job. You don't need to set this manually.

### CUDA Installations

CUDA toolkits are available at `/usr/local/cuda*/` on the compute nodes. If you need a specific CUDA version:
```bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

---

## 13. Environment Setup

### Python Version

CSLab runs **Python 3.13.9**. All code and dependencies must be compatible with this version.

**Notable compatibility issue**: Triton requires version >= 3.2.0 for Python 3.13 (older versions like 2.1.0 only support up to Python 3.11).

### Python Virtual Environment

Create a venv in your project directory and activate it in every job script:

```bash
# One-time setup (run on comps0.cs or apps0):
cd ~/my_project
python3 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

Then in every SLURM script:
```bash
source .venv/bin/activate
python my_script.py
```

**Always use `--no-cache-dir`** when installing packages. The pip cache fills up the home directory quota fast (~4 GB).

### Environment Variable Redirects

Many libraries cache large files to your home directory by default. On CSLab, the home directory has a strict quota, so you **must** redirect these caches to `/tmp`:

```bash
# Hugging Face models and tokenizers (can be several GB)
export HF_HOME=/tmp/hf_cache_$USER

# ir_datasets (can be 11+ GB for MS MARCO etc.)
export IR_DATASETS_HOME=/tmp/ir_datasets_$USER

# PyTorch Hub models
export TORCH_HOME=/tmp/torch_cache_$USER
```

Put these `export` lines in your SLURM scripts, **before** the `python` command.

**Important**: `/tmp` on compute nodes is local to that node and gets cleaned periodically. This means cached models will need to re-download on a different node or after cleanup. This is fine — it's just caching.

### Conda (if applicable)

```bash
# In SLURM script:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv
python my_script.py
```

---

## 14. Disk Quota and Storage

### Home Directory

CSLab home directories (`/h/<utorid>` or `/u/<utorid>`) have a **limited disk quota**. Things that fill it up fast:
- `pip` cache (`~/.cache/pip`) — can be 4+ GB
- Hugging Face model cache (`~/.cache/huggingface`) — can be 10+ GB
- `ir_datasets` cache — 11.5 GB for MS MARCO alone
- Conda packages
- Dataset downloads

### Preventing Quota Issues

```bash
# Always install without caching
pip install --no-cache-dir <package>

# Check your disk usage
du -sh ~
du -sh ~/.cache/*

# Clear caches when quota is full
pip cache purge
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface
```

### Scratch Space

CSLab has scratch space at `/cs/pools/fs4-scratch-01/`, but it is **read-only from all nodes** (login, submission, and compute nodes). **Do not** plan to use it for virtual environments, data, or output files.

For temporary large files, use `/tmp` on compute nodes (with `$USER` suffix to avoid collisions):
```bash
export MY_TEMP=/tmp/my_project_$USER
mkdir -p $MY_TEMP
```

### Summary of Storage Options

| Location | Writable? | Persistent? | Quota | Use For |
|----------|-----------|-------------|-------|---------|
| `~/` (home dir) | Yes | Yes | Limited | Code, venv, small configs |
| `/cs/pools/fs4-scratch-01/` | **No (read-only)** | Yes | N/A | Nothing (read-only) |
| `/tmp` on compute nodes | Yes | **No (cleaned regularly)** | Large | Library caches, temp data |

---

## 15. Time Limits

Every job must have a time limit. If your job exceeds it, SLURM kills it.

### Specifying Time Limits

```bash
#SBATCH -t 30              # 30 minutes
#SBATCH -t 3:00:00         # 3 hours
#SBATCH -t 1-0             # 1 day
#SBATCH -t 1-12:00:00      # 1 day and 12 hours
#SBATCH -t 5-0             # 5 days (CSLab maximum)
```

### Format Reference

| Format | Meaning |
|--------|---------|
| `MM` | Minutes |
| `HH:MM:SS` | Hours:Minutes:Seconds |
| `D-HH:MM:SS` | Days-Hours:Minutes:Seconds |
| `D-HH` | Days-Hours |

### CSLab Limits

| Limit | Value |
|-------|-------|
| Default time limit | 12 hours |
| Maximum time limit | 5 days (`-t 5-0`) |
| Default RAM per core | ~4 GB |
| Interactive idle timeout | 10 hours |

### Tips

- Request ~50% more time than you expect, but not wildly more — shorter jobs get scheduled faster
- Use `sacct -j <jobid> --format=Elapsed` after a job completes to see actual runtime for future planning

---

## 16. Fair-Share Scheduling and Billing

SLURM tracks resource usage to ensure fair scheduling across all CSLab users.

### How Billing Works

You are billed for resources **claimed** (requested), not just what you actually use:
- If you request 32G RAM but only use 8G, you're billed for 32G
- If you request a GPU for a 3-hour interactive session but only compute for 30 minutes, you're billed for 3 hours

### CSLab Billing Rates

| Resource | Rate |
|----------|------|
| CPU | 1 cpu-second per core per second |
| Memory | 1 cpu-second per 0.25 GB per second |
| GPU | **16 cpu-seconds per second** |

**GPUs cost 16x the CPU rate.** This means:
- Don't leave interactive GPU sessions open when idle
- Size GPU batch jobs carefully — don't request 5 hours if you need 30 minutes
- Prefer batch jobs over interactive sessions for reproducibility and efficiency

### Impact on Scheduling

Higher accumulated usage = lower priority when the cluster is busy. Your jobs will still run, but they may wait longer in the queue behind users with less accumulated usage.

---

## 17. Common Pitfalls

### "command not found: sbatch"
You're on `apps0` (the login node), not a submission node. SSH to `comps0.cs` first:
```bash
ssh comps0.cs
```

### Job Immediately Fails with No Output
- Check that the output directory exists (`mkdir -p results/`)
- Check `sacct -j <jobid>` to see the exit state
- Look for `slurm-<jobid>.out` in the submission directory

### OOM (Out of Memory) Kill
Your job used more RAM than requested. Solutions:
- Increase `--mem` in your next submission
- Reduce batch size or data loaded at once
- Use `sacct -j <jobid> --format=MaxRSS` to see actual peak usage and right-size

### Timeout Kill
Job exceeded its time limit. Solutions:
- Increase `-t` (max is `5-0` on CSLab)
- Add checkpointing to your code so you can resume
- Reduce the scope of work per job

### "Disk Quota Exceeded"
Home directory is full. This is the most common issue on CSLab. Solutions:
```bash
pip cache purge
rm -rf ~/.cache/pip ~/.cache/huggingface
du -sh ~ ~/.cache/*    # Find what's using space
```
Redirect library caches to `/tmp` in all SLURM scripts (see [Environment Setup](#13-environment-setup)).

### Job Stuck in Pending
```bash
squeue -u $USER    # Check the NODELIST/reason column
```
Common reasons:
- `(Resources)` — waiting for nodes to free up. Normal, just wait.
- `(Priority)` — other users have higher fair-share priority. Normal, just wait.
- `(QOSMaxGRESPerUser)` — you've hit the max GPUs allowed per user. Wait for current jobs to finish.
- `(ReqNodeNotAvail)` — requested node is down. Try a different GPU type or remove the node constraint.

**Tip**: If RTX 4090 nodes are busy, RTX A4500 nodes are the most plentiful and usually have shorter wait times.

### Virtual Environment Not Found
The venv path in your script doesn't match where you created it. Use a relative path (`source .venv/bin/activate`) and make sure you run `sbatch` from the project directory where the venv lives.

### CUDA Out of Memory (Different from SLURM OOM)
This is a **GPU memory** issue, not system RAM. SLURM won't report it as OOM — your Python process crashes with a CUDA error. Solutions:
- Reduce batch size
- Use mixed precision (`torch.cuda.amp` or `torch.bfloat16`)
- Use gradient checkpointing
- Use a GPU with more VRAM (RTX A6000 has 48 GB)

### HuggingFace / ir_datasets Fills Home Directory
These libraries cache to `~/.cache/` by default, which eats your quota. Always set environment variables:
```bash
export HF_HOME=/tmp/hf_cache_$USER
export IR_DATASETS_HOME=/tmp/ir_datasets_$USER
```

---

## 18. Useful Command Reference

### Job Submission
| Command | Purpose |
|---------|---------|
| `sbatch script.sh` | Submit a batch job |
| `srun --pty bash` | Start an interactive session |
| `srun <command>` | Run a single command on a compute node |
| `sbatch --wrap="<command>"` | Submit a one-liner batch job |

### Job Monitoring
| Command | Purpose |
|---------|---------|
| `squeue -u $USER` | Show your jobs |
| `squeue --partition=gpunodes` | Show all GPU jobs |
| `scontrol show job <id>` | Detailed info about a job |
| `sacct -j <id>` | Info about completed jobs |
| `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode` | Job stats with peak memory |
| `slurm_report` | CSLab custom: your jobs + remaining slots |
| `slurm_report -g` | CSLab custom: available GPU resources |
| `slurm_report -c` | CSLab custom: available CPU resources |

### Job Control
| Command | Purpose |
|---------|---------|
| `scancel <id>` | Cancel a job |
| `scancel -u $USER` | Cancel all your jobs |
| `scancel -u $USER -t PD` | Cancel only pending jobs |

### Cluster Info
| Command | Purpose |
|---------|---------|
| `sinfo` | Show all partitions and node states |
| `sinfo -s` | Summary of partitions |
| `sinfo --partition=gpunodes -o "%N %G"` | Show GPU types per node |
| `sinfo --partition=gpunodes -o "%l"` | Show max time limit |
| `sinfo -N -l` | Detailed per-node info |

---

## 19. Batch Script Templates

### Minimal CPU Job

```bash
#!/bin/bash
#SBATCH --partition=cpunodes
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH --output=results/cpu_job_%j.log

source .venv/bin/activate
python my_script.py
```

### Standard GPU Job (Any GPU)

```bash
#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 3:00:00
#SBATCH --output=results/gpu_job_%j.log

export HF_HOME=/tmp/hf_cache_$USER
export TORCH_HOME=/tmp/torch_cache_$USER

source .venv/bin/activate
python train.py
```

### RTX 4090 Job

```bash
#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 3:00:00
#SBATCH --output=results/gpu_job_%j.log

export HF_HOME=/tmp/hf_cache_$USER
source .venv/bin/activate
python train.py
```

### High-Memory Data Processing

```bash
#!/bin/bash
#SBATCH --partition=cpunodes
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH --output=results/data_job_%j.log

export IR_DATASETS_HOME=/tmp/ir_datasets_$USER
source .venv/bin/activate
python preprocess_data.py
```

### GPU Profiling with NVIDIA Nsight Compute

```bash
#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:30:00
#SBATCH --output=results/ncu_%j.log

source .venv/bin/activate

# Profile with Nsight Compute (ncu)
# --set full: detailed kernel analysis
# -o: output file (generates .ncu-rep file for Nsight GUI)
ncu --set full -o results/profile_output python my_microbench.py
```

### Quick Test Runner on GPU

```bash
#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:15:00
#SBATCH --output=results/tests_%j.log

source .venv/bin/activate
python -m pytest tests/ -v
```

Or as a one-liner without a script:
```bash
srun --partition=gpunodes --gres=gpu:rtx_4090:1 -c 2 --mem=8G -t 0:15:00 --pty bash -c "
    source .venv/bin/activate && python -m pytest tests/ -v
"
```

---

## Further Reading

- [SLURM Official Documentation](https://slurm.schedmd.com/documentation.html)
- [SLURM `sbatch` Reference](https://slurm.schedmd.com/sbatch.html)
- [SLURM `srun` Reference](https://slurm.schedmd.com/srun.html)
- CSLab cluster admin for questions about partition policies, quotas, or node maintenance
