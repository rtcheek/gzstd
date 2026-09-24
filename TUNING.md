# Calibrating gzstd for your machine

gzstd's defaults are chosen to be safe everywhere, not fastest anywhere. The fastest settings depend
on the machine: how many cores it has, how fast its disks are, how many GPUs it has and what it costs
to start them. gzstd can measure these and remember what it finds. This page explains how.

## The short version

```bash
gzstd --calibrate                      # once per machine: a minute or two
gzstd --adapt BIGFILE -o BIGFILE.zst   # from then on: every run, learning as it goes
```

If the machine has **two or more GPUs** and you compress with `--gpu-only`, also calibrate once on a
file like the ones you actually compress:

```bash
gzstd --calibrate /path/to/a/typical/large/file
```

Everything is stored in one file per user, `${XDG_CACHE_HOME:-~/.cache}/gzstd/profile.json`. Delete
it to start over. `--no-profile` makes a run neither read nor write it, which is what you want for
benchmarks.

## What `--adapt` does

`--adapt` watches each run while it happens and decides what limits it: reading the input, the
compression engines, or writing the output. It then tunes the parts that can change during a run:
reader threads, how much data is in flight, writer threads, GPU batch sizes, and which engine gets
work first. At the end of a clean run of 3 s or more, it saves what it measured, so the next run
starts from the answer instead of looking for it again.

- **Explicit flags always win.** If you pass `--gpu-devices 4` or `-T 16`, `--adapt` never
  overrides it.
- **The profile is per machine.** Entries are keyed by a hardware fingerprint (CPU model and core
  count, GPU names, kernel). Measured GPU values are also tied to the GPU driver version and are
  ignored after a driver change until new runs replace them.
- **Values settle over a few runs.** Each run moves a stored value halfway to what it just measured,
  so one unusual run cannot overwrite a machine's history.
- `-v` prints what it decided, as `[ADAPT]` lines.

`--adapt` is opt-in until v1.0.

## What `--calibrate` measures

`gzstd --calibrate` doesn't process any input. It measures the machine and records the results to
the profile, so that the first `--adapt` run already has them.

| command | measures |
|---|---|
| `gzstd --calibrate` | CPU and GPU compress and decompress rates, on generated data in memory. On a host with 2 or more GPUs, also what it costs to start each number of GPUs. |
| `gzstd --calibrate -o /mnt/data/NEWFILE` | also the write rate of that filesystem: it writes a scratch file there, syncs it, then deletes it. The path must not already exist. |
| `gzstd --calibrate FILE` | also the GPU compression rate at each number of GPUs, on your own data. See the next section. |
| add `--no-profile` | prints the measurements without saving them. |

Calibrate again after a hardware change, a driver update, or a move to a different kind of storage.

## How many GPUs to use for `--gpu-only` compress

Using more GPUs isn't free. Before gzstd can use any GPU, the CUDA driver starts every GPU the process
can see, one after another, and on some hosts that takes most of a second each. Past a few GPUs, the
host (its memory bandwidth and PCIe links), not the GPUs, also limits how fast data can be fed to
them. So using fewer GPUs can finish sooner, and how many is best depends on the size of the input.

Measured on a host with 256 CPU cores and 8 H100 GPUs, compressing 195 GiB from memory:

| GPUs visible to CUDA | startup cost |
|---|---|
| 1 | 1.5 s |
| 2 | 2.1 s |
| 4 | 3.6 s |
| 8 | 6.7 s |

| run | time |
|---|---|
| `--gpu-only`, all 8 GPUs | 18.7 s |
| `--gpu-only --adapt`, which settled on 2 GPUs | 13.2–14.7 s |

(On that host `--cpu-only` did the same job in 6.3–6.8 s. With that many cores, the CPU is the fast
path, and the default mode already uses it. `--gpu-only` is for hosts where CPU cores are scarce or
needed for something else.)

### How `--adapt` chooses

For each number of GPUs, the profile stores two measurements:

- **overhead**: the run's time outside GPU work: startup, the first read, the last write.
- **rate**: GiB/s while the GPUs were working.

The predicted time for an input is `overhead + size / rate`, and `--adapt` uses the number of GPUs with
the lowest prediction. The candidates are 1, 2, 4, and so on (doubling), below the number of GPUs you
have, plus all of them. This choice applies to a single-file `--gpu-only` compress. A command with
several input files uses all visible GPUs because its one CUDA startup is shared across files.
The current profile model supports up to 64 visible GPUs; larger fleets also use all visible GPUs.
If `CUDA_VISIBLE_DEVICES` contains entries gzstd cannot verify before CUDA starts (such as MIG
identifiers or a mix of numeric indices and UUIDs), it keeps that list unchanged rather than
narrowing it by a guessed count.

- **Exploring.** On an input of 8 GiB or more, a number of GPUs that has never been measured is
  tried first: all of them, then half, then a quarter, down to 1. Each exploring run is a normal run
  and gives correct output; it may just be slower than the best choice. A count becomes eligible for
  re-measurement after 20 compress runs; one stale count is selected per large run.
- **Smaller inputs never explore.** They use the best number measured so far.
- **Close results go to fewer GPUs.** If two choices are predicted within 3% of each other, `--adapt`
  uses the smaller number, which leaves the other GPUs for other work.
- **Input from stdin**, whose size gzstd can't know, uses the highest measured rate; within 3%, it
  uses fewer GPUs to avoid paying extra startup for a small rate difference.
- An ordinary `--adapt` run records a rate after at least 2 s of GPU work. Shorter runs record only
  the overhead. `--calibrate FILE` records its deliberately measured rate even when shorter.

`-v` shows the choice:

```
[ADAPT] GPU devices: 2 of 8 (predicted 13.5 s for 195.3 GiB (overhead 4.4 s, 21.5 GiB/s))
```

`--gpu-devices N` overrides it for one run.

### Calibrating the GPU count on your own data

To skip the exploring runs, give `--calibrate` a file:

```bash
gzstd --calibrate /data/typical-large-file
```

It compresses the file once for each candidate number of GPUs, and never modifies it. Each run is a
separate process, because only the first process to use a GPU pays its startup cost. It records each
number's overhead and rate, then prints the number it would choose for inputs a tenth of, the same as,
and ten times the file's size:

```
[CALIBRATE] gpu devices: --gpu-only compress of 195.3 GiB (/data/typical-large-file) at each count, one process each (each pays its own cuInit)
[CALIBRATE] gpu devices  1: overhead  3.19 s, busy  13.59 GiB/s
[CALIBRATE] gpu devices  2: overhead  4.43 s, busy  21.54 GiB/s
[CALIBRATE] gpu devices  4: overhead  5.06 s, busy  16.76 GiB/s
[CALIBRATE] gpu devices  8: overhead 11.48 s, busy  31.28 GiB/s
[CALIBRATE] gpu devices: best 19.5 GiB -> 1, 195.3 GiB -> 2, 1953.1 GiB -> 8
```

That is the 8-GPU host above. All 8 GPUs have the highest rate, but they also cost 11.5 s before
and after the work. For a 195 GiB input, 2 GPUs are predicted to finish in 13.5 s and 8 in 17.7 s,
so 2 is the choice. All 8 pay off only at about 2 TB. The whole calibration, including its other
measurements, took 1 min 37 s.

Choose the file carefully:

- **Similar to your real data.** GPU compression speed depends on the data. Data that barely
  compresses sends about four times as much back from the GPU, and was measured at 8.7 GiB/s against
  22.9 GiB/s for typical data on the same 2 GPUs.
- **Large: 60 GiB or more if you can.** The GPU batch size grows during a run, so a short run
  measures a lower rate. That's accurate for files that size but low for much larger ones.

Plain `--calibrate`, without a file, compresses generated data and records **only the overhead** for
each number of GPUs. A rate from generated data doesn't predict the rate on your data, so `--adapt`
learns the rates from your real runs instead.

### Which cards

When gzstd uses only some of the GPUs (`--gpu-devices N` below the device count, the count
`--adapt` chose, or the single card `--gds-only` and `--direct-stage` use), and you have not
set `CUDA_VISIBLE_DEVICES`, it asks NVIDIA's management library (NVML) for each card's
utilization and free VRAM, then picks the least-loaded cards. That costs about 0.4 s, and on a shared
machine it's worth it: a card that is busy with someone else's work can make a run many times slower.
In the default hybrid mode the choice is made only when the GPUs are actually started, while the CPU
is already working, so a run that ends up not using a GPU never pays it.

When it uses every GPU, it keeps CUDA's own order. Ranking all of them first costs the same 0.4 s, and in
every case measured it saved less than that, because the work spreads across all the cards anyway.
`--gpu-order=ranked` turns that ranking back on.

## Host setup: GPU startup cost

To see what GPU startup costs on your machine, compare one GPU against all of them, on a file too
small for the compression itself to matter:

```bash
head -c 256M /dev/urandom > /tmp/gz-probe.bin
time gzstd --gpu-only -q -f -k --gpu-devices=1 /tmp/gz-probe.bin -o /dev/null
time gzstd --gpu-only -q -f -k                 /tmp/gz-probe.bin -o /dev/null
rm /tmp/gz-probe.bin
```

If the second run takes seconds longer, most of that difference is startup cost, and it is paid by
every process that uses the GPUs. On the 8-GPU host above, the two runs took 1.9–2.5 s and 10.2 s.
Two more things about it were measured there:

- **Only the first process to open a GPU pays in full.** Starting all 8 GPUs took 6.97 s in a
  process that ran alone, and 0.44 s while another process already had all 8 open.
- **Persistence mode does not remove it.** `nvidia-smi -pm 1` or `nvidia-persistenced` keeps the
  driver loaded between jobs, and the cost was still there with it on.

So on a machine that runs GPU jobs back to back, a long-running process that keeps a CUDA context open
on every GPU saves most of that time for each job that follows. The costs: the context uses some
memory on each GPU (typically a few hundred MiB), and the process shows up in `nvidia-smi` as a GPU
user. gzstd doesn't provide such a process.

### If GPUDirect Storage is configured: the static BAR1 setting

On the 8-GPU host above, startup had risen from about 0.25 s per GPU to 0.75 s since an earlier
measurement. The change in between was the NVIDIA driver option that GPUDirect Storage's kernel P2PDMA
mode requires:

```
options nvidia NVreg_RegistryDwords="RMForceStaticBar1=1;..."
```

Check whether a host has it with `grep RegistryDwords /proc/driver/nvidia/params`.

With static BAR1, the driver's unified-memory module (`nvidia-uvm`, 570.207) adds the GPU's whole
BAR1 window, 128 GiB on an H100, to the kernel's peer-to-peer memory pool each time a process first
opens that GPU. When the GPU is released, it unmaps the window, but the pool entry is never removed.
Measured on that host:

- **Kernel memory leaks with every GPU process.** Each process start grew each GPU's
  `/sys/bus/pci/devices/*/p2pmem/size` by 128 GiB, and the kernel's `VmallocUsed` by 4 MiB per GPU.
  None of it comes back until a reboot. After 18 days of uptime that was 5,234 additions and 20.4 GiB.
- **Startup: probable, not yet proven.** Registering each GPU was one of the two slow driver calls
  seen with `strace`. Removing the setting and rebooting would show how much of the per-GPU startup
  cost it accounts for; that test has not been run yet.
- **HMM is not the cause.** Reloading `nvidia_uvm` with `uvm_disable_hmm=1` changed nothing (6.93 s
  against 6.91 s for 8 GPUs).

A resident process that keeps every GPU open avoids both costs, because a GPU is only registered
when its first user opens it. While one held all 8 GPUs, two further CUDA starts took 0.44 s each and
leaked nothing.

If you don't need GPUDirect Storage's peer-to-peer path, don't set static BAR1: `--direct-stage` gets
about 95% of `--gds-only`'s benefit without it (see GDS.md).

### What `--adapt` sees

Whatever the host does, `--adapt` measures the startup cost as part of each number of GPUs'
overhead. The measurement ends before process-exit teardown, so a device-count-dependent teardown
cost is absent from the prediction. If startup cost changes, a later re-measurement picks that up.

## Other tuning notes

- **Benchmarks:** use `--no-profile`, so that results don't depend on what earlier runs taught the
  profile.
- **Storage:** see `--help` for `--direct`, `--mmap` and `--direct-stage`. GDS.md covers GPUDirect
  Storage.
- **Everything `--adapt` stores** is listed under `--adapt` in `gzstd --help`.
