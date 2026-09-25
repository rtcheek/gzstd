# GPUDirect Storage (`--gds-only`) setup guide

## Read this first: you probably want `--direct-stage`

`--gds-only` reads your input from NVMe straight into GPU memory by peer-to-peer DMA, so the
uncompressed bytes never touch host memory. It is real, and it works — but it is narrow, it needs
four things from your platform, and **most of what it buys does not come from the peer-to-peer part.**

Decomposed on one host, cold, of 3.73 host CPU-seconds saved against the ordinary reader:

| where the saving came from | share |
|---|---|
| the O_DIRECT read landing directly in the GPU staging buffer | **3.55 s (95%)** |
| peer-to-peer DMA itself | 0.19 s (5%) |

`--direct-stage` is that 95%. It needs **none** of the four requirements below — no cuFile, no
`nvidia-fs`, no resizable BAR, no particular filesystem — and it runs anywhere you have a GPU:

```bash
gzstd --direct-stage BIGFILE -o BIGFILE.zst
```

Set `--gds-only` up if you want the last 5% and your hardware qualifies. Otherwise stop here.

Also worth knowing: **what either flag buys is not throughput.** Both paths saturate the same drive.
The win is host CPU and memory bandwidth handed back to whatever else the machine is doing, so on an
idle box it measures as very nearly nothing.

## Quick check

```bash
./gzstd-gds-check.sh --path /mnt/nvme
```

`--path` should be the filesystem you will actually read from. The script runs as an ordinary user
and changes nothing. Its verdicts are deliberately graded, because the evidence available on this
platform is:

| verdict | exit | meaning |
|---|---|---|
| `NOT READY` | 1 | **decisive** — the counter did not move, the run degraded, or the read failed |
| `LIKELY READY` | 0 | the read worked, the counter moved, and no competing GDS activity was observed during the idle sample |
| `INCONCLUSIVE` | 3 | the BAR1 evidence was unavailable or could not be attributed to this run |
| usage error | 2 | bad arguments |

**There is no plain `READY`.** Attribution would need a per-process routing signal, which cuFile
cannot currently provide here (see below), so claiming it would be exactly the over-reach this tool
exists to catch. `NOT READY` is the only verdict that is certain — which is fine, because that is the
one you act on.

## The four requirements

1. **A GPU whose PCI BAR1 aperture covers its VRAM.** BAR1 is the window the drive writes through.
   This means resizable BAR. Datacenter cards generally qualify; consumer cards are frequently fixed
   at 256 MiB and **cannot**, at any batch size. Check with
   `nvidia-smi -q | grep -A3 'BAR1 Memory'` and compare against total VRAM.
2. **The `nvidia-fs` kernel module**, loaded, and **version 2.26.6 or newer** (see below).
3. **cuFile userspace** — `libcufile`, from `gds-tools`.
4. **A filesystem cuFile accepts** — ext4 or xfs on local NVMe. Not tmpfs, not overlayfs, not NFS.
   cuFile also silently requires `O_DIRECT`, so a mount that refuses it cannot work.

Anything missing is a usage error (exit 2) naming the specific cause. gzstd refuses rather than
falling back, because a silent host bounce is exactly the failure this mode exists to avoid — and
the refusal always points you at `--direct-stage`.

## Installing

`nvidia-fs-dkms` ships in NVIDIA's CUDA repository. To avoid adding a repo that could also move your
GPU driver, fetch the single package:

```bash
curl -fLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nvidia-fs-dkms_2.26.6-1_amd64.deb
sudo apt install ./nvidia-fs-dkms_2.26.6-1_amd64.deb
sudo modprobe nvidia-fs
```

Adjust the distro path and version for your system. It depends only on `dkms` and any
`nvidia-*-dkms` driver package you already have, so it does not drag in a driver upgrade.

### Version matters: use 2.26.6 or newer

**nvidia-fs before 2.26.6 fails on current kernels, while looking completely healthy.** It marks its
internal shadow-buffer memory region `VM_IO`, and the kernel's page-pinning path (`check_vma_flags()`
in `mm/gup.c`) returns `-EFAULT` for any region flagged that way. Older kernels satisfied the pin on
a fast path that never consulted those flags; newer ones do not. NVIDIA removed the flag in 2.26.6.

The symptom is nasty precisely because nothing looks broken:

- the module builds and loads cleanly, and reports a healthy version;
- `/proc/driver/nvidia-fs/stats` reads fine;
- **every BAR1 map fails**, so every transfer bounces through host memory.

If you see this in `dmesg` or `journalctl -k`, upgrade nvidia-fs:

```
nvidia-fs:nvfs_mgroup_pin_shadow_pages:397 Unable to pin shadow buffer pages 1024 ret= -14
nvidia-fs:nvfs_map:1505 Error nvfs_setup_shadow_buffer
```

### If the module does not load at boot

NVIDIA's package ships `/etc/modules-load.d/nvidia-fs.conf`, and `depmod` works out that `nvidia.ko`
must load first, so systemd normally handles this. **Some vendor-repackaged builds omit that file**,
in which case nothing ever asks for the module and GDS is silently absent after every reboot. Check:

```bash
cat /etc/modules-load.d/nvidia-fs.conf     # should contain: nvidia-fs
journalctl -b -u systemd-modules-load | grep -i nvidia
```

If the file is missing, create it with the single line `nvidia-fs`.

### The kernel P2PDMA mode leaks kernel memory on every GPU process (570 driver)

cuFile's `use_pci_p2pdma` mode needs the NVIDIA driver option `RMForceStaticBar1=1`, which maps all
of VRAM into BAR1. With it set, the 570.207 driver adds each GPU's whole BAR1 window to the kernel's
peer-to-peer pool every time a process first opens that GPU, and never removes it. Measured on an
8-GPU host by rebooting with and without it:

| | `RMForceStaticBar1=1` | without |
|---|---|---|
| CUDA startup, 8 GPUs | 6.34–6.41 s | 1.90–1.96 s |
| kernel memory leaked per process | 4 MiB per GPU (20.4 GiB after 18 days) | none |
| GDS reads (cuFile's `posix=` counter) | all peer-to-peer (`posix=0`) | all POSIX, in either cuFile mode |

The startup cost is paid by every GPU program on the machine, not only by GDS. So on that driver it
is a straight trade: GDS peer-to-peer, or fast CUDA startup and no leak. Set it only if you need
`--gds-only`; `--direct-stage` needs none of this. TUNING.md, under "Host setup: GPU startup cost",
covers checking a host and keeping a resident process as a workaround.

## Verifying it actually works

**This is the part people get wrong, so it is worth being precise: almost every signal you might
reach for is unreliable.**

| signal | why it cannot prove peer-to-peer |
|---|---|
| `cuFileBufRegister` returns success | it succeeds in compat mode too |
| the module is loaded, stats file present | says nothing about whether maps succeed |
| throughput looks good | compat mode measured 4.917 vs 4.924 GiB/s — indistinguishable |
| gzstd's aligned-transfer count | counts *eligibility*, not routing |
| `properties.use_compat_mode` | echoes configuration, not behaviour |
| the BAR1 map counter moved | buffer *registration* moves it; measured moving (ok 2 → 6) while every read took the POSIX path |

**The one signal that settles it is cuFile's own per-process counter**, and since v0.17.76 it works
with gzstd. Enable statistics in a copy of the config, point cuFile at it, and read the log:

```bash
sed -e 's/"cufile_stats"[[:space:]]*:[[:space:]]*[0-9]*/"cufile_stats": 3/' \
    -e 's/"level"[[:space:]]*:[[:space:]]*"[A-Za-z]*"/"level": "INFO"/' /etc/cufile.json > ~/cufile-stats.json
CUFILE_ENV_PATH_JSON=~/cufile-stats.json CUFILE_LOGFILE_PATH=~/cufile.log \
    gzstd --gds-only BIGFILE -o BIGFILE.zst
grep -o 'Read: .* n=[0-9]* posix=[0-9]*' ~/cufile.log
```

`posix=0` with `n` above zero means every read was peer-to-peer; `posix` equal to `n` means every read
fell back to ordinary POSIX I/O. It is per process, so another GDS user on the machine cannot fake it.
The stats dump is written only at log level INFO. (Before v0.17.76 this crashed gzstd at exit: cuFile
writes these counters from its own library destructor, which faults when the library was loaded with
`dlopen`, as gzstd loads it. gzstd now re-runs itself once with the library preloaded whenever the
active config enables statistics; `-v` says so.)

**gzstd also reads cuFile's capability report before work starts** (v0.17.76). This is what
`/usr/libexec/gds/tools/gdscheck -p` prints as `NVMe P2PDMA` and `NVMe`. For a file on ext4 or xfs,
if neither is `Supported`, `--gds-only` refuses with exit 2. This gate addresses the measured local
NVMe case; the capability bits describe the driver, not the route of an individual file read.
The same gate checks each held ext4/xfs source during `--gds-only --tar` creation.
On an 8-GPU host with the 570 driver, `NVMe P2PDMA` read `Supported` only with the NVIDIA option
`RMForceStaticBar1=1` set. Without it both lines read `Unsupported`, and cuFile's counter confirmed
every read went through POSIX, including with `use_pci_p2pdma` switched off. See the P2PDMA section
above for what that option costs.

**The best available signal is the kernel module's own BAR1 map counter — but it is only half
reliable, and it matters which half.** The counter is SYSTEM-WIDE:

- **it did not move → decisive.** Nothing was routed peer-to-peer. gzstd's own preflight refuses on
  exactly this, and only this.
- **it moved → consistent, not conclusive.** Another GDS client on the same host moves the same
  counter, so movement is only attributable to you if nothing else was using GDS at the time.
- **it could not be read → inconclusive.** Missing evidence does not establish compat-mode routing.

`gzstd-gds-check.sh` samples the counter while idle first and reports an unattributable result
rather than a false positive. For proof, use cuFile's counter above.

Watch it across a real read:

```bash
grep Bar1-map /proc/driver/nvidia-fs/stats     # note ok=N
gzstd --gds-only BIGFILE -o /tmp/out.zst
grep Bar1-map /proc/driver/nvidia-fs/stats     # ok must have INCREASED
```

`ok=0 err=N` means every map failed and nothing went peer-to-peer — that reading is definite. gzstd
performs this same negative check itself before running, plus the capability check above, and
refuses when either fails. A `--gds-only` run that completes has cleared both, but only cuFile's
`posix=` counter proves the routing.

## Tuning notes

- **`--gds-only` implies `--gpu-only`.** With no host copy there is nothing for a CPU worker to
  compress, so the split is unavailable rather than a policy choice.
- **It raises the default GPU batch to 64 frames.** The per-frame content checksum runs on the
  device, and that kernel's throughput scales with frame count; a smaller batch would make the
  checksum, not the drive, the bottleneck. An explicit `--gpu-batch` still wins.
- **It uses one GPU and one stream by default.** Each stream registers its own input buffer into
  BAR1, which is expensive (~490 ms for a 1 GiB buffer), and the peer-to-peer output path requires
  exactly one device and one stream. `--direct-stage` has neither constraint and defaults to two
  streams so its reads overlap compute.
- **Archives are identical either way.** The per-frame XXH64 content checksum is computed on the GPU
  instead of the host, so output is byte-for-byte an ordinary zstd archive and stock `zstd` reads it.

## When it stops working after a kernel update

GDS depends on kernel internals that are not a stable interface, so a kernel upgrade can disable it
with nothing gzstd can do. The failure is usually silent at the platform level and loud only at
gzstd's refusal. Order of investigation:

1. `journalctl -k | grep nvidia-fs` — the module says why.
2. `cat /sys/module/nvidia_fs/version` — the **running** module. (`modinfo nvidia_fs` reads the file
   on disk, which may not be what is loaded.)
3. `grep Bar1-map /proc/driver/nvidia-fs/stats` — `ok=0 err=N` means maps are failing.
4. Upgrade nvidia-fs before suspecting anything else. The fix is usually a newer module, not a
   kernel rollback.

And if it cannot be fixed: `--direct-stage` was always going to give you 95% of it.
