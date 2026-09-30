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

1. **A GPU whose PCI BAR1 aperture covers its VRAM, on NVIDIA's open kernel module with static BAR1
   enabled.** BAR1 is the window the drive writes through.
   This means resizable BAR. Datacenter cards generally qualify; consumer cards are frequently fixed
   at 256 MiB and **cannot**, at any batch size. Check with
   `nvidia-smi -q | grep -A3 'BAR1 Memory'` and compare against total VRAM.
   - **The driver must be the OPEN kernel module** (`nvidia-driver-<branch>-open`). nvidia-fs is GPL
     and cannot link against the proprietary one: it builds, then fails to load with
     `Unknown symbol nvidia_p2p_get_pages`. `modinfo -F license nvidia` must say `Dual MIT/GPL`.
   - **It must run with static BAR1:**
     `options nvidia NVreg_RegistryDwords="RMForceStaticBar1=1;RmForceDisableIomapWC=1"` in
     `/etc/modprobe.d/`, then a reboot. The first key maps all of VRAM into BAR1; the second keeps
     that mapping uncached, without which the kernel's P2PDMA setup fails with `EBUSY`.
   - **Driver branch: 595 or newer is strongly recommended.** 570 works, but with static BAR1 every
     CUDA program on the machine starts about 3.4 s slower than on 595 (8 GPUs), and each one leaks
     kernel memory (see [below](#static-bar1-cost-by-driver-branch)).
   - **Tested:** nvidia-fs 2.26.6 with driver 570.207 (kernels 6.8.0-139 and 6.8.0-142) and with
     driver 595.91.07 (kernel 6.8.0-142). After a driver change it must be rebuilt; see "After ANY
     NVIDIA driver change" below.
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

### After ANY NVIDIA driver change, rebuild nvidia-fs by hand

**nvidia-fs compiles against the driver's own `nv-p2p.h`, and DKMS will not rebuild it for you.**
When the driver package changes, DKMS builds the new driver for every kernel, but nvidia-fs is
already "installed" for those kernels, so `dkms autoinstall` skips it. The old build then meets the
new driver's GPU page tables at the first GDS read. Measured going from driver 570 to 595, whose
page-table format changed from version 1 to version 2:

```
nvidia-fs:nvfs_pin_gpu_pages:1383 Incompatible page table version 0x00020000
kernel BUG at /var/lib/dkms/nvidia-fs/2.26.6/build/nvfs-stat.c:407!
```

That is a kernel oops. The process that made the read hangs, and the module can no longer be
unloaded because the crash leaks its references. Only a reboot recovers.

**Since v0.17.83, `--gds-only` refuses this before it touches a GPU** (exit 2), naming both driver
versions. It reads the DKMS build log below and also refuses a module that was rebuilt on disk but
is still the old build in memory. Other GDS programs (`gdsio`, your own cuFile code) have no such
check. Check which driver nvidia-fs was built against:

```bash
grep -h NVIDIA_SRC_DIR /var/lib/dkms/nvidia-fs/*/$(uname -r)/*/log/make.log
```

The path must name the installed driver version (`/usr/src/nvidia-595.91.07/...`). If it names an
older one, rebuild **before** the reboot that loads the new driver, for every installed kernel.
Pass `NVIDIA_SRC_DIR` explicitly, because nvidia-fs's Makefile otherwise takes the first `nv-p2p.h`
it finds under `/usr/src/nvidia-*`, which can be the old driver's while both are installed:

```bash
# The installed driver's DKMS version, and your nvidia-fs version (dkms status -m nvidia-fs).
DRV=$(dkms status -m nvidia | sed -n 's|^nvidia/\([^,]*\),.*|\1|p' | sort -V | tail -1)
for k in $(ls /lib/modules); do
  sudo NVIDIA_SRC_DIR=/usr/src/nvidia-$DRV/nvidia dkms build   nvidia-fs/2.26.6 -k "$k" --force && \
  sudo NVIDIA_SRC_DIR=/usr/src/nvidia-$DRV/nvidia dkms install nvidia-fs/2.26.6 -k "$k" --force
done
```

If the new driver is already loaded, the old nvidia-fs module is still resident after the rebuild.
`sudo modprobe -r nvidia_fs && sudo modprobe nvidia_fs` swaps it if nothing holds it; after a crash,
only a reboot will.

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

<a id="static-bar1-cost-by-driver-branch"></a>
**NVIDIA fixed this in the 595 driver branch, and it is now measured.** From its first release, the
open kernel module adds each GPU's window to the pool once and reuses it until the driver unloads.
The 570, 575, 580 and 590 branches still add the window at every registration (checked through
570.211.01 and 580.178.04). Same 8-GPU host, same kernel, static BAR1 on in both columns:

| | driver 570.207 | driver 595.91.07 |
|---|---|---|
| CUDA startup, 8 GPUs | 6.34–6.41 s | **2.97 s** |
| CUDA startup, 1 GPU | 1.49 s | **0.88 s** |
| P2PDMA pool | +128 GiB per CUDA start | 128 GiB per GPU, created once at first use, then flat |
| kernel memory (`VmallocUsed`) | +32 MiB per CUDA start | +32 MiB once, then flat |
| GDS reads and writes (`posix=` counter) | all peer-to-peer | all peer-to-peer |

On 595 the leak is gone and most of the startup cost with it. Some remains: 2.97 s against the
1.90–1.96 s of a 570 host WITHOUT static BAR1 (595 without it was not measured). The first CUDA start
after boot also takes longer (8.7 s here), because it builds the pools.

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
5. If the **driver** changed rather than the kernel, and the log shows `Incompatible page table
   version`, nvidia-fs was not rebuilt against it. See "After ANY NVIDIA driver change" above.

And if it cannot be fixed: `--direct-stage` was always going to give you 95% of it.
