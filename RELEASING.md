# Releasing gzstd

**Tagging is deployment.** A `v*` tag triggers the Actions portable build and hosts
auto-install it. There is no staging step, so everything below happens *before* the tag.

This checklist exists because several defects reached the tail end of a release arc
undetected — each one in something nobody routinely exercised. Every item below is here
because it caught a real bug, not because it seemed prudent.

---

## 1. Build BOTH configurations

```bash
cmake -B build            && cmake --build build -j$(nproc)                # GPU (default)
cmake -B build-cpu -DUSE_NVCOMP=OFF && cmake --build build-cpu -j$(nproc)  # CPU-only
```

**Do not skip the CPU-only build.** It is the configuration nobody compiles, and it has
hidden two separate defects in two days:

- **v0.15.32** — it did not *compile at all*, and hadn't since v0.15.28. A helper added
  for the GPU-engagement guard read an `Options` member that lives inside
  `#ifdef HAVE_NVCOMP`. Four versions shipped broken in that config.
- **v0.15.34** — GPU flags were handled backwards: `--gpu-only` was silently swallowed
  (so a script asking for the GPU got CPU compression and **exit 0**, and two conflict
  checks could never fire), while `--pinned` died as an unknown option. Separated-value
  args like `--gpu-batch 8` also left `8` behind as a positional.

The portable release bundles nvCOMP, so CPU-only is not what ships — but driver-less
hosts are real and the configuration is supported.

## 2. Test suites

Two runs, one per build configuration:

```bash
./gzstd-test.sh -e ./build/gzstd   # extensive, GPU build (superset of the default run)
./gzstd-test.sh ./build-cpu/gzstd  # CPU-only build
```

Both must be **0 failures**, and **neither** must show a drift note. As of v0.17.32 that is
a uniform rule: the suite adjusts its expected count for what the host could not run, so a
note now means a test was genuinely added or removed rather than "this machine differs from
the pre-tag box".

**The total legitimately varies by host**, because `TOTAL_RAN` counts pass + fail and a skip
is neither:

| host | extensive | default | reachable today? |
|---|---|---|---|
| GPU + GPUDirect Storage usable (the baseline) | 599 | 468 | **yes, this server** |
| GPU, GDS unavailable | 589 | 458 | yes, the workstation |  <!-- derived: baseline - EXPECTED_NOGDS_DELTA (11); measured 456 default at the v0.17.63 baseline of 467 -->
| no GPU (the CPU-only build) | not observed (`-e` adds no CPU-only cells) | 351 | yes |

The baseline row counts what RAN, so it is the `EXPECTED_TESTS` constant (600 / 469) minus the one
cell this server cannot provoke — the trivial-park cell needs a trivially-compressed batch to reach
a GPU, and the CPU pool here drains the fixture first. Since v0.17.64 that skip goes through
`skip_host`, which subtracts itself from the expectation, so the run says "as expected on this host"
instead of raising a drift note; a note again means only what this section says it means.

**GDS was unusable on both machines for part of 2026-09-04 and is working again on this server since
that afternoon.** It broke when the server moved to kernel 6.8.0-139: `nvidia-fs` still loaded, but
every BAR1 map failed, because nvidia-fs 2.24.3 marks its shadow-buffer VMA `VM_IO` and the kernel's
`check_vma_flags()` returns `-EFAULT` for that. NVIDIA removed the flag in **nvidia-fs 2.26.6**, and
installing that (`nvidia-fs-dkms 2.26.6-1`, from NVIDIA's CUDA repo, on the same 570.x driver)
restored peer-to-peer: `Bar1-map` went from `ok=0 err=517` to `ok=21 err=0`.

So the baseline row is produced by **this server only**. The workstation's `nvidia-fs` was removed
deliberately and it cannot do peer-to-peer regardless (256 MiB BAR1), so it lands on the
GDS-unavailable row and always will. If the GDS cells ever start skipping here again, suspect the
kernel module before the test: `cat /sys/module/nvidia_fs/version` (2.26.6 or newer) and
`grep Bar1-map /proc/driver/nvidia-fs/stats` (`ok` must be non-zero after a `--gds-only` run).

The GDS row is the six `--gds-only` cells that assert a successful run; they skip when the
host cannot do peer-to-peer, which since v0.17.32 includes a host with no `nvidia-fs` module
at all. The no-GPU row skips the whole GPU section as a group, and the GDS cells live inside
it, so those two deltas must never both be applied — the script's drift check uses `elif` for
exactly that reason. Both deltas live beside `EXPECTED_TESTS` at the top of the script with
their provenance; when adding or removing tests, update the **baseline**.

**The default `./gzstd-test.sh ./build/gzstd` was dropped from this checklist (2026-08-06)
because it is a strict subset of `-e`.** Every extensive gate in the script is
`if $EXTENSIVE; then … fi`; none exclude, so `-e` runs the default set plus the compat
sections. Re-measured on the 256-thread box 2026-09-18: **default 7.0 min, extensive 8.4 min,
CPU-only 1.6 min** — the 2026-08-06 figures (9.5 / 14 / 1.4) are superseded, and the arc that
removed fixed costs is why (teardown sleeps in v0.17.55, GPU ordering in v0.17.63). The extensive
run was timed while another tenant held six of eight GPUs at 100%: this suite is dominated by
kernel I/O, not GPU time (2m25s user against 8m10s sys), so GPU contention barely moves it. **The CPU-only run stays**: it is a different binary (~50 `HAVE_NVCOMP`
conditional regions compile the other way), so `-e` cannot substitute for it at any test
count, and it is the cheapest of the three.

**Pre-tag is the one place `-e` always runs.** Day to day the default run is the normal one
and `-e` is opt-in for substantial changes — see `CLAUDE.md`. A tag is deployment, so the
wider net is warranted here regardless of how small the change looked.

Never rebuild the binary while a suite is running against it.

## 3. Round-trip / byte-identity on real data

Suites use small fixtures. Before a tag, round-trip at least one large archive per shape
that has its own code path:

- a few huge files (bandwidth-bound writers)
- many small files (metadata-bound writers)
- one compression-heavy archive (exercises the decode pool rather than inline)

For anything touching the GPU path, include a run where the GPU **actually engages** —
a short job may skip it entirely and prove nothing. `-T 1` on a large input is a reliable
way to force engagement.

## 4. A second machine

The single hardest gap to close and the easiest to rationalise away. Varying
`CUDA_VISIBLE_DEVICES` on one box exercises GPU *count* (and is worth doing — 0/1/2/8 all
have distinct paths), but it does not vary PCIe generation, VRAM size, core count, kernel,
or filesystem. The `--direct` and GPU-verify defaults take a different branch below PCIe Gen4
that a Gen5 box never executes.

Any second host helps, including a GPU-less one.

## 5. Housekeeping

- Bump `GZSTD_VERSION` — every executable build gets its own version.
- CHANGELOG entry, including what was *measured*, not just what changed.
- **Check the commit message describes what the commit actually contains.** `1a17942`
  shipped with a message advertising work that landed afterwards; the message promised a
  feature the tree did not have. Cheap to verify, confusing forever if wrong.
- `--adapt` writes `${XDG_CACHE_HOME:-~/.cache}/gzstd/profile.json`. Benchmarks and tests
  that write to `/dev/null` or tmpfs can teach it a sink rate for a device that does not
  exist. If a host behaves oddly after a release, clearing that file is the first thing
  to try.

## 6. Periodically: a full code review

Not every tag. But the last stretch of work found several defects only at the tail end,
which is the signal that incremental review has drifted. Do it one angle at a time,
risk-ordered, banking each verdict before starting the next — a single sweeping pass over
~23 000 lines produces noise rather than findings.
