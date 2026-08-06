# Releasing gzstd

**Tagging is deployment.** A `v*` tag triggers the Actions portable build and hosts
auto-install it. There is no staging step, so everything below happens *before* the tag.

This checklist exists because several defects reached the tail end of a release arc
undetected — each one in something nobody routinely exercised. Every item below is here
because it caught a real bug, not because it seemed prudent.

---

## 1. Build BOTH configurations

```bash
cmake -B build            && cmake --build build -j$(nproc)              # GPU (default)
cmake -B build-nogpu -DUSE_NVCOMP=OFF && cmake --build build-nogpu -j$(nproc)   # CPU-only
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
./gzstd-test.sh -e ./build/gzstd         # extensive, GPU build (superset of the default run)
./gzstd-test.sh ./build-nogpu/gzstd      # CPU-only build
```

Both must be **0 failures**, and the extensive run must show **no drift note**
(`EXPECTED_TESTS` matching the number that ran). The CPU-only run reports a lower total
because GPU sections skip — that is expected; `EXPECTED_TESTS` is documented as assuming a
GPU is present.

**The default `./gzstd-test.sh ./build/gzstd` was dropped from this checklist (2026-08-06)
because it is a strict subset of `-e`.** Every extensive gate in the script is
`if $EXTENSIVE; then … fi`; none exclude, so `-e` runs the default set plus the compat
sections. Measured on the 256-thread box: default 9.5 min, extensive 14 min, CPU-only
1.4 min — dropping the default cut the pre-tag suite time from ~25 to ~15.5 min with no
loss of coverage. **The CPU-only run stays**: it is a different binary (~50 `HAVE_NVCOMP`
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
or filesystem. Asymmetric mode takes a different branch below PCIe Gen4 that a Gen5 box
never executes.

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
