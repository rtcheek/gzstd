# gzstd Optimization Changelog

**Covers:** v0.9.50 → v0.17.36  
**Test machines:**
- **Server:** 256-core CPU, 8× NVIDIA H100 (95 GiB VRAM each), NVMe ~3 GiB/s write
- **Workstation:** 256 GiB RAM, 24-core CPU, 2× NVIDIA RTX 2080 Ti (10 GiB VRAM each), NVMe ~1.8 GiB/s write

---


## v0.17.36 — the readiness checker claimed more than its instrument could support

An independent review of v0.17.33-35 returned five findings, all of which survived verification
against the tree. Four were mechanical. The fifth was the same defect this whole release series is
about, committed by the tool built to detect it.

### The blocker: a system-wide counter used as per-process proof

`gzstd-gds-check.sh` ended with a real `--gds-only` read and treated **any** increase in the
nvidia-fs BAR1 map counter as proof that *its* transfer went peer-to-peer. That counter is box-wide.
The source already said so, in the very function v0.17.33 modified:

> The counter is not proof in the positive direction -- a neighbour's GDS traffic moves it too --
> but it is reliable in the NEGATIVE one.

`--help` already called it *"eligibility/activity, not definitive per-read routing"*, and gzstd's own
preflight only ever uses it negatively, to refuse. The checker and `GDS.md` both used it positively,
contradicting the program they document. **The instrument was fine; the label on it was not** — which
is precisely the failure mode v0.17.29-34 exist to close, arriving through the one door nobody was
watching: the diagnostic tool itself.

**The first fix was wrong too, and the review caught that as well.** Sampling the counter while idle
and reporting an unattributable result is useful corroboration, but the branch that printed "cannot
be attributed" then fell through to `exit 0`, so the verdict line still said `READY`. A fix that
contradicts itself in the same file is worse than the original, because it reads as considered.

**The verdicts are now graded to the evidence that actually exists:**

| verdict | exit | meaning |
|---|---|---|
| `NOT READY` | 1 | **decisive** — no counter movement, a degraded run, or a failed read |
| `LIKELY READY` | 0 | the read worked, the counter moved, no competing activity during the idle sample |
| `INCONCLUSIVE` | 3 | the evidence was unavailable, or moved while another GDS client was active |

**There is deliberately no plain `READY`.** Attribution needs a per-process routing signal, and
cuFile's own counters cannot supply one here — their teardown crashes when the library is `dlopen`'d,
which is how gzstd must load it. Claiming `READY` would be the same over-reach. `NOT READY` remains
the only certain verdict, which is acceptable because it is the one a user acts on.

`INCONCLUSIVE` was mutation-tested against a **real** competing GDS client — a background loop of
`--gds-only` runs, not a stub — and returns exit 3; the same build returns `LIKELY READY` on an idle
host.

### The other four

- **The checker could clobber files in a shared `--path`.** It created a `mktemp` file and then used
  predictable `$tmp.zst` / `$tmp.err` beside it, either of which another user could pre-empt with a
  symlink; the stderr redirect would truncate the target and `-f` would overwrite it. It now works
  inside a mode-0700 directory, validated for ownership, mode and emptiness *after* `chdir`, using
  fixed names relative to that pinned cwd, and never deletes recursively.
- **An exit-0 CPU rebuild could be reported as success.** A GDS failure mid-run discards the output
  and rebuilds CPU-only at exit 0, and the counter may have moved during the failed attempt. The
  checker now rejects degradation and rebuild diagnostics before it considers the counter at all.
- **A positive but short probe read was accepted.** The v0.17.33 preflight refused only on a negative
  return. Since the real staged path requires every `cuFileRead` to return its complete frame, any
  non-exact result on a file large enough to serve the probe is decisive, and now refuses.
- **A bare `--path` or `--gzstd` looped forever.** `shift 2` fails when the value is missing, and
  without `set -e` the loop never terminated. Reproduced directly, then fixed.

### And the review's own patch had to be run, not just read

The hardening above left a `.gdscheck.XXXXXX` directory behind in the user's `--path` on every
successful run: its cleanup removed the three files it created, but **libcufile writes a `cufile.log`
into the working directory whenever it initialises**, so `rmdir` failed. Replaced with a depth-1
non-directory sweep, which stays as non-recursive as the named form while catching files neither side
enumerated. Only running it surfaced this.

Verified: the extensive suite passes **557/557 with zero failures and zero skips** — the documented
baseline, and the first run in which the `--gds-only` cells execute rather than skip, since GDS only
became functional on the development host earlier the same day. Both build configurations compile.

## v0.17.35 — the setup knowledge for --gds-only existed only in one person's head

`--gds-only` has always documented its four requirements in `--help`, and always refused with a
specific cause when one was missing. What it never told anyone was **how to get from "refused" to
"working"** — and a full day of this project's own time proved that path is not obvious even to
someone who wrote the flag.

Two additions, plus a pointer to them from the `--gds-only` help:

**`GDS.md`** — the setup guide. It opens by talking most readers out of the feature: of 3.73 host
CPU-seconds saved against the ordinary reader, **3.55 came from the O_DIRECT read landing in the GPU
staging buffer and 0.19 from peer-to-peer DMA itself**, and `--direct-stage` is that 95% with none of
the four requirements. It also states plainly that neither flag buys throughput — both saturate the
same drive; the win is host CPU handed back to whatever else is running.

Beyond the requirements and installation, it documents three things that are not written down
anywhere public and each cost real time here:

- **The nvidia-fs version gate.** Before 2.26.6 the module marks its shadow-buffer region `VM_IO`,
  and the kernel's `check_vma_flags()` returns `-EFAULT` for that, so the buffer cannot be pinned.
  Older kernels satisfied the pin on a fast path that never consulted VMA flags; current ones do not.
  **The module builds, loads, reports a healthy version, and fails every single transfer** — the
  guide carries the exact `dmesg` signature to recognise it by.
- **Vendor packages that omit `/etc/modules-load.d/nvidia-fs.conf`.** NVIDIA's own package ships it
  and `depmod` resolves the driver ordering unaided, so this normally needs no thought — but where
  that file is missing, nothing ever asks for the module and GDS is silently absent after every boot.
- **A table of five signals that cannot prove peer-to-peer** — registration success, module loaded,
  throughput, the aligned-transfer count, `use_compat_mode` — against the one that can.

**`gzstd-gds-check.sh`** — a read-only readiness check, running as an ordinary user, changing
nothing. It walks the requirements (BAR1 versus VRAM per GPU, module presence and version, cuFile,
filesystem and `O_DIRECT`) and then **settles the question with a real read, requiring the nvidia-fs
BAR1 map counter to move.** That ordering is the point: every check above the read exists to explain
a failure, never to certify success, because each one of them can be satisfied on a host where
nothing is routed peer-to-peer. When gzstd refuses, the script relays gzstd's own message rather than
maintaining a second opinion about the cause.

Its failure paths were mutation-tested rather than assumed: a tmpfs `--path`, a CPU-only binary and a
forced compat verdict each produce `NOT READY` naming the right reason, and the healthy host reports
`READY` with the counter movement quoted.

Documentation and a new script; no behaviour change beyond the added `--help` text.

## v0.17.34 — the check that guarded --direct-stage's whole claim could pass without looking

`--direct-stage` exists to be the portable path: no cuFile, so none of GPUDirect Storage's four
gates. That property was silently false once already — an independent review caught the staging slab
being registered through cuFile, which SUCCEEDS on a host with the gates open and fails on exactly
the hosts the flag exists to serve. Byte-identity cannot see it. The loaded-library set can, and the
pre-tag tool checked it like this:

```
while kill -0 $mpid; do
  grep -qa libcufile /proc/$mpid/maps && { mapped=yes; break; }
  sleep 0.25
done
[[ $mapped == no ]] && ok "--direct-stage never maps libcufile"
```

**That reports a PASS when it has observed nothing.** A run that finishes between two samples, a
mapping made and dropped inside one 250 ms interval, a `/proc` read that fails — every one of them
reaches the same `mapped=no` as a genuinely clean run. It cannot distinguish *absent* from *not
looked at*, which is the same defect class as this release series' other three: registration success
read as routing evidence, an absent module read as unknown, a failed probe read read as silence.

**`GZSTD_DEBUG_DSTAGE_GATE` replaces the poll with a rendezvous.** Set to a FIFO path, `--direct-stage`
blocks after staging completes and before teardown, so the test reads `/proc/<pid>/maps` at a moment
it chose. There is no timing in the handshake: `open(O_RDONLY)` returns exactly when the test opens
the write end ("staging is done, I am holding still"), and `read()` returns exactly when the test
writes ("I have read the maps, proceed"). Unset — every non-test run — is no syscalls and no gate.
Same construction and the same reasoning as `GZSTD_DEBUG_RM_GATE` (v0.15.9x), which replaced two
poll-and-hope tests for the same reason.

The check around it was rebuilt to match, in `tool_pretag_validate.sh`:

- **Never reaching the gate is now a FAILURE**, not a pass. Mutation-tested by pointing the script at
  a binary that does not honour the hook: it reports `--direct-stage libcufile check DID NOT RUN`,
  where the old poll would have reported a clean pass.
- **The control lives inside the tool.** `LD_PRELOAD=libcufile.so` forces the library in and the SAME
  detector must see it. Without that, "clean" is indistinguishable from a grep that can never match —
  a wrong pattern, an unreadable maps file, the wrong pid. Verified both ways here: clean when
  unforced, `mapped` when forced.

### Also: the GDS-usable test baseline is no longer reachable, and the docs now say so

`RELEASING.md`'s per-host table carried 557/417 as "the baseline" without recording that no machine
can currently produce it. Neither can: the workstation's `nvidia-fs` was removed deliberately, and
the server moved to kernel 6.8.0-139, where `nvidia-fs` still loads but the shadow-buffer pin returns
`-EFAULT` — the decision was to stay there and use `--direct-stage` rather than pin the kernel back
for a peer-to-peer path worth ~5%. The table now marks which rows are reachable and states the two
consequences: the baseline remains the number to edit when tests are added or removed (the deltas are
subtracted from it), and **`--gds-only` has no live coverage anywhere** — only its refusal contract
still runs, so changes to that code are unverified by test and want a hand read.

`tool_pretag_validate.sh` gained the matching gate: a one-shot host probe mirroring the suite's, so
its `--gds-only` sections SKIP rather than report failures for a correct refusal.
`PRETAG_FORCE_GDS_STATUS` exercises both branches from one host.

## v0.17.33 — the second stream is the double buffer, and a probe read that failed was read as silence

### `--direct-stage` stopped serialising read against compute

v0.17.31 documented, as a permanent trade, that this path "costs wall clock on a slower link": a
batch's staged reads are joined before that batch is launched, so it paid **read + compute** where
the ordinary reader pipelines a pool ahead of its workers and pays **max(read, compute)**. The
ROADMAP proposed double-buffering the staging slots to fix it.

**No new machinery was needed. A second StreamCtx already IS that double buffer** — it owns its own
device slab and its own pinned staging slots — and the only thing preventing one was a line that
forced every staged path to a single stream. **The justification for that line was never about
`--direct-stage`.** It cited two costs, both of which belong to `--gds-only`: per-stream BAR1
registration (~490 ms for a 1 GiB slab), which `--direct-stage` never pays because it makes no
cuFile call at all, and a third clause — *"this path is disk-bound at ~4.9 GiB/s, so the second
stream buys no overlap the drive can use"* — which is the same one-box generalisation already
corrected at the `--gpu-devices` default in v0.17.31 and the commit after it. It is false on both
measured hosts: 1.93 GiB/s of a 4.9 GiB/s drive here, and 1.04 GiB/s against a measured 2.0 GB/s
O_DIRECT ceiling on a PCIe Gen3 host whose writer reported `upstream-bound ... starved 96.0%`.

`--gds-only` keeps its single stream, and now for stated reasons: the BAR1 cost is real, and its
peer-to-peer output packer demotes to the ordinary writer unless the shape is exactly one device and
one stream.

Measured, 16 GiB cold, one device on every arm, three interleaved runs, flush inside the timed
region. Every range non-overlapping:

| arm | wall (s) | host CPU (s) |
|---|---|---|
| `--direct-stage --gpu-streams 1` (old default) | 8.79-8.92 | 4.89-5.86 |
| `--direct-stage` (two streams, new default) | **7.43-7.48** | 7.69-7.84 |
| `--gpu-only` (ordinary reader) | 7.56-7.72 | 13.70-13.92 |

**16% off the wall clock, and the flag now lands at or below the ordinary reader instead of behind
it** — the "do not expect it to go faster" caveat is gone. The second stream costs some of the CPU
win (43% below the ordinary reader rather than 62%), which is the trade; `--gpu-streams 1` buys it
back. Bringup sizes each stream against live free VRAM and auto-decrements the stream count if the
second will not fit, so a tight card degrades to exactly the old behaviour rather than failing.

Verified beyond wall clock, because byte-identity alone cannot check this path — a failed staged
read silently rebuilds on the CPU at exit 0: the `[DSTAGE]` instrument reports `256 frames,
4294967296 bytes`, the whole input, and both `[GPU0/S0]` and `[GPU0/S1]` appear at `-vv`.

### The preflight refused an absent module and ran on a broken one

A kernel move to **6.8.0-139** on the development server — past the 6.8.0-134/-138 nvidia-fs
regression this project already had on record — produced the exact shape the v0.17.29-32 gate exists
to kill, through a third door. nvidia-fs was still loaded, so v0.17.32's module check passed. Every
`cuFileRead` then returned `-1`.

The gate's verdict variable starts at "fine" and is only ever given a real value **when the probe
read returns its bytes**:

```
bool moved = true;                              // assume fine unless proven otherwise
if (pf.open(pfd) && pb.reg(pbuf, PB)) {
  if (pf.read_dev(pbuf, PB, 0, 0) == (ssize_t)PB) {   // -1 on this host
```

So the whole check was skipped and `--gds-only` ran: it did all its work through a path that could
not work, hit the failure on the first real read, discarded the output, rebuilt CPU-only, printed
the post-hoc *"did not run entirely on the peer-to-peer-eligible path"* warning — **and then wedged,
requiring a kill after 130 s.** The correct archive it produced was never the problem; running at
all was.

**"ABSENT is not UNKNOWN" was v0.17.32's lesson, and FAILED is not UNKNOWN either.** A read that
cannot be issued is not missing evidence, it is the answer. The fail-open rule stays right for what
it was written for — an unreadable counter, an unseekable input — and a **short** read still fails
open, because the input may simply be smaller than the 4096-byte probe, which is the accident that
hid a suite cell in v0.17.32. Only a negative return on a file large enough to have served the read
is decisive. `GZSTD_DEBUG_GDS_FAIL_PROBE_READ=1` forces the branch so it can be seen to fire on a
host whose cuFile works.

On the affected host `--gds-only` now exits 2 in under a second with a message naming the
kernel/nvidia-fs mismatch and pointing at `--direct-stage`, where it previously ran for 130 s and
hung. This also unhangs the suite's own `--gds-only` host probe.

### Two rationales that described one box

Both comment-only; the behaviour they justify is unchanged and still correct.

- The staged **batch cap of 64** was justified partly by the same "disk-bound at ~4.9 GiB/s" claim.
  Its real reason survives alone: the device XXH64 kernel's throughput scales at ~0.28 GiB/s per
  frame, so 64 frames sustain 17.8 GiB/s, past any drive this path has run against. Still a
  machine-tuned constant, and still the open general-goals item recorded for `--gds-only`.
- `ROADMAP.md`: the `--gds-only` refuse-vs-demote question is **settled as refuse**. The flag is never reached by accident — the user typed it, and it names one specific
  hardware and software path, so a silent demote does the opposite of what was asked. Where it
  refuses, the host does not meet the requirements, and the refusal names `--direct-stage`, which
  needs none of the four gates. The rejected alternatives are recorded in `ROADMAP.md` so the
  question is not reopened.

## v0.17.32 — the refusal only worked on hosts that could already be diagnosed

Removing `nvidia-fs` from a host that can never do peer-to-peer anyway — the reasonable
housekeeping after concluding GDS will not work there — showed that v0.17.30's refusal had a
precondition nobody had stated: **it only fires on a host that still has the module installed.**

The gate asks "did `cuFileBufRegister` succeed while the nvidia-fs BAR1 counter stayed still?".
With the module gone there is no counter to read, `gz_nvfs_bar1_read()` returns false, and the whole
block is skipped. `--gds-only` then ran to completion at **exit 0**, emitting a WARNING after the
fact that it "did not run entirely on the peer-to-peer-eligible path". That is the silent-success
shape the counter gate was added to kill, reached by a different door.

| host state | before | after |
|---|---|---|
| compat mode, nvidia-fs present | refuses, exit 2 | unchanged |
| **nvidia-fs absent** | **runs, exit 0, warns afterwards** | **refuses, exit 2** |

**ABSENT IS NOT UNKNOWN.** The block's own "fail open on anything unknown" rule is right for a
counter that merely cannot be read; it is wrong for a module that is not loaded, which is a positive
determination that no transfer can reach the GPU directly. A `/sys/module` check now settles that
before the counter probe runs, and the refusal names the missing module and how to install it.

### The control could not fire on the hosts it describes

`GZSTD_DEBUG_GDS_FORCE_COMPAT` exists so the refusal can be tested on hardware that cannot reproduce
it naturally. It lived **inside** the counter block, behind the same unreadable-counter guard — so on
a host with no nvidia-fs the switch that exists to make the gate testable was itself unreachable, and
the suite cell asserting the refusal failed there. It is now evaluated before any host gate. All
three refusals share one code path, so their wording cannot drift apart, and each still names
`--direct-stage`.

### And the suite probe trusted a zero exit

v0.17.31's host probe classified `rc == 0` as `engaged`. On this host that was exit 0 from a run
whose per-batch registrations were failing (`err 5016`) and whose BAR1 map never moved — so the four
`--gds-only` cells were handed a path that was not doing peer-to-peer. The probe now requires the
**absence of failure evidence**, not merely a zero exit, and is mutation-tested against five stub
shapes: a clean exit stays `engaged`, a demote message stays `demoted` (so hosts on the old contract
are still fully tested), and `cuFileBufRegister failed`, `Bar1-map never moved` and the
did-not-run-entirely warning each classify `refused`.

### A fifth cell, found by running the suite rather than by reasoning about it

`--gds-only -d accepts an archive of only empty frames` asserts exit 0 on a valid archive — the same
old-contract shape as the four gated in v0.17.31 — but it had **passed** on the compat-mode host and
so was never identified as one of them. It passed by accident: three empty frames is far under 4 KiB,
so the old counter probe's 4096-byte `read_dev` never completed and that gate failed open on the
small input. Refusing on an absent module regardless of input size removes the accident and exposed
the cell. Now gated like the rest.

**The general point:** a cell that passes is not evidence it does not encode a stale contract — it
may be passing for a reason unrelated to what it asserts. The four were identified because they
failed; this one needed the suite to be run.

### EXPECTED_TESTS is now host-aware, everywhere

A single constant could not describe every machine, because `TOTAL_RAN` counts pass + fail and a
skip is neither. The CPU-only run therefore printed a drift note on every single invocation — one
that was expected and correct, which is the fastest way to train a reader to ignore the one signal
that is supposed to mean *a test was added or removed*.

The check now subtracts documented, measured deltas for what the host could not run:

| host | extensive | default |
|---|---|---|
| GPU + GDS usable (baseline) | 557 | 417 |
| GPU, GDS unavailable | 552 | 412 |
| no GPU (CPU-only build) | not yet observed | 336 |

`-81` is the whole GPU section skipped as a group; `-5` is the `--gds-only` cells that assert a
successful run. **The GDS cells live inside the GPU section, so the two deltas must never both
apply** — the check uses `elif` for that reason. A host that needed an adjustment now says so
(`552 ran, as expected on this host (baseline 557, -5 GDS unavailable: refused)`) instead of
printing a note that looks like a defect. Verified by driving the check with stubbed capabilities
across all four host shapes plus a +1 and a -1 mutation, which are still reported.

`RELEASING.md` now states the rule uniformly: **neither** suite may show a drift note, on any host.

*Measured on a 24-core / 2x consumer-GPU host after `gds-guard-deploy.sh purge`:* `--gds-only`
compress exit 2 with the module named, forced-compat exit 2 naming `--direct-stage`, probe back to
`refused`, CPU-only suite **336 passed / 0 failed**, extensive **552/1** before the fifth cell was
gated and its raw command verified at exit 2 → SKIP afterwards.

---

## v0.17.31 — the fit test that fit, and five tests that outlived their contract

Both findings come from the second machine, and neither is visible on the development box.
A 24-core host with 2x RTX 2080 Ti (11 GiB, 256 MiB BAR1) ran the release for the first time.

### GPU decompress was silently dead on 11 GiB cards

`-d --gpu-only` and `-t --gpu-only` never touched the GPU there. At the default `batch=256` the
bringup pre-allocated its buffers, succeeded, and then died at the first device-checksum launch:

    [GPU0/S0] pre-alloc batch=256 comp=16MiB decomp=16MiB
    [GPU0] could not allocate VRAM reserve (non-fatal)
    [GPU0] fault: gzx_xxh64 launch (decompress verify): out of memory
    WARNING: all GPUs failed; finishing decompression on CPU (22 threads).

Output stayed correct and the exit code stayed 0, so the only symptom was the feature quietly not
happening — reported once, at `-vv`.

**The guard for this case existed and could not fire.** The halving loop's own comment names
"10 GiB consumer GPUs"; it only reacts to a `cudaMalloc` that *fails*, and every `cudaMalloc` here
succeeded. 256 frames x (16 MiB + 16 MiB) is ~8 GiB, which fits in 11 GiB. What did not fit was
everything allocated *after* the fit test: the VRAM reserve (a half-batch, i.e. half again as much
as the buffers), nvCOMP's temp workspace, and the verify kernel's scratch. The temp estimate fed
into the test was `per_stream_cap * 1024` — 256 KiB at batch 256 — which models none of them.

*Measured, 10817 MiB free, 16 MiB chunks:* batch 192 runs; batch 224 allocates 10752 MiB of
buffers-plus-reserve, leaves 65 MiB, and faults. The failure point tracks `batch * 48 MiB`, not the
`batch * 32 MiB` the fit test checked.

The compress bringup already caps itself at 80% of free VRAM via `cudaMemGetInfo`. Decompress now
does the same, with the reserve added to the per-slot cost. On the 11 GiB card that yields 176-177
and the GPU path runs; on a 95 GiB datacenter card it yields 1621, so it never binds and nothing
about the development box changes.

This also un-blocked the v0.17.15 multi-GPU rescue test, which had been reporting
`fault never fired: NOT TESTED` — its injection point sits in the GPU delivery loop, which was
never being entered.

### Five suite cells still asserted the pre-v0.17.29 contract

Until v0.17.28 a host that could not do peer-to-peer **demoted**: ordinary reader, exit 0. v0.17.30
made the same host **refuse** with exit 2. Cells written against the old contract — four asserting
exit 0 on a valid archive, one grepping stderr for the demote message — therefore failed on every
host without working GDS, which per `--direct-stage`'s own help is nearly everywhere. A sixth used
an unforced run as a negative control, which is invalid on a host that genuinely *is* compat mode.

That made `RELEASING.md`'s "both suites at 0 failures" unreachable off the development box.

The suite now probes the host once (`engaged` / `demoted` / `refused` / `nogds`) and those cells
skip rather than fail; `demoted` still counts as testable, so hosts on the old behaviour are
unaffected. The compat-mode cell keeps checking the forced arm and the `--direct-stage` pointer
everywhere, and only its negative control is gated. Build capability is read from `-V`, not from
`--help`, because a `USE_NVCOMP=OFF` binary documents `--gds-only` and rejects it at runtime.

### And the reserve now earns its VRAM on both paths

The reserve was allocated, held, and — outside one decompress site — never used. Three changes
make its comment true rather than aspirational.

**Bringup (decompress).** A reserve that will not allocate is the card saying it is full, one line
before the checksum kernel finds out the expensive way. It used to be logged `(non-fatal)` and
dropped. It now halves the batch, re-allocating every stream's buffers at the smaller size, and
repeats down to a floor of 1 before giving up and running without a reserve. There is no retry
counter: each pass strictly halves, so the floor already bounds the loop — the first version capped
at 10, which is invisible at 176 (8 halvings) and wrong at 1621, where reaching 1 takes 11.

**Mid-run (decompress).** Both per-batch allocation sites — the batch buffers and nvCOMP's temp
growth — used to throw, dropping the whole device to the CPU rescue over what may be a transient
shortfall. They now surrender the reserve and retry; failing that, they halve the pop and hand the
batch back to the queue (the frames are intact — inputs are not released until delivery) so the next
pop re-enters smaller. Only a single frame that will not fit is fatal.

**Mid-run (compress).** Compress allocates *nothing* after bringup: buffers and temp are both sized
once from fixed upper bounds, which is why its reserve had no consumer at all and its comment
described a recovery that did not exist. The one way a VRAM race can reach that path is a **launch**
that fails for want of memory — the same shape as the `gzx_xxh64` failure above. That status returns
on the submitting thread, and the batch is fully re-runnable there: `release_input()` frees only the
host copy, the bytes the launch reads are in `d_in_base`, and a second launch overwrites
`d_out_base`. So the launch now surrenders the reserve and retries once before the throw aborts the
output and rebuilds on the CPU.

*Verified by injection, since none of these can be reached on a healthy card:*
`GZSTD_DEBUG_FAIL_VRAM_RESERVE`, `GZSTD_DEBUG_FAIL_DECOMP_ALLOC` and
`GZSTD_DEBUG_FAIL_COMPRESS_LAUNCH` force each one. Measured: one injected decompress failure
recovers on the surrender alone; four recover with the surrender plus three shrinks and **no CPU
fallback**; permanent pressure walks to the floor and then falls back, exit 0 and correct output at
every step. One injected compress launch failure retries and completes on the GPU; two fall back to
the CPU rebuild. Every arm round-trips byte-identical.

`EXPECTED_TESTS` for the extensive run moves 548 -> 557. Skips are not counted, so a host where the
five GDS cells skip legitimately reports 552 — the same host-dependence already documented for the
CPU-only run.

---

## v0.17.30 — the seek table is a claim, and the preflight believed it

Seventh review pass, and the finding is in v0.17.29's own fix. That pass made the `--gds-only`
output preflight check **every** frame instead of only frame 0 — but it checked them against the
seek table, and never asked whether the table describes the file it is attached to.

A table is metadata the archive carries. It parses cleanly as long as its prefix sums are
self-consistent and the stream starts with zstd magic; nothing forces it to agree with the frames.
So the preflight armed peer-to-peer output on an unverified claim, the workers froze the output
topology when they started — **before** the staged producer read its first frame header — and the
producer's own "table disagrees with the file" demote arrived far too late to change anything.

Three forged tables, each on an archive `--cpu-only` decodes byte-identically:

| the table lies about | what the user got |
|---|---|
| frame boundaries (shifted one byte) | exit 4, `frame 2 starts at offset 5097` |
| decompressed sizes, frames checksummed | exit 4, `content-checksum mismatch at frame 0` |
| decompressed sizes, no content checksum | exit 2, `the GPU path failed` |

Nothing was silently corrupted — the checksum and nvCOMP's own size check caught the payload — but
all three refused a perfectly good archive, and the middle one blamed the GPU for a frame that had
decoded correctly.

### And a fourth, found by the review round that checked this fix

Verifying where a frame *starts* and what it *decodes to* still leaves the **compressed extent**
unverified. A table entry can begin at honest frame magic, declare the size that frame really
decodes to, and still span **two** concatenated frames — the prefix sums keep tiling the file, so
nothing notices that a frame has left the table.

This one did not fail loudly. Three checksumless frames whose table merged the first two:

| | exit | output |
|---|---|---|
| `--cpu-only` | 0 | 8209 bytes, correct |
| `--gds-only -d` | 0 | **4113 bytes — a whole frame silently gone** |
| `-t --gds-only` | 0 | reports the archive clean |

Silent data loss at exit 0, and `-t` vouching for it. A content checksum happens to catch the
merged case — the trailer read lands on the wrong frame's hash — which is exactly why reproducing
it needs `zstd --no-check`.

`gds_frame_extent_matches()` now walks the frame's block headers at both trust sites. Each block
header is 3 bytes and names the size of the block after it, so the walk **skips every payload by
offset** and never reads compressed data; reaching `Last_Block` exactly at the claimed extent
proves the entry describes one whole frame and nothing more.

*Cost, measured on the 65 GiB archive:* `-t` 22.34 → 23.38 s, `-d` 60.10 → 67.25 s. The `-d`
figure is doubled because the walk runs in the preflight and again in the producer. User CPU is
essentially unchanged (6.55 → 6.73 s), so what `--gds-only` is actually for — spending a fraction
of a core instead of fifteen — is untouched.

### A fifth: the table's own checksums were parsed and thrown away

The zstd-seekable format lets a table carry a per-frame content checksum. The parser read the
`Checksum_Flag` **only to size the entry stride** and skipped the checksum itself.

For an archive whose frames were written without their own checksum — `zstd --no-check` — that
table entry is the *only* byte-integrity evidence in the file. One flipped payload byte:

| backend | `-d` | `-t` |
|---|---|---|
| `--cpu-only` | exit 0, **wrong bytes** | exit 0, clean |
| `--gpu-only` | exit 0, **wrong bytes** | exit 0, clean |
| `--gds-only` | exit 0, **wrong bytes** | exit 0, clean |

Not a GDS defect at all — every backend, and latent in every released version. The checksums are
now kept and used wherever a frame carries none of its own: the CPU worker hashes the decoded
bytes, the GPU worker's device verifier is handed the table's value, and the staged producer puts
it on the Task. Only that case, deliberately — a frame with its own checksum was already verified
by zstd, and re-checking it against possibly-stale table metadata would fail sound archives to
close a gap they do not have. The map is keyed by **decompressed offset**, not sequence number,
because three different producers build these tasks and each numbers frames as it likes.

### The checksum fix missed every streaming decoder

The map was consulted only where a whole frame was decoded at once. Three routes decode a frame
in pieces instead — a frame with no declared content size, a single frame between 64 and 256 MiB,
and the unknown-size fallback tail — and all three sailed past it. Reproduced: a 3 MB frame piped
through `zstd --no-check` (so it declares no size and routes to the streaming decoder), one payload
byte flipped, **exit 0 and wrong bytes**, with the table's checksum sitting unused.

All three now roll the hash as the bytes go past and settle it at the frame boundary.
`decompress_from_buffer` also takes the output offset it starts at, since it is the *tail* route
for an archive whose leading frames another decoder already wrote, and the map is keyed by offset.
A base that is wrong simply misses the map and leaves those frames unverified — the behaviour
before any of this existed — rather than turning into a false mismatch on a sound archive.

### Three defects a second machine found that this one structurally cannot

A 2× RTX 2080 Ti box — 256 MiB BAR1, no resizable BAR — reports `Bar1-map n=7 ok=0 err=7` and
`Ops Read=0`. `--gds-only` has **never** been peer-to-peer there, at any batch size.

- **Registration success is not routing evidence.** The preflight probed `cuFileBufRegister` and
  it returns success in compat mode too, so the flag was accepted on a host where it can never do
  what it asks — run completes, exit 0, not one byte peer-to-peer. The preflight now issues a real
  `cuFileRead` and requires the nvidia-fs BAR1 counter to **move**. That counter is not proof in
  the positive direction (a neighbour's traffic moves it too) but it is reliable in the negative
  one, which is the direction used. It fails open on anything unknown — no readable counter, no
  seekable input — because none of those are evidence of compat mode.
- **A registration failure was reported as an allocation failure.** At any batch ≥ 16 that card
  printed `failed to allocate device buffers` with 10.8 GiB of VRAM free. BAR1 is the PCIe
  aperture, not VRAM, and the message sent diagnosis in the wrong direction. The two failures are
  now distinguished and the message names the aperture and points at `--direct-stage`.
- **A comment claimed a retry the code does not do.** It said an oversized batch "already retries
  at smaller sizes", so a 256 MiB-BAR1 card would "still run". It does not retry: the device is
  marked dead and the run demotes. At defaults that card does **zero GPU work** and exits 0.

*Deliberately not built: a BAR1-fitting batch autotuner.* On that card the hand-picked batch that
did proceed was still in compat mode — `ok=0`, `Ops Read=0` — so sizing the batch to fit the
aperture buys no peer-to-peer transfer at all. It would only suppress the demote and leave every
byte bouncing with no warning: a quiet lie in place of a loud failure, on precisely the hosts that
motivate it. `--direct-stage` is the portable path and needs none of GDS's gates.

`GZSTD_DEBUG_GDS_FORCE_COMPAT` forces the compat verdict, because a gate that has never been seen
to fire is not known to be a gate and the host that fires it is not the host this was written on.

### The batch was sized from our chunk size, not the archive's frames

`opt.chunk_mib` is the size gzstd would *write*; a foreign archive's frames can be larger, and the
slabs are allocated from the largest frame in the batch. Deriving the batch from `chunk_mib`
therefore registered up to four times the intended mapping on a 64 MiB-frame archive — the whole
quantity the cap exists to bound. The real maximum now comes from the seek table.

### The budget was applied per stream, per device

It bounded *each* registration rather than the process, so `--gpu-streams 2` quietly doubled the
footprint the cap promised. It is now divided across everything that registers concurrently. The
256 MiB floor is gone as well: a floor is a promise to exceed the policy on exactly the host that
can least afford it, and the batch bottoms out at one frame on its own. An unreadable
`/proc/meminfo` no longer falls back to the 4 GiB ceiling — that would restore the default on
precisely the hosts whose memory could not be read.

### The registration budget is derived from the host now, not assumed

`GDS_VERIFY_REGISTER_BUDGET` was a hard 4 GiB, and host RSS tracks registered bytes at about 4.2×.
`-d` registers **two** slabs — the NVMe→VRAM read target and the VRAM→NVMe write source, opposite
ends of one pipeline, so they are necessarily live together — which is why its peak is double
`-t`'s. Measured at the ceiling: **16.2 GiB for `-t`, 32.3 GiB for `-d`**, on any host, because the
code never consulted memory at all. The advice to "pass `--gpu-batch` on a smaller machine" is not
a memory policy; it is a footgun with documentation.

The budget is now `min(4 GiB, MemAvailable / 17)`, the divisor chosen from the linear fit so peak
RSS lands near half of what the host actually has free. The ceiling still wins on a large machine —
this server derives 85 GiB and clamps, leaving `-t` at 23.19 s against a 23.38 s baseline — and an
explicit `--gpu-batch` still overrides everything. Forcing the cap down confirms the fit from a
second direction:

| budget | wall | user | peak RSS |
|---|---|---|---|
| 4 GiB (ceiling) | 23.19 s | 6.40 s | 16.2 GiB |
| 1 GiB | 33.64 s | 18.49 s | 4.2 GiB |
| 512 MiB | 47.94 s | 33.52 s | 2.2 GiB |

That cost is why the ceiling stays where it is: a smaller budget means more per-batch
synchronisation, and CUDA's default sync spin-waits. `GZSTD_DEBUG_GDS_BUDGET_MIB` forces a value,
which is the only way the reduced-budget path gets exercised on a machine that never triggers it.

**Both** paths now hold the table against the file before trusting it. The frame header declares the
two things the table asserts — that a frame begins here, and how much it decodes to — so the output
preflight and the staged read producer each check magic and `ZSTD_getFrameContentSize` per frame,
and a frame that declares nothing is not taken on trust either. A table that survives cannot
misplace output.

*Divergence from the proposed patch:* it offered an authoritative frame walk before arming output.
That reads the whole stream — the same 65 GiB objection that shaped v0.17.29 — to guard a case that
is nearly always damage or forgery. Verifying the claim costs **18 bytes per frame**, about 150 KB
on that same archive, and closes the hole just as completely. The honest path is unchanged: a real
archive still arms staged read and peer-to-peer output with no decline, and archives compressed
from a pipe keep the fast path, since gzstd declares each frame's size regardless of its source.

The suite cell is mutation-proven against a pre-fix binary, which fails both forgery shapes.
## v0.17.29 — the preflight proved one frame and trusted the rest, and a registration outlived its buffer

Sixth review pass. Five findings, two of them serious, and one of them a defect that v0.17.27's own
change had introduced.

### A valid archive was rejected after partial output

The `--gds-only` output preflight exists so an unsuitable archive demotes to the ordinary writer
*before* anything is written — its own comment says silent truncation is "the worst failure this
code could have". It checked **only frame 0**.

Frame *k* starts at the sum of the decompressed sizes before it, so frame 0's alignment proves
nothing about frame 2. A legal concatenation of 4096, 1001 and 17 byte frames starts them at 0,
4096 and 5097: it passed the preflight, GDS wrote a prefix, and the worker's alignment guard then
killed the run at frame 2. Reproduced exactly.

Now every frame but the last must be a whole number of 4 KiB blocks, checked from the seek table —
a trailer read and some arithmetic.

*Divergence from the proposed patch:* it walked frame headers when no seek table is available.
Advancing frame-to-frame needs each **compressed** size, so that walk reads the entire stream —
65 GiB before any work starts on the archive this was measured against. A table-less archive now
declines peer-to-peer writes instead. That costs an uncommon case (foreign archive, no table,
conveniently aligned frames) nothing but the fast writer; gzstd's own archives always carry a
table.

### A cuFile registration outlived the VRAM it described

`DecompStreamCtx::free_device()` freed `d_comp_buf` on the line **above** `gds_in_reg.dereg()`, so
cuFile kept a BAR1 mapping of memory already returned to the driver — on every teardown *and every
batch resize*, not just at shutdown. `d_decomp_buf`/`gds_out_reg` were always ordered correctly;
the bug arrived with the second registration and inherited the first one's placement. Both slabs
now deregister before either is freed.

### A failed write was reported as a usage error

A short or failed `cuFileWrite` threw into GPU recovery — but GDS output has no CPU sink, so
`gpu_only_cpu_fallback` refused with "re-run without `--gds-only`": exit 2, saying nothing about
the write that actually failed. The partial output was never committed, so this was a diagnostic
defect rather than corruption. Four sites now `die(EXIT_IO)`, matching what the compress path has
always done for the same condition.

### Write-behind had stopped existing

Worse than the vacuous test that revealed it. Coalescing (v0.17.27) runs as a pre-pass and wrote
its span synchronously, so a batch that formed a single run never populated `wb_jobs` — **no
background thread was ever created**, and `GZSTD_DEBUG_GDS_WRITE_BEHIND=1` silently did nothing on
ordinary archives, which always coalesce. The regression test still passed, because it checked for
the "ENABLED" banner that prints when the switch is *read*.

The two now compose, and the composed shape is the better one for the hosts the switch exists for:
one large asynchronous write per batch overlapping the next batch's reads, rather than thousands of
small ones. Proven by a marker printed **from inside the thread** — absent by default, present with
the switch, with coalescing on or off.

### A data race in the byte-verify kernel

`gzv_compare_kernel` set its mismatch flag with a plain store, arguing the race was benign because
every writer stores the identical value. Agreeing on the value does not make concurrent
unsynchronised writes defined. It only executes on the mismatch path, so `atomicExch` costs nothing
on a healthy run. Corruption detection and the absence of false positives were both re-verified.

### Three regression cells, ported by whether they can actually fail

The review's suggested cells assert against its own implementation's message strings, so they were
adapted rather than copied — and one was demoted on the evidence:

- **`--gds-only -d demotes on misaligned frame starts`** — a genuine regression test for the
  preflight defect above. Mutation-proven: the pre-fix binary exits 4 with no output on this valid
  archive and names `frame 2 starts at offset 5097`; the fixed one exits 0 byte-identical.
- **`--gds-only -d accepts an archive of only empty frames`** — kept, but named for the shape it
  guards rather than as a regression test for the zero-size slot floor. The review expected this
  cell to prove that fix reproduced; **it does not on this hardware**, because `cudaMalloc(size 0)`
  succeeds here and a build without the floor passes identically (verified). That fix stays in the
  same category as the stride padding: right by argument, not demonstrated.
- **`--gds-only read report does not claim alignment proves routing`** — fails if the vacuous
  "N aligned, 0 unaligned" wording ever returns to the staged-read report.

### Release status

Not tagged. Extensive suite 553/553 with no GPU or GDS cells skipped, CPU-only 335 + 72 skipped.
Still outstanding before a tag: the large round trips re-run against this preflight change, and
second-machine checks — `--gds-only` has only ever run on one host.

---

## v0.17.27 — 8,343 writes become 33, and the write phase reaches the device ceiling

Fifth review pass, this one performance-only, and the idea that paid is the **inverse** of one this
project had already rejected.

### Coalesced decompress writes — now the default

Adjacent frames occupy adjacent VRAM slots *and* adjacent file intervals, so a batch's writes are
one contiguous transfer. They were being issued one `cuFileWrite` per frame anyway — **8,343
synchronous calls on a 65 GiB archive where 33 would do**.

Measured on that archive, cold, output verified by md5 against the reference:

| | calls | `cuFileWrite` | rate | wall | sys |
|---|---|---|---|---|---|
| per frame | 8,343 | 43.90 s | 2.85 GiB/s | 68.78 s | 36.66 s |
| **coalesced** | **33** | 35.63 s | **3.48 GiB/s** | **60.77 s** | **24.39 s** |

3.48 GiB/s is ~3.74 GB/s against `dd`'s 3.7 GB/s write-alone ceiling — this takes the write phase
from 77% of what the device can do to all of it. Re-run under a harness that *asserts* no CPU
fallback or seek-table demotion occurred: 69.55 s → 60.94 s, both arms `GPU-ok`. `-t`, which writes
nothing, is correctly unmoved (22.35 s vs 22.24 s).

**This is not the write fan-out rejected earlier.** That issued *more concurrent* requests and lost
to read/write contention on one drive. This issues *fewer and larger* ones. The 2.7 GB/s
concurrent-write ceiling that explained the fan-out's failure never applied to the default path at
all, because the default does not overlap reads with writes — a distinction this changelog
previously got wrong.

Combining it with write-behind was also tried, since collapsing 8,343 tiny writes into 33 large
ones changes the overlap opportunity completely: **60.69 s vs 60.77 s, no interaction.**

### Compress: profiled at scale for the first time, and it is structurally sound

The pure peer-to-peer branch returned before the ordinary drain accounting, so profiling was blind
exactly where packing and writes happen. With that instrumented: on the same 130 GiB input,
`cuFileWrite` is **17.92 s of 59.59 s** and pack/finalize sync is **0.02 s**. Compress does *not*
have decompress's per-frame problem — its P2P path already packs frames before writing.

Coalescing the compress-side reads into one ~1 GiB request per batch is a **trade, not a win**, and
reproducible with interleaved reps: wall 59.39–59.67 s → 61.69–65.28 s, but user CPU
**2.80–2.97 s → 0.26–0.28 s**. Ten times less user CPU for 4–10% more wall clock. Left opt-in as
`GZSTD_DEBUG_GDS_COALESCE_READS=1`; it is the same shape of trade as the staged read itself.

### Skipping the checksum pass when nothing carries a checksum

The decompressor launched the full device XXH64 pass even when every frame in the batch had
`Content_Checksum_flag == 0`, then ignored the results. Now conditional.

gzstd always writes frame checksums, so this can only help **foreign** archives. Magnitude
unmeasured: the detail line that would size it prints only on the staged paths, and a stock
`zstd --no-check` archive has no seek table, so it runs `--gpu-only` instead. Verified only that it
changes no verdict — on a checksumless archive `--gpu-only` and `--cpu-only` agree on both a clean
and a corrupted copy (both exit 0 on the corrupt one, which is what `--no-check` means: there is no
content checksum for either backend to fail).

### Corrections to earlier entries

- The "missing 25% of the batch loop" was an arithmetic error. The phases account for 61.46 of
  64.26 s — 4.4% unassigned. The old `D2H transfers` figure was never write-only; it spanned status
  readback and the checksum pass (43.90 s of writes + 1.86 s of post-decode work).
- `cuFileDriverOpen`'s 2.5 s is **already amortised across input files** — it is a function-local
  static, called once per process. The `-vv` line reprints the stored duration for each input,
  which was misread as it running again. Only the ~0.3 s buffer registration repeats per file.

---

## v0.17.25 — a valid archive rejected, and the other half of the v0.17.24 fix

Fourth review pass over the GDS work.

### An all-empty batch rejects a legal archive

An empty zstd DATA frame is legal, and the ordinary producers pass its
`decomp_size` of 0 straight through — demoting from the seek table does **not** keep such a frame
away from the GPU, as v0.17.24's comment now says. If every frame in a batch is one, `max_decomp`
is 0, which survives both roundings (`(0 + 4095) & ~4095` is still 0) and reaches `ensure_buffers`
as `cudaMalloc(batch_n * 0)`. Under `--gds-only` there is no CPU fallback to recover with, so a
valid archive is refused.

The logical sizes stay 0 — nvCOMP is still asked for zero bytes and the checksum kernel still
hashes zero bytes — but the physical slot now has a 4 KiB floor so the slab exists, and a
zero-length `cuFileWrite` is skipped.

**Not reproduced.** The trigger needs empty frames reaching `gpu_decomp_worker` *while GDS output
is engaged*, and an all-empty archive produces no output bytes, so the output path declines before
the two can coincide. `cudaMalloc(0)` is not a valid allocation and the floor costs one page, so
the fix stands on construction rather than on a demonstration.

### v0.17.24 fixed half of the accounting

The CPU discard path got both `wrote_bytes` and `tasks_done`; the **GPU verify branch** got only
`wrote_bytes`. So every frame verified on the device was missing from the progress meter's frame
count, and the totals depended on where the work happened to run. Both paths now credit exactly
what `writer_thread` retires a frame with in TEST mode.

### The write-behind realloc boundary discarded a write failure

`ensure_buffers()` correctly joins `wb_thr` before `free_device()` releases the slab it is DMAing
out of — no use-after-free — but it then cleared the job list **without inspecting `wb_err`**. The
thread cannot throw (an exception crossing a `std::thread` boundary calls `std::terminate`), so a
failed `cuFileWrite` is recorded there and was silently dropped at this boundary: frames gone,
output short, nothing reported. The pre-kernel join always checked it; this one now does too.

### `-vv` read diagnostics accumulated across input files

`g_gdsv_read_*` were process-lifetime counters, so `gzstd -vv -t --gds-only A.zst B.zst` printed
B's line with A's calls, bytes and time folded in — and the derived ms/call was wrong for both.
Measured before the fix: second file **256 calls, 1063.7 MiB**; after: **128 calls, 531.9 MiB**,
matching the first. Reset with the transaction they describe.

### Alignment claim, restated more strictly

The v0.17.22 stride padding stands, but the earlier wording overstated what was shown. The four
concatenated 1,000,003-byte frames *do* exercise the defect — slot bases land at 0, 3, 6 and 1
modulo 8, and the frames are long enough to enter the 64-bit `__ldg` loop. The accurate claim:

> The caller violated the kernel's documented 8-byte stride precondition; the H100 tolerated the
> reproduced misaligned accesses, and no functional failure was observed there.

An alignment-checking sanitizer or different hardware is likelier to make it visible than any
other archive shape.

---

## v0.17.24 — a test that could not fail, and two comments that claimed more than the code delivered

Third review pass. One real defect, and two places where the source asserted something untrue.

### `--gds-only -t`'s CPU rescue reported the wrong size

In TEST mode `writer_thread` credits both `tasks_done` and `wrote_bytes` for every frame it
retires. The discard path added in v0.17.22 has no writer, so dropping the frame also dropped its
accounting — while the GPU verify branch kept crediting its own. A run that verified part of the
archive on the device and then lost the GPU therefore reported only that prefix. Measured:
**8,388,608 bytes reported against 1,073,741,824 actual**, a 128x under-report, because exactly one
frame had been delivered before the fault. The integrity verdict was always right; the size, ratio,
throughput and stats JSON were not.

### The test for that path could never have failed

Worth recording separately, because it invalidates earlier claims in this changelog. The verify
branch ends with `continue`, skipping the loop tail where `GZSTD_DEBUG_FAIL_GPU_DECOMP_LAST`
lives — so **the fault hook could not fire for `--gds-only -t` at all**, and every "GPU fault →
CPU rescue" check run against the verify path was silently exercising nothing. The hook now also
fires from the verify branch, and only with it in place does the under-report above become
visible: without the hook, the fixed and unfixed binaries produce identical output.

### Two comments that asserted invariants the code did not maintain

- `gzx_xxh64_kernel` documents its precondition — `stride` a multiple of 8 for aligned 64-bit
  loads — and claimed it "holds for every caller … always a whole number of MiB". True of the
  compress-verify caller; **false of the decompress one**, which passes the largest declared frame
  size in the batch. Four independently-compressed 1,000,003-byte frames concatenated produce an
  odd stride, with no seek table involved. The caller was fixed in v0.17.22 to round up; the
  comment now says the precondition holds *because the caller maintains it*, not because the shape
  guarantees it. (On this H100 the unfixed binary passes that archive anyway — misaligned loads are
  tolerated in hardware, which is why it looked harmless.)
- The v0.17.22 seek-table comment justified demoting empty data frames as keeping zero-length
  frames out of the GPU batch path. **That is not what it does**: the ordinary producers set
  `t.decomp_size` straight from `ZSTD_getFrameContentSize` with no zero filter, so such a frame
  reaches `gpu_decomp_worker` either way. The real reason is that admitting one to the seek table
  creates a zero-width interval and a duplicate `u_off` that every consumer of that table would
  then have to handle. Corrected, with the wrong version noted so it does not come back.

### Also

`gpu_only_cpu_fallback`'s `discard_results` default argument is gone — a caller that omits it is a
caller that has not decided whether its pipeline has a writer, which is precisely what went wrong
in v0.17.22. The compress caller states `false` explicitly.

Still accepted on reasoning rather than demonstration: the multi-file wrong-file read closed in
v0.17.22. The recipe needs a second input whose `O_DIRECT` reopen fails while the first archive's
cuFile handle is still live; every filesystem reachable here supports `O_DIRECT`, tmpfs included.

Suites: 419/419 GPU, 335 passed + 72 skipped CPU-only.

---

## v0.17.23 — the fix in v0.17.22 reintroduced the bug it fixed, on the demote path

Found by sending v0.17.22 back for a second review pass. This is a defect in
yesterday's fix, not in the code it fixed.

v0.17.22 stopped `--gds-only -t`'s CPU rescue from retaining the whole decompressed output by
having `cpu_decomp_worker` discard each frame once `ZSTD_decompress` had validated it. It decided
whether to discard by reading `g_gds_verify_active`.

**That global is read at three different times, and it changes in between.** The writer-spawn
decision reads it before the producer runs. A producer that then demotes — no usable seek table —
calls `gds_decomp_read_close()`, which clears it. The worker reads it later still. So:

```
gate sets verify_active = true      ->  writer NOT spawned
producer finds no seek table        ->  demotes, clears verify_active
CPU workers read verify_active      ->  false, so they PUSH
                                    ->  into a ResultStore with no writer
```

which is an unbounded accumulation and then a hang — the exact failure v0.17.22 existed to
prevent, reached through a path it did not consider. The race also runs the other way: a worker
that reads `true` after a demotion would discard frames the pipeline was supposed to deliver.

The sink topology is now **frozen where the writer decision is made** and passed explicitly to
every pool that produces results — `cpu_decomp_worker`, `gpu_decomp_worker`, and the
`gpu_only_cpu_fallback` rescue. One decision, made once, so no two pools can disagree about
whether a writer exists. The non-GDS call sites pass `false` explicitly rather than leaning on a
default, and `gpu_decomp_worker` uses the same value for its own delivery (`C.gds_verify`) as for
the rescue it may start.

Verified on the shapes that exercise it: a truncated-table and a forged-table archive both demote
and return 4, matching `--cpu-only`; clean and corrupt archives with **no seek table at all**
return 0 and 4; normal staged `-t` and `-d` are unaffected and byte-identical; and the injected
GPU-fault rescue still reports a correct verdict.

Suites: 419/419 GPU, 335 passed + 72 skipped CPU-only.

---

## v0.17.22 — seven defects from the first independent review of the GDS work

v0.17.13 through .21 shipped without an outside read. This is the first Codex pass over them.
Every finding below was verified against the source before being changed, and the code compiled
between each fix.

### A forged seek table could make `-t` report OK on a corrupt archive

The worst of the set, and reachable by corrupting **four bytes**. `parse_foreign_seek_table`
treated a table entry with `dsize == 0` as a skippable frame: it advanced the running compressed
offset but never added the frame to the extent lists. The structural check that follows
(`c != tstart`) still passed, because the byte count was still accounted for.

So zeroing the **last data frame's** `dsize` field removed that frame from everything built on the
table while leaving the table valid — and `--gds-only -t`, which reads frame extents from exactly
that table, verified every frame except the one that had been hidden. Mutation-tested: on an
archive with a corrupted final frame *and* that frame hidden, the old code reports **exit 0** and
the fixed code reports 4, matching `--cpu-only`.

Now the entry's own magic decides. A real skippable must also agree with the table about its
physical extent; anything else — including a legal zstd data frame that decompresses to zero
bytes — demotes to the frame walk, which decodes everything and reports the damage.

*Deviation from the proposed patch, deliberately:* the review's version admitted empty data frames
into the extent lists with a duplicate uncompressed offset. That introduces `decomp_size == 0`
frames into the GPU batch path, a shape it has never seen. Demoting closes the same hole without
creating a new one.

### Staged read state leaked between input files

`decompress_nvcomp` runs once per input, but `g_gds_read_active`, the cuFile handle and two
descriptors were only ever cleared on the demote path. After a **successful** staged run they
stayed live, so the next input inherited an active flag and a handle to the previous archive; the
two descriptors were also overwritten without being closed, leaking two per file. Now torn down at
entry *and* exit, by one helper.

The descriptor leak is plain in the source. The wrong-file read it enables is real by inspection
but **was not reproduced**: every per-file failure path reachable here either re-opens the handle
or routes away from `decompress_nvcomp` entirely.

### The checksum kernel could read past its buffer, and off unaligned addresses

Two separate defects in v0.17.14's device-side verification:

- It was launched with `d_actual_sizes` — what nvCOMP *produced* — **before** any status had been
  inspected. For a chunk nvCOMP failed on, that slot holds nothing trustworthy, and a large value
  makes the kernel read past `d_decomp_buf`, raising an illegal memory access that poisons the
  CUDA context before the ordinary per-chunk failure handling runs. It now hashes the
  header-declared sizes, which are bounded by the slot size by construction. Nothing is lost: the
  result is only consumed for a chunk whose status is success *and* whose actual size equals its
  declared size.
- `gzx_xxh64_kernel` reads frames as 64-bit words, so every slot base must be 8-byte aligned. The
  stride was only rounded on the `--gds-only` **write** path. gzstd's own archives hide this
  because their frames are whole MiB, but a foreign archive read through its seek table can have
  arbitrary frame sizes. Padded to 8 bytes; logical sizes are unchanged.

### Three defects in the v0.17.21 read fan-out

- `place_frame` stored into the plain `bool staged_batch` from up to eight threads. Every store
  wrote the same value and the read happened after the join, so it could not misbehave in
  practice — but it is a data race, and therefore undefined. Computed once in the owning thread
  from a count it already takes.
- The `rd_err` check lived **inside** the fan-out block. A batch with a single staged frame, or a
  run with the fan-out disabled, takes the sequential path instead — where a short `cuFileRead` was
  recorded and then ignored, letting nvCOMP run on a partial or stale slot. A frame with no
  content checksum could then yield wrong bytes of the right declared length. Checked on both
  sequential paths now.
- `--gds-only -t` spawns no writer, so nothing drains the ResultStore. A GPU fault sending work to
  the CPU rescue therefore retained the entire decompressed output in RAM — 130 GiB on the test
  archive. The rescue now drops each frame after `ZSTD_decompress` has validated it (that
  validation *is* the verification) and releases the throttle permit the writer would have freed.

**That last fix nearly introduced a worse bug.** The discard is keyed on `g_gds_verify_active`,
which was cleared only on entry to `decompress_nvcomp` — so a later input routed to
`decompress_cpu_mt` would have inherited a stale `true` and silently written an **empty output
file**. Caught while checking the fix; the exit-path teardown closes it, and that exact sequence is
now tested.

Suites: 419/419 GPU, 335 passed + 72 skipped CPU-only. Both build configurations compile.

---

## v0.17.21 — the read fan-out that a 1 GiB corpus said was worthless is worth 15%

v0.17.19 unstarved the batches, which invalidated every performance verdict reached before it —
all of them measured while batches were 3 frames instead of 256. This release re-tests the whole
list on the real 65 GiB archive. Two verdicts flip.

### Read fan-out: rejected once on 1 GiB, now the default

`cuFileRead` is synchronous per call, so one thread gets the **single-request** rate and no more.
Measured: a single thread sustains 3.97 GiB/s while `dd` shows the device doing ~4.9 GiB/s.

| fan-out | `-t` wall, 130 GiB |
|---|---|
| 1 | 25.94 s |
| 4 | 22.83 s |
| **8** | **22.28 s** |
| 16 | 22.08 s |

Default is 8; 16 buys 0.2 s. Reads now measure **4.93 GiB/s — the device ceiling**.

**This exact change was built, measured and reverted earlier in the arc**, correctly, because on a
1 GiB corpus reads were 0.52 s of a 7.84 s run and parallelising 6.6% of the work changed nothing.
At real scale reads are 17 s of 26 s. *A read-side idea cannot be judged on a corpus where reads
are 7% of the work* — which is the same lesson as v0.17.19's, arrived at from the other direction.

It also explains why raising cuFile's own `execution.max_io_threads` to 32 measured as a no-op,
twice: that setting parallelises **within** one request, not **across** separate ones.

### The v0.17.18 temp bound is not an optimisation, it is load-bearing

It shipped on 1 GiB evidence as "5–10%". A/B at real scale, batch 256:

| | wall | sys | host RSS |
|---|---|---|---|
| shape-derived bound | 25.76–25.94 s | 12.2 s | 17.0 GB |
| exact per-batch query | 29.25–29.33 s | **1625 s** | **138 GB** |

Those numbers understate it, because the second arm is not a slower GPU run at all: it **faults
the device** (`cudaStreamSynchronize(decomp): an illegal memory access`) after ~256 frames and
finishes the archive on 96 CPU threads, exiting 0 with correct output only because the v0.17.15
CPU rescue catches it.

The exact query remains in the code as the fallback for a shape the bound does not cover, so a
failure of `GetTempSizeAsync` now **throws a diagnosable error** naming the batch size instead of
falling through into the path measured to fault the GPU.

### Verdicts that survived re-testing

| idea | at scale |
|---|---|
| more streams | worse at both memory levels (see v0.17.20) |
| write fan-out | **worse**, not neutral — 69.31 s → 71.15 s (4 threads) → 76.54 s (8) |
| cuFile config tuning | no effect, now confirmed with reads dominant |
| `--chunk-size` | worse — it shrinks the batch and never changes read granularity |

**The read/write asymmetry explains all of them.** A single thread could not saturate reads, so
fanning those out reached the ceiling. Writes were *already* at theirs (2.86 GiB/s), and reads and
writes share one drive — `dd` gives 3.7 GB/s write-alone but 2.7 GB/s concurrent — so every
write-side idea only takes bandwidth from reads already in flight.

### Write-behind: kept, off by default

`GZSTD_DEBUG_GDS_WRITE_BEHIND=1` overlaps a batch's writes with the next batch's reads. At scale it
is **no result** — off 67.90–69.05 s, on 67.65–68.42 s, overlapping ranges — where hiding the
13.25 s read phase inside the 45.62 s write phase should have been worth ~19%.

It is kept rather than reverted because the code is correct when enabled (byte-identical output,
md5-verified at 130 GiB, with joins before the kernel, before a realloc, at teardown and on the
fault path) and **the reason it does not pay is a property of this host, not of the design**: one
drive serving both directions. On a host with reads and writes on separate devices the contention
that cancels it disappears. The diagnostic to watch is the `-vvv` `D2H transfers` rate — if it
drops from 2.86 GiB/s with the switch on, the drive is the limit; if it holds, the read phase is
genuinely being hidden.

### `cuFileDriverOpen` explained, and no longer silent

It costs ~2.5 s and is the first thing a `--gds-only` run does, so the tool looked hung. Measured
standalone: cold 2.136 s; with the CUDA context warmed first, `cudaFree(0)` 1.511 s and
`cuFileDriverOpen` 0.567 s. The costs are **additive** — ~71% of it is the cuInit it performs
internally, which any GPU path pays anyway (`--gpu-only`'s entire fixed cost is 1.90 s of the same
thing). Only ~0.57 s belongs to cuFile. That also explains why backgrounding it bought nothing
earlier: the background thread and GPU bringup serialise on the same context creation.

Nothing to reclaim, so it now says what it is doing instead: a transient
`[GDS] opening the cuFile driver ...` that the progress bar paints over, on a terminal only, and
suppressed by `-q`. The timing itself is reported at `-vv`.

Also fixed: the `[GDS] verify reads` line divided bytes by **summed** thread time, so the fan-out
made reads look 5x slower than serial. The misleading aggregate rate is gone.

### Cumulative, on the archive that started this

| | `-t` | `-d` |
|---|---|---|
| as first reported | 175.95 s | 72.63 s |
| **now** | **22.15 s** | **~68 s** |

**7.9x on `-t`.** Output md5-identical to `--cpu-only` across 139,964,108,800 bytes throughout.

---

## v0.17.20 — the compress progress meter, and the batch knee re-measured where it matters

### `--gds-only` compress jumped to 100% long before it finished

Same defect as the decompress meter in v0.17.17, in the mirror path, and it survived that fix
because the two paths account for progress differently.

The staged compress producer credited `read_bytes += file_size` — the **whole input** — the moment
it finished enqueuing. Those Tasks only *name* regions of the file; not one byte had been read.
Enqueuing is nearly instant, so the meter hit 100% almost immediately and then sat there.

The GPU worker does credit frames as it consumes them, but that accounting is gated on
`view_ptr && direct_buf < 0` — a staged Task has neither, so staged frames were skipped entirely
and the bulk credit was the only one that ever fired. The producer's credit is gone, and the
worker now counts each staged frame once its `cuFileRead` has actually delivered the bytes.
Verified climbing 0% → 50% → 100% on a 4 GiB input and finishing at exactly 4.00 GiB, archive
byte-identical.

### The registration budget, measured against streams instead of guessed

With batches no longer starved (v0.17.19), the batch/stream question could finally be asked
properly — holding **total registered slots constant** so memory is not a confound. `-t`, 130 GiB,
one device, cold:

| slots (host RSS) | 2 streams | 1 stream |
|---|---|---|
| 512 (33.8 GB) | 27.81 s | **25.90 s** |
| 256 (17.0 GB) | 29.49 s | **25.98 s** |

**One stream wins at both memory levels**, so the deficit is the streams themselves and not the
memory they cost. That vindicates the `region_staged() ? 1` decision, which until now rested on a
1 GiB-era measurement.

And 256 is the knee: `1×512` beats `1×256` by 0.08 s — noise — for another 16.8 GiB. So
`GDS_VERIFY_REGISTER_BUDGET` goes 2 GiB → **4 GiB** (batch 256), and stops there.

`-t` on the user's own 65 GiB archive, cumulative across this release series:

| | wall |
|---|---|
| as first reported | 175.95 s |
| + `util_scale` fix, 1 GiB budget (v0.17.19) | 36.15 s |
| + 2 GiB budget (v0.17.19 as shipped) | 30.46 s |
| **+ 4 GiB budget (this release)** | **25.69 s** |
| *(reference)* `--gpu-only` | 28.49 s |
| *(reference)* `--cpu-only` | 18.80 s |

**6.8x overall**, and `--gds-only` now beats `--gpu-only` outright. The 17 GB of host RSS is a real
cost and is chosen for a host that has it; `--gpu-batch` overrides the budget entirely, and
`--gpu-batch 64` restores the old 4.4 GiB footprint for a smaller machine.

### Also re-measured, since v0.17.19 invalidated every earlier verdict

The staged peer-to-peer read was A/B'd at scale for the first time
(`GZSTD_DEBUG_GDS_NO_STAGED_READ=1`): **24.80 s without it against 30.46 s with**, but 195.22 s of
system CPU against 11.29 s. Turning it off is *faster* and burns 17x the kernel CPU — the page
cache does the work instead. It stays on: this flag exists to spend less CPU, and that is now a
number rather than an assumption.

The compress worker's identical `util_scale` line was gated the same way as decompress and
**measured to do nothing** — batches came out median 8 / mean 10.2 / max 32 either way, and
`--gds-only` compress (which forces one stream) already takes full 64-frame batches. Reverted; the
same line is not the same defect. Forcing `--gpu-streams=1` on `--gpu-only` compress was also
tried: it does not unstarve the batches and costs 19% of wall clock, because the second stream is
overlapping intake with compute.

---

## v0.17.19 — a multi-GPU fairness heuristic was starving single-GPU batches to 3 frames

`gzstd -t --gds-only` on a 65 GiB / 8344-frame archive took **175.95 s**, against
`--cpu-only`'s 18.80 s, with 161 s of that in **user** CPU and one core pegged. Every
optimisation in v0.17.16–.18 was measured on 1–4 GiB corpora, and none of them could see this:
the defect is invisible below a few hundred frames.

The batch intake scales the pop size by GPU utilisation:

```
util_scale = max(0.05, (100.0 - util.gpu) / 100.0);
pop_n      = max(1, pop_n * util_scale);
```

Its own comment scopes it correctly — *"a GPU at 50% utilization gets half the batch → finishes at
roughly the same time as idle GPUs → results arrive in order for the writer"* — which is an
argument about **several** GPUs sharing a queue. With one worker there is nothing to finish
together with, and the scaling **feeds back on itself**: a busy GPU scores high utilisation → gets
a smaller batch → needs more launches for the same work → stays busy → `util_scale` sits on its
0.05 floor forever. 64 × 0.05 = 3.2.

Measured, one device, cold, 130 GiB of data:

| | before | after |
|---|---|---|
| median batch | **3 frames** (mean 5.6, max 64) | **64** |
| batches | 1495 | 131 |
| GPU kernel | 94.87 s (63.5 ms/batch, nearly all launch overhead) | **8.21 s** |
| per-batch sync | 63.00 s | 7.39 s |
| staged reads | 17.09 s @ 3.83 GiB/s | 17.03 s @ 3.84 GiB/s — unchanged, always at the ceiling |
| **`-t` wall** | **175.95 s** | **36.15 s** |

Confirmed twice before the fix was written: raising `--gpu-batch` — which `util_scale` simply
multiplies — halved the runtime each time it doubled (auto 168 s → 128: 76 s → 256: 44 s).

`--gpu-only` escapes the worst of it because its auto-tuner is free to grow toward 256 and partly
cancels the scaling. The staged path **pins** that tuner (v0.17.16), so nothing offset it there —
this release's own change is what exposed the trap.

Gated on `gpu_worker_count > 1`. The compress worker carries the identical line and presumably the
identical defect; it is deliberately left alone, because it has not been measured at scale.

### The registration budget was also chosen on too small a corpus

`GDS_VERIFY_REGISTER_BUDGET` was set to 1 GiB in v0.17.16 on the strength of a 1 GiB archive.
Re-measured at 65 GiB with the batches no longer starved:

| budget | batch | wall | user | host RSS |
|---|---|---|---|---|
| 1 GiB | 64 | 35.89 s | 16.69 s | 4.4 GiB |
| **2 GiB** | **128** | **28.85 s** | 9.08 s | 8.6 GiB |
| 4 GiB | 256 | 25.83 s | 4.98 s | 17.0 GiB |
| 8 GiB | 512 | 24.98 s | 2.99 s | 33.8 GiB |

Raised to 2 GiB. Returns fall off a cliff after 128: 64→128 buys 7.0 s for 4.2 GiB, while 256→512
buys 0.85 s for 16.8 GiB, and this has to be affordable on a workstation rather than only on a
host with 1.4 TiB. `--gpu-batch` still overrides.

### At the scale that matters

`-t`, the user's own invocation with all 8 GPUs visible (`--gds-only` narrows to one device by
design, so the fix engages): **175.95 s → 28.80 s, 6.1x**, user CPU 161.07 s → 8.74 s.

`-d` on the same archive, output **md5-identical** to `--cpu-only` across 139,964,108,800 bytes:

| `-d` | wall | total CPU | cores busy |
|---|---|---|---|
| `--gds-only` | 72.63 s | **41.1 s** | **0.57** |
| `--cpu-only` | **51.90 s** | 807.5 s | 15.6 |

`--cpu-only` still wins wall clock by 1.4x; `--gds-only` does the same work with **19.6x less
CPU**. On an idle 256-core host that trade is uninteresting, and on a machine doing anything else
it is the entire point of the flag. This is the first measurement in this arc taken at a scale
where that shows.

**The lesson, and it applies to three constants introduced in this release series:** every one was
tuned on a corpus ~1/100th the size of the real workload, and every one was wrong there. A GPU
batching path cannot be characterised on an archive that fits in a few batches.

---

## v0.17.18 — the per-batch temp-size sync, and the 50% of reads nobody asked for

`-vvv` put 38% of the GPU decompress batch loop in "syncs, temp queries,
readbacks" — a larger share than the decompress kernel itself. Most of it was one
call.

nvCOMP offers two ways to size the decompress workspace.
`nvcompBatchedZstdDecompressGetTempSizeSync` inspects the compressed frames already
on the device: exact, but it costs a `cudaStreamSynchronize` on **every batch** —
and when its answer grew, the slab was freed and reallocated, so every staged frame
had to be **read from the drive again**. That is where `192 reads for 128 frames`
came from: half the read traffic on that workload existed only to refill a buffer
the temp query had just invalidated.

`nvcompBatchedZstdDecompressGetTempSizeAsync` takes only
`(chunks, max_uncompressed_chunk_bytes)`, touches no device memory and needs no
stream. It returns a conservative bound, which is exactly what you want for
*allocation* — and it is what the `--verify` path has always used to size its own
workspace before any data exists. Sizing temp from that bound up front means the
temp always fits, so the per-batch query and its sync never run and the grow path
(with its re-read) never fires. The exact query stays as a fallback for a shape the
bound does not cover.

| | before | after |
|---|---|---|
| read calls for 128 frames | **192** | **128** |
| `-d --gds-only`, cold 1 GiB | 4.37–4.46 s | **3.96–4.14 s** |
| `-t --gds-only`, cold 1 GiB | 3.41–3.50 s | **3.31–3.35 s** |
| "syncs / other" share of the batch loop | 38% | ~25% |

**The wall-clock win is 5–10%, not 38%, and the reason is worth recording.**
Allocating the conservative bound up front consumes VRAM, and the VRAM-fit loop
answered by shrinking the batch from 64 frames to ~26 — so total kernel time rose
0.145 s → 0.329 s across more, smaller launches, giving back much of what the
removed syncs saved. Forcing the batch back measured 4.12 s against 4.14 s auto:
overlapping ranges, no result, not worth chasing. The redundant read traffic is
gone either way, and that part is unambiguous.

Writes are now the largest block at ~36% — blocking `cuFileWrite` calls on the one
host thread that also reads and launches kernels.

Unchanged and verified: `-t` agrees across `--gds-only` / `--cpu-only` /
`--gpu-only` (clean 0, two corrupt archives 4), `-d` byte-identical on all three,
`--gpu-only` and `--hybrid` unaffected, both build configurations compile.

---

## v0.17.17 — `--gds-only -d` becomes NVMe → VRAM → NVMe, and the progress meter stops reading past 100%

Until now only the **write** half of `--gds-only -d` was peer-to-peer. Compressed frames came in
through the ordinary host reader and an H2D copy, so every compressed byte still crossed host
memory on its way to a device that then wrote its output straight back to the drive. The
seek-table producer built for `-t` in v0.17.16 supplies the missing half.

`-d` needed no ordering work to use it: offset-addressed writes already bypass the ordered writer,
and the producer was already carrying each frame's `out_off` from the table. The one flag was
split in two — `g_gds_read_active` (frames are file regions read by `cuFileRead`; `-t` and `-d`)
and `g_gds_verify_active` (TEST only: verified on the device, never copied D2H).

Cold, 1 GiB, **one device on every arm** — the device-count confound has produced a wrong answer
four times in this arc and is controlled here:

| `-d` | wall | user | sys | host RSS |
|---|---|---|---|---|
| **P2P read + P2P write** (this version) | 4.37–4.42 s | **0.49–0.58 s** | 3.33–3.41 s | 8.67 GiB |
| host read + P2P write (v0.17.16) | 6.30–8.51 s | 2.12–4.57 s | 5.97–6.04 s | 17.7 GiB |
| `--gpu-only`, no GDS at all | **2.92–3.23 s** | 1.21–1.26 s | 3.65–3.90 s | **1.16 GiB** |

Against the previous `-d`: **1.4–1.9x wall, 4–9x less user CPU, half the RSS**, and the
run-to-run instability is gone (6.30 vs 8.51 s on the old path).

**`--gpu-only` still wins wall clock at this size** — 2.92–3.23 s against 4.37–4.42 s — while GDS
wins total CPU (3.90 s vs 4.91 s). That is the same verdict as everywhere else in this arc: GDS
buys CPU efficiency, not throughput. It is measured on 1 GiB against one H100 on fast NVMe, which
is the size where fixed costs dominate and therefore the least representative case for a decision.

`GZSTD_DEBUG_GDS_NO_STAGED_READ=1` runs `--gds-only` through the ordinary host reader, so the
peer-to-peer read can always be measured against its own absence on one binary. Without that
control the read could not be separated from the batch pinning that landed beside it — and the two
were worth very different amounts.

### The progress meter read past the size of the file

`--gds-only -d` on a 65 GiB archive reported `in: 100.0% 106.98 GiB @ 28.14 GiB/s` — a total
larger than the file, at a rate no drive produced.

The staged producer credited every frame's compressed size at **enqueue** time, while the GPU
worker separately credits each batch as it **consumes** it. Both fired. At `out: 59.3%` that was
65.38 GiB banked up front plus ~41.6 GiB actually consumed — exactly the 106.98 GiB reported, and
the "28 GiB/s" was simply the whole archive appearing before any work had happened.

The worker's count is the correct semantic — bytes processed, not bytes enqueued — so the
producer's increment is gone. Output was byte-identical throughout; this was only ever cosmetic.
But it was **invisible on every test corpus used to develop the feature**: at 1 GiB the producer
finishes almost instantly, `in:` snaps to 100%, and it looks plausible. It took a 65 GiB archive,
where producer and consumers are separated in time, for it to show at all.

### Measured, for the optimisation work that follows

`-vvv` on a 1 GiB decompress, 2 batches:

| phase | time | share of the batch loop |
|---|---|---|
| staged reads | 0.132 s | 13% |
| **decompress kernel** | **0.145 s** | **14%** |
| writes (`cuFileWrite`) | 0.347 s | 34% |
| syncs, temp queries, readbacks | ~0.38 s | 38% |

**The GPU is busy 14% of the batch loop**, which is what watching nvtop sputter actually looks
like. The reads are not the constraint (4023 MiB/s aggregate, 1.03 ms/call). `gpu_streams` is
forced to 1 for every `region_staged()` path, so read, kernel and write cannot overlap: nothing
reads while the kernel runs and nothing decompresses while frames go out. One stream × 64 frames ×
16 MiB × 2 slabs is ~2 GiB of a 95 GiB card in play.

Also visible: **192 reads for 128 frames** — a temp-buffer grow reallocated the slab and forced a
re-read of one batch, so ~50% redundant read traffic on this workload.

Not addressed here. Recorded because the next step is pipelining, and a bigger buffer alone will
not fix it: a larger batch still runs read → kernel → write in series and simply idles the GPU for
longer stretches instead of more often.

---

## v0.17.16 — `--gds-only -t`: NVMe → VRAM → verify on the device, and the payload never reaches host memory

`gzstd -t --gds-only` used to refuse outright: *"--gds-only decompress needs a real output file"*.
That rule is about writing by offset into a registered handle, and `-t` writes nothing — it was
never a rule about `-t`. The flag now drives the **read** side instead.

Frame extents come from the archive's seek table, each frame is `cuFileRead` straight into VRAM,
nvCOMP decompresses it there, and the device-side XXH64 added in v0.17.14 is compared against the
frame's own trailer. No D2H, no writer thread, no host copy of the payload.

**Why the seek table is required.** The ordinary decompress producer finds frame boundaries with
`ZSTD_findFrameCompressedSize`, which walks each frame's block headers *in a host buffer* — so it
must pull the compressed stream into host RAM just to learn where the next frame begins. Reading
it again into VRAM afterwards would move every byte twice and make "NVMe → VRAM" a fiction. The
seek table gives exact `(offset, csize, dsize)` per frame for free. Every gzstd archive carries
one (v0.14.92), as does any zstd-seekable archive; anything else **demotes with a warning** rather
than failing or pretending.

**Compressed frame starts are almost never 4 KiB-aligned** — measured on real archives, only frame
0 is. Peer-to-peer needs alignment on the file offset, the device offset *and* the length, and an
unaligned `cuFileRead` does not fail, it silently bounces through host memory. So each frame is
read as an aligned superset and nvCOMP is handed a pointer to where the frame actually starts
inside that window.

### The performance work, which was mostly undoing my own mistakes

Reported from a 65 GiB archive: **151 MiB/s, 62 s user against 2.8 s system** — one core pegged
while the drive idled. Two hypotheses were wrong and are recorded so nobody retries them:

- **Serial `cuFileRead`.** The compress path fans its reads across threads and documents why
  ("synchronous per call"), so a fan-out was built here too — and measured to do **nothing**
  (7.63–7.94 s serial vs 8.01 s fanned out). Instrumentation said why: the reads are **0.52 s of a
  7.84 s run**, 159 calls at 1282 MiB/s aggregate. They were never the constraint. The fan-out was
  reverted rather than kept on a hypothesis measurement had refuted; the counters it took to find
  that out are kept and print at `-vv`.
- **Buffer registration.** 4 calls, ~115 ms total. Not it either.

The actual cause was **a bound added in this same version**. Host RSS tracks the *registered*
bytes (`--gpu-only` allocates identical device slabs for free — it just never registers them), so
the compressed slab was capped at 256 MiB. That forced batch=16, and a smaller batch means more
per-batch synchronisation, and **CUDA's default sync spin-waits in userspace**. Tidy-looking
memory hygiene bought a pegged core. The auto-tuner then made it worse by exploring *smaller*
batches still, so it is now pinned and locked on this path.

Cold, caches dropped per run, 1 GiB archive:

| | wall | user | sys | host RSS |
|---|---|---|---|---|
| 256 MiB budget, tuner free | 8.36 s | 5.43 s | 2.77 s | 1.28 GiB |
| 1 GiB budget, tuner free | 5.68 s | 2.70 s | 2.90 s | 4.42 GiB |
| **1 GiB budget, tuner pinned** | **3.41–3.50 s** | **0.32 s** | 2.87–2.98 s | 4.42 GiB |
| *(control)* `--gpu-only`, 1 device | 2.99 s | 1.15 s | 3.35 s | — |

**2.4x wall and 17x user CPU against where it started**, now below `--gpu-only` on user time. The
1 GiB budget is a measured knee, not a round number: past it the mapping grows 4x to buy nothing
(batch 256 was *slower*). The 4.42 GiB of host RSS is the honest price of the mapping.

### What this does NOT claim

Peer-to-peer **routing** is not claimed. Alignment counters show eligibility, not routing — only
cuFile's `posix=` counter discriminates, and it was not run here. What is demonstrable is that the
payload never enters host memory on gzstd's side.

`--cpu-only -t` remains far faster in wall clock at this size (0.40 s cold on 1 GiB). Much of what
looks like GPU slowness is fixed init: measured on a near-empty archive, `--gpu-only` costs
**9.66 s with 8 devices visible and 1.78 s with one** — for `-t`, the effective fix is not
initialising eight GPUs to check one file.

### Safety, and three defects found while building it

- **cuFile silently needs `O_DIRECT`** — a buffered `FILE*` registers fine and then reads through
  the page cache. The archive is reopened through procfs so `O_DIRECT` cannot leak into the
  caller's handle.
- **The CPU rescue segfaulted** on staged Tasks, and on the runs where it did not it "verified"
  garbage and **exited 0** — the worst possible answer for a command whose entire job is to say
  whether an archive is intact. Staged Tasks are now re-read from the archive by the CPU path, so
  the rescue produces a real verdict (exit 4) instead of refusing with a usage error.
- **A second upload path** (after a temp grow reallocates the slab) also copied from a staged
  Task's non-existent host pointer. Both paths now go through one routine; staged frames are
  re-*read*, since their bytes only existed in the slab that was just freed.

`gpu_only_cpu_fallback` also used to die without ever saying *why* the GPU failed; the cause is
now logged.

Suite gains `--gds-only -t verifies on the device`, which keeps three outcomes distinct on
purpose: a demoted run **skips** (GDS has four platform gates and any of them can be shut), a run
that silently never engaged **fails as NOT TESTED**, and only an engaged run may pass — then it
must also reject a corrupted archive.

---

## v0.17.15 — the multi-GPU CPU rescue could never fire, so a bad archive wedged instead of failing

`gzstd -t --gpu-only` on a corrupt archive exits **1** with `internal error: writer stuck —
workers_done but frame 0 of 1 missing` instead of the data error every other backend reports.
Reproduced 8 times out of 8 — this is deterministic, not a race. `--hybrid` and `--gpu-only
--gpu-devices=1` are correct on the same file, which is why it survived this long.

When a GPU decompress worker faults it re-enqueues its undelivered frames *"so other GPUs can
pick them up"*, and the last worker to fail runs `gpu_only_cpu_fallback` so those frames still
have a consumer. The trigger was `fails == gpu_worker_count` — counting only **failures**. But a
worker that drains the queue and leaves exits *normally* and never touches that counter. On any
multi-GPU run at least one device finishes cleanly, so the count is capped below the total
forever and **the rescue can never fire**. The re-enqueued frames sit in the queue with nothing
left to consume them, and the writer waits for a frame that will never arrive.

Single-GPU runs were correct by accident: with one worker, "the one that failed" and "the last
one out" are the same thread. The defect needs a device count above 1 and at least one clean
drain — that is, the normal configuration on a multi-GPU host.

Fixed by keying the rescue on **exits** rather than failures, in a separate counter incremented
on all three exit paths (clean drain, VRAM-skip at init, fault). `gpu_failures` keeps its old
meaning because the bringup barrier reads it — counting clean exits there would release that
barrier early. The last worker out rescues only if `queue->drained()` is false, so a healthy run
never spawns a fallback pool.

Corrupt archive, 2 frames, 8 GPUs visible:

| config | before | after |
|---|---|---|
| `--cpu-only` | 4 | 4 |
| `--hybrid` | 4 | 4 |
| `--gpu-only --gpu-devices=1` | 4 | 4 |
| **`--gpu-only`** (8 devices) | **1 — wedged, 8/8 reps** | **4** |

**What actually triggers it in the wild**, since this is not a hypothetical path: a corrupt
archive that walks nvCOMP off the end of a buffer. The first device reports
`cudaStreamSynchronize(decomp): an illegal memory access was encountered` and the poisoned CUDA
context then cascades — the remaining devices fail with `CUDA-capable device(s) is/are busy or
unavailable`. That is a **throw**, so it takes the re-enqueue path and wedges. A plain content
checksum mismatch does *not* wedge: v0.17.14 made that `die(EXIT_DATA)`, which exits before any of
this machinery runs. The distinction matters for testing — see below.

### A second issue, found while building a deterministic test for the first — and NOT a live bug

`C.delivered` — the count the catch block uses to decide which frames of the in-flight batch are
still undelivered and must be re-enqueued — was reset to 0 **immediately before the nvCOMP
decompress launch**. Every throw site in batch *setup* runs earlier: device allocation, the three
H2D copies, nvCOMP temp sizing, and the re-upload after a temp grow. A fault there, on any batch
after the first, would find `C.delivered` holding the **previous** batch's count and slice that
many frames off a freshly popped batch, re-enqueueing only the remainder.

**Then the instrument said otherwise, so here is what is actually true.** A probe printed
`delivered` and `batch.size()` from inside the catch on every fault. Across 9 runs and 4 device
and stream configurations, the erase **never once fired**:

```
[SCRATCH] catch: delivered=16 batch=5  -> no erase
[SCRATCH] catch: delivered=16 batch=15 -> no erase
[SCRATCH] catch: delivered=0  batch=16 -> no erase
```

The guard is `C.delivered <= C.batch.size()`, and the stale count is reliably *larger* than the
fresh batch, because batches shrink as the queue drains. A run injecting setup faults produced
byte-identical output 15 times out of 15. An earlier 1-in-3 wedge that prompted this investigation
did not recur (0 of 6) and the probe shows no erase on those runs, so it was a flake and not this
mechanism.

So this is **not** a data-loss bug and should not be cited as one. It is a correct-by-accident
invariant: the code was relying on "the next batch is always smaller" without saying so anywhere.
Resetting `C.delivered` at batch **intake** — where it always describes the batch actually held —
makes the guard unnecessary rather than load-bearing. Kept because it is a one-line move that
removes a dependency on an accident, not because it fixes an observed failure.

**Checked and ruled out**, so nobody re-derives it: the trivially-compressed batch path also
calls `queue->re_enqueue()` and then exits the worker with a comment reading *"Let CPU workers
drain the queue alone"* — which under `--gpu-only` is nobody. It is unreachable rather than
wrong: the whole block is guarded by `if (sched)`, and `sched` is constructed only when
`!opt.gpu_only && opt.hybrid`, so under `--gpu-only` it is dead code and under `--hybrid` the
CPU pool it hands off to actually exists.

The compress worker shares the `gpu_only_cpu_fallback` mechanism but not the defect: its mid-run
catch deliberately does **not** re-enqueue — it raises `g_gpu_aborted` and lets the driver
rebuild CPU-only from the still-present input — so it has no stranded-frame case to strand.

---

## v0.17.14 — GPU decompress never checked the content checksum, and wrote corrupt output at exit 0

`gzstd -d --gpu-only` on a bit-rotted archive produced wrong bytes and exited 0. `gzstd -t
--gpu-only` on the same archive reported OK. This is present in tagged v0.17.11 and every release
before it, and has nothing to do with the peer-to-peer work; it was found while establishing a
baseline for a device-side verification feature.

The GPU path checked two things: nvCOMP's per-chunk status, and the produced size against the
header-declared size. **Neither can see wrong bytes of the right length** — the frame header
declares the size, nvCOMP emits exactly that many bytes, and both checks pass. The CPU path gets
content-checksum validation free from `ZSTD_decompressStream`; the GPU path never had an equivalent.

Measured with single-bit flips in a frame payload:

| | exit | result |
|---|---|---|
| `-d --cpu-only` | 4 | correctly rejected |
| stock `zstd -d` | 1 | `Restored data doesn't match checksum` |
| **`-d --gpu-only`** | **0** | **67,108,864 bytes, differing at byte 50,770** |

**41 of 41 offsets tried.** It is not a corner case; it is the normal outcome for payload
corruption. `--hybrid` and the default path happened to catch it here only because frames that land
on CPU workers are checked — on a machine where the GPU takes every frame, nothing is.

Each frame's XXH64 is now recomputed **on the device** over the bytes nvCOMP produced, and compared
against the checksum in the frame trailer. `d_actual_sizes` is already the per-frame length array
the kernel needs and output frames sit at a uniform stride, so this is one launch per batch with no
extra host traffic. The expected value is read while the compressed frame is still on the host,
gated on the Content_Checksum_flag; frames without it are left alone rather than falsely reported
as verified.

### Why it dies rather than throwing, and the pre-existing bug that decided it

The neighbouring size-mismatch check throws, which re-enqueues the frame so another worker can
retry — correct there, because a size mismatch can be a transient device fault. **A content-checksum
mismatch is deterministic**: the same bytes hash the same way on every device, so a retry cannot
change the answer.

Routing it through that machinery is also unsafe. The fault path re-enqueues the frame and runs the
CPU fallback only once `fails == gpu_worker_count`; with eight devices, the workers that never
touched the bad frame exit cleanly, the count never completes, and the re-enqueued frame has no
consumer. Observed: `--gpu-only` across eight devices ended in `internal error: writer stuck` and
exit 1, while the same archive on one device or under `--hybrid` correctly reported a data error.
So the mismatch now calls `die(..., EXIT_DATA)`, which matches the CPU path, stock zstd, and this
project's own exit table.

**That multi-GPU fallback gap is a separate, pre-existing defect and is NOT fixed here.** Any
deterministic per-frame GPU failure under `--gpu-only` with several devices can wedge the same way.
It has never surfaced because nothing produced one. It needs its own change and its own testing.

### Verified

Clean archives pass and corrupt archives report exit 4 under `--cpu-only`, `--hybrid`,
`--gpu-only --gpu-devices=1` and `--gpu-only` alike; `-d` writes no output rather than corrupt
output; 41 of 41 corruptions detected.

## v0.17.13 — the review of v0.17.12 found a corrupt archive behind an ordinary flag

v0.17.12 shipped with 22 targeted validation checks and two clean suites, and still had a path that
wrote a corrupt archive and exited 0. An independent review found it. Nothing had been tagged, so
nothing deployed.

### `--gds-only --no-direct` wrote the seek index over the start of the archive

The peer-to-peer path writes through its own O_DIRECT descriptor, obtained by reopening the
verified output through procfs. The ORIGINAL buffered `FILE*` therefore never moves: its position
stays at 0. `append_tar_index()` then wrote the seek trailer through that `FILE*` — at offset zero,
on top of the first frames. `-l` reports "not a valid zstd stream"; decompression warns that the
first frame has no content-size header.

**This was introduced by v0.17.12's own fix for the MISSING index**, hours after it. `--no-direct`
had been checked by hand *before* that change, seen to pass, and never re-run; the validation suite
did not cover the flag at all. The buffered stream is now positioned at the peer-to-peer prefix
length before the append.

### `--verify-engine=cpu` verified nothing and exited 0

The CPU `VerifyPool` consumes host `FrameBuf` objects that the writer normally produces. The
peer-to-peer branch returns before creating any, so the pool was fed zero frames and reported
success. Also reachable through AUTOMATIC CPU selection on older PCIe hardware, where the GPU
verifier is not chosen. Peer-to-peer output is now demoted when verification resolves to the CPU
engine — that restores the D2H traffic, but only because a CPU verifier was explicitly asked for.

### Two shapes that could not work were not refused

`--gpu-devices=2` and `--gpu-streams=2` died mid-run on the in-order sequence assertion (exit 1,
partial output removed — it failed safe, but it should never have started). An `O_APPEND`
descriptor, as produced by `gzstd -c … >> existing.zst`, was accepted even though cuFile writes by
absolute offset and append semantics would ignore it. Both now demote to the ordinary writer with a
warning before any work begins.

### The technique that found these, which is the reusable part

The review was asked one question: *enumerate everything `writer_thread` and
`DirectWriter::finalize` do, and for each say whether the peer-to-peer path does it, deliberately
skips it, or misses it.* It walked nineteen responsibilities. Two of the three defects came out of
that enumeration rather than from testing. **Bypassing a component means inheriting its bookkeeping;
enumerate the bookkeeping instead of discovering it.** On this path that lesson has now cost four
defects: throttle permits, the seek table, the CPU-verify tap, and the buffered append cursor.

Also corrected: `wrote_bytes` omitted the four-byte checksum trailer per frame and `total_out` was
never updated on this path. Reporting only, but wrong.

The harness is now 26 checks and covers explicit multi-stream, `--no-direct`, append-mode
redirected stdout, and forced CPU verification — the four shapes that were missing.

## v0.17.12 — --gds-only compress is now pure peer-to-peer: NVMe to VRAM to NVMe

The read half has been peer-to-peer since v0.17.0. The write half was not: every compressed frame
came back to the host through a pinned slot, was memcpy'd into a frame buffer, and went out through
the ordered writer. This finishes the path. No payload byte enters host RAM in either direction.

Measured on 130 GiB, cold, same device:

| | before | after |
|---|---|---|
| wall | 67.55 s | 59.93 s |
| **user CPU** | 15.54 s | **2.69 s** |
| total CPU | 49.38 s | 34.53 s |

Proven by cuFile's own per-process counters rather than by alignment, which cannot distinguish
peer-to-peer from a silent bounce: `Read n=768 posix=0 unalign=0`, `Write n=8 posix=0 unalign=0`.
`zstd -t` validated all 139,964,108,800 bytes.

**The gain is 5.8x on user CPU, not the ~0.3% predicted.** The D2H transfer really is nearly free --
0.39-0.91 ms inside a 150-220 ms batch -- but costing the transfer missed the chain behind it: the
pinned-slot memcpy, the frame buffers, the ResultStore, the ordered writer. Removing a copy removes
everything downstream of it.

### How the alignment problem is beaten, which is the design worth keeping

`cuFileWrite` routes peer-to-peer only when file offset, device offset AND length are all 4 KiB
aligned. Compressed frames are variable length, so writing them individually at packed offsets would
leave nearly every one misaligned and cuFile would silently bounce it through host memory -- correct
bytes, none of the point. So frames are **packed contiguously into a registered device buffer and
flushed in whole 4 KiB multiples**: one aligned write covering many frames, with the sub-block
remainder carried to the front for the next flush. A CUDA kernel (`gzp_finalize_kernel`, new in
`gpuverify.cu`) stamps each frame's zstd Content_Checksum_flag and 4-byte trailer in place, work the
host used to do after the D2H.

**The output is byte-identical to the ordinary writer.** No padding frames, no format change.

### Four defects, three of them invisible to a round-trip

* **Wedged at 16.9% with the output frozen.** Bypassing the ordered writer means inheriting its
  bookkeeping, and the FrameThrottle permit release is the invisible part: workers acquire per frame
  and only the writer releases. This is v0.17.1 a second time -- the identical leak wedged
  `--gds-only` decompress at 97.8%.
* **Truncated stream after 193 frames.** The finalize kernel edits frames in place, and it ran once
  per batch -- so every frame a MID-BATCH flush had already written went out with the checksum flag
  clear and an uninitialised trailer. Only an input large enough to force a mid-batch flush exposed
  it; the single-flush case passed cleanly.
* **`--gds-only --preallocate` produced a ZERO-BYTE ARCHIVE AT EXIT 0.** `DirectWriter::finalize()`
  trims the reservation back to `logical_written_`, which is 0 because this path writes the file
  itself and never touches that writer. Our `ftruncate` set the correct size and finalize then cut
  it to nothing. Fixed by telling the writer the true length rather than refusing the combination:
  preallocate and peer-to-peer are compatible, since the reservation is exactly what the writes land
  in.
* **Every archive silently lost its seek index.** `writer_thread` records each frame's on-disk size
  so the seek-table geometry can be pinned; with no writer the table was empty and no trailer was
  emitted. `-l` read 384 frames / 0 skips against the ordinary path's 385 / 1 -- O(1) listing and
  random access gone. The drain records it now.

The last two were found only by compressing the same input down BOTH output paths and requiring
byte-identical archives. That check is worth more than any round-trip: the archive decompressed
perfectly while its index was missing.

### Also measured, and it says the CPU cost that remains is not ours

GDS's marginal kernel cost is **0.112 s/GiB, identical to plain O_DIRECT** (fitted across 5/20/80
GiB gdsio transfers, separating 14.8 s of fixed setup). It redirects the DMA destination; it does
not remove the syscall, the block-layer submission, or the DMA setup. Of gzstd's 34.5 s, roughly
22 s is the irreducible cost of moving 196 GiB across the block layer. No cuFile transfer mode
beats the one already in use: batch and async variants all cost MORE kernel CPU, and every mode
issues the same number of DMAs, so batching groups submissions rather than transfers.

Not adopted, having been built and measured: **ping-pong output banks**, where a writer thread
drains one bank while the drain packs the other. It worked exactly as designed -- stream-wait fell
from 53% to 20% -- and was **3.5% slower**, because the 18 seconds it saved reappeared as reads
collapsing from 4.88 GiB/s (the device ceiling) to 2.81 once writes overlapped them. One saturated
NVMe has nothing to give; serialising is the better trade here. It may still pay on a host with
input and output on separate drives, which is untestable on this one.

## v0.17.11 — GPU device selection was correct; the reporting made it look broken

### The investigation, which found no defect

`--gds-only` and `--direct-stage` both pin themselves to one GPU, and every log line from such a
run calls it `GPU0`. That looked wrong: this box has three cards with 95 GB free and five with 81,
and starving a device did not move the choice. It appeared that selection was ignoring free VRAM.

It is not. When a run asks for a subset of devices, gzstd ranks them through NVML **before any CUDA
call exists**, names the winner by UUID in `CUDA_VISIBLE_DEVICES`, and only then lets CUDA start --
which is what makes `--gpu-devices` faster than paying `cuInit` for eight contexts. CUDA renumbers
whatever it can see from zero, so the chosen card is *always* device 0 from inside the process. The
constant `GPU0` was the renumbering, not a stuck selection.

Verified by asking the driver rather than the program: `nvidia-smi --query-compute-apps` reports the
live process on `GPU-d628d3f1-…`, which is index 4 -- one of the 95 GB cards, i.e. exactly the
roomiest device the documented rule should pick. Selection has been right the whole time.

Two earlier attempts to disprove it failed for reasons worth recording, because both are traps this
repo already documents. `cudaMalloc` does not commit VRAM, so a hog that "allocated" 78 GiB in 4 ms
starved nothing and moved no ranking. A version that also `memset`s does commit -- and then starved
CUDA device 0 while `nvidia-smi` showed the drop on index 4, which is the NVML-to-CUDA index
mismatch that is the whole reason this codebase correlates by UUID and never by index.

### What was actually wrong: nothing said which card it picked

That is not cosmetic. It cost a full investigation into a bug that did not exist, and on a shared
machine the physical identity is the only way to know where a job landed. Selection now says so at
`-v`, naming the device, how it was chosen, and why every later line will disagree:

    [GPU] selected GPU-d628d3f1-… (least busy, ties to most free VRAM); CUDA renumbers this set from 0, so later lines say GPU0

The reason is reported too, because the three paths are not equally trustworthy: an NVML ranking, a
blind guess at the tail of `/proc`'s list when the sampler has not answered, and a fall back to raw
indices on a driver-less box. Only the first is a measurement. An explicit `CUDA_VISIBLE_DEVICES`
is reported as the user's own choice.

### Ranking now combines utilization and free VRAM instead of letting util dominate

Three separate sort predicates ranked GPUs -- the pre-cuInit NVML pass, `select_best_gpus`'s NVML
fast path, and its CUDA fallback -- and all three used utilization as the primary key with free VRAM
only breaking ties. Any difference in util, however small, therefore outranked any difference in
memory however large: a card 1% busy with 2 GiB free beat an idle one with 95 GiB. For a worker that
wants a registered input slab plus nvCOMP scratch, that is the wrong trade.

It had not bitten, and the reason is the interesting part: NVML was assumed to report 0% on busy
devices, so util tied at zero everywhere and free VRAM decided by accident. **That assumption is
wrong** -- a card driven to saturation reports 100% and was correctly ranked last -- which means the
old rule was one working signal away from making bad choices.

All three sites now call one helper. Each device is ranked twice, independently: utilization
ascending and free VRAM descending, both by DENSE rank so a rank is a position in the preference
order rather than a function of how many devices happen to tie. The device with the lowest sum wins.

**The tie-break was wrong in the first version of this change, and measurement caught it.** Breaking
ties on free VRAM looks natural. With the three 95 GiB cards driven to 100% and the five 81 GiB
cards idle, every device scored 1 -- busy-and-roomy as 1+0, idle-and-smaller as 0+1 -- and that
tie-break handed the job to a fully saturated GPU, which is worse than the rule being replaced. Ties
now go to the idler card. The asymmetry is real rather than a preference: memory beyond what a
worker needs buys nothing, while utilization is contention for the SMs we are about to run on.

Verified across three states, with each device named by UUID so index confusion could not corrupt
the setup: all idle still picks a 95 GiB card; the three roomy cards saturated now picks an idle
81 GiB card rather than a busy 95 GiB one; and a card worse on both axes ranks last.
`GZSTD_DEBUG_GPU_RANK=1` prints the table -- device, util, free, both ranks, and the sum -- because
ranking happens before any log sink exists and its inputs change run to run.

### Names that outlived their meaning

`g_gds_input_fd` was the ordinary O_DIRECT descriptor **both** staged backends read with `pread`;
the name implied a cuFile coupling it has not had since v0.17.9, and it cost an independent
reviewer's attention to point out. It is now `g_stage_input_fd`. The Task fields `gds_off` and
`gds_len` name a region of the input file, which is not a GPUDirect Storage concept at all --
`src_off` and `src_len`. `is_gds()` asked whether a Task carries host bytes, not whether cuFile is
involved -- `is_staged()`. Everything still called `gds_*` is genuinely cuFile-specific: the flag,
the registration, the BAR1 accounting, the tar staging pool, the decompress output handle.

Identifier renames only; verified by diffing a mechanically-renamed copy of the previous source
against this one, whose only surviving difference is the new log line.

## v0.17.10 — the measurement hook had the defect its own finding fixed everywhere else

Second independent review round over v0.17.9. It returned one low-severity finding and no further
portability, data-loss or deadlock defect, which is the falling-severity signal this project treats
as the readiness test: the previous round found something that would have made `--direct-stage` fail
outright on the hosts it exists for.

The finding is in `GZSTD_DEBUG_GDS_FORCE_BOUNCE`, the hidden measurement arm that produced the 95/5
attribution `--direct-stage` was built on. That arm substitutes an ordinary pread and a `cudaMemcpy`
for `cuFileRead`, so it has exactly the exposure the previous round fixed in the production path:
its fan-out threads never called `cudaSetDevice`, and CUDA's current device is host-thread-local, so
they operated on device 0 whatever device owned the slab. It also called `checkCuda`, which throws,
and a throw escaping a `std::thread` calls `std::terminate` rather than reaching the abort and
CPU-only rebuild. Triggered only with the environment variable set and a selected GPU that is not
device 0; the failure was loud rather than silent, which is why it is a low finding and not a
blocking one.

**Worth recording as a pattern rather than an incident: the instrument had the same defect as the
code it was built to measure, and was fixed a round later.** A debug hook that substitutes a
different mechanism inherits every hazard of the mechanism it substitutes.

**The fix is verified by inspection and by the measurement arm still producing byte-identical output;
its actual trigger is NOT exercised.** Reproducing it needs the selected GPU to be something other
than CUDA device 0, and device selection on this host would not move off device 0 even with that
device starved of VRAM. Two attempts to force it failed in ways worth recording, because both are
documented traps in this repo: `cudaMalloc` alone does not commit VRAM (78 GiB "allocated" in 4 ms
with the free figure unmoved, so nothing was starved and nothing moved), and once a committing
version was used, the device it starved was CUDA device 0 while `nvidia-smi` showed the drop on
index 4 -- the NVML-to-CUDA index mismatch this codebase already knows about and correlates by
UUID rather than index everywhere it matters.

### What the class audit found, which is the more useful result

The round was scoped to a class rather than a diff -- *where else does the portable path depend on a
property only a GDS-capable host has?* -- because the previous round's finding was invisible to every
test that can run on this machine, which has all four GDS gates open. Nothing further turned up, and
the negative result is worth writing down:

* Every cuFile entry point is gated on `gds_only`: the preflight, slab registration, file
  registration, and the CLI library check.
* The shared device checksum uses ordinary CUDA allocations, streams and kernel, and that kernel is
  compiled into every nvCOMP build including forward-compatible `compute_75` PTX.
* The 64-frame batch floor and the one-stream default are checksum and pipeline choices, not GDS
  host gates.
* `--direct-stage` teardown reaches `GdsBuf::dereg()`, but its registration pointer is null, so no
  cuFile function is called.
* On a staged-read failure the rebuild forces `pass_opt.cpu_only` and CPU dispatch never enters
  `compress_nvcomp`, so a `direct_stage` flag left set on the rebuilt pass is inert.

One naming wart it surfaced and this release does not fix: `g_gds_input_fd` is now the ordinary
O_DIRECT descriptor shared by both backends, and the name implies a coupling that no longer exists.

### And one question answered by narrowing rather than widening

The previous round gated its `cudaSetDevice` fix on `direct_stage` alone, which looked like it might
be incomplete. It is not, and the reason is checkable: `cuFileRead` receives the base pointer that
was registered on the owning device plus an explicit byte offset into it, so device identity comes
from the registration and the normal `--gds-only` fan-out performs no CUDA Runtime call at all. The
declaration at the top of `GdsFile::read_dev` already says the base must be the registered pointer.

## v0.17.9 — --direct-stage: the 95% of --gds-only that needs none of its hardware

v0.17.7 decomposed what `--gds-only` actually saves. Of 3.73 host CPU-seconds against gzstd's
ordinary reader, 3.55 came from an O_DIRECT read landing directly in the GPU staging slab and 0.19
from peer-to-peer DMA itself. The four gates that feature demands -- a resizable-BAR GPU whose BAR1
aperture covers its VRAM, the nvidia-fs kernel module, a filesystem cuFile accepts, and a kernel
without the shadow-buffer pin regression -- buy the last 5%. `--direct-stage` is the other 95%,
and it needs none of them.

It reads each frame with O_DIRECT into page-locked host memory, pushes it once into the same slab,
and hashes it with the same device-side XXH64 kernel. Everything else is shared: the producer
emitting region-naming Tasks, the 64-frame batch floor, one GPU stream, the implied `--gpu-only`.
A new predicate, `region_staged()`, marks the sites that are about staging rather than about
cuFile, so the two backends differ in exactly one branch in one lambda.

### Measured, one device on every arm, five cold runs, 6 GiB

| arm | host CPU | wall |
|---|---|---|
| ordinary reader | 6.18-7.06 s | 4.90-6.97 s |
| **--direct-stage** | **3.69-3.87 s** | **5.03-5.17 s** |
| --gds-only | 4.59-4.77 s | 5.78-6.00 s |

**About 44% less host CPU than the ordinary reader, and it beats peer-to-peer as well** -- every
range non-overlapping. The flag was only ever meant to come close to `--gds-only` without its
hardware; it is cheaper, because it does not pay `cuFileBufRegister` and does not carry cuFile in
the per-read path at all. v0.17.7 predicted exactly this and said so: *the O_DIRECT arm still pays
cuFileBufRegister, so a native version would likely be cheaper still.*

An earlier draft of this entry reported 4.73-4.82 s and called the difference from `--gds-only`
indistinguishable. That measurement was taken against a binary carrying the registration defect
below, which cost it 761 ms it should never have spent.

**The first version of this measurement said 3.4x and was wrong.** Plain `--gpu-only` defaults to
all eight devices while both staged paths default to one, so the baseline was inflated by device
count -- the same confound that produced a fake 2.4x in v0.17.0 and is recorded as one of the three
in this arc. Every arm above is pinned to one device.

### Byte-identity cannot verify this flag, so the report carries exact bytes

A failed O_DIRECT read throws, and the throw lands in the existing abort -> discard -> CPU-only
rebuild. The archive therefore comes out byte-identical at exit 0 whether the flag did its work or
silently gave up -- the same shape as a cuFile compat fallback. `[DSTAGE]` now reports the exact
staged byte count so a caller can compare it against the input size.

That instrument is mutation-tested: removing the O_DIRECT round-up is caught on all six unaligned
sizes and on none of the three aligned ones, where the round-up is a no-op. Two earlier mutants
survived and were the useful part -- one showed the tail clamp is unreachable on a static file
because EOF bounds the read (it is kept for an input growing under us), and the other showed that a
MiB-rounded report cannot tell 4095 staged bytes from zero.

The unaligned tail is the known trap here: v0.15.63 records O_DIRECT compress reads failing on every
input whose size was not 4096-aligned, hidden for four versions because every generated corpus is
MiB-sized. Eight tail shapes are covered, and this kernel does enforce the rule -- a 4095-byte
O_DIRECT pread returns EINVAL.

### A staged-read failure was announced as a GPU fault

The recovery was already right and already loud: a failed staged read throws into the existing
abort, discard and CPU-only rebuild, and the rebuild line runs at error verbosity so it survives
`-q`. The EXPLANATION was wrong. That line assumed any non-verify rebuild was a GPU fault, so an
O_DIRECT read the filesystem refused produced:

    WARNING: a GPU faulted -- discarding output and rebuilding CPU-only from the original input

No GPU faulted. A user seeing that goes to `journalctl -k` for Xid errors and driver versions, for a
problem that is neither -- while the actual event is that the whole file was silently recompressed
the slow way at correct output and exit 0. Both staged backends now record the real reason, and the
rebuild says so:

    WARNING: --direct-stage: pread failed (Input/output error) at offset 83886080
      Input I/O or filesystem error.  Attempting recovery with a CPU-only rebuild
      from the original input (attempt 1)...

It names the error and the recovery rather than arguing with the reader about what the fault is not.
Every branch of that line must end mid-sentence, because a shared suffix appends
`from the original input (attempt N)...` -- the first version of this one closed its own sentence
and printed `...skip the retry from the original input (attempt 1)...`, which is how the wording got
looked at twice.

`GZSTD_DEBUG_DSTAGE_FAIL_FRAME=N` forces frame N's read to fail, because this path is otherwise
reachable only on a filesystem that accepts an O_DIRECT open and then refuses the read -- which a
test cannot arrange. Verified at all three verbosities, and with the control that matters: a real
GPU fault, injected with `GZSTD_DEBUG_FAIL_GPU_AFTER`, is still reported as a GPU fault. Without
that control the change would be a relabel rather than a discriminator. The flag is reset per output
file, so a multi-file run cannot let one file's I/O error relabel the next file's genuine fault.

### --direct-stage was registering its slab with cuFile -- found by independent review

The flag exists to need no cuFile. It was calling `cuFileBufRegister` anyway, because the
allocation that sets up the shared device checksum was converted to the shared predicate wholesale
and the BAR1 registration sits in the same block. On this host that was merely 761 ms of waste per
run. **On the hosts the flag is FOR -- no nvidia-fs, no resizable BAR, no cuFile-approved
filesystem -- the registration fails, the stream context fails to allocate, and the run dies
printing a message naming `--gds-only`, a flag the user never passed.**

It could not have been caught here. This machine has all four GDS gates open, so the wrong call
succeeds. Registration is now gated on `gds_only` while the checksum allocation stays shared.
Verified by the log line the registration emits: present under `--direct-stage` before the fix at
761 ms, absent after, and unchanged under `--gds-only` at 157-161 ms.

**A consequence worth stating, because it is now a property rather than an intention: with that
call gone, `--direct-stage` never loads libcufile at all.** Checked directly -- the library does not
appear in `/proc/<pid>/maps` for the life of the run, and under a cufile.json with statistics
enabled the flag exits 0 and writes no `cufile.log`, where `--gds-only` on the same config exits 139
in libcufile's own destructor. So `--direct-stage` is structurally immune to the v0.17.8 defect,
not merely unaffected by it in practice. The suite now asserts the library stays unmapped, so a
future cuFile call reintroduced into this path fails a test rather than quietly costing the
portability the flag is named for.

A second defect from the same review: the reader fan-out threads performed the H2D without ever
calling `cudaSetDevice`. CUDA's current device is host-thread-local, so those threads operated on
device 0 regardless of which device owned the slab -- and `--direct-stage` selects one device by
free VRAM, which is frequently not device 0. **The sibling `--tar` assembler already carried the
fix and a comment naming the hazard**, which is the fourth time in this codebase that a defect's
correction was already written in a mirrored path. Related: those threads called `checkCuda`, which
throws, and a throw escaping a `std::thread` calls `std::terminate` rather than reaching the
abort and CPU-only rebuild. CUDA failures there now route through the same error channel as a
failed read.

### Two defects found while wiring it

The input descriptor was closed only under the `gds_only` arm of the teardown, so `--direct-stage`
leaked one fd per run. And the read-error throw prefixed its message with `--gds-only:`, so a
`--direct-stage` failure reported a flag the user never passed.

### Scope

Compression only, and not with `--tar`. The decompress direction of `--gds-only` writes VRAM to
NVMe through cuFile and has no O_DIRECT equivalent to stand in for; `--tar` composes each frame from
many member extents rather than one contiguous region. Both are refused with exit 2 naming the
reason rather than accepted and ignored. `--adapt` files the settled batch under its own
`settled_batch_dstage` key, for the reason v0.17.6 split the GDS key out: one latest-wins key let an
ordinary batch of 8 overwrite a measured 64.

## v0.17.8 — the only way to prove --gds-only did what it says was the one setting that crashed it

`--gds-only` reports `[GDS] N aligned transfers, 0 unaligned` and gzstd has been treating that as
evidence the peer-to-peer path ran. It is not. The counter is computed from alignment alone -- file
offset, device offset and length all on a 4 KiB boundary -- and says nothing about how the bytes
actually travelled. Mutation-testing it settles the point: under `GZSTD_DEBUG_GDS_FORCE_BOUNCE=1`,
which sends every frame through an ordinary host pread and a cudaMemcpy, the same run still prints
384 aligned transfers, 0 unaligned. The `--help` text already said this ("Alignment and the
system-wide Bar1-map counter show eligibility/activity, not definitive per-read routing"); nothing
in the tooling acted on it.

The one per-process discriminator is cuFile's own `posix=` counter, which needs `cufile_stats` set
to a non-zero value in cufile.json. Turning it on segfaulted gzstd at exit, every time:

    Thread 1 "gzstd" received signal SIGSEGV
    #0  libcufile.so.0
    #9  _dl_call_fini (elf/dl-call_fini.c:43)
    #10 _dl_fini (elf/dl-fini.c:114)
    #11 __run_exit_handlers (status=0)

`status=0` -- the compression had already succeeded and the archive is byte-identical. The crash is
in libcufile's ELF destructor, which is where it defers its counter dump.

**This is not gzstd's defect and gzstd cannot fix it.** A 17-line C program that dlopens libcufile,
calls cuFileDriverOpen and returns from main reproduces it with no CUDA, no I/O and no gzstd code;
the same probe linked against libcufile instead of dlopening it exits cleanly. The crash is
conditional on the library having been dlopened, and gzstd has to dlopen it -- a DT_NEEDED entry
resolves before main, so linking libcufile would make the portable binary refuse to START on every
host without GDS installed, which is nearly all of them.

| libcufile loaded via | cufile_stats=0 | cufile_stats != 0 |
|---|---|---|
| linked | exit 0 | exit 0 |
| dlopened | exit 0 | **SIGSEGV** |

What gzstd *was* getting wrong is next to it. The driver was opened and deliberately never closed,
under a comment reading "the teardown cost buys nothing in a process that is about to exit". That
holds only while statistics are off. With them on, cuFileDriverClose is what flushes the counters,
and skipping it meant they were never written at all -- so the one measurement that could confirm
the feature works was unobtainable. Same config, same data, only the binary differing:

| binary | exit | cufile.log | Read counters |
|---|---|---|---|
| before | 139 | 10.5 KB | none |
| after | 139 | 12.7 KB | `n=768 posix=0` |

The exit code is unchanged and will stay 139 until NVIDIA fixes the destructor. The counters now
exist, and `posix=0` across 768 reads is the first direct confirmation in this repo that
`--gds-only` transfers are not silently taking the POSIX compat path.

The driver is closed from a scoped guard in main, so it runs on the exception paths too and always
before exit() reaches the loader's finalizers.

**Also: `--gds-only` still has no automated suite coverage** (it needs hardware the suite cannot
assume) and this release does not change that. It does replace the pre-tag round-trip script, which
was written at v0.15.40, hardcoded that version, and could not run at all -- it referenced an
undefined variable four times and died under set -u before its first check.

## v0.17.7 — 95% of what --gds-only saves is O_DIRECT, not peer-to-peer DMA

v0.17.6 recorded `--gds-only` cutting host CPU 44% against `--gpu-only` on a cold cache. That number
is real, but it compares peer-to-peer DMA against gzstd's ORDINARY reader, which goes through the
kernel read path. It never asked what the peer-to-peer part specifically contributes.

`GZSTD_DEBUG_GDS_FORCE_BOUNCE` now also covers the non-tar read site, whose offsets and lengths are
whole MiB by construction and therefore genuinely peer-to-peer eligible (178 aligned transfers to 1
unaligned, that one being the final partial frame). That makes the three-way comparison possible on
one file, one geometry, cold cache, interleaved:

| arm | wall | host CPU |
|---|---|---|
| gzstd's ordinary reader | 4.70-4.81 s | 7.55-7.79 s |
| O_DIRECT pread + H2D into the same staging slab | 4.62-4.74 s | 4.05-4.19 s |
| peer-to-peer cuFileRead | 4.62-4.72 s | **3.90-3.97 s** |

Wall time is identical across all three; every range overlaps. The CPU decomposition is not:

* total saving over the ordinary reader: **3.73 s**
* attributable to O_DIRECT reads straight into the staging slab: **3.55 s (95%)**
* attributable to peer-to-peer DMA itself: **0.19 s (5%)**

**So the four gates this feature needs -- a resizable-BAR GPU whose BAR1 covers its VRAM, the
nvidia-fs module, a filesystem cuFile accepts, and a kernel without the pin regression -- buy the
last 5%.** The other 95% needs none of them: it is an O_DIRECT read landing directly in the buffer
the compressor already owns, which any host can do. The `odirect` arm above still pays
cuFileBufRegister and the staging setup, so a native implementation would likely be cheaper still.

This does not retract the v0.17.6 measurement; `--gds-only` does roughly halve host CPU against the
ordinary path. It reattributes it. And it settles the tar scratch region for good: that design exists
to turn unaligned extents into aligned ones, and aligned-versus-unaligned is now measured at 4% of
host CPU with a small wall-time penalty, against read amplification that reaches 41x on a
100-byte member.

The lesson is the one this feature keeps teaching: **a control arm has to be the thing you are
actually claiming to beat.** Three times in this arc a conclusion rested on a comparison that was not
measuring what it appeared to -- a device-count confound, a page-cache confound, and now an
attribution confound.

## v0.17.6 — --adapt filed a GDS batch under the ordinary key

Round 3 of the Codex review, scoped OPEN over its own round-2 edits as shipped. It returned one
medium finding, no data-loss and no deadlock, and confirmed the four pieces of reasoning I had
flagged as unverified. Severity across the three rounds went data-loss/deadlock, then five of eleven
fixes rejected, then this.

### The finding

`--adapt` kept one `settled_batch` key for both the ordinary GPU path and `--gds-only`. They do not
have interchangeable batch geometry: GDS registers its input slab through BAR1 and runs the device
checksum, and the ordinary path does neither. v0.17.5 stopped GDS from CONSUMING the shared key,
which cured the poisoning but also made GDS discard its own measured verdict on every run.

Split into additive `settled_batch` and `settled_batch_gds`, matched across save, load,
driver-change invalidation and prior application. Additive, so no schema-epoch bump; an older
profile simply lacks the GDS key and the cold start applies.

Measured on this host after the split: the ordinary path settles at **4**, `--gds-only` at **64**.
That 4 is precisely what had been overwriting the GDS starting point.

The stamp is applied in `adapt_profile_save`, where the Options and the observation are both in
hand — not at the call sites. There are five, and the first attempt at this set it at two of them,
which turned out to be `--tar` extract and `-t` rather than plain compress. An observation that
reaches the profile without knowing which geometry produced it is the whole bug.

### Reasoning confirmed, and framing corrected

* `adopt_external_prefix`: the happens-before holds. The flag and `plain_fd_v_` are written under
  `mx_`, every op is enqueued under the same mutex, and each drain locks it before taking an op.
* Staging slots: liveness holds, but "every slot is explicitly released" was too strong of me. On
  terminal abort, tasks abandoned in the reorder map are not individually released; that is safe
  only because `set_done()` wakes all acquirers and the pool outlives the producer and worker joins.
* `/proc/self/fd/N` is identity-stronger, not unconditionally stronger: it adds a procfs and
  current-permission dependency `dup` does not have. Input fails loudly, output demotes.
* The staging backoff carries no partial state between attempts: allocation failure leaves the base
  null, registration failure frees it, and the free list is populated only after complete success.
* Checksum overlap remains permitted but unproven. `ev_comp_end` is recorded after the join, so it
  cannot distinguish `max(checksum, compress)` from their sum.

### Still open

* **The GDS cold start is still a machine-tuned constant.** Batch 64 comes from this box's
  0.28 GiB/s-per-frame checksum and 4.9 GiB/s drive. The new per-path prior lets a measured value
  replace it after a qualifying run, but the cold start itself remains against the house rule.
* `--tar --gds-only` is still mostly not peer-to-peer. See the measurement below for what that is
  now known to cost, and why the obvious fix is not cheap.

### Measured, then fixed: an unaligned cuFileRead costs more than reading it ourselves

`--tar` puts a 512-byte header before every member, so member data does not land on the 4 KiB grid
and cuFile cannot route it peer-to-peer. It does not fail on such an extent -- it degrades
internally. The question was whether that degradation costs anything, and whether the aligned
device-scratch region sketched in the round-2 review was worth building to avoid it.

Fixture: 40 members of exactly 16 MiB - 512 bytes, so every tar entry occupies exactly 16 MiB and
every extent sits 512 bytes off the grid -- **0 aligned, 80 unaligned** -- with no tiny-file
registration overhead to confound it. Two arms on identical geometry (same inode, offsets, staging
buffers, one GPU, one stream, batch 32, output to /dev/null), cache dropped before every rep, 3
interleaved reps, `GZSTD_DEBUG_GDS_FORCE_BOUNCE=1` selecting the read-it-ourselves arm. Both arms
are checked byte-identical, so only the routing differs.

| host CPU, default readers | cuFile | read ourselves |
|---|---|---|
| 40 x (16 MiB - 512) | 4.33 s | **3.51 s** |
| 6 x 100 MB (unaligned file offsets) | 3.91-4.03 s | **3.47-3.54 s** |

Wall time is indistinguishable throughout (3.5-3.6 s, fully overlapping). The CPU ranges do not
overlap: handing cuFile an extent it cannot route costs **12-23% more host CPU for no wall-clock
benefit**. Unaligned extents are now routed deliberately, and the measured result on the same
fixtures is 4.33 -> 3.59 s and 3.92 -> 3.63 s, tracking the read-it-ourselves arm.

**THE FIRST ATTEMPT AT THAT ROUTING CORRUPTED ARCHIVES**, and the reason is the point. A registered
member is open O_DIRECT precisely so cuFile can do peer-to-peer, and O_DIRECT cannot service an
unaligned pread either: every extent short-read, reported "file changed as we read it", and left the
frame's zero fill in place. The tar stream stopped matching the ordinary path at exactly a chunk
boundary. Reading correctly means reading an ALIGNED WINDOW into an ALIGNED buffer and copying the
interior -- round the offset down, round the length up, copy from the middle.

That is the same machinery the review proposed putting on the DEVICE side as a scratch region. On
the host it is about fifteen lines, and it captures the entire measured win. **So the scratch region
is not deferred, it is unnecessary**: there is no remaining gap for it to close on this hardware,
where wall time is set by the drive and the only cost in question was host CPU.

**The measurement instrument had the same defect as the first fix, and nearly produced a reported
number.** `GZSTD_DEBUG_GDS_FORCE_BOUNCE` corrupted the 100 MB fixture while producing correct output
on the 16 MiB-512 one -- because that fixture's *file* offsets happen to be aligned even though its
device offsets are not. The fixture the experiment was designed around was the single case where the
broken control looked healthy. It is caught now by asserting both arms produce byte-identical
archives as part of the measurement, which is the check that found it.

Verified: extensive suite **548/548** in 8m36s and the CPU-only build **335/335** in 1m26s, both
configurations compile clean, plus the full GDS trigger set from v0.17.5.

## v0.17.5 — the review round that rejected five of the previous round's fixes

Round 2 of the independent Codex review, scoped to adjudicate its own earlier findings rather than
re-review the code. It rejected **five of eleven** fixes as incomplete and one of my claims as
factually wrong. Three of the rejections were reproduced here before adopting anything.

### Two more silent data-loss paths, both now closed

* `-d --gds-only --preallocate` produced a **0-byte file at exit 0**. Peer-to-peer writes leave
  `DirectWriter::logical_written_` at zero, so finalisation saw a completed file as unwritten and
  truncated it. DirectWriter can now adopt an externally written prefix as its logical cursor --
  without punching a hole, unlike `seek_forward()`.
* Decompressing several files in one run, where an earlier file used GDS and a later one declined
  it, left the later file inheriting `active=true`: it suppressed its own writer and failed at
  exit 2 with its output removed. Per-file state is now reset unconditionally on entry and cleared
  during teardown.

### A comment of mine was simply false

The output-descriptor adoption carried a comment claiming that toggling `O_DIRECT` on a `F_DUPFD`
duplicate leaves the caller's `FILE*` alone. `F_DUPFD` shares one open file DESCRIPTION, so it does
no such thing -- confirmed against the kernel, where setting the flag on the duplicate flips it on
the original. That is what made `--no-direct` demotion write unaligned through a descriptor that had
silently become direct.

Both the input and output descriptors are now reopened through `/proc/self/fd/N`, which gives an
independent open description AND cannot resolve a substituted pathname. That is strictly stronger
than the `st_dev`/`st_ino` verification it replaces: the held descriptor selects the inode, so the
identity check is unnecessary rather than merely redundant.

### The staging deadlock was still live

`queue.set_max_depth()` sees the queue but not the tasks held in the tar assembler's reorder map.
The interleaving: reader 0 claims sequence 0 and is descheduled before acquiring a slot; later
readers consume every slot and deposit sequences 1..N into the reorder map; the pusher waits for the
missing head; the queue never reaches its cap, so the depth escape never fires.

Readers now acquire a staging slot BEFORE claiming a sequence number. This is the same head-of-line
shape as the multi-reader deadlock closed in v0.15.66-67, reintroduced in a new pool.

### The sizeless tail did not need refusing after all

v0.17.4 turned a silently vanishing tail into a usage error, on the reasoning that a peer-to-peer
prefix could not be handed to the streaming decoder. That was wrong: adopting the prefix as
DirectWriter's logical cursor, and routing an unaligned continuation through an independent plain
descriptor, appends correctly. The combination works now instead of being rejected.

### Also

* The staging slab was allocated on whatever device the main thread happened to leave current.
  Device probing can finish on the last CUDA device while GPU ranking selects another, so the
  worker's device-to-device copy could cross devices with no peer access. Allocation now selects the
  chosen device explicitly.
* `--adapt` with a profile carrying `settled_batch=8` overwrote the GDS compress batch of 64 after
  it was set. GDS is now excluded from that prior.
* The fixed 32-slot staging pool was a machine-tuned constant justified with this host's checksum
  and drive rates -- against the house rule in `AGENTS.md`. It now starts from the requested batch
  geometry and backs off only on real allocation or registration failure.
* The adaptive reader controller charged `t.data.size()`, which is zero for every staged GDS task,
  so `--tar --gds-only` fed the reader governor nothing but zeroes. It uses `t.len()` now.
* A sizeless FIRST frame returned before GDS teardown, leaking the cuFile handle and descriptor and
  leaving state set for the next file.
* The counters called alignment eligibility "peer-to-peer" and tar member extents "frames", and a
  final partial plain-file read counted as peer-to-peer despite an unaligned length. They now
  classify aligned versus unaligned, and the report no longer implies that alignment or a box-wide
  BAR1 delta proves routing.

Codex also shipped one bug of its own: the input-reopen edit dropped `opt.input` from a string
concatenation, leaving `const char[] + char*`, which failed to compile.

Verified: extensive suite **548/548** in 8m35s and the CPU-only build **335/335** in 1m26s, both
configurations compile clean. Every trigger above was reproduced before the fix and re-tested after,
along with the staging deadlock at `--gpu-batch` 64 and 256, a 20000-member tar, and `--adapt` with
and without `--tar`. `--tar` archives remain byte-identical to the ordinary path.

Still open: `--tar --gds-only` remains mostly not peer-to-peer. The scratch-region design for edge
blocks is specified but deliberately not implemented -- the deciding input is what cuFile's bounce
path costs against an ordinary pread, which has not been measured, and a 100-byte member would see
roughly 41x read amplification if the answer is wrong.

## v0.17.4 — the rest of the review, and the discovery that --tar --gds-only mostly is not peer-to-peer

Follow-up to v0.17.3, working through the remaining findings of the independent Codex review.

### --tar was reporting peer-to-peer transfers it was not making

The counter said every successful `cuFileRead` was a peer-to-peer transfer. It is not: the device
offset has to sit on a 4 KiB boundary, and tar puts a 512-byte header in front of every member, so
member data lands off the 4 KiB grid and cuFile routes it through a bounce while still returning
success. Instrumenting the offsets actually passed:

| archive | reads | unaligned |
|---|---|---|
| 20000 small members | 20000 | **20000 device offsets** |
| 6 x 100 MB members | 40 | **28 file offsets** |

So `--tar --gds-only` is substantially NOT peer-to-peer today, and the reporting said otherwise.
It now counts only reads cuFile can actually route peer-to-peer, and says so at default verbosity:
`6 frames peer-to-peer, 35 bounced through host memory`. The non-tar path is unaffected -- its
offsets are whole MiB by construction -- and still reports zero bounced.

Making tar genuinely peer-to-peer needs an aligned device scratch region for the edge blocks. That
is not in this version.

Worth recording that cuFile's own routing counter could not be used to establish this: setting
`cufile_stats: 3` segfaults libcufile inside gzstd's threading, though the env var alone is
harmless. The offsets had to be measured directly.

### A sizeless tail after a peer-to-peer prefix now fails loudly instead of vanishing

An archive whose later frames carry no content-size header falls back to the CPU streaming decoder
for the remainder. Under `--gds-only` that tail **silently never appeared** -- 100 MB of a 130 MB
output, exit 0.

It resists a small fix. Peer-to-peer writes are absolute-offset and never advance the descriptor,
and repositioning it is not enough: when DirectWriter owns the output it tracks its own logical
offset, so an `lseek` on the fd changes nothing, and its `seek_forward()` cannot be borrowed because
that PUNCHES A HOLE over the range -- which here is the prefix GDS just wrote. So this is now a
usage error naming the workaround, and `die()` removes the incomplete output so there is no
half-file to mistake for a good one.

### Correctness

* The plain GDS producer reopened `opt.input` by name, reintroducing the ABA window the `--rm` work
  closed: the path can be replaced between the caller's open and this one, so GDS would archive one
  file while `--rm` deleted the identity recorded for another. The reopened descriptor is now
  verified against the already-open one. Verified rather than adopted, because dup'ing shares one
  open file description -- setting `O_DIRECT` on the dup would set it for the caller's `FILE*` too,
  and a later buffered read through it would start failing on alignment.
* Assembler threads never called `cudaSetDevice`. CUDA's current device is thread-local, so they
  operated on device 0 whatever device owned the staging slab. Unreachable today, since the path is
  single-device and that device is 0 -- which is the point of writing the invariant down.
* The assembler parks in `GdsStagePool::acquire()` waiting for slots that dead workers will never
  release. It now gets the same abort wake `DirectReadPool` has, for the same reason.
* The peer-to-peer decompress stride is rounded to 4 KiB, not merely its file offset: slot i sits at
  i * alloc_decomp, so an unaligned stride puts every slot after the first off the boundary. The
  rounding also reserves the padding the last frame's write needs.
* `-o /dev/null` decompress became fatal when O_DIRECT was made mandatory, because /dev/null refuses
  it. It demotes to the ordinary writer now, like every other destination cuFile will not take.
* The degradation warning printed "bounced through host memory0 members cuFile would not take" when
  nothing had been refused; the optional clause is now built separately.

Verified: extensive suite **548/548** in 8m27s and the CPU-only build **335/335** in 1m27s, both
configurations compile clean. `--tar` archives remain byte-identical to the ordinary path.

## v0.17.3 — --tar --gds-only, and the benchmark that said the whole thing was pointless

`--tar` creation now works under `--gds-only`. A tar frame cannot be one file region -- it
interleaves member data with headers gzstd synthesises -- so the assembler composes each frame in
registered VRAM instead: `cuFileRead` for member extents, a small host copy for the headers, and a
device-to-device copy into the compressor's slab. Extraction is still rejected, for a different
reason that was checked rather than assumed: it writes a whole tree, while GPUDirect Storage
addresses one registered file handle by offset.

Archives are byte-identical to the ordinary `--tar` path, for large files and for 20000 small ones.

### The performance claim in v0.17.0 was measured wrong

v0.17.0 reported `--gds-only` about 2.4x faster than `--gpu-only`. That compared one GPU against
eight, and eight is much slower on this box. Corrected for device count it inverted: GDS looked
1.5x SLOWER. Both numbers were wrong, and the second one for a better reason -- **the input was
100% resident in page cache**, so GDS was reading NVMe while the ordinary path read from RAM.

Cold cache, storage-equivalent, 2.79 GiB, 3 interleaved reps, one device and one stream on both:

| | wall | host CPU |
|---|---|---|
| `--gds-only` | 4.57-4.94 s | **3.90 s** |
| `--gpu-only -d1 -s1` | 4.71-4.88 s | 6.98 s |

Wall ranges overlap -- no measurable difference -- and host CPU is **44% lower**. The ordinary
path's CPU nearly doubles when the cache is cold (3.50 -> 6.98 s); that is the kernel read path,
and it is exactly the term GDS removes. On warm input the comparison is not storage-equivalent and
should not be used.

### Two bugs were most of the gap, and neither was structural

* The per-stream tuner started at `DEFAULT_GPU_BATCH_CAP` (8) while the slab was registered for 64,
  so batches averaged 5.6 frames. The GPU checksum's throughput is ~0.28 GiB/s PER FRAME IN THE
  BATCH, so a 5.6-frame batch hashes at ~1.6 GiB/s and cost ~55 ms every batch. Registering 64
  slots and filling 8 of them was the worst of both. Batches are now ~45 frames over 4 batches.
* The checksum kernel was given its own stream so it could overlap the compressor, and it did not
  overlap at all: the readback targeted a pageable `std::vector`, and `cudaMemcpyAsync` to pageable
  memory is SYNCHRONOUS with respect to the host, so the submitting thread blocked on the hash
  before it could enqueue nvCOMP. The stream was fine; the destination was the problem. The
  checksums now land in page-locked memory.

### Correctness

* `-d --gds-only -f` over an EXISTING file wrote a 0-byte file and exited 0. Overwriting goes
  through an atomic `<out>.gzstd.*.tmp`, so opening `opt.output` by name wrote the data to the final
  name and then let the rename install the untouched temp on top of it. The output descriptor is now
  adopted from whoever owns it -- and `out` is NULL whenever DirectWriter adopted it, which is the
  default here, so `fileno(out)` segfaulted until that case was handled. Same
  path-resolved-twice window `DirectWriter::adopt_fd` already existed for.
* cuFile refuses some valid destinations, including that atomic temp. Rather than fail the job, an
  output it will not take now falls back to the ordinary writer: the user asked for a decompressed
  file, and the acceleration is the negotiable part.
* Per-file GDS output state is reset. A second, shorter file inherited the first file's high-water
  mark and was ftruncated up to it.
* Three CUDA calls in the tar assembler ignored their return status. That is a silent-corruption
  path rather than a missing diagnostic: **the frame's checksum is computed on the GPU from the same
  buffer after it is filled**, so a failed zero-fill or header copy yields stale VRAM with a checksum
  that MATCHES the stale bytes -- passing `zstd -t` and every round-trip check gzstd has.
* `--tar --gds-only` with more than one GPU is rejected. The staging pool is one allocation
  registered on one device; the forced `--gpu-devices=1` hid it, but an explicit override made a
  worker copy device-to-device from another device's pointer with no peer access.
* A 64-frame batch against a 16-slot staging pool deadlocked: the worker waited in
  `wait_for_batch_or_cap` for frames the assembler could not supply while the assembler blocked on
  slots that could not come back. The queue is now told the staging depth, and the pool is 32 slots.

`--gds-only` degradation is now reported at DEFAULT verbosity, not under `-v`. A compat fallback
returns correct bytes at nearly the same speed, so a degraded run is otherwise indistinguishable
from a healthy one; it now says how many frames went peer-to-peer, how many bounced, how many
members cuFile refused, and whether the nvidia-fs `Bar1-map` counter ever moved.

Six of these came from an independent Codex review, including the two efficiency bugs and the
`-f` data loss. It also supplied the page-cache objection, in a DISAGREEMENTS section that only
exists because the review scope asks for one.

Verified: extensive suite **548/548** in 8m32s and the CPU-only build **335/335** in 1m31s, both
configurations compile clean. The CPU-only run failed first on the endian source guard, which
flagged `cudaHostAlloc(&ptr, sizeof(unsigned int) * n)`. That one is genuinely what the ALLOW list
is for -- the same out-param allocator shape as `cudaMalloc` beside it -- so it was exempted by
callee with a reason, and the widened exemption was then mutation-tested to confirm the guard still
catches a real `pread(fd, &magic, 4, 0)`.

`--gds-only` still has NO suite coverage: it needs hardware the suite cannot assume, so everything
above rests on hand verification. That is why it remains EXPERIMENTAL and out of the short `-h`.

## v0.17.2 — what --gds-only did on a host that cannot run it, which is most of them

The mode needs a GPU whose BAR1 aperture covers its VRAM, the nvidia-fs module, a filesystem cuFile
accepts, and a kernel clear of the pin regression. Most hosts fail at least one of those, so the
common case is not the happy path — it is a host that *nearly* qualifies. v0.17.1 handled that
badly, and this fixes it.

### And what --gds-only does on a host that cannot run it

The likeliest configuration in practice is a host that *nearly* qualifies, and v0.17.0 handled it
badly. Forcing the condition (capping cuFile's pinnable device memory reproduces a BAR1 aperture too
small for the slab) produced:

    gzstd: --gds-only: cuFileBufRegister failed (err 5016)      ... x7
    [GPU0] insufficient VRAM for even 1 stream at batch=1 - skipping device
    WARNING: all GPUs failed; finishing compression on CPU (96 threads).
      Falling back for data safety: ... so the output is complete and correct
    gzstd: ERROR: ZSTD error: Src size is incorrect
    exit=4

Three separate failures compounding. The retry blamed VRAM, when err 5016 is
CU_FILE_INVALID_MAPPING_SIZE and the real constraint is the BAR1 aperture. The all-GPUs-failed path
then handed the queue to the CPU rescue pool, announcing that the output would be complete and
correct. And the pool inherited --gds-only Tasks, which carry NO host bytes -- len() is the frame
length but ptr() is an empty vector -- so it read undefined memory, caught only because ZSTD noticed
a size mismatch. The run then exited 4, a DATA error, blaming the user's input for what was purely a
host configuration problem.

Now: a preflight registers a small device buffer before any task is queued, so a host that cannot do
GDS at all fails immediately with the actual reason and exit 2. The probe deliberately does not try
to prove the full slab fits -- it separates "cannot do GDS" from "this batch is too big", which the
setup retry already shrinks for, so a small-BAR1 card that can serve a modest batch still runs. If
the GPU path fails anyway, gpu_only_cpu_fallback now refuses the job rather than mis-serving it, in
both directions: compress Tasks have no host bytes, and decompress bypasses the ordered writer, so
nothing would drain the ResultStore the pool pushes to. The CPU worker carries a matching assertion
for a case that should now be unreachable. Every one of these exits 2, because the remedy is the
same in all of them: this host cannot run --gds-only.

The pipe case is unchanged in behaviour but now exits 2 rather than 1, and names the command that
hits it: tar -I with --gds-only, or any shell pipe. cuFile rejects a pipe or FIFO at registration
with CU_FILE_INVALID_FILE_TYPE, before any transfer -- it needs a regular file, because the drive
DMAs into a BAR1 window for a block range of one. A stream assembled in host memory has already made
the trip GDS exists to avoid.

Verified: extensive suite 548/548 in 8m36s and the CPU-only build 335/335 in 1m33s, both
configurations compile clean.

The CPU-only run earned its keep again, failing 334/335 on a source guard: check-endian-reads.sh
flagged size_t(4) << 20 in the new preflight as a possible host-order read of an on-disk field. A
false positive — it is just 4 MiB — but the fix is to stop writing the pattern rather than widen the
checker's ALLOW list, so it reads 4 * ONE_MIB now. A lint exemption that grows every time it fires
stops being a lint.

## v0.17.1 — --gds-only decompress wedged at 97.8%, because bypassing the writer leaked its permits

v0.17.0 shipped a deadlock. `--gds-only -d` on a 65 GiB archive stopped dead at 97.8% with every
thread parked in a futex and no forward progress. It was never released — no v0.17.0 tag was cut, so
the portable build never ran and no host auto-installed it — but the commit is on main, so this is
the fix on top rather than a rewrite of it.

The decompress intake acquires one FrameThrottle permit per popped frame, and the only thing that
hands them back is the writer's AsyncWritePool, one per frame it writes. v0.17.0 bypasses the writer
entirely, because peer-to-peer writes go to absolute offsets and need no ordering — and in doing so
it inherited that release and never performed it. Every frame leaked a permit, so the run wedged the
moment the throttle was exhausted: **8156 frames delivered against an 8192-frame throttle**, which
is exactly the 97.8% it stopped at. Anything smaller never reaches the limit, which is why it
survived every test written for the feature and took a real archive to surface.

The permit is now released at the write site, where the frame is on disk — precisely the condition
the writer releases on.

### Bypassing a component means inheriting all of its responsibilities

They have to be enumerated, not discovered. writer_thread does four things that matter here:
releases throttle permits, updates wrote_bytes, updates tasks_done, and truncates the output at
close. Three were handled when the writer was bypassed. The one missed was the only one with **no
visible effect until a queue fills** — the failure mode is invisible at every size a test would
plausibly use. The rest of writer_thread has since been audited; there is no fifth.

### And a diagnosis failure worth more than the bug

The permit-leak theory was correct on the first pass and was discarded because the test written for
it passed. That test paired 2862 frames with the *default* 8192-frame throttle, and separately
paired a floored 2048-frame throttle with a 36-frame file. Neither combination can exhaust anything,
so it could not have failed. Reproducing the wedge needs both knobs at once: `--throttle-frames=1`
(which floors to 2048) against an archive with more than 2048 frames, which `--chunk-size 1` builds.
**A passing test only disproves a theory if the test could have failed.**

Verified: extensive suite 548/548, CPU-only build 335/335, both configurations compile clean. The
reproduction above hangs on v0.17.0 and completes byte-identical on this build; compress was checked
for the same leak class and does not have it, because its writer is still spawned.

## v0.17.0 — --gds-only: the drive writes into VRAM, and the CPU never sees the bytes

GPUDirect Storage moves the input from NVMe into GPU memory by peer-to-peer DMA, so the
uncompressed bytes never enter host memory. `--gds-only` implies `--gpu-only` — with no host copy
there is nothing for a CPU worker to compress, so the split is not a policy choice, it is
unrepresentable. Both directions are covered: compress reads NVMe → VRAM, decompress writes
VRAM → NVMe.

**Shipped as EXPERIMENTAL** and omitted from the short `-h` listing. Two reasons beyond novelty:
the mode has no automated test coverage, because it needs hardware the suite cannot assume; and it
can degrade invisibly, since a cuFile fallback to a host bounce buffer runs at 4.917 GiB/s against
4.924 for the real thing. `-v` prints the nvidia-fs `Bar1-map` delta so a run can be checked.

The path is narrow on purpose. It needs a GPU whose BAR1 aperture covers its own VRAM (resizable
BAR; consumer cards are typically 256 MiB and can never qualify), the nvidia-fs kernel module, a
filesystem cuFile accepts, and a kernel without the shadow-buffer pin regression.

Rejected as usage errors: --cpu-only and --hybrid, stdin input, stdout output, a build without
nvCOMP, and --tar in either direction — creation assembles the archive as it is written so there is
no file to read, and extraction writes a tree rather than the single registered file GDS addresses.

### What it buys — not throughput

Both paths saturate the same drive. 572 MiB, 5 interleaved repetitions, server class host:

| | wall | host CPU |
|---|---|---|
| compress `--gds-only` | **3.62–4.35 s** | **3.29 s** |
| compress `--gpu-only` | 9.66–9.77 s | 10.19 s |
| decompress `--gds-only` | **3.62–4.25 s** | **4.64 s** |
| decompress `--gpu-only` | 9.71–9.77 s | 11.02 s |

Ranges do not overlap. Against raw `gdsio` on the same host the peer-to-peer read cost 0.49 host
CPU-seconds per GiB against 0.62 for the ordinary path, with *user* CPU falling about 6x — the copy
through host memory stops happening. On an idle machine this measures as very nearly nothing; the
win is contention resilience.

### The content checksum had to move to the GPU

Every gzstd frame carries zstd's XXH64 content checksum, which is what makes a GPU archive
self-verifying. It was computed on the host during H2D staging, from a buffer that no longer
exists. Hashing on the host would mean copying every byte back over PCIe — exactly the traffic this
feature removes — so the hash now runs where the data already is (gzx_xxh64_kernel in gpuverify.cu).

**Its parallelism is capped at four per frame, and that is the algorithm's limit, not the
implementation's.** XXH64 keeps four accumulators, each a strictly sequential chain
acc = rotl(acc + in*P2, 31) * P1, and rotl does not distribute over multiplication, so no closed
form composes two stripes. The chain cannot be split, tree-reduced or skipped ahead.

The useful consequence is a clean model: wall time is set by the CHUNK size alone and is nearly
constant in frame count (13.8-14.0 ms at 4 MiB from 8 to 512 frames; 54.6-56.5 ms at 16 MiB), so
throughput is ~0.28 GiB/s per frame in the batch, chunk-independent and linear to 512 frames:

| frames | 8 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| GiB/s | 2.3 | 9.0 | 17.8 | 35.4 | 71.2 | 144.3 |

Against a ~5 GiB/s drive that needs roughly 18 frames per batch. Do not size this kernel from a
small batch and conclude it is slow.

**Tried and it did nothing: unrolling the main loop 8x for memory-level parallelism** (57 → 56.4 ms,
noise). The prediction was that dependent-load latency was going unhidden. It was wrong — this is a
genuine dependency chain of emulated 64-bit multiplies at ~150 cycles per round. The unroll is
harmless and stays; the experiment does not need repeating.

Correctness is checked by GZSTD_DEBUG_XXH_SELFTEST=1, which compares the kernel against the CPU
xxh:: implementation over every length from 0 to 200 (exhaustive across the tail logic) plus 17
large sizes. It was mutation-tested four ways; the lane-2-seed mutation leaves exactly 32 of 201
lengths matching, which are lengths 0-31 — the ones that skip the accumulator path entirely.

### One device, one stream

cuFileBufRegister is what maps VRAM through BAR1 so the drive can target it, and it is expensive:
~600-950 ms for a 4 GiB slab uncontended, degrading to ~3.9 s with several running at once. Since
this path is disk-bound at ~4.9 GiB/s and one server-class GPU compresses well above that, a second
device cannot make the job faster, only slower to start. Wall time scaled almost linearly with
device count — 5.1 s at one device, 30.9 s at eight — while the reads stayed near 4.9 GiB/s
throughout. Defaults are one device, one stream, batch capped at 64, all overridable.

### The defect worth recording: silent truncation

Decompress writes at absolute offsets, so frames need no ordering and the ordered writer is
bypassed. It cannot merely be starved: the writer ftruncates the output to what it personally
wrote, so handing it empty frames would delete everything GDS had written.

Bypassing it made gzstd own the final length, which recreated the same hazard one level up. An
archive whose frames are not 4 KiB multiples tripped a mid-run alignment throw; the GPU-abort path
caught it and rebuilt the whole output correctly on CPU — and then the finalising ftruncate cut
that correct output back to the GDS high-water mark. A 30 MB archive came back 1000003 bytes long
with exit 0.

The fix settles the shape before the output is touched: peek the first frame, and if its
decompressed size is not a multiple of 4 KiB, quietly use the ordinary writer instead. Workers read
the resolved decision, never the flag. The per-frame check survives as a die_data(), never a throw,
so it can never race a fallback that owns the same file. A mid-run failure that hands a half-written
file to a rebuild path writing the same file is not a recoverable error, it is a corruption
mechanism.

### Fixed in passing: out_off was always zero on the parallel reader

stream_frames_to_queue_mt never set Task::out_off; the serial stream_frames_to_queue always had.
That reader is chosen for any seekable input over 128 MiB, --keep-going included — and --keep-going
prints out_off .. out_off + decomp_size for each unrecoverable region, so every damaged range has
been reported as starting at offset 0. Fixed with the same running prefix sum the serial path uses.

Verified: extensive suite **548/548** (8m40s) and the CPU-only build **335/335** (70 GPU tests
skipped, 1m30s), both configurations compile clean. The GDS path itself has **no suite coverage** —
it needs hardware the suite cannot assume — so it was verified by hand: round-trip byte-identical
through gzstd, stock zstd -d and the CPU-only build, zstd -t validating the GPU-computed checksum,
and 11 edge sizes from 0 to 100 MB.

## v0.16.9 — --adapt senses the core budget, because "the box is busy" is the wrong signal

The GPU backend wins on a machine that is short of CPU cores, and loses on one that is not. That
much was assumed. What was not known is *where* the line sits, or what signal finds it — and the
obvious candidate, "is the box busy", turns out to be wrong.

### The measurements

195.3 GiB from tmpfs, cores varied with `-T`, best of 2:

| cores | 4 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|
| `--cpu-only` | 30.62 s | 15.11 | **10.12** | **7.78** | **5.75** |
| `--hybrid` | 17.11 | **13.69** | 11.94 | 9.55 | 8.11 |
| `--gpu-only` | **14.71** | 14.21 | 14.51 | 14.21 | 13.02 |

`cpu_time × cores` is nearly constant — 122, 121, 121, 124, 138 core-seconds — so the CPU backend
scales as ~K/cores, while the GPU backend is **flat**. The crossover is therefore K_cpu/K_gpu
≈ 8.7 cores, and the winner really does change between 8 and 12.

### Why "busy" is the wrong signal

A full contention sweep, with load supplied by concurrent gzstd instances, found **no crossover at
all**: every backend degraded by the same factor (1.8 / 1.7 / 1.76x) up to 73% CPU busy. The
reason is that a 256-core box cannot starve gzstd of cores — even at 87% busy, ~33 remain, far
above the crossover.

Worse, "busy" conflates two different things:

| load type | `--cpu-only` | `--gpu-only` |
|---|---|---|
| memory-heavy (gzstd compressing), 73% busy | 1.8x slower | **1.7x slower** |
| core-only (arithmetic spinners), 87% busy | 1.9x slower | **1.23x slower** |

The GPU path is largely immune to **core** contention and fully exposed to **memory-bandwidth**
contention, because its H2D staging competes for the same bandwidth the CPU compressors use. So a
busy-ness trigger predicts the wrong thing in one of those two cases. Available **cores** predicts
both.

### What changed

`adapt_avail_cores()` reports the cores a run can actually expect: the affinity mask (taskset,
cgroup, container), reduced by the current load as a *fraction*. Contention is applied as a
fraction and never subtracted — a first version subtracted a system-wide load average from a
per-process affinity count and reported **1 core available instead of 8** under `taskset -c 0-7`,
because it charged the whole machine's load against an 8-core allowance.

The profile now records `overall_gibs_cpu_cores` beside each cpu-only rate, because that rate is
meaningless without the budget that produced it. At decision time the stored rate is rescaled to
this run's budget, clamped so more cores than were measured can never inflate it (cpu-only
saturates). The GPU side is left unscaled: hybrid does degrade with fewer cores, but far less, and
scaling both would put hybrid at 0.94 against cpu's 1.01 at 8 cores and pick cpu-only — which is
measurably wrong there. The state table for all four cases is written into the code.

Observed on one machine, one profile:

```
247 cores:  cpu prior 30.76 -> 30.54 GiB/s  ->  defaulting compress to --cpu-only
  8 cores:  cpu prior 32.64 ->  1.01 GiB/s  ->  defaulting compress to --hybrid
```

`--adapt` only. Runs without it are untouched, and profiles written before this release simply
lack the core count, which skips the adjustment — so no schema epoch bump is required.

The endian source-guard needed a `sched_getaffinity` exemption: a CPU mask taken from the kernel
matches the `&addr` + `sizeof` shape the check looks for, but has no on-disk representation and
no byte order. Mutation-tested — a genuine host-order `pread` is still caught, so the exemption
did not widen into a hole.

Verified: extensive suite **548/548** (6m46s), both build configurations compile clean, plus
byte-identical round-trips on all three backends, GPU decompress, the GPU-fault CPU rebuild, and
stock `zstd -t`. Non-`--adapt` runs are unchanged, including `--version` at 0.00 s.

**Measured on one machine.** The shape generalises — cpu-only scales ~1/cores, the GPU path does
not — and each machine derives its own crossover from its own recorded rate. The ~8.7-core figure
is this host's; a box with a slower GPU or faster cores lands elsewhere.


## v0.16.8 — sample the GPUs in the background, and pick the idle one

v0.16.7 ranked devices by asking NVML inline, which cost ~347 ms at the moment the answer was
needed. NVML is in fact much faster than CUDA — started at the same instant, it has a full table
for all eight GPUs at **0.390 s** while the first GPU is not usable until **1.354 s** (one device
visible) or **3.391 s** (eight). The information was never the slow part; asking for it at the
wrong moment was.

`GpuMonitor` publishes a table of every GPU's utilization and free memory, refreshed on a 250 ms
timed condition-variable wait, and device selection consults that table instead of probing.

It starts **at the point of use, not at process entry.** Starting it in `main()` was tried first,
on the theory that argument parsing would pay for the NVML init. It cost every invocation
**~370 ms** — including `--version` and `--cpu-only` runs that never touch a GPU — because the
destructor joins a thread blocked in `nvmlInit`, and it doubled the test suite from 6m50s to
11m47s. That is the same defect this project already fixed once for `cuInit`: **never charge GPU
setup to a run that will not use a GPU.** The early start was theoretically free but measured as
unmeasurable, so nothing is given up by starting it where its answer is consumed — which is also
exactly where v0.16.7 paid for its inline probe.

Three properties the code has to keep, each learned the hard way:

- **Rows are keyed by UUID and PCI bus ID, never by index.** NVML enumerates in PCI order and
  CUDA defaults to fastest-first; every one of the eight devices mismatched on this host. The
  UUID form is also what `CUDA_VISIBLE_DEVICES` accepts, so a device can be named before any
  CUDA index exists.
- **Utilization is smoothed** as `(3*old + new)/4`. A single reading is not trustworthy —
  `nvidia-smi` was repeatedly observed reporting 0% on a device that was demonstrably compressing.
  A sampler can average over time; one inline probe never could. The first sweep takes the raw
  value so the table is usable immediately.
- **NVML is never shut down.** Its init is refcounted and costs ~290 ms only on the 0 to 1
  transition; dropping the count makes the next user re-pay in full.

### Waiting for the first sweep, and when not to

Selection blocks on the sampler's first sweep, because starting on a contended GPU costs seconds
on a shared machine and the wait does not. It is **skipped entirely when the run wants every GPU
the box has** — ranking decides nothing then, and `/proc/driver/nvidia/gpus` answers "how many"
for free without CUDA or NVML. If the sampler has not answered (or there is no NVML at all),
selection falls back to `/proc` UUIDs taken from the **end** of the list, on the reasoning that
device 0 is what every other tool grabs first. That is a guess, and labelled as one.

### On the performance claim: there isn't one

None of these variants is separable on this host. 1.9 GiB, `--gpu-devices=1`, six interleaved
reps each:

| variant | seconds |
|---|---|
| v0.16.7, inline NVML probe | 2.92–3.15 |
| background sampler, 250 ms cadence | 2.84–3.02 |
| background sampler, 2000 ms cadence | 2.87–3.01 |

All three overlap. The run-to-run spread of about ±0.15 s on a 3 s run is larger than the
90–350 ms effects involved, so **this release claims no startup win** — and equally, the sampler
and its cadence cannot be shown to cost anything. What the monitor provides is a smoothed,
continuously fresh table for decisions taken after bringup, where the alternative was a single
unreliable sample. It is infrastructure for choosing and switching devices during a run, not a
speedup.

Verified: extensive suite **548/548** in **6m51s** — the same runtime as v0.16.7, which is the
proof that the process-entry regression is gone (it had taken 11m47s) — and both build
configurations compile clean. The `USE_NVCOMP=OFF` build caught a real compile error along the
way: `g_gpu_monitor` lives inside `HAVE_NVML`, so its call site needed guarding.


## v0.16.7 — --gpu-devices was slower than not using it, and NVML was reading the wrong GPU

Two GPU-selection defects, found by measuring a flag nobody had timed and then by asking whether
an index meant what it looked like.

### --gpu-devices reduced parallelism without reducing cost

`cudaGetDeviceCount` triggers `cuInit`, and `cuInit` initialises every **visible** device. So
capping the device count inside gzstd bought the smaller fleet while still paying for the whole
one. At 24.2 GiB the flag was **1.5x slower** than achieving the same thing with
`CUDA_VISIBLE_DEVICES`:

| devices | `--gpu-devices=N` before | same devices hidden by hand |
|---|---|---|
| 1 | 8.02–8.10 s | 5.49–5.64 s |
| 2 | 6.72–6.98 s | 4.49–4.52 s |
| 4 | 6.00–6.03 s | 4.26–4.44 s |

`apply_backend_defaults` now sets `CUDA_VISIBLE_DEVICES` before the first CUDA call, so the
driver never initialises devices the run will not use. It needs no device count to do this — the
chicken-and-egg being that counting them is what costs — because CUDA ignores indices that do not
exist, and devices can be named by `GPU-<uuid>` before any index exists. After: **5.46–5.51 /
4.45–4.49 / 4.22–4.39 s**, identical to hiding them by hand.

Which devices, not just how many: `select_best_gpus` already ranked by NVML utilization and free
VRAM, but it maps NVML to CUDA through `cudaGetDeviceProperties`, so it cannot run until CUDA is
up — by which point `cuInit` has charged for everything anyway. The same ranking now also runs
*before* CUDA, through NVML alone, so the choice survives and the saving is collected. An explicit
user `CUDA_VISIBLE_DEVICES` is never second-guessed: the first N of **their** list is taken,
since they have already said which devices they want.

### NVML indices are not CUDA indices, and three call sites assumed they were

NVML enumerates in PCI order; CUDA defaults to fastest-first. On this box `nvml[0]` is PCI
0000:01:00 while `cuda[0]` is PCI 0000:81:00 — different physical GPUs. NVML also ignores
`CUDA_VISIBLE_DEVICES` entirely: with it set to a single device, NVML still reports all eight.

Three sites passed a CUDA device index straight to `nvmlDeviceGetHandleByIndex`: the compress
utilization probe, the decompress one, and the watchdog JSON dump. **All eight devices resolved
to the wrong GPU** — a permutation, so not one was accidentally right:

```
cuda  |  naive nvmlByIndex(cuda_id)  |  PCI-mapped (correct)
 0    |  GPU-976d6911...             |  GPU-d628d3f1...
 3    |  GPU-c56a5d03...             |  GPU-976d6911...
```

The effect was that `util_scale` — which shrinks the next batch when a GPU is busy, the one
feature written specifically for shared machines — throttled against an unrelated device's load.
Under a user-set `CUDA_VISIBLE_DEVICES` it read a GPU the user had explicitly excluded.

`gz_nvml_handle_for_cuda()` maps through `cudaGetDeviceProperties` PCI bus ID to
`nvmlDeviceGetHandleByPciBusId`, the route `select_best_gpus` already used correctly, caching
positive and negative results so a failed lookup is not retried every batch. The two remaining
index call sites are genuine NVML enumerations and are unchanged.

**The rule this leaves behind: correlate NVML and CUDA by UUID or PCI bus ID, never by index.**

### Measured while doing this, and recorded because it is not obvious

`cuInit` costs roughly **950 ms plus 250 ms per visible device** — 1009 ms with one visible,
2960 ms with eight — and it is charged for devices that are never touched. `cudaMemGetInfo` is
free once a context exists (0.0 ms) and reflects another process's allocations immediately and
exactly. And the visible device set is frozen at `cuInit`: re-setting the environment variable
or calling `cudaDeviceReset` does not reveal more devices, so the device set is a one-shot
decision made before the first CUDA call.

Decomposed with the driver API, essentially all of that is `cuInit` itself: `cuDeviceGet` and
`cuMemGetInfo` are 0.0 ms, and context creation afterwards is ~150–200 ms. A bare binary linking
only the CUDA runtime pays the same, so none of it is gzstd's. **Getting one GPU usable has a
~1.03 s floor** (six reps: 2790, 1869, 1974, 1019, 1037, 1026 ms — it settles after other CUDA
activity subsides). The only levers are how many devices are made visible, and what that second
is allowed to overlap with. Note this is an H100 PCIe host on driver 570.207; a ~1 s
single-device `cuInit` is high, so the figure should be re-measured elsewhere rather than assumed.

Verified: extensive suite **548/548** (6m56s), both build configurations compile clean.


## v0.16.6 — bring the GPUs up one at a time, so the first one starts working immediately

v0.16.5 measured CUDA context creation at 1.386 s for eight devices and concluded it was
irreducible. The first half of that is true and the second half was a bad inference.

The measurement behind it created all eight contexts **concurrently** and found they finish
together, 1.21–1.37 s, with no device usable before the last. That is a fact about *concurrent*
creation, not about the driver. Creating them **sequentially** gives a staircase:

| device | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| usable at | 0.148 | 0.299 | 0.444 | 0.586 | 0.732 | 0.872 | 1.011 | **1.155 s** |

Sequential creation also finishes the whole fleet **sooner** — 1.155 s against 1.386 — because
the contended path wastes work on top of delivering nothing early.

Context creation serialises inside the driver no matter what we do, so gzstd cannot make it
cheaper. What it can do is stop paying for it as dead time. `g_gpu_ctx_init_m` holds context
creation to one device at a time, in both the compress and decompress workers, so each device
begins compressing the moment it is ready instead of waiting for the fleet. On eight devices that
recovers **4.19 device-seconds** of otherwise-idle GPU time.

Two details matter and both were measured:

- **The lock must be narrow.** A first version also held the nvCOMP `GetMaxOutputChunkSize` query
  inside it and pushed the last device from 1.35 s to 1.68 s for no benefit. Only creation is
  serialised; VRAM probing and allocation stay concurrent.
- **`cudaSetDevice` is lazy**, so `cudaFree(0)` is what actually forces the context to exist. The
  current device is per-thread state and outlives the lock, so nothing after it needs holding.

In-app the staircase is now visible in the `-vv` init breakdown — `ctx=` 329, 152, 500, 807,
1024, 1238, 1430, 1610 ms — with GPU0 compressing at ~0.38 s instead of ~1.35 s.

End-to-end, `--gpu-only`, three reps, min–max:

| input | v0.16.5 | v0.16.6 |
|---|---|---|
| 1.9 GiB | 4.68–5.75 s | 4.47–4.71 s |
| 8.0 GiB | 5.10–5.35 s | **4.69–4.89 s** (non-overlapping, ~9%) |
| 24.2 GiB | 5.81–6.02 s | **5.27–5.73 s** (non-overlapping, ~6%) |
| 195.3 GiB | 14.89–17.80 s | 13.06–15.85 s (medians 16.16 to 13.17) |

### Still open, and unchanged by this

Per-device init cost is linear in device count while throughput is sublinear, so **engaging every
visible GPU is still not the right default**: measured best device counts are 1 GPU at 1.9 and
8.0 GiB, 2 at 24.2, 4 at 48.4, and a tie between 2 and 8 at 195.3. The size gate answers whether
to use a GPU at all and never how many. Staggering makes the eight-device case cheaper; it does
not make it correct.

### A side effect on the suite, and why it cost nothing

Small fixtures now finish before the second device exists, so most GPU tests engage a single
device — the emergent form of the same win, and measurably faster. That also deleted the suite's
*accidental* multi-GPU coverage. It cost nothing only because that accident had already been
turned into a deliberate test: the multi-GPU dispatch case pins two devices, uses the
`GZSTD_DEBUG_GPU_ALL_READY` rendezvous so both must claim a first batch before either claims a
second, and counts devices that **completed** batches rather than ones that merely initialised.
It still reports *2 GPUs used*. Its own comment called this shot — *"an accident stops covering
you the moment someone changes an unrelated default"* — and it is now the only multi-GPU
coverage there is, so it should not be weakened or made conditional.

Verified: extensive suite **548/548** (6m55s), both build configurations compile clean.


## v0.16.5 — hand the input back while the run is still using it

v0.16.4's host timeline put the input unmap at **5.131 s of a 15.5 s eight-GPU run — 33%** — and
showed it was pure tail latency: `writer drain` read 0.000 s immediately after it, so there was
nothing left to overlap with.

`madvise(MADV_DONTNEED)` over already-consumed regions does not make that work cheaper, it
**moves** it. Measured on a 24.2 GiB tmpfs mapping (`tool_madvbench.c`):

| | madvise | final munmap |
|---|---|---|
| plain munmap | — | 0.694 s (28.7 ms/GiB) |
| madvise all, then munmap | 0.700 s | **0.012 s** |

**Granularity is irrelevant** — 16 MiB, 64 MiB, 256 MiB, 1 GiB and 4 GiB steps all cost 0.70 s.
This is per-byte page-table work, not a per-call or TLB-shootdown effect, so there is no batch
size to tune and no reason to fear issuing many small calls. Two prior concerns — a TLB-IPI storm
and mmap-lock contention — were both wrong for a single-threaded retirement loop.

So the entire win is overlap, and `InputRetirer` takes it: `release_input()` reports the consumed
view, a contiguous-prefix watermark advances over out-of-order releases, and a dedicated thread
advises the closed prefix while compression runs. The thread exists precisely so the expensive
call never lands on a worker's critical path.

| segment, 195.3 GiB, 8 GPUs | v0.16.4 | v0.16.5 |
|---|---|---|
| release input map | 5.131 s (33%) | **0.136 s (1%)** |
| compress (workers) | 7.812 s | 10.927 s |
| writer drain | 0.760 s | 0.000 s |

End-to-end, 4 interleaved reps, min–max, both non-overlapping:

| | v0.16.4 | v0.16.5 |
|---|---|---|
| `--gpu-only` | 16.96–18.90 s | **13.85–16.23 s** (~13%) |
| `--cpu-only` | 7.48–7.64 s | **6.20–6.67 s** (~15%) |

The CPU path gains more proportionally: same fixed unmap cost, shorter run.

### Why release_input() is an acceptable trigger

The mapping is `PROT_READ` + `MAP_PRIVATE`. Nothing is dirty, so `MADV_DONTNEED` only drops PTEs,
and a read of a retired page re-faults it from page cache and sees identical bytes. **Retiring a
region too early is therefore a performance bug and never a correctness bug** — which is what
makes it safe to drive this from a per-task "done with its input" signal rather than from a
global barrier.

The thread's lifetime is structural rather than positional: `MmapRegion::reset()` stops the
retirer before `munmap`, so an exception unwinding past the compress drivers' explicit stop
cannot leave it advising an address range the kernel has already handed to something else.

**Not fully hidden.** The compress window grew 7.812 to 10.927 s, so roughly 3.1 s of the 5.0 s
reappeared inside the run: advising retired pages contends with faulting new ones. The net is
clearly positive and measured, but retiring with a deliberate lag, or throttling the advise
thread, may recover more.

Verified: extensive suite **548/548** (6m51s), both build configurations compile clean, and
byte-identical round-trips on `--gpu-only`, `--cpu-only` and `--hybrid` including a full
195 GiB run, plus stock `zstd -t`, `--verify`, the GPU-fault CPU rebuild, and `--no-mmap` with
the retirer correctly inactive.


## v0.16.4 — a timeline for the host, and it corrects two claims from v0.16.3

Every performance counter in this program measured time *inside* the compress loop. That is
why none of them could explain the multi-GPU behaviour: measured solo against seven concurrent
sibling processes, nvCOMP kernel time was **identical** (92.5 vs 92.6 ms/batch) and total GPU
batch time grew 13%, while end-to-end degraded 2.5x. Roughly 6.5 s per process was falling
*between* the counted phases, where nothing was looking.

`-vvv` now prints a **Host timeline** that segments the whole operation — startup and CUDA init,
queueing the input, compression, releasing the input mapping, writer drain, teardown. Seven
timestamps, one clock read each, inert below `-vvv`.

### What it found, in the configuration users actually run

One process, eight GPUs, 195.3 GiB in tmpfs:

| segment | `--gpu-batch=128 --gpu-streams=1` | default |
|---|---|---|
| startup + CUDA init | 2.932 s (20%) | 1.836 s (12%) |
| queue the input | 0.001 s | 0.001 s |
| compress (workers) | 6.925 s (46%) | 7.812 s (50%) |
| **release input map** | **5.151 s (34%)** | **5.131 s (33%)** |
| writer drain | 0.000 s | 0.760 s (5%) |

### Two corrections to what v0.16.3 recorded

**Compression scales far better than the end-to-end figures implied.** 195.3 GiB inside the
6.925 s compress window is **28.2 GiB/s**, against 32 GiB/s ideal — eight times the measured
single-GPU rate of 4.04. That is ~88%. Per-GPU *during compression* is 5.84 → 3.53 from one
device to eight (~60%), not the 38% the previous release's table reported. **That table folded
fixed overheads into the rate and should not be quoted as a compression-scaling number.**

**Moving the unmap earlier did not overlap it.** v0.16.3 relocated `mmap_region.reset()` to just
after the worker join expecting it to hide behind the writer drain. The timeline shows
`writer drain = 0.000 s` immediately after it: the writer had already finished, so there was
nothing left to hide behind. The ~8% that change measured was real but it was not the mechanism
claimed for it. Unmapping costs **~26–29 ms/GiB** and is constant across scales — 0.697 s for
24.2 GiB, 5.151 s for 195.3 GiB.

### And the multi-process story is startup, not bandwidth

Same process, solo versus alongside seven siblings: startup and CUDA init **1.000 → 7.370 s**,
compression 4.145 → 6.937 s, unmap 0.697 → 0.741 s. CUDA and driver initialisation serialise
across processes; host memory bandwidth is not implicated. Raw tmpfs reads sustain 69.0 GiB/s
aggregate across eight concurrent readers while the pipeline uses about 12.7.

So the remaining GPU-path work is specific: stop paying ~26 ms/GiB to unmap the input — retire
it incrementally during compression, or avoid building one enormous mapping — and attack startup.
Compression itself is close to done.

Instrumented in the GPU compress driver only. `compress_cpu_mt` mmaps too and shows the same
shape; its marks are deliberately deferred rather than added untested alongside a release.

Verified: extensive suite **548/548** (6m48s), both build configurations compile clean.


## v0.16.3 — a back-off that never backed off, and teardown charged to the tail

Two independent fixes, both found while chasing where the wall clock actually goes
rather than where the compression happens.

### The VRAM back-off has never run

`free_stream_buffers_only()` frees *buffers*, but it did so with `C = StreamCtx{}`, which
wipes every field in the context. Two of them are read immediately afterwards.

**`per_stream_batch`** is what the VRAM retry loop halves and re-tests. Zeroed, the loop's
`C.per_stream_batch <= 1` guard fired on the **first** failure, every time — so the
documented 256 → 128 → 64 back-off never executed once, and its "VRAM insufficient, reducing
batch to N" line could never print. A single transient allocation failure abandoned the whole
device with `insufficient VRAM for even 1 stream at batch=1`, a message that was itself false:
the batch was 256 and had never been halved. The retry then handed batch 0 to
`allocate_stream_buffers()` on a null stream.

**`stream`** is destroyed by the caller, not by this function. Zeroed, every
`if (C.stream) cudaStreamDestroy(C.stream)` became a silent no-op and the stream leaked on
both teardown paths.

Both are now preserved alongside `stats`/`last_adjust`. **This matters most on small cards,
which are exactly where a back-off is supposed to save the run** — on a 95 GiB H100 the first
allocation simply succeeds, which is why eight of them never exposed it.

Mutation-tested in both directions, by forcing the first two allocations to fail: unfixed, one
failure skips the device; fixed, the batch halves to 128 and the retry proceeds with the stream
handle intact.

### The input mapping was unmapped after the run instead of during it

`gdb` and `perf` are both unavailable on this host (`ptrace_scope=1`,
`perf_event_paranoid=4`), so the post-work window was profiled by sampling
`/proc/PID/task/*/wchan`. It shows several threads in uninterruptible sleep on
`__vm_munmap` while RSS drains — kernel page-table teardown, charged entirely to the tail.

`mmap_region.reset()` now runs immediately after the workers join, in **both** compress paths,
so the teardown overlaps the writer drain. The lifetime argument is the one already made by the
adjacent `g_direct_read_pool = nullptr`: only Tasks hold views into the mapping, every Task is
consumed by a worker, the workers are joined, and the writer holds compressed output, which is
copied rather than viewed.

195.3 GiB in tmpfs, 4 interleaved reps, min–max:

| | before | after |
|---|---|---|
| `--gpu-only` | 16.67–17.82 s | **15.48–16.24 s** (~8%, non-overlapping) |
| `--cpu-only` | 7.50–7.60 s | 7.38–7.69 s (overlapping — neutral) |

The CPU path is neutral because that run is too short to leave much writer drain to hide the
unmap behind.

### What the measurements actually said, which was not what was being optimised

A controlled scale-up — **constant 24.2 GiB per GPU**, batch pinned at 128, one stream — shows
per-device throughput collapsing as devices are added:

| GPUs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| total GiB/s | 4.04 | 6.86 | 10.04 | 12.31 |
| **per GPU** | **4.04** | 3.43 | 2.51 | **1.54** |
| efficiency | 100% | 85% | 62% | **38%** |

**One GPU sustains 4.04 GiB/s; eight deliver three times the work of one.** Per-device
efficiency is not the problem — sharing is. Every staging and drain change up to this point was
tuning the single-device number. Not yet isolated: the single host reader, host-memory/PCIe
aggregate bandwidth, the single writer thread (zero-scan is 94.9% of its busy time), queue and
ResultStore lock contention, and in-order head-of-line blocking (17–22%, averaging 306 frames
stuck at eight devices).

A single-device batch sweep also puts the optimum at **batch=128 (4.03 GiB/s)**, with **256 —
where the shared auto-tuner settles — measurably worse** (5.99–6.02 s against 6.52–6.78,
non-overlapping). Left alone deliberately: the right batch may move once sharing is fixed.

Verified: extensive suite **548/548** (6m44s), both build configurations compile clean, and
byte-identical round-trips on `--gpu-only`, `--cpu-only` and `--hybrid` plus stock `zstd -t`,
`--verify`, and the GPU-fault CPU rebuild.


## v0.16.2 — the drain was barriering the whole device, and two D2H fixes that lost

v0.16.1 named the symptom: workers block 75–82% on `wait_idle_stream`, behind their own drain
thread. This finds the mechanism, and it is one line of CUDA semantics.

**gzstd creates its streams with `cudaStreamCreate` — *blocking* streams.** Anything issued on the
legacy null stream is therefore an implicit **device-wide barrier**: it waits for every other
stream on the device, and every other stream waits for it. And `gpu_drain_batch` read every frame
back with a *blocking, null-stream* `cudaMemcpy` — two for the metadata plus **one per chunk**, so
at batch=256 that is **up to 258 full-device barriers per batch**. `gpu_verify_check` added two
more under `--verify`, and the decompress drain had the identical defect.

That is why the three GPU stages measured as strictly serial, summing to 98 ms/batch instead of
overlapping, and why `--gpu-streams=2` had never bought anything: a second context cannot overlap
anything through a device-wide barrier. The "single stream avoids context-switch overhead"
default was measuring this defect, not context switches.

**The fix** issues every readback as `cudaMemcpyAsync` on the batch's own stream, closed by a
single `cudaStreamSynchronize` — so frames are stamped and published in a second pass, because a
copy has not landed until the sync. No page-locking, no extra allocation, no init cost.

Measured at **195.3 GiB** (see below on why that size), 4 interleaved reps, min–max:

| config | seconds |
|---|---|
| baseline `--gpu-streams=2` | 17.54–19.23 |
| **fixed `--gpu-streams=2`** | **16.12–17.09** |
| baseline `--gpu-streams=4` | 17.76–18.64 |
| **fixed `--gpu-streams=4`** | **15.56–16.32** |

Both matched pairs are non-overlapping. Every *baseline* stream count overlaps every other —
extra streams really were inert before.

### The stream default was split, because the two directions measured opposite ways

| | compress / `-t` | decompress |
|---|---|---|
| new default | **2** | **1** |
| sweep, 195 GiB | s1 17.00–18.95, s2 16.14–16.91, s4 15.72–16.80, s8 14.77–16.49 | s1 13.23–13.69, s2 14.40–14.88, s4 16.11–17.46, s8 21.68–23.19 |

Compress: s1 is clearly worst and s2/s4/s8 are statistically tied, so take the cheapest — extra
streams split one VRAM budget into smaller per-stream batches, which costs most on a small card.

Decompress gets monotonically worse, every neighbour non-overlapping, and the reason is
structural: **decompression expands.** Its D2H moves 195.31 GiB against 53.97 GiB of H2D, making
D2H the dominant stage (21.3 s of 41.5 s of batch time) — and a device has one PCIe link, so more
streams do not add bandwidth, they just split the same link and widen out-of-order completion
against an in-order writer (head-of-line blocking 34.1%, average 131 frames stuck). There is
nothing to hide the dominant stage behind. Compress shrinks 59.60 GiB to 16.51 GiB and has a
39.7 ms/batch kernel for transfers to overlap with, which is why it benefits and decompress
cannot.

`gpu_streams == 0` already meant "auto" and is resolved in one place where the mode is known, so
this needed no new user-set flag — and it mirrors the existing `DEFAULT_GPU_BATCH_CAP` /
`DEFAULT_GPU_DECOMP_BATCH_CAP` split.

### Benchmark at 195 GiB, not 20 — the old size was measuring ramp-up

`--gpu-only` on this box is **3.25 GiB/s at 19.53 GiB, 6.55 at 59.60, and 10.7 at 195.3**. The
batch tuner climbs 8 → 256 over the run and 1.4 s of CUDA init is a 23% tax on a 6 s run. Every
GPU figure this project has recorded from a ~20 GiB file was measuring the ramp, so v0.16.0's
headline — "the GPUs deliver 3.00 GiB/s, 4.2% of kernel capability" — was an artefact of
benchmark size. Run-to-run spread here is ~10%; nothing under about four interleaved reps is a
result. A single 64 GiB pair reported the exact opposite of the four-rep answer during this work.

### Two D2H rewrites that were built, measured, and rejected

Both made the transfer dramatically faster and both lost end-to-end. Recorded so neither is tried
a third time.

**A pinned staging slab** (async D2H into page-locked memory, then `memcpy` into the frame):
3.25 → 2.40 GiB/s, a 26% regression. It turned one transfer into a transfer plus a ~160 MiB host
memcpy per batch, and the transfer was never the expensive part.

**`cudaHostRegister` over the frame buffers themselves** — the only shape with no extra copy.
Mechanically it worked: D2H 2.72 → 6.88 GiB/s, per-batch total 60.7 → 45.3 ms. End-to-end it was
*monotonically slower the more memory was page-locked* — 0/2/4/8/16 slots per stream measured
5.81–6.29 / 6.34–6.70 / 6.58–6.82 / 7.06–7.50 / 8.05–8.49 s — and it still lost at 195 GiB
(18.33–19.61 vs 15.89–17.80) with the interleaving isolated out.

The reason is a rule worth keeping: **page-locking costs about as much per byte as the copy it
accelerates** — `cudaHostRegister` runs at ~0.21 ms/MiB, is serialised driver-wide (eight threads
on eight devices took the same wall time as one thread doing all of it), and is charged per-byte
rather than per-call, so a slab cannot amortise it either. Meanwhile the D2H it replaces was
already hidden behind the drain thread. It can only pay where D2H is genuinely on the critical
path and the buffers are reused enough, which is not this pipeline. The patch is kept out of tree
against a future where compression ratio is poor enough that D2H dominates.


## v0.16.1 — per-worker phase accounting, and it names the bottleneck in one line

v0.16.0 established that the devices are starved with nothing upstream blocking, and left the
root cause open after four wrong hypotheses. This closes it by measuring instead of guessing.

The watchdog already marked *which* phase a GPU worker was in, at exactly the right places — it
just never recorded how long it **stayed** there. Accumulating on the existing transitions needed
no new call sites and no new hazards, and the answer is unambiguous:

```
[GPU2] phases (1.21s total):
   wait_idle_stream        0.94s (78%)
   intake_wait_batch       0.00s ( 0%)
   intake_acquire_permits  0.00s ( 0%)
   submit_h2d_kernel       0.26s (22%)
```

Every worker, 75–82%, blocked on `wait_idle_stream` — waiting for a free `StreamCtx`. Time spent
waiting for input: **zero**. The devices were never short of work; they were blocked behind their
own drain thread returning a context.

**The structure explains it and corrects v0.16.0's conclusion.** There is one `drainer` thread per
device serving N `StreamCtx`, so extra streams give the worker more contexts to fill but they all
funnel through a single drain — which is why `--gpu-streams=2` moved nothing (3.13 vs 3.21).
v0.16.0 concluded "the drain is off the critical path" because fixing a 90 ms defect there gained
nothing end-to-end. Both observations were real; the inference was wrong. The drain **is** the
per-device serializer: fixing the checksum realloc simply moved its 90 ms into the D2H copies,
which then contended with the worker's H2D on the same device. Drain stayed ~100 ms/batch against
~25 ms of submit, so the worker waits ~75% either way.

The full chain, now measured end to end: worker blocks 78% on `wait_idle_stream` → one drain
thread per device holds the context ~100 ms/batch → the drain is dominated by blocking, pageable,
per-chunk `cudaMemcpy` contending with the worker's H2D → and one async slab into registered
memory measures **52.78 GiB/s against 21.57** for blocking pageable, single-device.

Also fixed here: the phase mark for the H2D upload sat *after* the kernel launch, so that batch's
staging was charged to whatever the worker had been waiting on beforehand. Marked before the work
now.

`-vvv` only; one clock read per transition when enabled, and the watchdog store is untouched
either way.

### Tried and reverted: a page-locked D2H staging slab (**26% slower**)

Recorded because this is the obvious next move and it loses. The phase accounting above says
shrink the drain, and the transfer numbers say page-locked async D2H is 50.00 GiB/s against
21.57 blocking — so: allocate a bounded staging slab (`d2h_pinned_base`, a field that already
existed unused), issue all readbacks async into it, one `cudaStreamSynchronize`, then `memcpy`
out into the frame buffers. Bounded at 256 MiB/stream with its own budget, deliberately *not*
gated on `pin_mode` (that flag is an H2D verdict, and applying it to D2H would be the
one-flag-two-questions mistake).

**Measured: 3.25 → 2.40 GiB/s.** Byte-identical output, purely a performance loss. Reverted.

**Why, and it is the same trap as `--pinned on`:** today the readback lands *directly in the
frame buffer* — one transfer. Staging makes it two, adding a **host memcpy of ~160 MiB per
batch**. That copy costs more than the faster DMA saves. It is exactly why `--pinned on` is worse
for H2D (12.33 vs 23.38 GiB/s) — a number quoted in the very comment written above the new code,
while making the same mistake in the other direction.

**The rule this yields: on this path, any design that adds a host-side copy loses**, however much
faster the DMA becomes. The transfer was never the expensive part — 96.4 GiB/s is available
across eight devices and the pipeline uses 1.4.

So the only shape left is async D2H **directly into the frame buffers**, i.e. `cudaHostRegister`
over the `out_pool` rather than a separate slab. That is harder than it looks: `FrameVec` buffers
are `shared_ptr`s held by the writer for arbitrary time, and any `resize()` past capacity
reallocates and silently invalidates the registration. It needs every pooled buffer reserved to
`max_out_chunk + 4` once and never grown — enforced structurally, not by a comment.

One aside worth recording: the D2H loop's `resize()` was briefly suspected of zero-filling. It
does not — **v0.13.39** replaced `FrameBuf`'s allocator with `default_init_allocator` precisely to
kill that memset. A benchmark that modelled the memset overstated today's cost by ~33%.

## v0.16.0 — the GPUs are not slow; two staging defects, and a lesson about which one mattered

Opens the GPU performance chapter. `--gpu-only` delivers **3.0 GiB/s** while the eight devices'
kernels are capable of **71.8 GiB/s aggregate** (8 × 8.98, from the per-batch records) — against
**19.34 GiB/s** for all 256 CPU cores on the same input. The hardware is ~3.7× the CPU and the
pipeline was getting 4.2% of it out, with devices busy only ~1.4 s of a 6.5 s run.

Measured with the source in RAM and the sink `/dev/null`, deliberately: this host's NVMe reads at
4.56 GiB/s, so a storage-to-storage run here cannot *measure* a GPU win. That is a property of
this box, not of the design — gzstd has to be fast on machines we do not own.

**Two defects, and the difference between them is the point.**

**The frame checksum reallocated every frame.** `gpu_frame_add_checksum` did `resize(csz)` then
four `push_back`s. `size == capacity` at that moment, so the first `push_back` reallocated the
whole frame and copied every byte — megabytes of alloc-and-copy to append four bytes, 32 frames
per batch, on eight drain threads at once. It was **60–93% of the entire drain phase**
(`drain 97.3 ms = copy 7.5 + ck 89.7`). Now the buffer is sized `csz + 4` once and the trailer is
written in place: **0.007 ms**.

**End-to-end gain: zero.** The drain runs on its own thread, overlapped with the next batch, so a
90 ms defect there cost nothing at all. *A phase being slow does not make it a bottleneck* — and
this one looked exactly like the bottleneck.

**The content checksum ran between the H2D copies.** A full XXH64 pass over every input byte,
inline in the upload loop, so each chunk was: block in the copy, hash, block in the next copy.
17.0 ms/batch against 19.7 ms of copying — **46% of a region that *is* on the critical path**. The
copies are host-synchronous, so that thread is blocked in the DMA anyway and the hashing is free
if it happens elsewhere; per-chunk hashes are independent and on a GPU run the CPUs are ~79% idle.
Now computed on up to eight helper threads, joined before `release_input()`, which is what keeps
the source pointers valid. **+8% (3.00 → ~3.25 GiB/s).**

Those helper threads needed an RAII joiner: `checkCuda` throws on CUDA error and `gpu_worker`'s
catch turns that into the abort → CPU-only rebuild, but `std::thread`'s destructor calls
`std::terminate()` while joinable — so without it every *recoverable* GPU fault would have become
a hard process abort. Verified with `GZSTD_DEBUG_FAIL_GPU_AFTER`: forced fault, exit 0, no abort,
byte-identical output after the rebuild.

**`-vvv` now breaks both staging regions down** (`drain … = meta + acq + size + copy + ck + push`,
and `h2d-region: copy + checksum`). That instrumentation is what found both defects; the phase the
logs call "d2h" was never the copy.

**Ruled out by measurement — recorded so they are not re-investigated:** D2H bandwidth (pageable
does 96.4 GiB/s across 8 devices; the app used 1.4), H2D bandwidth (78–87 GiB/s), mmap vs
anonymous source for DMA (within 10%), cross-device contention on pageable transfers (it *scales*,
21.6 → 96.4 GiB/s from 1 → 8 devices), batch/stream/chunk tuning (every setting lands 3.1–3.35),
the FrameThrottle (peak 2.3%, zero blocks), and the queue byte cap (a documented no-op for mmap
views). One of those was a claim this project had previously made about `resize()` zero-filling —
`FrameVec` uses `default_init_allocator`, so it never did.

**The root cause is not yet found.** The devices are starved and nothing upstream blocks. Next is
per-worker phase accounting using the existing `wd_*`/`WatchPhase` hooks, which distinguish
spinning from blocked — not another hypothesis. Four hypotheses were wrong in a row here, each
costing a benchmark, which is the argument for instrumenting before guessing again.

## v0.15.97 — the suite got 54% faster, and the last HIGH turned out to be GNU tar's contract

### `--rm`'s remaining blind spot is not a defect — measured against GNU tar 1.35

An independent round called v0.15.96 NOT SAFE TO TAG: a same-size rewrite that leaves the
reported mtime unchanged passes both the identity and content-stat checks, so bytes absent from
the archive can still be deleted at exit 0. Before treating that as a defect, the reference
implementation was measured. A 1.5 GB source modified in place during the read:

| case | GNU tar 1.35 | gzstd |
|---|---|---|
| modified mid-read, mtime advances | `file changed as we read it`, exit 1, source kept | refused, exit 3, source kept, entry named |
| same-size rewrite, mtime restored | **exit 0, source removed, no warning** | same |

**GNU tar uses the same size+mtime test and has the same blind spot.** gzstd already matches it
on the detected case and is stricter in reporting. Closing it here would make `--rm` diverge from
the behaviour GNU tar defines for this flag, and cost a full re-read of every source at removal.
So it is deferred as an opt-in enhancement, with the design recorded in `ROADMAP.md`, and the
promise stated as *stat-visible stability* rather than content identity.

The reviewer accepted the measurement, withdrew "ordinary concurrent activity" as a description
of the trigger — an ordinary writer advances mtime and is caught — and revised its judgement:
this round's findings are lower severity than the last, and **the surface is converging**. That
is the first convergence signal in this arc, and it came from measuring the reference rather
than arguing about severity.

### One representation omitted from a shared rule, for the third time

Wholly sparse **OLDGNU** members store size zero, so they never entered the reader path and
skipped assembly-time validation entirely: a file changed after `probe_sparse()` could produce a
stale all-hole archive at exit 0, and `--rm` would then delete the changed inode. Ordinary, empty
and PAX-sparse members were all covered; this one representation was not — the same shape as the
earlier misses, which is why the deferred design in `ROADMAP.md` starts with a single
`ArchivedSnapshot` abstraction rather than with fingerprints. Descriptor validation now lives in
one helper that every path calls, including zero-stored-byte members via their header path.

### The suite: 15m03s → 6m56s

CUDA initialisation scales with **visible devices**, not with work — 20 ms cpu-only, 1210 ms
gpu-only on one device, 4550 ms on eight, for a 2 MB input. The suite now pins
`CUDA_VISIBLE_DEVICES=0` (`GZSTD_TEST_ALL_GPUS=1` restores all), and the `--cpu-share` round-trip
sweep drops from five values to three. Measured: **extensive 15m03s → 6m56s, a 54% cut**, with
CPU-only unchanged at 1m27s as the control. The estimate beforehand was ~3 minutes; the gap was
counting GPU-forcing *tests* rather than *invocations*, since most compress **and** decompress and
so paid device init twice. `--rm input-shape matrix` fell 24.5s → 8.8s without losing a single
assertion.

Three bugs in that work, all found by review rather than by the green suite:

- **A test that could be green without testing its subject.** The new multi-GPU cell counted any
  `GPUx/Sy` token — including initialisation and zero-batch summaries — then *skipped* when only
  one device did work. It now counts only completed non-zero batches and fails rather than skips.
  A second flaw surfaced in the fix itself: readiness alone did not guarantee two workers got
  work, since one could claim every batch. A test-only rendezvous (`GZSTD_DEBUG_GPU_ALL_READY`)
  now holds each device after its first claim until the other has claimed or failed. It passed on
  its first execution, driving 2 GPUs in 4.6s.
- **The device pin overrode a caller's deliberate choice.** The guard tested `${VAR:-}`
  (non-empty) instead of `${VAR+x}` (set), so running the suite with `CUDA_VISIBLE_DEVICES=""` to
  simulate a GPU-less host got GPU 0 exposed anyway.
- **Bash 4 still forked `date` per result line.** A `/proc/uptime` monotonic fallback removes it;
  bash 5 continues to use `$EPOCHREALTIME`.

## v0.15.96 — an inode is not its contents, and five rounds of identity work never noticed

Every fix in v0.15.90 through v0.15.95 bound an identity: the removal record, the member read,
descriptor reuse, the sparse map, the symlink target, the pinned input. All of them compared
`(st_dev, st_ino)`. **A writer that rewrites a file in place keeps `(st_dev, st_ino)`.** So the
whole arc's guarantee — that `--rm` removes only what the archive holds — was never true of a
file whose *contents* changed after it was read:

```
gzstd --rm -o f.zst f          # with the removal gate held open
  → overwrite f in place, touch its mtime
  → f deleted, archive holds the OLD bytes, exit 0, no warning
```

Reproduced against the released v0.15.95 binary. **No attacker is required** — an ordinary
concurrent writer between the read and the removal is enough, which is a log being appended to
or an editor saving during a backup. That places it on the always-fix side of the threat-model
line, not in the defended-where-cheap tier where the rest of this arc lived.

`--rm` now records the size and nanosecond mtime of the inode it compressed and refuses to
remove an entry whose snapshot moved, restoring the input and exiting non-zero with the archive
kept. Both forms are covered: plain `--rm` compares against the `fstat` taken when the input was
opened, and `--tar --rm` against Pass B's stat, with the reader and the sparse probe validating
the same snapshot so a change between the walk and the read cannot reach the archive either.

**The bound is stated rather than implied**, because this arc's recurring defect is a comment
promising more than the code: size+mtime detects a writer who is not hiding. It does **not**
detect a same-size rewrite with a restored mtime, or one hidden by coarse filesystem timestamp
granularity. Byte-level certainty would mean re-reading and digesting every member. `--help`
says so too.

`ctime` is deliberately excluded: unlinking one archived hardlink bumps the shared inode's ctime
before the next name is visited, so including it would make every multiply-linked archive refuse
its own second alias. Data mtime does not move when a name is removed.

**Found by the OPEN half of a review round.** The adjudication half — "are the five fixes we
agreed actually applied?" — returned all five confirmed. A round scoped only to that question
would have reported a clean bill on a version that silently deletes unarchived data. It is the
second time in this arc that an open pass caught what a narrow one structurally could not.

Two more from the same round, both about a guarantee being half-applied:

- **A wholly sparse PAX member can synthesise its map without ever calling `read_seg()`.** The
  reader's rejection of a changed member relied on that function to raise the error flag, so for
  those members the rejection was recorded nowhere and the removal phase still ran. Found by
  asking whether a reader failure *actually* stops removal, rather than assuming it.
- **A regression test that could not fail for the right reason.** `1c-9f` checked only that the
  substituted pathname still existed and ignored the exit status, so an implementation that
  refused *every* removal would have passed it. It now requires a non-zero exit and verifies the
  archived inode survived under its moved name.

### The suite times itself now, and the profile said something surprising

The old `[Ns]` came from a `run_test` wrapper used at 55 call sites, so it covered 102 of 546
results and accounted for **443 s of a 913 s run**. Timing the *interval between result lines*
instead needs no per-test edits and includes the fixture work before each test, so the intervals
tile the whole run: coverage is now **100%** (903 s of 903 s). `-T` adds the slowest tests and
that coverage figure; `GZSTD_TEST_TIMING_FLOOR=0` shows every line.

Two bugs in the instrument itself, both worth recording. The first version had callers do
`dt=$(result_ms ...)`, which runs the function in a **subshell** — the state update was discarded
and every interval was measured from process start, reporting ~108 s per test and 15164% coverage.
Absurd output caught it; a plausible wrong number would not have, which is the argument for the
coverage line existing. The second: `now_ms()` forked `date` on every call and made the run 18 s
slower — an instrument costing a fifth of what it was built to find. `$EPOCHREALTIME` removed the
exec, and a later review caught that `local x=$(...)` still forks a subshell regardless.

**The profile then refused to name a villain.** The top 10 tests are 16% of the run and the top 50
are 50% — deleting the ten slowest buys 2.4 minutes of 15 and costs real coverage. What it did
show is that **CUDA initialisation scales with visible devices, not with work**: on the 8-GPU host,
a 2 MB input takes 20 ms cpu-only, 1210 ms gpu-only with one device, and 4550 ms with eight. About
850 ms fixed plus ~470 ms per device, paid again by each of ~89 GPU-forcing invocations, to
initialise seven GPUs the fixtures are far too small to use. The suite now pins
`CUDA_VISIBLE_DEVICES=0` (`GZSTD_TEST_ALL_GPUS=1` restores them all).

That trade removes *incidental* multi-GPU coverage — nothing ever asserted on it — so it is
replaced by a deliberate test that opts back in to every device and checks the work actually
reached more than one, rather than that the command exited 0. And the `--cpu-share` round-trip
sweep drops from five values to three: the endpoints are the only distinct paths, and 0.25/0.75
were 22 s spent re-evaluating arithmetic while the split's *response* to the value is measured
separately at both extremes.

## v0.15.95 — one identity check was not enough; there were five things that decide the data

v0.15.93 bound `--tar --rm`'s removal record to the inode the walker saw, and that closed the
trigger it was given. Three independent review rounds then showed the record was the *only*
thing bound. Everything that decides **what actually goes into the archive** still resolved the
member by pathname, long after the walk — so the two halves could describe different objects
while each looked correct on its own.

Five operations decide member data. All five are now bound to the identity the layout recorded:

| operation | what went wrong |
|---|---|
| the member read | opened by pathname, checked only `S_ISREG`; a same-size swap put another file's bytes under this member's header while `--rm` deleted the original |
| descriptor reuse | cached by pathname alone, so a later member with the same `src` silently inherited a descriptor validated for a *different* record |
| `--sparse` geometry | hole map taken from whatever the pathname opened; applied to the real file it stores a partial member with no short read, so nothing is flagged |
| the symlink target | `lstat` then a separate pathname `readlink`, so the archived target could come from a link the record does not describe |
| the directory type | emitted from enumeration's `is_dir` while the record took its type from a later `lstat`; disagreement archives a data-less directory member and unlinks a regular file |

Each was reachable the same way — a writer with access to the source directory — and each ended
the same way: the archive holds one object's data while `--rm` deletes another's inode, exit 0.
The member read was reproduced directly (a same-size substitution during assembly puts
`VICTIMXX` under the original's header) and confirmed by mutation.

**And one that needed no attacker at all.** `gzstd --tar --rm -f -o a.zst d/.` recorded the root
as `d/.`, whose basename the removal loop skipped silently — so `d` survived, and the run
printed `removed 2 archived source entries` having removed one, and exited 0. A trailing dot a
user is entitled to type. Paths are normalised before recording, an unusable basename is now a
counted failure rather than a silent skip, and the count is taken at the `unlink` itself. The
first attempt at that count was a no-op that read like a fix: `size() - kept` printed inside the
`kept == 0` branch, where it is `size()` by construction.

A missing parent directory is benign only when **we** removed it earlier in the same reverse
walk; tracked, not assumed. Treating every missing parent as benign would have turned a renamed
parent — sources still alive under the new name — into a silent exit 0, which is a worse failure
than the false exit 3 it was fixing.

`--rm`'s own input is now held, not just remembered. `open_input_pinned()` recorded
`(st_dev, st_ino)` and closed the `O_PATH` handle, leaving the inode free to be released and its
number reissued to a replacement that would then satisfy the comparison. The descriptor is kept
until the removal decision, so the inode cannot be freed and the recorded pair stays an identity
rather than a number. Tar cannot do the same — a held descriptor per walked entry would exhaust
the table on a real tree — and that asymmetry is now stated in the code instead of implied.

**Accepted, and written down rather than glossed:** the compare-then-unlink window (no
unlink-by-inode syscall exists); that an entry of the same *removal class* can be substituted
into it, which is broader than "same type"; that a hardlink reached through a swapped parent
component is a distinct case from that window; and that `--acls/--xattrs` metadata is gathered
by reopening the path and is not identity-bound. Every one of those was found by a reviewer
reading a comment that claimed more than the code delivered — the failure mode this whole arc
has been about.

## v0.15.94 — both tests for the deletion defects only ran when they happened to win the race

The two cells that guard the code which deletes files were each racing for a window microseconds
wide, and the suite was green either way.

The `--tar --rm` cell (new in v0.15.93) polled for the output file, swapped, and hoped that landed
between "the archive is installed" and "the first source is unlinked". On a tree of three small
files it did not: it **passed on the GPU build and skipped on the CPU-only build**, so the only
test covering the only code in gzstd that deletes a tree did not execute at all on the
configuration where this project's defects hide.

The plain `--rm` type-change cell — which guards the *shipped* v0.15.91 security fix — had the
same structure and bought its window by brute force: it compressed **400 MB of `/dev/urandom` on
every run of every configuration**, and still skipped when it lost. Widening a race is not closing
one, and it was charging both suites for the privilege.

The window is now explicit for both. `$GZSTD_DEBUG_RM_GATE` names a FIFO; when set, gzstd blocks
on it once the output is complete and installed and before the source is touched, on both `--rm`
paths. The test opens the write end — which returns *exactly* when gzstd opens the read end, so
the swap cannot land too early, which is the half a poll loop cannot get right — performs the
substitution, writes a byte, and gzstd proceeds. **There is no timing in the handshake at all.**
Both cells now run on every build, deterministically, on ten-byte inputs; the 400 MB member is
gone from both suites. Unset — every non-test run — is no syscalls and no gate.

**Verified by mutation, not by observing green.** Each gate was checked by reintroducing the
defect it exists to catch:

- disable the dev/ino check and take `AT_REMOVEDIR` from the observed type again (v0.15.93's
  defect) → the tar cell fails 3 of 3 on the CPU-only build, where it used to skip;
- restore the consumer's type-dispatch, `S_ISLNK(quarantined) ? in_lid : in_id` (v0.15.91's
  defect) → the plain cell fails 3 of 3 **at exit 0**, the original signature, on a ten-byte
  input that the old 400 MB version needed bulk to catch at all.

**The rule this makes concrete:** never gate a suite test on wall-clock timing — if a test must
act inside a window, the program opens the window. The failure mode is not a false alarm but a
silent *non*-failure, and `skip` is where it hides, invisible to a failure count. And a test that
has never been seen to fail is not known to be a test: mutate it and watch it fail.

## v0.15.93 — the new deletion code had the defect it was written right after fixing

The `--tar --rm` removal list stored a **path string**. At removal it `stat`ed whatever was at
that name and used *that* entry's type to choose `unlink` vs `rmdir`. So:

```
mv d/a.txt d/saved.txt        # move the archived file aside
mv victim.txt d/a.txt         # drop an unarchived file at the name
```

deleted `victim.txt`, **a file that was never archived**. Reproduced. That is v0.15.90's finding
reintroduced in brand-new code written hours after fixing it, and the comment above the function
claimed the protection — "only what was archived is removed", "a path whose type changed is left
alone" — while the code implemented none of it. The comment was the more dangerous half.

The record now carries **dev+ino and the archived type**, both re-checked through the parent
descriptor before anything is removed. A name that no longer leads to the archived inode is left
alone and counted; `AT_REMOVEDIR` comes from the recorded type rather than the observed one, so
the agreement is structural instead of incidental. The replacement chooses nothing: not whether
it is removed, and not how.

The feature's own tests could not catch this — every one of them is a substitution-free run, so
they exercised only the paths that were already correct. There is now a test that performs the
swap.

**Documented, deliberately not fixed:** the identity proof is an `fstatat` and the removal is an
`unlinkat` *by name*, so a different-uid writer with write access to the parent can substitute a
**same-type** entry between the two and have that one unlinked. There is no unlink-by-inode
syscall, and another recheck is not a fix — it is one more pair of lookups for an exchange to sit
between, the same reasoning already recorded for the abnormal-exit cleanup window. Quarantining
every member by `renameat2`, the way plain `--rm` treats its single input, was rejected: a rename
per archived entry across a whole tree, to narrow a window measured in microseconds, is not the
cheap self-contained defence the threat-model line asks for. The bound that makes it acceptable is
that the attacker chooses only the *victim*, never the *treatment* — `AT_REMOVEDIR` still comes
from the record. The function's comment now says so, rather than listing only the properties it
does deliver: understating the code is how the defect above shipped.

**The lesson, and it is not "be more careful":** a fix does not generalise to code written after
it. The producer/consumer split from v0.15.90 was re-derived here from scratch and re-broken the
same way, because the new code was reviewed against *its own* comments rather than against the
defect class. New destructive code needs the class checklist run over it explicitly.


## v0.15.92 — `--tar --rm` stops lying

`gzstd --tar --rm` accepted the flag, exited 0, printed nothing, and **kept every source the
user asked it to remove**. `--rm` set `opt.keep`; nothing in the tar path ever read it. A
destructive flag that silently no-ops is worse than one that errors, and this one had been
quietly not-deleting since `--tar` shipped.

It now removes the archived sources, and `--remove-files` is accepted as an alias — GNU tar's
spelling of the same intent. Both are documented in `-h` and `--help`.

**Three properties, and the last two matter more than the first:**

1. **Only what was archived is removed.** The list is built by the walker as it adds members, so
   a file created during the run — which is not in the archive — is never a candidate. Removal
   is not driven by the command-line source paths.
2. **It is never a recursive force-delete.** Non-directories are `unlinkat`ed and directories
   are `rmdir`ed, nothing more, walking the list in reverse so children go before parents. So
   anything left behind — excluded by `--exclude`, created mid-run, or skipped — *keeps its
   parent directory alive* instead of being destroyed with it. `ENOTEMPTY` is the designed
   outcome, not an error to work around.
3. **Nothing is removed unless the run was clean.** A non-zero exit covers a failed archive;
   `g_tar_had_errors` covers the archive that *succeeded while skipping a member* — unreadable,
   vanished, or changed mid-read. Removing sources in that case destroys data the archive does
   not contain, which is precisely what the `--verify` work exists to prevent; the flag whose
   whole job is deletion must not reintroduce it.

Entries that could not be removed are named individually, and the run exits 3 with the archive
kept: a destructive step you asked for and did not get must not exit 0.

**Measured against GNU tar 1.35 side by side, not presumed.** The common case matches; the failure
modes deliberately diverge, and that divergence is the point:

| case | GNU tar 1.35 | gzstd |
|---|---|---|
| clean run | tree removed | tree removed — same |
| excluded file present | file + parent dirs survive, named, exit 2 | same, exit 3 |
| unreadable member (archive incomplete) | **deletes what it archived** | **removes nothing** |
| SIGINT mid-run | **36 of 40 sources already destroyed** | **all 60 intact**, partial archive removed, exit 130 |

GNU tar removes each file *as it archives it*; gzstd defers until the archive is complete and
installed, and is all-or-nothing. So "just like GNU tar" is true for the happy path and
deliberately false for every failure path.

Verified by behaviour, not inspection: a clean tree is removed and the archive still
round-trips; an excluded file survives *and so does its parent*; an unreadable member yields
`--rm: the archive did not complete cleanly; no sources were removed` with every source intact;
a single-file source does not take its parent directory with it; and without the flag nothing is
ever removed.

**Not changed, and worth knowing:** `-d --tar --rm` still does not remove the archive it just
extracted. That is the same silent-no-op shape on the decompress side, it is pre-existing, and
it is a separate destructive behaviour that deserves its own decision rather than being folded
in here.


## v0.15.91 — the reviewer's fix was incomplete too, and a mirrored path went unfixed

A follow-up pass put v0.15.90's four fixes back to the reviewer. It confirmed two and **rejected
two as incomplete** — both correctly.

**`--rm` broke procfs magic links, and narrowing the gate did not fix it.** v0.15.90 added a
fallback for a missing `/proc`, but the reviewer supplied a trigger neither of us had
considered: a magic link as *input*. `readlink` on `/proc/self/fd/3` for a pipe returns
`pipe:[908921]`, which is not a pathname, so the `openat` of that text fails:

```
gzstd     -f -o out.zst /proc/self/fd/3   → exit 0, archive written
gzstd --rm -f -o out.zst /proc/self/fd/3  → exit 3, nothing at all
```

`--rm` was the only difference. The reviewer's own fix narrowed the gate on `!opt.to_stdout`,
which does not reach `--rm -o FILE` with a pipe magic link — still no archive. The shipped fix
**fails soft instead**: fall back to a plain open, abandon the identity claim, and let `--rm`
refuse to delete with a clear message. That does reintroduce a race over *which data gets
compressed*, but not the destructive one, because the failed proof stops before quarantine.

**The `--overwrite` mirror of the recovery-location fix was left unfixed.** v0.15.90 stopped a
failed `--rm` removal from asking a *deletion* where the file was, and left the identical
pattern in the `--overwrite` path. That is this project's own "diff mirrored code paths" rule,
missed inside a fix for a finding about exactly that.



## v0.15.90 — the fix was half a fix: the consumer let the replacement pick its own test

An independent review round on v0.15.89 returned **NOT SAFE TO TAG**, and it was right. The
previous release fixed the *producer* of the `--rm` identity and left the *consumer* choosing
which identity to compare **based on the replacement's current type**:

```
if (S_ISLNK(quarantined))  compare against the pinned link
else                       compare against the compressed target
```

So the attacker picked the test. Given `link -> real`, the exchange

```
mv link saved-link
mv real link
```

makes the name a regular file whose inode **is** the compressed target's — it passes the
non-symlink branch, and `real`, the actual data file, is **deleted at exit 0** with the user's
relocated link left dangling. Reproduced before fixing, and it still reproduced against
v0.15.89's "closed" ABA fix, because that fix never touched this decision.

The comparison is now against the pinned entry and nothing else. `in_lid` is the inode
`open_input_pinned()` pinned — the link for a symlinked input, the file otherwise — which is by
construction the entry `--rm` removes. One identity, one question. **A shape matrix structurally
cannot catch this** (every cell keeps one type from start to finish), so there is a separate
type-change test that polls for observable state, never a fixed sleep, and skips explicitly if
compression finishes before the swap lands.

### Also fixed from the same round

**`--rm` no longer requires procfs.** The pinned open reaches a non-symlink entry's data through
`/proc/self/fd/N`, so in a chroot or a minimal mount namespace `--rm` began failing on inputs
that were perfectly readable — something the plain `openat` it replaced never did. It now falls
back to opening the name and proving it is the pinned inode. That is safe here in a way it is
*not* for the symlink branch: it compares the **same fact** from both lookups and demands
equality, so an A→B→A gains nothing. `GZSTD_DEBUG_NO_PROCFS` forces the branch, because the real
trigger needs a user namespace this host restricts and the fallback would otherwise ship
untested.

**A failed removal no longer asks a deletion where the file is.** On `unlinkat` failure the
caller used to call `release()` and report the *original* pathname when it succeeded. A
`release()` that removes a **substituted** empty directory also succeeds — so the message sent
the user back to a path that did not have their file, while the real quarantine still holding it
sat under a name nothing reported. The quarantine path is now reported unconditionally.

**`release()` keeps its descriptor until the `rmdir` actually succeeds.** Closing it first made a
transient FUSE/NFS failure permanent: the destructor's retry had no descriptor, could not
establish ownership, and silently stranded an empty `.gzstd-q.*` while the command exited 0.

### What the reviewer confirmed rather than faulted

No path runs the `--rm` block without a successful pinned capture; dangling-symlink behaviour is
byte-identical to the old `openat` (exit 3, same diagnostic, nothing removed); relative target
resolution through the held parent is correct including when that parent is renamed, and for
`.`, `/`, and absolute targets; the readlink growth loop does not accept truncation; `errno` is
preserved across `close(efd)` on every failing path.

Suite: **405** default / **538** extensive, plus **322** on the `USE_NVCOMP=OFF` build.


## v0.15.89 — one lookup for two questions, and a measurement that cancelled a project

### Fixed — the one accepted hole from v0.15.88 is closed

`--rm` on a **symlinked** input was defeatable by an ABA exchange. The entry's identity (the
link, because that is what gets deleted) and the data's identity (its target, because that is
what gets compressed) were established by **two independent lookups of the same name**, so
swapping A→B→A between them satisfied both while they described different entries.

`open_input_pinned()` now resolves the user's name **exactly once**:

```
openat(parent, base, O_PATH|O_NOFOLLOW)   pins the ENTRY — fstat of it is provably that entry
  symlink  -> readlinkat(pinned, "")      the target is read FROM THE PINNED LINK
              openat(parent, target)      (a relative target resolves against the parent, which
                                           is where the link lives — so that is correct)
  otherwise-> open("/proc/self/fd/N")     a new description on the SAME inode, no name left
```

**The two cases cannot be unified, and the obvious one-liner does not work.** Re-opening
`/proc/self/fd/N` when `N` is an `O_PATH|O_NOFOLLOW` handle on a *symlink* fails `ELOOP` —
exactly the case the fix exists for. That was measured, not assumed; the design recorded as
"the concrete fix" before this release would have failed on every symlinked input. The readlink
buffer **grows until a short read** rather than trusting `st_size`, which some filesystems
report as `0` for symlinks. The pinning is gated behind `--rm` being in play, so a
many-small-files run pays nothing.

**`QuarantineDir::release()` no longer removes a directory it cannot prove is its own.** The
interior was already proven private (owner + `0700` through the held fd), but the *basename*
stayed movable afterwards, so an unconditional `unlinkat(name, AT_REMOVEDIR)` could remove a
substituted directory while ours — possibly still holding the user's input — was stranded under
a name nothing reports. The held descriptor now decides. The residual is bounded and stated:
`AT_REMOVEDIR` refuses a non-empty directory, so the worst case is removing an empty directory
that was not ours, never data.

**An identity flag that was set but never read.** `--overwrite`'s quarantine match compared
against `overwrite_id` without consulting `overwrite_id_ok`. Both paths that skip the `fstat`
currently `die()`, so it was latent rather than live — but a check whose result nothing reads is
how an identity test goes missing with no test failing. It is now required.

### Tests

**The `--rm` input-shape matrix asserts what got COMPRESSED**, which its predecessor did not.
The old test checked exit status and what survived — so a change that deleted the right entry
while reading the *wrong* data passed it. That is the same blind spot the output matrix had, and
it is the column that pins this fix down. Seven cells: plain, hardlink, and symlinks that are
relative, absolute, pointing outside the input directory, into a subdirectory, and chained — plus
a dangling link, which `O_PATH|O_NOFOLLOW` opens happily and which must still refuse.

Building the matrix immediately found a bug **in the matrix**: the chain case pointed at a link
another cell had already deleted, so it failed for a reason that had nothing to do with the code
under test. A matrix cell must not be able to fail because of another cell.

Suite: **403** default / **536** extensive, plus **322** on the `USE_NVCOMP=OFF` build. All
three green.

### Not covered, and worth saying plainly

No test here, or anywhere in the arc, uses an actual different-uid attacker. This box cannot
produce a second UID (no root; AppArmor sets `kernel.apparmor_restrict_unprivileged_userns=1`,
so `unshare -Ur` fails). The UID does not change what gzstd observes — it only decides whether
the attacker *can* perform the substitution, which inside the quarantine is settled by `0700`
plus ownership. The structural coverage is the matrix.

### Measured — GPUDirect Storage was investigated and rejected

Direct SSD→GPU compression was evaluated on the 8×H100 server and **is not worth building**:

| | |
|---|---|
| NVMe cold sequential read, `O_DIRECT` | **4.9 GB/s** |
| gzstd's *slowest current* H2D path | **25 GiB/s** |

GDS removes a host bounce from a link that already has a **5× margin over the storage feeding
it**, so it cannot raise any ceiling; its remaining benefit is CPU cycles, and compress uses 724%
of 25600% available. It would also cost a `nvidia-fs-dkms` install and a reboot with the AMD
IOMMU moved out of `DMA-FQ` mode. Not spent.

The measurement did find something real, which is recorded for the next release rather than
rushed into this one. `--gpu-only` compress runs at **2.03 GiB/s against `--cpu-only`'s 4.62** —
the CPU path saturates the drive and the GPU path does not reach half of it. The `h2d` region
holds a full XXH64 pass over every byte *and* a `cudaMemcpyAsync` from pageable memory, which is
host-synchronous, so nothing overlaps. Registering the reader pool — already allocated as one
contiguous 2 MiB-aligned region — measures **53.29 GiB/s against 25.17**, portable across all
eight GPUs from a single call, while `--pinned on` measures **16.94** and is worse than doing
nothing. That change also carries a trap: `release_input()` currently depends on pageable copies
being synchronous, so pinning the source without moving the release behind the H2D event turns it
into a use-after-free against an in-flight DMA.


## v0.15.88 — closing the no-attacker defects, and drawing the line

Two more review rounds on the quarantine machinery. The findings sorted cleanly along a
distinction this release makes explicit, and `AGENTS.md` now records it as policy:

> A defect reachable with **no attacker** is always fixed. A **same-uid** peer is out of scope —
> it can delete your files outright and no permission scheme helps. A **different-uid writer
> with write access to your input or output directory** is defended where the defence is cheap
> and self-contained, and **documented** where it is not.

### Fixed — reachable with no attacker

**A setgid bug introduced by the previous round's own fix.** The quarantine directory's privacy
proof tested `(mode & 07777) == 0700`, but a directory created beneath a setgid parent inherits
`S_ISGID` and comes out `02700`. So `--overwrite` and `--rm` **failed outright in a conventional
`2770` collaborative spool** — precisely the deployment the hardening exists to protect. The
test now masks `0777`. Verified against a real `2770` directory.

**Stranded entries were reported at the wrong path.** `release()` correctly refuses to forget a
quarantine it could not remove, but the callers discarded that result: after a failed
`unlinkat`, `--rm` named the original input while the file sat under `.gzstd-q.*/in`, and
`--overwrite` did the same, including on a failed restoration where it claimed "nothing was
removed". Both now report where the entry actually is. A FUSE or NFS `EIO` is sufficient to
reach this — no hostile writer required, which is why it was treated as seriously as a HIGH.

**Two `fstatat` calls per named input that only `--rm` consumed.** Now gated on `--rm`. This was
the one avoidable cost in the whole transaction: nothing else here runs per file without being
needed.

### Documented, deliberately not fixed

`--rm` on a **symlinked** input, in a directory a hostile different-uid writer controls, can be
defeated by an **ABA exchange**: put symlink A at the name, let the no-follow stat record it,
swap in B pointing at the real inode so the following stat passes, then restore A — both
observations are individually true and the pair is meaningless. An ordinary recheck cannot beat
ABA; closing it requires the entry itself to be the starting object of the target open. Recorded
in `AGENTS.md` as accepted, with instructions not to re-file it.

### Cost, measured by inspection rather than assumed

Asked directly where the hardening lands:

| | frequency |
|---|---|
| per frame / chunk / block | **none** |
| per tar-create member | **none** |
| per named input/output file | held parent descriptors, registration, one `renameat` |
| quarantine + `getrandom` | only for an existing `--overwrite` target, and `--rm` |

Large-file steady-state throughput does not contain any of it, and a `--tar` creation over a
large tree pays one output transaction for the archive rather than one per member. Plain
multi-file runs pay it linearly per file; "invisible" is not certified without measurement on
metadata-cold or networked storage, and is not claimed here.

For calibration, measured with `strace`: stock zstd 1.5.7 does `stat`/`unlink`/`open` by path
through `AT_FDCWD`, with no `O_EXCL`, no `O_NOFOLLOW`, and `--rm` as a bare `unlink()`. That
does not excuse a defect, and it does not establish that gzstd's remaining exposure is lower —
only that the baseline this tool is judged against has none of these protections.

---
## v0.15.86 — a private directory, and the destructive step that trusted a name twice

Round 13's two HIGHs were one root cause: **"rename aside, prove it is ours, unlink"** is three
operations on a name another writer can substitute between. A random quarantine name does not
close that — an inotify watcher learns it the moment the rename lands.

**Both deletions now happen inside a private directory.** `--overwrite` and `--rm` create a
0700 subdirectory beside the target, move the entry into it, validate it there, and unlink it
there. Same filesystem, so the rename is atomic and cheap, and `--overwrite` keeps its one-copy
space property.

What that defends is stated in the code rather than implied: a **different-uid** writer with
access to a shared output directory is fully excluded, which is the threat these findings
describe. A **same-uid** process is not, and cannot be by any permission scheme — nor does it
need a race, since it can delete the files outright.

**Cleanup now proves the leaf, not just the directory.** Owning the directory descriptor stopped
the fd from coming to mean a different directory; it said nothing about the entry. A writer
could rename our partial away, drop their file at that name, and an unconditional `unlinkat()`
deleted theirs. Cleanup now compares a **held descriptor** against the current entry and removes
it only on a match.

Not a remembered inode number, which was the first attempt: `sig_atomic_t` is 4 bytes on this
target and `ino_t` is 8, so that would have silently compared truncated halves of every inode.
Holding the descriptor keeps both `stat` structs local to the handler, where their width is
nobody's problem.

### The defect inside the fix, again

The first version of that guard did nothing, because there were **two** unlink sites — one in
`cleanup_tmp_file()` and an inline copy in the signal handler — and the guard went into one. The
attack still succeeded and the new test said so. There is now a single implementation called
from both paths.

That is this project's signature defect appearing *inside a fix for a mirrored-path finding*.
Three more of the same class were caught by testing during this round: a symlink `--rm` broke
because an entry moved into a subdirectory has its relative target broken (`--rm` removes the
LINK, so it must be validated by the link's own inode — "is this the data I compressed" is a
question about the target, and one identity cannot answer both); a quarantine directory leaked
because `die_io` calls `exit()`, which does not unwind; and the width bug above.

### For context, measured rather than assumed

`strace` on stock zstd 1.5.7 doing the same work:

```
stat("out.zst")                                        ← by path, follows symlinks
unlink("out.zst")                                      ← by path, unconditional, no identity check
openat(AT_FDCWD, "out.zst", O_WRONLY|O_CREAT|O_TRUNC)  ← no O_EXCL, no O_NOFOLLOW, no held dirfd
```

and `--rm` is a bare `unlink()` by path. **Every race raised against gzstd in this arc exists
unmitigated in the reference implementation.** That does not make the findings wrong — they are
real, and they are fixed — but it places the remaining exposure below what zstd ships today.

Two earlier claims corrected while measuring: `zstd -f` is **unlink-first**, not truncate in
place (the inode *number* is recycled, which is what made it look otherwise), and its symlink
output survives its target only as an accident of unlinking first.

Suite: **402** (535 extensive), green on both build configurations.

---


## v0.15.84 — the reviewer wrote the patch, and two product decisions

Round 11 returned five HIGHs and named an exact correction for each, so the arrangement was
inverted: **the reviewer wrote the code and I adjudicated it.** Eleven rounds of me writing the
fixes had produced a regression in three consecutive releases, so the experiment was worth
running.

It applied eight corrections. **Six went in as written**, each verified against its own defect:
the hardlink omission (a hardlink output has the same inode as its source, so the pre-existing
archive must be excluded as a directory *entry* — parent dev/ino plus basename — not as an
inode); `--rm` and `--overwrite` quarantine ordering; the `--adapt` profile transaction on a
held directory; `--calibrate` confirming its unlink; and the dangling-symlink registration.

**One was rejected in part.** It deleted `cleanup_tmp_file()` outright, taking the SYNCHRONOUS
fatal-path cleanup along with the asynchronous one — so a disk-full compress started leaving a
partial archive behind, against a test that asserts otherwise. The async reasoning was right; a
signal handler must not use a BORROWED descriptor, because the owner can close it and the next
`open()` reuses the number. But `die() → exit()` does not unwind main, so the descriptor is
still the right one there. The fix is to stop borrowing, not to stop cleaning up.

**And it broke the endian guard** with a `getrandom` call — a legitimate hit, not an on-disk
read, needing an exemption. It could not have known: there is no toolchain in the review
sandbox, which is why it was told to say the edits were unbuilt rather than claim otherwise.

Its summary said the 30-cell matrix expectations were unchanged. That was true of the
ASSERTIONS and false of the BEHAVIOUR — `-o dangling-symlink -f` now replaces the link instead
of following it. The matrix asserted exit code, survival and completeness but not **what the
output name became**. There is a KIND column now, and the new behaviour is better: dangling is
consistent with every other symlink shape instead of being the one case that writes to a path
the user never named.

### Two product decisions, measured against stock zstd

**`--overwrite` is unlink-first again.** The quarantine held the old output until installation,
which closed a race but cost the flag both properties it exists for — peak usage became old +
new, and a 400 GiB archive would need 800 GiB free. It now quarantines and unlinks
**immediately**: `renameat2(RENAME_NOREPLACE)` is atomic, so the entry is moved aside, proven to
be the one identified, and removed — same space and time profile as a plain unlink, and a file
another process creates at that path still cannot be deleted. The trade that remains is the
honest one, and it is now in `-h` and `--help`: **not atomic**, so a failed run leaves neither
old nor new.

**SIGINT removes the partial output, matching zstd.** Measured, not assumed: `zstd -f`
interrupted mid-compress removes its output and exits 2. gzstd deliberately **diverges on write
errors**: zstd leaves the partial behind, gzstd removes it (the suite has asserted that for a
long time). A truncated `.zst` is not a checkpoint — neither tool can resume one — so keeping it
only invites someone to mistake it for output. One further place gzstd does NOT match:
interrupted while overwriting, zstd destroys the old archive because it truncated in place;
gzstd's `-f` is atomic, so the old archive survives. Copying that would be a downgrade.

Also corrected while testing: the help claimed `--overwrite` produces "a NEW INODE". The inode
NUMBER is recycled by the filesystem — measured — though the substantive claim holds (the old
inode is released; a hard link to the old archive keeps its content and the link count drops).

### The tests, including two of my own that were not fit to ship

Codex added five matrix cases; the hardlink-descendant one catches its own HIGH. I added two
more for the product decisions — and the first versions were **timing-dependent**, using
`sleep 2` before acting. One extensive run failed exactly two tests; both are now event-driven,
waiting until the temp file exists so the signal is guaranteed to land mid-write, with an
explicit skip if the work finishes first. Five consecutive extensive runs clean since, with the
SIGINT test confirmed to run rather than skip.

A flaky test is worse than no test: it teaches people to re-run until green, which is exactly
how a real intermittent failure gets waved through. Both were also checked to still FAIL
against the design they were written to prevent — the `--overwrite` one initially passed on
both designs, guarding the documentation and not the semantics, and needed a mid-run assertion
to distinguish them.

Suite: **401** (534 extensive), green on both build configurations.

---


## v0.15.81 — the matrix first, then the seven

Round 10 returned four HIGHs and answered the question I had asked about my own previous fix:
*"is 'follow for the clobber guard, NOFOLLOW for identity' the right split, or have I now got
it backwards somewhere else?"* Backwards somewhere else.

**The archive silently omitted a source file.** The clobber guard follows (correct — the open
follows), but the archive-identity registration was also a path-based `stat()`, which follows —
so with the output a symlink to a source file, the SOURCE's inode was registered as "the
archive", and the tar walk skips any inode on that list:

```
ln -s root/data out.tar.zst
gzstd -f -o out.tar.zst --tar root     → exit 0, root/data NOT in the archive
```

Nothing destroyed, success reported, backup quietly incomplete — the worst shape this tool can
take. The walker uses `lstat`, so the registration must not follow either: the opposite answer
from the clobber guard twelve lines above it. That is the third appearance of one root cause in
this area — **a single predicate answering two questions is wrong for one of them** — after
`seq` (data frames vs any frames) and `opt`/`pass_opt` (request vs pass).

### The matrix came first, and immediately earned it

Before fixing anything: five output shapes (absent, regular, symlink→outside, symlink→source,
dangling) × three force modes × {plain, `--tar`} = **30 asserted cells**, each checking the exit
code, what survives, and — for `--tar` — whether the archive is COMPLETE. That last column is
the one that matters: an archive can succeed, destroy nothing, and still be missing a file.

It flagged exactly one cell red, confirming the finding and bounding it. Then it caught **two
regressions in the fixes themselves**, before they shipped:

- Keying the new `O_EXCL` on the wrong stat would have put it on `-o /dev/null` and on dangling
  symlinks, breaking both. It keys on the NAME stat instead — a name that already held something
  was never the no-clobber case.
- The first `--rm` identity check compared `lstat` against a descriptor `open()` had followed,
  so **every symlinked input** started refusing `--rm`.

The two previous releases each shipped a silent defect in this exact area because nothing
enumerated these shapes. Two more were caught within the hour of the matrix existing.

### The seven items

1. **Archive identity** registers through the held directory with `AT_SYMLINK_NOFOLLOW`.
2. **The no-`-f` create is atomic.** Stat-then-open is not: a symlink dropped in that window was
   followed and an unrelated file truncated with no `-f`. `O_EXCL` makes the decision and the
   act one operation.
3. **`--calibrate` creates AND unlinks through a held parent.** `O_EXCL` protects creation, not a
   later unlink by pathname; I had called this out of scope and that was wrong.
4. **Abnormal cleanup is descriptor-bound.** The signal/atexit path stored a full pathname and
   `unlink()`ed it — after a parent exchange that deletes an unrelated file. "Best effort"
   licenses leaving a partial file, not removing someone else's. `unlinkat` is equally
   async-signal-safe.
5. **`--rm` deletes the file it compressed**, verified by inode rather than by re-resolving the
   name at the end of the run.
6. **The fallback copy reads its temp through the transaction** — binding only the destination
   is half a transaction.
7. **Only `ENOENT` means absent**, and a pre-unlink identity check that cannot look is fatal.

Suite: **394** (527 extensive), green on both build configurations.

---


## v0.15.80 — two questions, one stat, and a `-f` guard that stopped guarding

Found while preparing v0.15.79 for review, not by the review. Converting the output path to
`fstatat` I used `AT_SYMLINK_NOFOLLOW` for the existence and type test, reasoning that a symlink
is a symlink and not the thing it points at. That is correct for **identity** questions and
wrong for the **clobber guard**, because a symlink is not a regular file:

```
                      -o symlink-to-existing-file, NO -f
v0.15.78              exit 3, target intact          ← correct
v0.15.79              exit 0, TARGET DESTROYED       ← regression
v0.15.80              exit 3, target intact
```

`-o link-to-file` stopped counting as "an output already exists here", skipped the `-f`
requirement entirely, and truncated whatever the link pointed at. Silent, exit 0.

There are two different questions and one stat cannot answer both:

- **"Does a NAME exist here?"** — `AT_SYMLINK_NOFOLLOW`. Used for the symlink notice and to
  confirm the unlink removed the link rather than its target.
- **"Will I DESTROY existing data?"** — follows, because the plain-compress open follows. This
  is what gates `-f`.

Following is safe for the guard *precisely because* every destructive step below it is
`*at()`-based and `O_NOFOLLOW`: the guard may ask about the target while the operations refuse
to touch it. Verified against v0.15.78 across four shapes — symlink without `-f`, symlink with
`-f`, dangling symlink, plain existing file — behaviour identical on all four, and the full
round-6-to-9 aliasing matrix still refuses with both legitimate patterns still working.

A regression test now covers it. Nothing caught this because every other test on that path uses
real files, and the guard only fails for a symlink; the suite grew by 34 tests across this arc
without once pointing `-o` at a link to something that mattered.

The general lesson is the one this arc keeps producing in new costumes: **a single predicate
answering two questions is wrong for one of them.** `seq` counted data frames and was asked
whether any frame existed; `opt` described the request and was asked what the pass was doing;
one stat described a name and was asked about its target. The fix each time is two names for two
facts, not a cleverer single check.

---
## v0.15.79 — the output becomes one transaction

Round 9 confirmed the doubt this release started from, and made it worse than stated. The
output's parent directory was resolved **five separate times** between deciding the
destination was safe and writing it:

```
path_containment()   opens the parent to walk        ← the verdict came from here
fs::exists(output)   resolves the whole path again
open(output,O_PATH)  again, for the identity check
fs::remove(output)   again — and this one destroys
open(dir) in open_output_verified()                  ← v0.15.78 fixed only this one
```

Each lookup asked the filesystem afresh, so an attacker who can swap an intermediate
component — no root, no mount, `renameat2(RENAME_EXCHANGE)` in a loop — could have the guard
approve one directory while the unlink destroyed a file in another. v0.15.78 moved the *final*
open to a descriptor, which was the last of the five and therefore the least useful one.

**The destination directory is now opened once and held for the whole transaction.** Existence
and type (`fstatat`, `AT_SYMLINK_NOFOLLOW`), the containment walk, the identity check
(`openat`, `O_NOFOLLOW`), the unlink (`unlinkat`), the create, the atomic temp, the install
(`renameat`, both ends through the same descriptor) and the fallback recreation all go through
that one `fd`. The directory that was approved is the directory that gets written into by
construction, not by re-checking. The final component never contains a slash, so no `*at()`
call can walk out of the held directory.

Two things fell out of doing it properly. The symlink-target resolution added in v0.15.78 is
gone — `unlinkat` removes a link rather than its target and the create uses `O_NOFOLLOW`, so
the target is unreachable either way; that was also the last caller of `weakly_canonical` on
this path, a mount-blind function that had been guarding against a mount-based alias. And
`exists` now means *the name exists*, not *its target exists*, so a dangling link no longer
reads as absent and a link to a source no longer reads as a regular file.

Round 9's other findings, all confirmed by inspection or measurement:

**A third fail-open `fstat`, in the function whose other two were just fixed.** Written as
`fstat(...) == 0 && S_ISREG(...) && size > 0`, a failed stat silently skipped the truncate — so
a shorter archive written over a longer one kept the old tail, exited 0, and `--rm` deleted the
input. It reads as a type test rather than a safety check, which is exactly why it survived two
passes over the same function.

**`--sync-output` did not sync what it installed.** The temp is fsynced before the rename; when
the rename fails, the fallback copies into a *different inode*, closed it unsynced, and then
removed the durable temp. Durability has to follow the bytes the user ends up with.

**The endian check failed twice more — nine and ten.** An exemption applied to a whole logical
line, and because a multi-line lambda folds into one logical line, a single permitted
`cudaMalloc` silenced every forbidden call in the same body. Exemptions now **subtract**: each
is deleted from the text and the remainder re-tested, so an allowed call can only ever excuse
itself. And the pattern required the width to *follow* the address while the comments claimed
order independence, so `copy_n(src, 4, reinterpret_cast<char*>(&magic))` was missed; both orders
now match. Verified against all eight catchable shapes; the one known hole — a width held in a
variable — is documented rather than implied away.

**The check now runs in `gzstd-test.sh`**, not only in the release workflow. A host-order read
behaves perfectly on every machine here, so the first thing that could catch it was a tag,
which is far too late for a check that takes well under a second.

M2 was confirmed as having crossed its boundary, and the judgement that `PassOpt{opt}` should
remain constructible was endorsed: preventing deliberate misuse would mean restructuring
ownership for no gain against the two historical accidental misreads.

Suite: **390** (523 extensive), green on both build configurations.

---
## v0.15.78 — one abstraction boundary further

Round 8 assessed the four structural mechanisms from v0.15.77 rather than the findings, and
returned the most precise verdict of the arc:

> *"Structural fixes are not inherently just larger patches for the same problem; **M3
> demonstrates real closure**. Here, however, **two of four mechanisms stop one abstraction
> boundary short**."*

M3 (separate frame counters) closed its class outright. M2 closed the wrong *read* and left the
wrong *argument*. M1 made a rigorous decision about a **pathname** while the destruction happens
through a **descriptor**. This release moves the two short mechanisms across that boundary.

**The output invariant now attaches to a descriptor, not a name.** Identity comparison can only
reject an object it has been told about, and for a directory source only the directory's own
inode is captured — enumerating every descendant would mean walking the tree twice. So a
symlink appearing at the output name and pointing at a file *inside* a source opened cleanly:
the victim's inode is not the directory's inode, the forbidden check passed, and `ftruncate`
destroyed it. No `--overwrite` required, and the same window reopens after `--overwrite`'s
unlink. Identity could never close that, because the set is incomplete by construction.

When tar sources are being protected, the final component is now opened through a **held parent
descriptor with `O_NOFOLLOW`**, so there is no name left for anyone to swap and a symlink at
that position is refused rather than followed. A plain compress still follows a symlink output,
which is deliberate and unchanged. Measured deterministically with a *dangling* link — the same
state the race produces, since `fs::exists` is false and the absent-output route runs:
v0.15.77 created a 281-byte archive inside the source tree through the link; v0.15.78 refuses.

Two more fail-open checks in the same function are closed: a failed `fstat` proceeded to the
truncate having proven nothing, and the post-unlink recheck treated *any* `lstat` failure —
not only `ENOENT` — as "the name is gone".

**`build_pass_verify_pool(opt)` no longer compiles.** v0.15.77 moved the pool builder into a
function where the request-level `opt` is out of scope, which fixed the wrong read and left the
wrong argument: both objects are `Options`, so passing the wrong one at the call site was still
type-correct. A narrow `PassOpt` wrapper — explicit constructor, no conversion from `Options` —
now makes that a compile error: *"could not convert 'opt' from 'Options' to 'PassOpt'"*.
Deliberately narrow, per the reviewer's own recommendation: the retry driver and the pool
builder, not a project-wide conversion.

**The endian check caught its fifth form — its own.** It missed `pread(fd, &magic, 4, off)`,
because the pattern required the address-of to be the *first* argument, while the script's
comment claimed it caught `pread`. Fixing that produced two further mistakes worth recording,
since the point of this file is that hand-written patterns fail: removing the first-argument
constraint made every `a && b` match and the check drowned in its own noise; the `[^&]&` guard
added to fix *that* could not match `memcpy(&x, …)`, because there is no character between the
paren and the ampersand for it to consume — so it silently stopped catching the very first
form on its own list. It is now anchored on **argument position** (an address-of argument
follows `(` or `,`, which `&&` never does), folds multi-line calls into logical lines, and is
verified against all five historical forms plus a member expression.

It also **stops overstating itself**: a width held in a variable is not matched, and neither is
a struct overlay, a union, or an `istream::read`. A clean run means "no instance of the known
shapes", not "no host-order read" — the `static_assert` is what actually makes big-endian
unreachable. A check that claims more than it does is worse than one that is merely narrow.

Also: a symlink *source* is captured with `O_NOFOLLOW`, so capture and walker agree by
construction — following it added the link's target directory to the containment set and
refused a safe output inside it, when the archive was only ever going to contain the link;
`--calibrate` no longer prints `profile recorded:` with an empty path when there is no cache
directory; and the rename-failure fallback no longer writes through a symlink. That last one is
the v0.15.70 lesson again — an emulation must reproduce every effect of the operation it stands
in for, and a rename *replaces a name* rather than writing through it, so the fallback now drops
the name and creates it `O_EXCL|O_NOFOLLOW`.

Suite: **389** (522 extensive), green on both build configurations.

---
## v0.15.77 — fixing the pattern instead of the findings

Round 7 rejected five of my round-6 fixes and asked the question that mattered more than
the list: *is this converging?* Its answer was precise —

> *"The original defect set is converging, but this fix series is still generating fix-local
> follow-ons… The failures cluster around **invariants being represented by the wrong state
> or checked at only one spelling of a path**."*

— and it prescribed four structural changes rather than five patches. This release makes
those changes. Each one turns a class of defect into something that cannot recur silently,
rather than repairing the instance that was found.

**A safety check that cannot reach a verdict must say so.** The `--tar` containment guard
returned a bool, which forced every way of NOT KNOWING — an unopenable parent, a chain past
the depth bound, a failed `fstat` — to be reported as "outside", i.e. as permission to
destroy. That is the same fail-open shape as the `weakly_canonical` version it replaced,
which silently skipped whenever canonicalization errored: I fixed the mechanism in v0.15.74
and reproduced the failure mode in a new spelling. It is now tri-state — `INSIDE`, `OUTSIDE`,
`UNKNOWN` — and only `OUTSIDE` proceeds. There is exactly one exit that may return `OUTSIDE`:
having walked every ancestor to the filesystem root.

**And the destructive step's FAILURE must be fatal.** Round 7's trigger needed no root, no
bind mount and no race:

```
ln -s root/data alias/out ; chmod a-w alias
gzstd --overwrite -o alias/out --tar root      → exit 0, root/data destroyed
```

The containment walk cleared `alias/out` — its own parent chain really is outside the source
— then `fs::remove` failed on the unwritable directory and **the failure was ignored**, so the
symlink survived and the open followed it into `root/data`. The old comment said the fopen
"will return its own error"; it did not, because the open succeeded on the wrong file. That
unlink is not "make room", it is what neutralises a symlink at the output name, and a failed
neutralising step is now fatal. The containment check separately resolves an existing output
symlink, so both halves have to fail before anything is destroyed. Measured: 27 bytes of
source content intact where v0.15.76 replaced them with 227 bytes of archive.

**The `opt` / `pass_opt` misread is now a compile error.** The compress driver holds both what
the user asked for and what this pass is doing; they diverge the moment a GPU fault rebuilds
CPU-only. Both spellings compile and the wrong one usually behaves, which is why the same
misread landed twice — v0.15.74 for *whether* to build a verify pool (the rebuild went
unverified), v0.15.76 for *how to size it*, three lines below the first fix. Pool construction
now lives in a free function that receives **only** the pass's options. `opt` is not in scope
there. Anything genuinely request-level has to be passed as an explicit parameter, which
forces the question to be answered rather than assumed.

**`seq` counted DATA frames and was being asked "did any frame exist at all".** Two different
questions, one variable, and both possible answers were wrong at different times: the original
guard let trailing garbage behind a skippable-only stream pass, and widening it in v0.15.75
then rejected a *valid* skippable-only stream — so `printf '\x50\x2a\x4d\x18\0\0\0\0' | gzstd
-t` exited 4 while the identical bytes in a named file exited 0. Both readers now track `seq`
and `total_frames` separately. A stream of complete skippable frames is valid on every route
(stock zstd agrees); a truncated stub is fatal on every route.

**A mechanical check replaces the audit that kept missing sites.** `scripts/check-endian-reads.sh`
fails CI on any host-order read of an on-disk integer, and it exists because inspection
demonstrably does not work here: v0.15.75's sweep left two sites that `pread` straight into a
`uint32_t` (a grep for `memcpy` cannot see a `pread`), and round 7 found a third,
`memcpy(&m32, magic, 4)` — my audit pattern had been `memcpy\(&[a-z_]+,` and the **digit** in
`m32` fell outside the character class. The first draft of the checker then failed its own
test by listing readers *by name*, missing `pr(&magic, 4, pos)` where `pr` is a local lambda —
the same mistake a third time. And the first shape-based draft required a *literal* width, so
`pr(&magic, sizeof magic, pos)` — the same defect written the way a careful author would write
it — passed: four consecutive patterns, each of which looked complete. It now matches any call
whose first argument is the address of a scalar and whose arguments state a width, a bare 4/8
**or** a `sizeof`. Inclusion is by shape; **exclusion is by callee name** (three CUDA/time
functions, each with a reason), on the asymmetry that a wrong exclusion costs a false positive
somebody has to justify while a wrong inclusion ships a corrupt archive. Verified against all
four historical forms, and clean on the current tree.

Also closed, from the same round: a redirected stdout is registered as the archive's identity,
so `gzstd --tar root > root/a.tar.zst` no longer stores a zero-length member for itself
(`S_ISREG` is the whole test — a pipe has no inode to collide with); `--adapt` warns when it
has **no cache directory at all** (`HOME` and `XDG_CACHE_HOME` both unset returned before the
four instrumented failure points, so the warning added for them never fired for the first
failure path); `--keep-going` now recovers on the single-frame streaming route, which had
called `die_data` unconditionally while telling the user to re-run with `--keep-going` —
measured, 2,995,004 of 3,000,000 bytes recovered at exit 7 where it previously recovered
nothing at exit 4; and `--verify-engine=gpu` on an empty input says that the 13-byte frame was
checked on the CPU instead of printing "engine: GPU" and going quiet.

Six of round 7's items were **confirmed**, including the streaming frame-tracking added in
v0.15.75 that I had flagged as the fix I trusted least.

Suite: **387** (520 extensive), green on both build configurations.

---
## v0.15.76 — the same misread, three lines below its own fix

Not a review finding. Sweeping the compress retry loop for other `opt.` reads that should be
`pass_opt.` — the distinction v0.15.74 fixed for the verify pool's *existence* — turned up two
more in the pool's own **sizing**, inside the block that fix had just edited:

```
const int vmax = opt.cpu_only ? clamp(cores/16, 2, 16) : clamp(cores/8, 4, 32);
size_t frame_est = max(opt.chunk_mib, 1) * ONE_MIB;
if (!opt.cpu_only) frame_est = min(frame_est, GPU_SUBCHUNK_MAX);
```

On a CPU rebuild after a GPU fault, `opt.cpu_only` is still false. So `vmax` took the hybrid
branch — *"the GPUs carry the compression, so the CPU pool has spare cores"* — while the CPU
was in fact carrying it, and `frame_est` clamped to the GPU subchunk size for frames that are
now `chunk_mib`-sized. Sizing only, not correctness: the rebuild still verifies, just with a
pool ceiling and queue depth computed for a pass that is not the one running.

Worth recording for what it says about the defect rather than its severity. `opt` and
`pass_opt` are both in scope, both spellings compile, and the wrong one usually behaves — so
the misread survived the fix written for it, three lines away. The rule is: any read
describing **what this pass is doing** must be `pass_opt`; any read describing **what the user
asked for** must be `opt`. That is not enforceable by the type system as the code stands, and
a structural fix (a pass-scoped view that makes the wrong read not compile) is the real
answer if this recurs a third time.

Also: the second all-zeros word scan (`blk_zero_4k`) is annotated like the first, recording
that it is deliberately not an `rd_le64` — it tests a word against zero, which is the same
value in every byte order, and reads a data block rather than an on-disk field. Both are now
labelled so the next endianness audit does not have to re-derive it.

Suite: **383** (516 extensive), green on both build configurations.

---
## v0.15.75 — closing the round-6 remainder, and one reader for one format

v0.15.74 fixed round 6's two HIGH findings and left four smaller ones open on the grounds
that none were tag blockers. They are closed here, along with the duplication that produced
two of them.

**Eighteen open-coded host-order reads became one reader.** Every integer gzstd reads from a
file is little-endian, and the correct shift-based reader already existed in the tree — three
times, copy-pasted, as identical `rd_u32` lambdas — while eighteen other sites did
`memcpy(&v, p, 4)`, which is a HOST-order read that merely happens to be right on x86. There
is now a single `rd_le32`/`rd_le64`, and the three duplicate lambdas are gone. The change is
net-negative in lines and free at runtime (every compiler folds it back to one load).

The justification is not portability. Two of those parsers are **mirrored walks of the same
on-disk structure**, and this project's most expensive findings have all been mirrored paths
drifting apart — round 6 alone produced two more. One reader for one format removes the
class. Big-endian correctness is a side effect, and the sweep immediately paid for itself:
converting the `memcpy` sites left a pair in the `-l` buffered walk that `pread` **straight
into a `uint32_t`**, which is the same host-order read spelled differently, and is exactly the
kind of site a grep for `memcpy` cannot see.

**Big-endian is now declared unsupported at build time** rather than implied to work. A
`static_assert` fails the build on a big-endian target. The readers are byte-order explicit
and the format handling would likely survive a port — but "would likely survive" is not worth
shipping untested, and there is no big-endian hardware here: no s390x, no qemu-user, no cross
toolchain. The one platform that would exercise it in 2026 is IBM Z, which nvCOMP does not
build for. The previous state was worse than either honest position: a code comment asserting
big-endian intent for XXH64 while the upfront magic check could not read its own format.

**`-l` accepted trailing junk that `-t` rejected.** Append one, two or three bytes to a valid
archive and `-l` exited 0 while `-t` exited 4 on the same file. Both `-l` walks got there
independently — the buffered one returned an explicit `fsize - pos < 4` tolerance, the mmap
one looped on `pos + 4 <= fsize` and simply ran out with no post-loop verdict at all. They
agreed, and they were both wrong; two mirrored implementations reaching the same wrong answer
is not corroboration. A valid stream tiles its file exactly, and both now require that.

**The damaged-trailer verdict no longer depends on the route.** An archive whose first frame
carries no content-size header, or exceeds 256 MiB, goes to the single-frame streaming decoder
instead of the frame splitter — and that decoder had no truncated-skippable exception, so
identical damage was recoverable or fatal depending on `--chunk-size`. It now tracks the frame
in progress (its first bytes and true length) and asks the same shared predicate, counting
completed DATA frames only, so a stream that is nothing but a broken trailer stays fatal there
too. Verified on a sizeless piped-`zstd` stream, which takes that route without needing a
256 MiB frame: `-d` recovers byte-identically and warns, `-t` fails.

**`--adapt` could fail to persist forever without saying so.** All four failure paths — no
lock, no temp file, write, rename — logged at `-v` only. On a read-only cache directory, or a
filesystem whose `flock` does nothing, the profile never converged and the user saw only that
`--adapt` didn't seem to do anything, which at default verbosity is indistinguishable from
"still learning". It is now a one-shot warning at default verbosity (surviving `-q`, silenced
by `-qq`), naming the specific cause. Silent when the save succeeds.

Also: the dead `ZSTD_isError` line round 6 flagged in the sliding-window verifier is gone.

**Four regression tests**, three of which discriminate against v0.15.74 (`-l` on a short tail
0 → 4, `-d` on the streaming route 4 → 0, and the `--adapt` warning absent → present); the
fourth guards behaviour that was already correct and is labelled as a guard, not a fix.

Suite: **383** (516 extensive), green on both build configurations.

---
## v0.15.74 — a guard that compared paths, and the one run that skipped verification

Round 6. Six of the eight fixes from v0.15.73 were **rejected**, but the more useful number
is the other one: *"I found no additional independent production defect outside the rejected
fixes."* After five rounds of finding decade-old defects, the sixth found none — every
finding here is a hole in the previous patch or in the question asked about it. The review is
now reviewing the diff, not the codebase.

**The `--tar` containment guard compared PATHS, so a bind mount walked straight through it.**
v0.15.73 stopped `--overwrite` from placing the archive inside a directory source by testing
whether `weakly_canonical(source)` was a path prefix of `weakly_canonical(output)`.
Canonicalization resolves symlinks and knows nothing about **mounts**:

    mount --bind root alias
    gzstd --overwrite -o alias/data --tar root      # unlinks root/data through the alias

`canonical("root")` is not a prefix of `canonical("alias/data")`, so the check passed and the
unlink destroyed the source. No race, no fault injection. Containment is now decided by
**identity** — walking the output's parent chain upward with `openat("..")` and comparing
`(st_dev, st_ino)` at each level. A bind mount shares the superblock, so the alias and the
original report the same inode and the walk sees through it. The string form also skipped the
check silently whenever canonicalization failed; there is nothing to fail now. Every
constructible aliasing shape (symlinked parent, symlinked source, `-C`-resolved, directory
source) still refuses, and the legitimate case still works. *The bind-mount trigger itself
could not be constructed here — no root, and user namespaces are disabled — so that one shape
is reasoned, not measured.*

**`--verify` checked nothing on the one run that most needed it.** The verify pool is built
per pass, but its condition read the ORIGINAL `opt`: *"gpu_verify is set, so the GPU worker
does the verifying, and this pass needs no CPU pool."* True — until a GPU fault or a failed
GPU verify discards the output and rebuilds CPU-only. `pass_opt.cpu_only` flipped;
`pass_opt.gpu_verify` did not. So the rebuild had no CPU pool (opt said the GPU verifies) and
no GPU worker either, and the run that exists *because* verification failed was the one run
that went unverified — exit 0, and with `--rm` the source deleted afterwards. Measured: a
clean `--cpu-only --verify` prints `[VERIFY] … 1 frames (7.63 MiB) checked`; the rebuild
printed no `[VERIFY]` line at all. Both halves are fixed — the condition now reads `pass_opt`,
and the rebuild clears `gpu_verify` — because either alone is a no-op.

This is the v0.15.72 lesson on a second axis. That one was a feature attached to one of four
*paths*; this is a condition that stopped holding once the state changed underneath it. The
question to ask is not only "which paths does this reach" but "does its predicate survive the
state change".

**Four bytes of garbage passed as a valid archive.** `printf '\x50\x2a\x4d\x18' | gzstd -t`
printed OK and exited 0. Zero frames plus a non-empty leftover fell between three checks that
each excluded it: the truncated-skippable tolerance required `seq > 0`, the trailing-bytes
rejection required `seq > 0`, and the no-frames rejection required `leftover == 0`. Both
zero-frame shapes — no bytes, and bytes that never became a frame — now go through one branch.

**The deliberate divergence is now split by command, which is what it should always have
been.** gzstd tolerates a truncated trailing *skippable* frame because that is its own
seek-table trailer and it carries no user data. The reviewer accepted the payload-safety
reasoning and rejected the scope, in one line worth keeping: *"`-t`'s job is integrity
validation, not payload salvage."* So `-d` and `-l` still recover every data byte, and `-t`
now fails — a physically truncated file is not intact, however much of it is recoverable.
The recovery warning also moved from `-v`-only to **default verbosity**: a warning nobody
reads at the default is the same false success in a quieter font. The parallel reader, which
suppressed the failure and said nothing even at `-vvv`, now warns identically — the tolerance
decision and the warning both live in one shared helper, because three readers answering this
question separately is precisely how it went wrong.

**A dangling symlink stopped being archivable.** v0.15.73's fail-closed source capture used a
symlink-*following* `O_PATH`, which is ENOENT on a broken link — so `gzstd --tar dangling`
became a hard error against a source the walker happily archives (it `lstat`s, as GNU tar
does). It now retries with `O_NOFOLLOW`, capturing the link itself. A genuinely missing source
is still fatal.

**`-f` archived the archive.** The output is created before the walk, so it is just another
file in the tree: `gzstd -f -o root/a.tar.zst --tar root` stored a zero-length member for
itself, and a repeated backup nested the previous archive inside the new one. The walk now
skips the archive by identity — the file being written and, on the atomic path, the existing
archive its temp will replace. Done in Pass C, where everything has been `lstat`ed already, so
it costs nothing; doing it in Pass A would have added a syscall per leaf to the one pass that
deliberately avoids them. GNU tar reports this case the same way ("file is the archive; not
dumped").

**Three of the round-5 regression tests asserted the wrong thing** and this entry fixes them
too — the point the reviewer made is that a test which passes for the wrong reason is
indistinguishable from one that works. The trailing-frame test checked `-l` on the empty file
but not on the appended malformed tail; the archive-inside-source test checked only the exit
status while the archive contained itself; the GPU-verify test checked a final round-trip,
which a trustworthy CPU compressor satisfies whether or not a verifier ran. All three now
assert the observable. Three new tests cover the lone-stub stream, the `-d`/`-t` split with
its default-verbosity warning, and the dangling-symlink source.

Two fixes were **confirmed**: the sliding-window verifier's frame-boundary requirement (the
zero-length-buffer worry was impossible — `note_output` returns early on `n == 0`), and the
`--adapt` profile lock's skip-the-save behaviour.

Still open and deliberately not in this patch: `-l` accepts 1–3 trailing bytes that `-t`
rejects (both walks stop at `pos + 4 <= fsize` with no leftover check); the skippable
tolerance is route-dependent above a 256 MiB first frame; and XXH64 is now endian-correct
while ~13 other on-disk reads are still host-order `memcpy`, so big-endian remains
non-functional either way — that one wants a decision (declare BE unsupported) rather than a
patch.

Suite: **380**, green on both build configurations.

---
## v0.15.73 — false-success in the hot path, and an ordering bug that made last round's fix a no-op

Round 5, open mandate. Seven findings — three HIGH — and two of them were in code nobody had
touched in five rounds.

**`gzstd -t` passed a damaged archive.** Append the four-byte zstd magic `28 b5 2f fd` to a
valid file: stock zstd says *"premature end"*, gzstd said **OK and exited 0** — and
`gzstd -d --rm` decompressed the intact prefix and then deleted the archive. Two readers
tolerated up to 8 trailing bytes as "could be padding"; the sequential splitter was worse
still, accepting an ARBITRARY amount behind a `-v` warning. An empty `.zst` was accepted too.
Fixing it took **three** decompress paths, not the two reported: the empty-file case routes to
the sizeless-frame streamer, so repairing the splitter and the parallel reader moved nothing.
zstd's verdict and gzstd's now agree on every malformed shape tested.

**One deliberate divergence, and it is the interesting part.** Making trailing bytes fatal
broke gzstd's documented behaviour of falling back to the decompress walk when its own index
trailer is damaged. The resolution is not leniency but a distinction: **a skippable frame
carries no user data**, so a clipped one at EOF cannot hide missing content, while a truncated
DATA frame can. A truncated trailing skippable frame now warns and falls back; everything else
is fatal. gzstd is therefore *more* permissive than stock zstd for exactly one case — its own
optimization trailer — which is a conscious trade against the drop-in goal, recorded at the
predicate.

**Last round's `--tar` fix was a no-op, and resolution was only half the reason.** The capture
opened the RAW source strings while creation resolves them against the positional `-C`
directory, so `--overwrite -o root/data --tar -C root data` still destroyed the file. Fixing
resolution alone **still destroyed it**: `--overwrite` unlinks the output BEFORE
`open_output_verified()` runs, so the fd-identity check was comparing against a file already
deleted. The check had to move ahead of the unlink. Sources now also fail closed when they
cannot be identified, and `--overwrite` refuses to place the archive inside a source directory
(`-f` still may — that path renames rather than unlinking).

**The sliding-window verifier accepted truncated frames.** Matching digests and byte counts are
not sufficient: `ZSTD_decompressStream` can emit every plaintext byte and still return a
positive "more input required" hint, which a frame missing its checksum does exactly. It now
requires a clean frame boundary, as the older frame verifier always did.

Also: the empty frame v0.15.72 introduced bypassed `--verify` (`-T1 --verify empty` produced a
valid archive and reported zero frames checked); `-l` listed an empty file as a tidy zero-frame
row and trusted an attacker-controlled skippable size without bounds-checking it in the mmap
walk; the `--adapt` profile lock proceeded unlocked on failure, reinstating the lost-update
race it exists to prevent; and the in-tree XXH64 read host-endian lanes, which would have
produced spec-violating checksums on a big-endian host. XXH64 was re-validated 57/57 against
libzstd after that change.

Six of the nine round-4 fixes were confirmed. Suite: **377**, green on both build
configurations — including a regression the suite caught and this entry describes.

**Three regression tests added**, closing the gap that produced most of these findings: a
truncated trailing frame and an empty `.zst` are rejected (asserting `-t`, `-d` and `-l`);
`--verify` reports a non-zero checked count on `-T1` and `--sliding-window`, not merely exit 0;
and empty input yields a valid frame on every compress path, cross-checked against stock
`zstd -t`. All three fail on the pre-fix binary — the assertions were verified to discriminate,
not just to pass.

---
## v0.15.72 — a feature attached to one of four write paths, and the checksum we never owned

Round 4, open mandate again. Nine findings: four HIGH, four MEDIUM, one LOW — and an
explicit answer to the question of whether the remainder was below the shipping bar:
*"the remaining source-truncation race and the two silent integrity failures are
independently tag blockers."*

**`--verify` was verifying NOTHING on two of the four compress paths.** The verifier taps
the ordered writer's in-order drain, and `-T1` and `--sliding-window` do not use an ordered
writer — so they created a verification pool, checked zero frames, and exited 0. gzstd
printed its own proof and nobody read it:

    -T1 --verify              → [VERIFY] ... 0 frames (0.00 B) checked
    --sliding-window --verify → [VERIFY] ... 0 frames (0.00 B) checked
    default --verify          → [VERIFY] ... 1 frames (3.88 MiB) checked

With `--rm` that removed the source after an integrity check that never happened. `-T1` now
submits its frames directly. `--sliding-window` cannot — its whole output is ONE frame
emitted incrementally, so there is no completed frame to hand over — and it gets a
**streaming verifier** instead: a dedicated thread decompresses the bytes as they are
written while the compressor digests the input, and the two rolling XXH64 digests must
match. Constant memory, which matters precisely here, since this mode exists for very large
windows and buffering the plaintext to compare would defeat it. (The obvious alternative,
rejecting `--verify --sliding-window`, was declined: the user asked for verification and an
extra thread is what that costs.)

**gzstd never owned an XXH64 — and that was a load-time portability defect.** It had the
plumbing to ATTACH a content checksum (`gpu_frame_add_checksum` flips the flag bit and
appends four bytes) but not to COMPUTE one; the value came from `extern "C" ZSTD_XXH64`.
libzstd compiles xxhash in but does not always EXPORT it: Ubuntu's libzstd 1.5.5 exports
zero XXH symbols, so a dynamically linked GPU build died with `undefined symbol:
ZSTD_XXH64` **before running any GPU code**. Release builds are static, which is why this
never bit in the field. XXH64 is now implemented in-tree — validated byte-for-byte against
libzstd's own across 57 cases (lengths 0–8192, three seeds, streaming fed in 1/3/7/32/33/100
byte slices) — and both the GPU frame checksum and the new stream verifier use it. The
binary now carries no XXH reference at all and loads against a stock distro libzstd.
Note this never affected CHECKING: libzstd validates the frame checksum internally on every
decompress, which needs no exported symbol. Only our producer-side computation did.

**The same write-path divergence hid a second bug, wider than reported.** An empty input
produced a ZERO-BYTE output — not a zstd stream at all — because the frame-parallel paths
queue nothing and the writer writes nothing. The review found it in `-T1`; checking all four
paths found `cpu_mt` and `hybrid` too, with `--sliding-window` correct only by accident
(`ZSTD_e_end` flushes a frame regardless). Real zstd emits 13 bytes. Fixed once in
`run_one_pass`, the single point every path funnels through, rather than four times.

**`--tar` could still truncate a named source through an absent-output race.** The previous
fix compared sources only when the output already existed, so for an absent output another
process could create that name as a symlink to a source between check and open. Sources now
have their identities captured — and `O_PATH` handles held, so the inodes cannot be recycled
— before anything touches the output, and the OPENED FD is compared against them.

**Redirected stdout still kept an old tail for excluded destinations.** The `/dev/` and
`(deleted)` pathname tests were proxies for "is this a real file" and are obsolete now that
the descriptor is adopted: a regular file under `/dev/shm`, or an unlinked regular fd, was
excluded, took the buffered path, and kept the tail of a longer previous file. `S_ISREG`
already answers the only question that matters. Truncation failure is now fatal rather than
falling back — falling back reproduces exactly the corruption being prevented.

Also: the buffered `--verify` rebuild ignored `ftruncate` failure, so a shorter CPU rebuild
could overwrite the prefix of a rejected archive and leave stale bytes after it (the verify
pool only checks the frames the rebuild produced); calibration cleanup did `lstat` then
`unlink` as separate pathname operations, and now unlinks at creation and measures through
the fd, leaving no name to race; global PAX extended metadata (`SCHILY.xattr.*`, ACLs,
SELinux) still applied to one member, where only the scalar fields had been separated; the
`--adapt` profile temp was a predictable `profile.json.tmp.<pid>` opened with a
symlink-following `ofstream`; and concurrent `--adapt` saves lost each other's observations,
now serialised by an advisory lock on a sidecar (never on `profile.json` itself, which is
replaced by rename and would drop the lock with the inode).

Six of the nine prior fixes were confirmed, and every judgment call flagged as uncertain was
validated: exit 2 for the tar alias, the `setxattr`-through-the-magic-link security property,
the broad `incomplete` promotion, the index-builder revert, and `S_ISREG || S_ISBLK` as the
persistence-bearing set.

Suite: 374, green on both build configurations.

---
## v0.15.71 — a third review round, an open mandate, and a regression I had shipped

Round 2 was scoped to adjudicating its own prior findings. Round 3 was deliberately left
**open** — adjudicate the five, then look anywhere — on the reasoning that if there are real
flaws we want to know now rather than after a `v*` tag auto-installs on the hosts. It found
nine, including one that v0.15.70 had introduced.

**The regression was mine.** v0.15.70 taught `DirectWriter` to adopt an already-open
descriptor instead of re-resolving a pathname. For redirected stdout that reproduced the
*data path* of the `open(..., O_TRUNC)` it replaced but not the **truncation**:

    exec 1<>old.zst; gzstd --direct -c input
      v0.15.68 →   305,504 bytes, verifies
      v0.15.70 → 9,000,025 bytes, CORRUPT   (new output over the prefix, old tail intact, exit 0)

Fixed by truncating fd 1 before adoption. **Adopting a descriptor must reproduce every effect
of the open it replaces, not just the writes.**

**And v0.15.70's commit message was wrong.** It claimed adoption meant "the name is never
resolved twice". `DirectWriter::plain_fd_()` was still doing `readlink` → `open(that string)`
for every unaligned tail, so a rename after the readlink sent the file's final bytes
positionally into somebody else's file. It now opens `/proc/self/fd/N` directly, which
resolves through the kernel's descriptor table and has no substitutable name.

**`--tar --overwrite` could delete a source before archiving it.** Tar mode leaves the input
`FILE*` null, so neither the path check nor `open_output_verified()`'s fd comparison ever saw
a tar source. `gzstd --overwrite -o data --tar data` unlinked `data` to make room for the
output, then archived the empty file it had just created — printing `2.21% (10.00 KiB =>
226.00 B)` and exiting 0. The output is now compared against every named `--tar` source.
Writing the archive *into* a source directory stays legal, because that is a common pattern
and is not destructive.

**`--calibrate -o` kept the destructive symlink race** the main output opener had already
lost: existence check, then `fopen(...,"wb")`, then an unconditional `unlink` of the name.
Now created `O_CREAT|O_EXCL|O_NOFOLLOW`, and the cleanup removes only an inode it created and
still owns. Measured: v0.15.68 followed a dangling symlink and created through it; the fixed
build refuses.

**The unknown-size frame fallback still turned `pread` failure into a clean EOF**, resizing to
whatever prefix arrived and returning success — so an `EIO` in a multi-frame archive dropped
the trailing frames, exited 0, and `--rm` then removed the archive. Now fatal, with `r == 0`
before the known `fstat` size treated as unexpected EOF rather than success.

**Two findings were pushed back on, and the record says so.** The xattr finding's stated
trigger is **not constructible** — Linux refuses `user.*` xattrs on symlinks and FIFOs
outright. A direct C probe confirmed the mechanism anyway: `lsetxattr("/proc/self/fd/N")`
returns EPERM because it targets the procfs entry, while `setxattr` on the same path lands on
the real inode. So attributes on symlinks/FIFOs/devices were silently dropped, reachable only
for privileged namespaces. Fixed — and the old comment claiming the `l` prefix was what kept
the attribute on the link is corrected: the `O_NOFOLLOW` on the `openat` is what does that.
Separately, the global-PAX fix initially went into two parsers; the index builder turned out
to reject `'g'` headers outright and defer to the authoritative reader, so that half was dead
code and was reverted rather than shipped.

**Global PAX (`g`) headers applied to only the first member.** `'x'` and `'g'` shared one
pending state that is cleared after each entry. Verified against GNU tar with a hand-built
archive (GNU tar does not emit `'g'` for ordinary `--format=pax` output): reference gives both
members `mtime=1234567890`; v0.15.68 gave the second `mtime=999`; fixed matches the reference.

**`--keep-going --tar` reported "unverified" over a truncated tree.** A corrupted frame can
produce its full declared length and still destroy a tar header with the bytes it produced;
the parser then stops, but `g_damage.incomplete` was only ever set when a frame yielded fewer
bytes than declared. Measured on a corrupted archive: both builds extract 2 of 6 members,
v0.15.68 exits **6**, v0.15.71 exits **7**. Also, `fsync_fd_ok()` forgave `EINVAL` for block
devices — persistent by definition — and let `fstat`/`fclose` clobber the errno it reported.

Three regression tests added (374 / 507 / 293 + 70 skipped), all for the deterministic findings: `--tar` output
aliasing a source, global PAX scope, and the earlier `-l` header. Those were one-liners the
suite had simply never tried — a coverage gap, not exotica.

---
## v0.15.70 — the re-review rejected five of eleven fixes, each with a trigger

v0.15.69's fixes went back to the same independent reviewer, asked not to "review it again"
but to **confirm or reject each of its own prior findings as fixed**, with my specific doubt
about each stated up front. It confirmed six and **rejected five**, every rejection naming a
concrete trigger. It also found **no new regressions** in the 531 changed lines.

**1 — the identity check had a TOCTOU window.** `fs::equivalent` only runs when the output
already exists, so for a NEW output there was a gap between "does not exist" and
`fopen(path, "wb")` in which another process could create that name as a symlink to the
input. Now the output is opened **without `O_TRUNC`**, the resulting **fd** is `fstat`ed
against the input's fd, and only then truncated — nothing is destroyed even if the path
changes underneath. Symlink outputs still work; a symlink aimed at the input is caught like
any other alias.

**2 — `--direct` threw away the protection `O_EXCL` had just bought.** The temp is created
`O_EXCL|O_NOFOLLOW`, and then the O_DIRECT path did `stat(write_path)` + `open(write_path,
O_TRUNC)` with neither flag. No prediction needed: an inotify watcher sees the temp appear,
unlinks it, and drops a symlink to a victim before the second open. `DirectWriter` gained
`adopt_fd()`, which takes over the descriptor the caller already holds and verified, so the
name is never resolved twice. Three call sites converted — file output, redirected stdout
(now adopting fd 1, with the `readlink` used only to *reject* unsuitable targets), and the
GPU-fault/`--verify` rebuild path, which used to close the verified fd and reopen by name.

**3 — five `fread` sites were not the complete set.** The sequential decompression splitter
converted zero to EOF without checking `ferror`, then returned success for whatever complete
frames it had already parsed: an `EIO` partway through a multi-frame archive silently dropped
the trailing frames, and `--rm` then removed the archive. The non-mmap `-l` reader had the
same hole. Both now check. Measured before/after: v0.15.68 exits **0** on an unreadable
decompress and an unreadable `-l`; v0.15.70 exits 3 on both.

**5 — `EINVAL` was suppressed from the errno alone.** That is right for a pipe or a terminal,
which have nothing to persist, but a FUSE filesystem can return `EINVAL` for a **regular
file** — and `--sync-output` then reported success, renamed the output over the target and
removed the source without ever obtaining durability. `fsync_fd_ok()` now `fstat`s the
descriptor and only forgives `EINVAL` on a non-regular target. The same audit found
`--calibrate` ignoring both `fflush` and `fsync`, which could persist a sink rate for work
that never reached the device; it now refuses to record a measurement it could not make
durable.

**11 — the fixed-share bringup barrier could hang forever.** The compress GPU catch set
`g_gpu_aborted` and explicitly discarded the failure counter (`(void)gpu_failures;`), so a
GPU that threw *before* `register_gpu_stream` — a bad `cudaSetDevice`, a stream or event
creation error — satisfied neither barrier term and parked the main thread permanently.
Decompress's catch had always counted. The compress catch now counts and signals, and both
barriers additionally release on `g_gpu_aborted` as a safety net, since the failure mode here
is a permanent hang and an aborted pass is discarded anyway.

**Confirmed fixed and left alone:** the `ftruncate` propagation (coverage verified complete
across `--tar` create and redirected stdout), the `EPERM` device-node failure, the `--rm`
failure report, the three zero-length write loops (sibling set confirmed complete), and the
sorted `re_enqueue`.

**And I was wrong about the tar hang.** v0.15.69 recorded finding 8 as unreproducible and
kept the escapes only as hardening. The reviewer withdrew "confirmed" as a statement of
reproduction but **not the finding**, and identified what my rebuttal missed: after the pusher
drains the contiguous *claimed* prefix it waits for the first **unclaimed** index, and
checking abort before `next_chunk.fetch_add` guarantees only that claimed indices are
deposited — not that every index is ever claimed. The trigger needs `--read-threads 1` and
many more chunks than the claim window, which none of my seven configurations produced. The
fix was warranted; the reasoning I published for keeping it was not.

Suites: 372 / 505 / 291 + 70 skipped, green on both build configurations.

---
## v0.15.69 — an independent whole-codebase review before tagging, and the seven ways output could be lost

24 commits had landed since the last tag with no full-codebase review, and tagging is
deployment here. An independent reviewer (a different model, primed with `AGENTS.md` and
given the tree but not this project's conclusions) was pointed at all 25,918 lines, plus a
separate pass auditing the built-in help against the parser. Verdict: **NOT SAFE TO TAG**.

**Seven CRITICAL findings, every one of them "destroys or corrupts data and exits 0."**
Two were reproduced verbatim before any fix was written:

- **`--rm` with input and output naming the same file deleted both.** `gzstd --rm -f -o data data`
  printed `75.37% (3.86 MiB => 2.91 MiB)`, exited 0, and left an empty directory: the temp
  was renamed onto the input path, then the post-success unlink removed that same path.
  Now rejected before the output is opened, by FILE IDENTITY rather than string compare, so
  a symlink or hard link to the input is caught too.
- **The atomic-overwrite temp file was a fixed name opened with `fopen("wb")`.** Pre-creating
  `archive.zst.gzstd.tmp` as a symlink made gzstd truncate and write through it, then rename
  the symlink over the output — arbitrary file clobbering from a guessable name in a shared
  directory, confirmed by turning `archive.zst` into a symlink and overwriting a bystander
  file with 2 MB of compressed data. Now a unique same-directory name created
  `O_CREAT|O_EXCL|O_NOFOLLOW`, which also stops two concurrent writers sharing one temp
  inode. The mode is restored to `0666 & ~umask` afterwards: an `O_EXCL` create at 0600 gets
  renamed over the output, so shipping it would have silently tightened every archive's
  permissions.
- **`fread` errors were read as clean EOF at five sites.** A read failing partway through a
  file ended the stream, the prefix compressed "successfully", the archive was installed,
  exit was 0 — and with `--rm` the source was then deleted. Demonstrated with an input that
  fails immediately: v0.15.68 produced a 0-byte archive and exit 0; now exit 3 with the
  incomplete output removed. The sliding-window path checks the SHORT-read branch too,
  because a short read is what ends that stream.
- `DirectWriter::finalize()` ignored `ftruncate` failure, accepting an output with a
  zero-filled tail past the real data. `--sync-output` ignored `fsync` failure at both the
  buffered and O_DIRECT sites, so the durability the user asked for could silently not
  happen before the rename and the source deletion. Non-root extraction skipped device nodes
  on `EPERM` with only a `-v` message, producing a silently incomplete tree at exit 0 (GNU
  tar errors here too). A failed `--rm` was discarded entirely; it now reports and exits 3,
  keeping the completed output.

**Two HIGH:** three tar-extract write loops spun forever on a zero-length `write()` return,
now treated as I/O failure like `pwrite_all` already did; and `re_enqueue()` still pushed
whole batches to the deque front, which reintroduced exactly the seq inversion v0.15.66
exists to prevent as soon as two GPU workers requeued concurrently — the gap left by that
fix, in the path whose own comment first described the hazard. It now inserts by seq.

**Also:** `--gpu-devices` accepted negative values that could drive the worker count below
zero; `-T-5` and `-T=-5` were silently treated as auto while the separated `-T -5` was
rejected; `--no-progress` was implemented by lowering DEFAULT verbosity, so `-v --no-progress`
in either order left the meter running; a `-l` input-open failure reported exit 4 (data
error) instead of 3. The two fixed-share GPU-bringup loops that polled at 1 ms now wait on a
condition variable, so the project's own no-sleep-in-scheduling-paths rule finally holds.

**One reported finding did NOT reproduce and is recorded as such.** The review called a
GPU-fault hang in `--tar` assembly "confirmed", and it was one of the four stated reasons for
NOT SAFE TO TAG. Seven fault-injection timings (`GZSTD_DEBUG_FAIL_GPU_AFTER` = 0, 1, 3, 8,
20, 50, with and without `--adapt`) all completed on the pre-fix binary. The abort check runs
BEFORE `next_chunk.fetch_add`, so a reader that has already claimed an index still deposits
it and the pusher's exact-index wait stays satisfiable. The abort escapes were added anyway —
that liveness argument is invisible from the wait site and one edit away from being wrong —
but the comments say plainly that the hang was not observed.

**The help was audited against the parser: 33 discrepancies, 12 HIGH.** The parser accepts
143 flag spellings; short help documented 79 and long help 89, with no phantom flags. Four of
the 33 were fixed in code rather than in prose (above). The rest are corrected here, the
largest being a new COMPATIBILITY OPTIONS section: the "drop-in-compatible replacement" claim
was unqualified while roughly 30 zstd options were accepted and silently ignored — including
`--no-check`, which does not disable checksums — and dictionary training warns and then
compresses normally, producing no dictionary. Also corrected: `--overwrite` creates a new
inode rather than truncating in place (hard links keep the old content); `--read-threads` does
not disable the cold-input read probe and is ignored entirely if O_DIRECT wins it; `-M` is not
a hard memory cap on compression; `--throttle-frames` is raised to the GPU batch floor;
`--gpu-only` still uses CPU for rescue and verify, and exits 2 (not 5) with no GPU; GPU tuning
flags imply `--hybrid`; `-T 0` reserves I/O cores rather than being uncapped; and `--calibrate`
existed only in short help.

`AGENTS.md` lost ten "known-open" entries that the tree had already closed — both review
passes independently flagged them, and stale known-opens actively mislead the next reviewer.

Suites green after every individual fix (371 default), and **371 / 504 / 290 + 70 skipped**
on both build configurations at the end.

---
## v0.15.68 — the last unsorted queue turned out to be unreachable, so it is gone instead of guarded

v0.15.66 and v0.15.67 left one loose end on record: `RescueQueue::push` still appended
unsorted, the same shape as the bug that wedged `TaskQueue`. It was deliberately left alone
as "a low-volume GPU-failure fallback". Going back to sort it found something better —
**nothing ever calls it.**

One instance, declared in `decompress_nvcomp` and passed to `gpu_decomp_worker`, which never
references the parameter. The only method invoked on it anywhere in the file is `set_done()`.
No producer, no consumer. Three separate comments already say why: the rescue path *was*
deleted — *"the old rescue queue / re-enqueue / gpu_only_cpu_fallback machinery was
deleted"*, *"the old rescue/fallback machinery (which re-compressed the tail only to throw it
away) was deleted"*, and *"(No CPU 'rescue' pool: a GPU fault aborts the whole compress pass
…)"*. The code that used the queue went away; the plumbing did not.

Sorting an unreachable push would have added a comment describing a hazard that cannot occur
— which is the exact species of claim that hid the original deadlock for months. The class,
its instance, the unused worker parameter, the call-site argument and the `set_done()` call
are removed instead: **73 lines deleted, 2 added.** No behaviour change; a GPU fault on
decompress already finishes on CPU inside the worker and keeps its output.

Also annotated in `ROADMAP.md`: the v0.12.0 entry's claim that the throttle is *"deadlock-free
by construction (FIFO queue guarantees the writer's next frame is always in-flight)"*, and the
v0.14.70 entry's *"submit order = pop order = seq order"*. **Those two sentences are the
premise that was asserted in four places and enforced in none**, and they read as settled
design rather than as an assumption. Both now carry what actually enforces them, and the
warning not to restate the guarantee without naming the mechanism.

Suites: 371 / 290 + 70 skipped, green on both build configurations.

---
## v0.15.67 — v0.15.66 fixed two of the three cycles, and the third was the one written down first

v0.15.66 was validated on the 256-thread box, which is where it was predicted to matter most. It reproduced there — **5/5 hangs on the pre-fix binary** on the deterministic cell, all 99 threads in `futex_wait_queue`, `rchar` frozen, `wchar: 0`. Then the same cell was run against the **fixed** binary: **2/30 hangs.** The fix was incomplete.

The residual is not noise, and it does not scale with the thing v0.15.66 addressed. Warm, `--chunk-size 1`, `--throttle-frames=32`, 40 runs per cell:

| readers | 4 | 8 | 12 | 16 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|---|
| hangs / 40 | 0 | 0 | 1 | 7 | 13 | 25 | 34 |

**Scaling with READER COUNT is the signature of the claim-ahead window, not of queue ordering** — and it points straight back at the cycle this project's own root-cause note described first, before cycles A and B were found and fixed. `pooled_read_chunks` claims its chunk index (`next_idx.fetch_add`) **before** it acquires a buffer and preads. So frame *k* can be claimed and **not yet queued at all** while workers commit every permit to frames > *k*. The writer needs *k*, so it never releases a permit; every worker parks in `acquire()`. **Sorted insert orders what is in the queue — it cannot order a frame that is not there to sort.**

The fix is the principle v0.15.66 already applied to the four output pools, applied one level up to the permits themselves. `FrameThrottle::acquire_or_overdraft` takes a permit, or — when none is available and the frame at the head of the queue is the one the in-order writer is waiting for — **takes it anyway**, letting `permits_` go negative. Both paths decrement, so the writer's later `release(1)` stays symmetric and no caller needs to know which it took. `TaskQueue::push` keeps a pointer to the throttle and pokes it after each insert (with the queue lock **released**, so the only lock order is throttle → queue and there is no ABBA cycle), because in the wedge case no `release()` will ever come to signal the CV on its own.

**At most one overdraft is outstanding** (`permits_` never goes below -1). That bound matters: `-M` makes the budget a user-visible memory cap, and an unbounded version measured `peak=83/32` — a 2.6x overshoot. Bounded, it is `peak=33/32`, exactly one frame. One is sufficient, because the cycle needs exactly the writer's next frame to get through, and once that frame is written its release repays the overdraft.

**Every failing cell goes to zero**: 48 readers 0/40 (was 34/40), 32 readers 0/40 (was 25/40), 24 readers 0/40 (was 13/40), 12 readers 0/30 (was 2/30), plus 0/30 at 64 readers and 0/30 at 96 readers/16 permits — harsher than anything previously tested. Round trip byte-identical and stock `zstd -t` validates the output. `-vv` now reports `head_of_line_overdrafts=` so the mechanism is observable rather than inferred.

**Reachability, measured rather than assumed.** The default path on the 256-thread box never wedged, pre- or post-fix — on a warm regular file it takes **mmap**, not the pooled multi-reader, and its compress sink is fast enough that reads never outrun it. The failure needs a small permit-to-reader ratio *and* small frames, which `-M` reaches with no debug knob: `-M 32 --chunk-size 1 --read-threads 24` hung **10/30** before this change and 0/30 after. So the earlier claim that a high core count is the risk factor was wrong — **the precondition is a slow sink relative to reads**, which is why the 24-thread box is where it bit at default settings.

Also corrected: the GPU-path `acquire_out_buf` header comment still asserted the blocking behaviour and the ascending-seq premise that v0.15.66 had already removed from the code beneath it. That species of stale comment is what hid this bug for months.

---
## v0.15.66 — multiple readers had quietly broken the invariant everything else assumed

v0.15.65 owed one test to the low-core box: does the new measured-noise-floor gate over-suppress where the real differences are small and the distributions tight? It does not — the floor measures **1.028-1.115x there against the 256-core box's 1.16-1.27x**, it self-calibrates exactly as designed, and all five runs probed both directions and correctly settled on 3. Re-running the sweep to establish that, however, wedged for 100 minutes on a three-second job.

**`--cpu-only` compress of a WARM source with >=9 pooled readers deadlocks permanently**, ~20-25% of runs, every thread in `futex_wait`. Cold never does: the device is slower than compression, so the buffer pool never saturates. Warm reads run at 12-15 GiB/s against a ~2.8 GiB/s sink and the pool sits pinned at its limit, which is the precondition. This went unseen because `n_readers = max(3, min(12, hw/8))` is **3** on a 24-thread box — below the threshold. **On a 256-thread box it is 12, so that machine runs the default path inside the failing range**, in the mode its own A/B calls fastest for compress.

**One false premise, stated in four places and enforced in none:** *"frames are pushed in seq order, so the writer always has the oldest in-flight frame to write, drains it, and frees a slot."* A single reader made that true for free, which is why it was only ever asserted. The multi-reader pooled reader races, so seq k can be pushed after k+8 — and two independent circular waits follow.

**Cycle A, the per-worker output pool.** `out_pool` is a fixed per-worker partition of the throttle budget (`budget / n_workers` = 4 slots) whose slots are freed only by the in-order writer. A worker pops 100-103, fills all four, then pops 95 — and needs a fifth slot for the very frame the writer is waiting on. The wait is circular *inside one worker*. Fixed by making all four such pools (compress worker, decompress worker, both GPU-path ones) **overdraft rather than block**: allocate a fresh buffer when no slot is free. Safe because the FrameThrottle permits, not the pool, are the real global bound on in-flight frames — the pool is a recycling cache, and treating it as a budget is what deadlocked.

**Cycle B, the throttle permits — and the first fix missed it.** Workers take a permit *before* popping and pop from the front, so they can spend every permit on frames > k while k sits deeper in the out-of-order deque; the writer never advances, so no permit is ever returned. `TaskQueue::re_enqueue` already describes this exact wait and defends its own path with `push_front`. Fixed at the source instead of hardening each consumer: **`TaskQueue::push` now inserts sorted by seq**, scanning from the back — arrival is near-sorted, so it is O(1) amortised and exactly O(1) for the single-reader, stdin and re-enqueue paths that were already ordered.

**Cycle B is the reason to distrust a clean default-configuration result.** With only the pool fix, the defaults looked perfect — 0/15 at 9 readers, 0/10 at 12 — while `--throttle-frames=44` (two slots per worker) still hung 1/10. That harsher cell is the only thing that exposed the second cycle.

**After both fixes: 0 hangs in 72 runs**, including 0/25 on the cell that survived the first fix, 0/8 at 16 readers, and 0/6 with the throttle disabled entirely. Round trip byte-identical on both builds. No cost: cold compress 14.12 s median, identical to v0.15.65; warm 9 readers 3.26 s against 3.31 s before. Suites **370 / 503 / 289 + 70 skipped**, green on both build configurations.

A timeout-guarded regression test covers it (128 MiB warm, 12 readers, `--throttle-frames=8 --chunk-size 1`): **5/5 hangs on the pre-fix binary, 0/5 after.** `RescueQueue` pushes unsorted too — same shape, but it is a low-volume GPU-failure fallback and was deliberately left alone.

---
## v0.15.65 — the reader controller was grading noise, and now measures the noise first

v0.15.64 left an open question: the reader objective is not unimodal on the low-core box (a valley at 2 hides the optimum at 1), and the fix was to be judged here, where the range is 6-32 from a start of 12. Mapping the curve here answered a different question instead.

**On this box there is no hill to climb — only noise.** Cold buffered 16 GiB, median-of-5: readers 6 → 2.90 (2.79-3.09), 9 → 2.59 (2.56-3.20), 12 → 2.70 (2.64-3.27). The ranges overlap almost entirely, and a first median-of-3 pass had reported a confident "peak at 9" of 3.29 GiB/s that simply evaporated with more samples. Five identical `--adapt` runs settled at 12, 12, 6, 12, 12 — and the *same* settled value of 12 produced 3.14 and 2.57 GiB/s, a 22% spread that the reader count cannot explain.

**The mechanism, from `GZSTD_DEBUG_RD_CTL=1`:** a probe is graded by comparing one 250 ms window against one earlier window, against a fixed 5% margin. Consecutive *baseline* windows at an unchanged reader count came in at 3.561 and 4.121 GiB/s — **16% apart with nothing changed**. The margin sat far below the floor, so `KEEP 12 -> 11 rate 3.683 (was 3.248)` recorded a 13% "win" between two counts a median-of-5 says are identical.

**So the gate now measures the floor instead of assuming it.** The last four baseline windows are kept, and a step must beat the spread they already show — `max(5%, observed spread)`. Where the signal is clean the gate stays at the 5% floor; where it is mush it demands more than the mush. Self-calibrating per machine *and* per regime, which no constant can be. Verified cold: the controller now measures a 1.16-1.27x floor, raises the bar to match, and **holds at 12 across three runs** instead of wandering. `GZSTD_DEBUG_RD_CTL=1` prints the floor and the requirement it produced.

**What this does not do**, stated plainly: it is verified only for the *suppression* half. The other half — that a genuinely structured objective still gets found — cannot be shown here, because the one regime on this box with a large real effect (copy-bound, warm, sink removed: 3 → 5.72 vs 16 → 16.23 GiB/s) never reaches the controller at all. The v0.15.61 regime gate correctly holds there, since at the default 12 that run is compute-bound. **The low-core box is where the suppression could over-fire**, because its real differences were 6-17% with tight distributions. That is the test this change still owes.

---
## v0.15.64 — the reader search was right, and still could not reach the answer

**`GZSTD_DEBUG_RD_CTL=1` prints one line per reader-controller decision.** The teardown
line reports only the settled count, and that number cannot distinguish "probed both
directions and correctly rejected them" from "never probed at all" — they print identically
whenever a search returns to where it started. Reading that difference was the whole
diagnostic below, and it was not previously possible from outside the process.

**The reader-count objective is not unimodal, so the hill-climb cannot reach the optimum.**
The controller takes a step, measures it, and keeps it only if it pays — which assumes one
hill. Measured cold and buffered on a 24-thread Gen3 box, 32 GiB, median of 3:

| readers | 1 | 2 | 3 (default here) | 6 | 9 | 12 |
|---|---|---|---|---|---|---|
| wall | **15.15 s** | 18.93 s | 16.14 s | 16.45 s | 16.58 s | 16.86 s |

The best count is 1, and a **valley at 2** — 17% worse than either neighbour — sits between
it and the default. The trace shows the controller probing 3→4 (2.061 → 1.999 GiB/s,
reverted) and 3→2 (2.061 → 1.811, reverted), then settling: both rejections are correct
measurements, 3 is a genuine local optimum, and no ±1 climb starting there can ever see 1.
The implementation is doing exactly what it was designed to do; the design is what is wrong.
Left open rather than patched — the fix (evaluate the floor explicitly, or coarse-scan
before climbing) should be judged on the many-core box first, where the range is 6–32 from a
start of 12 and a valley would cost more. In practice this costs nothing on the box that
found it: the default path takes O_DIRECT at 11.27 s, better than any buffered count.

**Baseline refreshed now that both M.2 slots link Gen3 x4.** Same 10.36 GiB / 8,612-file
tree, flush-inclusive (`sync` inside the timed region), warm source, median of 3:

| | before (one slot at Gen1) | now |
|---|---|---|
| `--tar` create → the refitted drive | 11.42 s | **5.41 s** |
| `--tar` create → the always-Gen3 drive | 8.10 s | **5.42 s** |
| `--tar` extract, refitted drive | 9.69 s | **5.37 s** |
| `--tar` extract, always-Gen3 drive | 6.79 s | **5.51 s** |

The ~30% drive-to-drive asymmetry is gone; both sustain ~1.9 GiB/s end-to-end. The refitted
drive accounts for its own improvement, but the always-Gen3 column moved too — that part is
software, v0.15.55 → v0.15.64 (v0.15.61's converged-reader fix alone was worth 4.93 → 4.63 s
on this shape). Notably **create now matches what the same tree costs in tmpfs** (5.44 s),
so create is no longer storage-bound at all on this machine; extract still is (1.50 s in
tmpfs, ~72% of its wall time is the device).

---

## v0.15.63 — O_DIRECT refused almost every real file, and the flag took the slow door

Both defects were found by pointing the read path at repaired hardware. A low-core Gen3
box had been carrying a source NVMe linked at PCIe **Gen1** (866 MB/s); refitted to Gen3 x4
it sustains **2.9 GB/s**, and at that rate the cold read probe changes its mind — it adopts
O_DIRECT where it had declined. Everything below was hiding behind a decision that had
never been taken on this machine.

**The first O_DIRECT read of a real file failed at 98.8%.** A 32 GiB cold compress, no
flags, exited 3 with no output. The last block of a file whose size is not a multiple of
4096 is partial, and an O_DIRECT `pread` returns exactly that partial block. The
short-read retry loop (v0.15.41, added because a short read *mid-file* was being treated as
EOF and silently truncating) classified any unaligned short read as unresumable and failed
the job — including the one that is simply the end of the file. It could not tell the two
apart because it did not know the input's size. It does now: a short read that ends exactly
at EOF is a complete final chunk; anything else still fails loudly. **This rejected 4095 of
every 4096 real inputs** on `--direct-read`, and on the default path for any cold input the
probe adopted. It survived four versions because every generated test corpus is MiB-sized
and therefore aligned — the regression test added here deliberately uses sizes that are not.

**Asking for O_DIRECT by name was 1.50x slower than the probe choosing it.** With the tail
fixed, the same 32 GiB cold input ran 11.77 s on the default path and **17.65 s** with
`--direct-read` — the same physical read path, both single-stream O_DIRECT. The flag had
its own branch in the GPU/hybrid producer that read at host-chunk granularity into one
scratch buffer and memcpy'd every subchunk into a Task, on the reasoning that "one owning
buffer per host read doesn't map to one Task". The pooled reader had since dissolved that
constraint — read at GPU-chunk granularity and one pool buffer *is* one Task, zero-copy —
and `compress_cpu_mt` was converted; this producer, the one a plain `gzstd FILE` uses, was
not. The flag now takes the same pooled call the probe adopts: **17.65 s → 11.27 s**, with
`[READER] task-copy` falling from 23.1% to 0.0%. `--adapt` reaches this branch too (it sets
`direct_read` from its measured prior without setting the user flag), so a machine that
learned O_DIRECT wins had been re-applying it through the slow copy on every subsequent run.

**The premise that core count drives the O_DIRECT *read* regression is retired.** The
rationale for building the read path measure-not-rule cites `--direct-read` regressing
compress 20–40% on this low-core box. That figure needs a provenance note: the measurements
on record matching it are for `--direct` — O_DIRECT **output** — at v0.13.22 (compress
−20% to −40% here, +50% to +100% on the server), and no equivalent `--direct-read` compress
dataset for this box appears in this log. The two flags are independent and always have
been. Whatever its provenance, the read-side claim does not reproduce. Re-measured cold,
32 GiB, median of 3, output to `/dev/null` so no sink clamps it:

| read path | median | rate |
|---|---|---|
| default (probe decides → O_DIRECT) | 11.54 s | 2.79 GiB/s |
| forced `--direct-read` | **11.27 s** | **2.86 GiB/s** |
| forced buffered pool | 16.06 s | 2.01 GiB/s |
| forced mmap | 18.67 s | 1.73 GiB/s |

O_DIRECT **wins by 1.42x** on the machine that was the sole evidence for the opposite rule,
and reaches the drive's raw `dd` ceiling. What produced the old ranking was the crippled
link plus the copy above — not the core count, which predicts nothing here. The probe's
verdict is correct on both machines on record, and it was measurement, not the rule, that
caught the rule being wrong. Under `--adapt` this converges without flapping: SOURCE_BOUND
every run, O_DIRECT adopted every run, 11.42–11.64 s across three.

**`--direct` (O_DIRECT output) is NOT re-measured by any of this** and its v0.13.22
asymmetry stands unchanged. Every run above wrote to `/dev/null`, which removes the write
path from the measurement entirely — deliberately, so that a ~2.0 GiB/s sink could not clamp
three read paths that differ well above it. The Gen4+ auto-`--direct` gate is untouched.

**The reader-count sweep separates now that the device is not the wall.** At Gen1 speeds
1 reader tied 12 and the sweep was worthless. At 2.9 GB/s, cold and buffered, it is
monotonic — 1 reader **15.25 s**, 3 (this box's default) 16.07 s, 6 16.45 s, 9 16.58 s,
12 16.86 s. More readers cost throughput against a source this fast, which agrees with the
warm finding at v0.15.57. The lever is moot for cold compress here in any case: O_DIRECT is
single-stream by design and beats every buffered configuration by 3.9 s.

Suites at v0.15.63: **369/369 default, 502/502 extensive, 288 passed + 70 skipped
CPU-only**, both build configurations.

---

## v0.15.62 — telemetry that said things it could not know

Three small honesty fixes, plus two validation results that overturn assumptions this project has been carrying.

**A virgin profile announced a driver change that never happened.** The invalidation guard is `!have_drv || prev_drv != fp.driver` — correct, because a *missing* key must also clear GPU priors. But on a machine's first-ever run it printed `GPU driver changed ( -> 570.207); clearing GPU-derived priors`, which reads as damage on exactly the run where nothing can be damaged, and the clear it announced was vacuous. Only the wording is now conditional; the clear still happens either way. Verified in both directions: silent on a virgin profile, and a forced `999.999 -> 570.207` is still reported.

**`actions: none` after printing `probing +1 parallel writer` twice.** Not the contradiction it looked like. The governor sets a probe *request*; the summary reports whether a writer thread actually *engaged*. Nothing engaged, so "none" was literally true — the telemetry simply had no way to say "asked, never happened", which is the more interesting fact. Asked / engaged / kept are now three distinguishable states.

**Extraction does not fsync, and the help now says so.** `--adapt`'s writer pool is documented as sized from the writers' measured busy/starved split, without saying what "busy" measures. Measured: a 10.36 GiB extract to a 0.81 GiB/s drive **returned in 1.63 s** while the kernel spent a further **13.19 s** flushing after the process exited (zero `fdatasync` call sites; O_DIRECT exists for the leaf open but is not the default). So on any box whose RAM absorbs the output, that busy share is the copy into page cache and the sink-bound grow has no device pressure to find. This documents the measurement limit — GNU tar does not fsync either and the behaviour is unchanged — but it also means the v0.15.27 +26% figure carries a precondition: write volume must outrun the cache.

**Validation: two carried assumptions were wrong.**

*The residency-keyed backend prior is NOT fast-fabric-only.* The concern was that a Gen<4 box takes cpu-only before residency is consulted, making v0.15.36 unreachable there. It is reachable: the prior is consulted first and `return`s on a hit, and the static Gen<4 rule is the cold-start fallback — the code even computes `static_picks_cpu` to know what the fallback would have chosen. Confirmed on Gen3 hardware: run 2 of a fresh profile logged `profile prior (exploring hybrid; cpu-only measured 7.00 GiB/s end-to-end)`, overriding the Gen3 default, then settled back on cpu-only after measuring hybrid slower. All four buckets populate — `overall_gibs_cpu_warm 2.58` / `_cold 1.03`, `overall_gibs_hybrid_warm 2.00` / `_cold 1.03` — and the unkeyed blend (1.80 / 1.51) describes neither, which is the whole argument for bucketing. **Trap for future benchmarking: residency is deliberately not probed for non-regular output, so measuring to `/dev/null` silently disables residency keying entirely.**

Suites at v0.15.62: **366/366 default, 499/499 extensive, 285 passed + 70 skipped CPU-only** (the extensive run matters here: this version edits `--help` text).

*The replaced GPU is sound.* First multi-GPU exercise since the card was replaced: `--gpu-only` compress and decompress each brought up 2 device workers, an 8.05 GiB round-trip was byte-identical, hybrid round-trip byte-identical, and zero faults or CPU rescues — no sign of the fault arc the previous card produced.

---

## v0.15.61 — a converged reader controller stopped paying to re-learn what it knew

Searching and verifying are not the same problem, and the step size was answering only the first. The proportional step `max(1, cur/2)` exists so a COLD controller can cross its whole range inside a bounded round budget — `±1` measured settling at 14 where 24 was worth 16% more (v0.15.44). But a controller **seeded from the profile** already holds a converged answer for its key; its job is to confirm that is still a local optimum. Probing proportionally from a settled 4 means probing 2 and 6 — and 2 measured **7.31 s** against the optimum's 4.63 s. The probe was most of what a converged run paid.

The step is now `1` when seeded and proportional when not. Nothing changes for a cold start, so the big-box search behaviour is untouched. A seeded controller can still travel — one step per run, re-seeding from where it lands — so a genuinely moved workload is followed across a few runs instead of re-explored inside every one.

Measured, cold `--tar` create of a 10.36 GiB / 8,676-entry tree, converged steady state:

| | wall |
|---|---|
| v0.15.60 (proportional probe) | 4.93 s |
| **v0.15.61 (seeded ±1)** | **4.73 s** |
| forced `--read-threads=4` (the optimum) | 4.63 s |

The standing tax on an already-correct answer falls from ~6.5% to ~2.2%. Convergence costs one extra run (3 → 3, then 3 → 4, then stable), which is the right trade: slightly slower to find, materially cheaper once found.

**What the storage actually costs, flush-inclusive.** The earlier figures in this file were taken without a flush inside the timed region, which lets a slow sink hide behind dirty page cache — a 4.56 GiB archive cannot durably reach a 0.81 GiB/s device in the 5.32 s once recorded. Re-measured with `sync` inside the timing, on the same tree:

| | create | extract |
|---|---|---|
| Gen1-linked NVMe (0.81 GiB/s) | 11.42 s | 9.69 s |
| Gen3-linked NVMe | 8.10 s | 6.79 s |
| tmpfs (no storage at all) | 5.44 s | **1.50 s** |

Extract in RAM runs at **6.91 GiB/s**, within noise of the measured CPU compress ceiling of 6.97 GiB/s — so the pipeline itself is not the constraint and storage is ~78% of extract wall time even on a healthy drive. That is independent corroboration of the v0.15.58–60 verdict: these runs are source-bound, and were being classified compute-bound.

---

## v0.15.58–60 — the governor could not see three of its own reader paths

The `--adapt` governor decides everything from one number per subsystem. Its source-busy fraction is `reader_io_ns` summed across readers and divided by `reader_threads` — and **all three terms were wrong, in three different places**. Every fix here corrects a *measurement*; not one constant was retuned, because once the signal was right the existing policy already chose correctly.

**1. The reader divisor was stale for the whole run (v0.15.58).** `Meter::reader_threads` was written only at teardown, *after* the readers had joined, so the governor normalised every window by the count the pool STARTED with. The error scales with how far the pool has moved, and it goes both ways: shrink 3→2 and a fully saturated reader measures 0.67 against a 0.85 threshold and reads as compute-bound; grow 12→24 on a big box and it measures ~2x busy and manufactures a source-bound verdict that is not there. `ReaderPoolCtl::bind_live()` now publishes the active count on every `set_want`, before waking the readers, so the window that grades a step divides by the count that was actually running during it. Measured: a cold device-bound read went from `source-bound 3% / compute-bound 89%` to `source-bound 91% / compute-bound 1%`, and the `[READER]` line's impossible **105.7%** became 91.5%.

**2. `--tar` create fed the governor nothing at all (v0.15.59).** `tarx::assemble` set `reader_threads` but never touched `reader_io_ns`, `reader_copy_ns`, `reader_blocked_ns` or `reader_parse_ns`. All four stayed at zero, so `rbusy` was identically 0 and **a tar create could never classify SOURCE_BOUND however read-bound it was** — the same structural blindness already documented in the code for the mmap reader. The member `pread` now feeds `reader_io_ns` (timed whenever a Meter exists, not just under `-vvv`, because the governor needs it at every verbosity) and the push-frontier wait feeds `reader_blocked_ns`. The controller's own `wait_turn` park is deliberately NOT counted: a parked reader is inactive by the controller's choice, not blocked by the pipeline, and charging it there would understate `rbusy` exactly when the pool is shrinking. Measured on a 10.36 GiB / 8,676-entry tree: `compute-bound` became `source-bound 47%`, the `[READER]` line began to exist at all, and `source-latch(gpu-batch)` — a source-bound action that previously could never fire — now does.

**3. `-d --tar` extract was blind twice over (v0.15.60).** Every compressed byte of an extract is read through `pread_seek_frame`, which counted `read_bytes` but not `reader_io_ns`; and the decode pool never published its reader count at all, leaving the governor's divisor at its default of **1** while ~15–22 readers preaded concurrently. Either bug alone produces a wrong verdict, and together they would have produced a right-looking one for the wrong reason — a ~15x inflated `rbusy` forcing source-bound regardless of truth. The pread is now timed, and the pool publishes its live active count from the same spawn/retire events that resize it (it starts high and contracts, so a write-once count cannot work). Measured with the archive on a Gen1-linked NVMe — 4.56 GiB at 0.81 GiB/s, i.e. 5.6 s of a 5.8 s run, with writers starved 79.1% — the regime went from `compute-bound 48%` to `source-bound 46%`, matching both the `[WRITER]` and `[READER]` verdicts. Extraction stayed byte-identical across 8,612 files.

Suites at v0.15.61: **366/366 default, 499/499 extensive, 285 passed + 70 skipped CPU-only**, and a 10.36 GiB / 8,612-file archive round-tripped byte-identical.

**What this bought, and why no parameter was touched.** With the signal correct, the reader controller converges to the measured optimum on its own. On cold `--tar` create of that tree it settles at **4** and holds; a forced sweep under the same flags puts the peak at exactly 4 (2 → 7.31 s, 3 → 5.24 s, **4 → 4.63 s**, 6 → 5.61 s, 9 → 6.52 s). Notably the optimum is 4 *with* `--adapt` and 6 without it, so measuring the wrong configuration would have sent a tuning effort at the wrong target. The remaining gap is that a converged controller still pays ~6% exploration every run (4.93 s auto vs 4.63 s forced) — recorded as a ROADMAP row rather than papered over with a constant.

---

## v0.15.57 — the reader-pool controller climbed the wrong quantity

Found by the first execution of the reader controller's low-core range. `n_readers` starts at `max(3, min(12, threads/8))`, so a 24-thread box runs the controller in **1–9 starting at 3**, a band that does not overlap the server's 6–32 at the bottom and had never been exercised.

**It ratcheted to its ceiling and persisted the result.** Across runs it walked 3 → 4 → 6 → 9, pinned there, and wrote `read_threads_settled: 9` into the profile to seed every later run — while a fixed sweep put the optimum at **3**. Warm, median-of-7, the distributions do not overlap:

| readers | samples | median |
|---|---|---|
| **3** | 5.20 5.22 5.22 5.22 5.23 5.24 5.28 | **5.22 s** |
| 9 | 5.33 5.34 5.34 5.35 5.35 5.38 5.40 | 5.35 s |

**The cause is not noise, and proving that mattered.** The first fix attempted was a confirm-window — a step had to clear the 5% margin in two consecutive windows before being kept. It changed nothing; the climb was identical. That is the evidence that each step's measured gain is *real*: the controller maximises **reader** throughput, and on a pipeline that is not reader-bound, extra reader threads genuinely do raise reader throughput — by taking CPU from the workers and buffering further ahead. The signal anti-correlates with run time. Warm on this box the pipeline is compress-bound at ~6.97 GiB/s (saturating by `-T 16`) and the buffered pool is memory-bandwidth-capped near 5.7 GiB/s, so readers 1→16 span only ~4% and none of it is the reader's to win.

The controller now gates on the governor's published regime, stepping only while `SOURCE_BOUND` and reverting an in-flight probe if the regime leaves it. **It was the only `--adapt` acting site that did not consult the regime** — the others already did — and it is the only one whose metric can improve while the run gets slower. A size reached while genuinely source-bound is held rather than reverted to base: it was earned, and a later compute-bound stretch has not shown it wrong. Verified: warm now holds 3 across five runs and persists 3.

**No automated test covers this, and that is a real gap.** The controller only supervises while the reader is actually running, so exercising it needs a multi-GiB input held under active read for several seconds — the suite's fixtures finish in well under a tick. Forcing a regime with `GZSTD_DEBUG_ADAPT_REGIME` is deterministic but produces no reader-pool activity at that size, so the hook alone cannot stand in. The fix is verified by measurement instead (warm: holds 3 across five consecutive runs and persists 3, where the previous build walked to 9 every time; cold: still engages and steps), which is weaker than a regression test and should be treated as such.

**Scope, stated honestly:** this stops the controller doing harm, but its reach depends on the regime classifier, and that classifier is under-reporting I/O-bound in at least two measured cases (see the new ROADMAP row — a cold read with `[READER] io 105.7%, blocked 0.0%, "reader saturated"` still recorded 89% compute-bound). Where a genuinely reader-bound run is misclassified, the controller now correctly stands down but also cannot help. That is a separate defect, upstream of this one, and deliberately not bundled.

Also established, and not previously verified on such a machine: the code's claim that the reader optimum is **3 on a 24-thread box** is correct, and the residency inversion holds on Gen3 hardware — warm mmap 6.91 GiB/s beats buffered 5.74, while cold the order reverses to buffered 0.83 > mmap 0.79 > O_DIRECT 0.70 GiB/s.

A measurement caveat worth recording with those numbers: the source drive used for the cold cells is linked at **PCIe Gen1** (2.5 GT/s against a Gen3-capable link, confirmed stuck under sustained load), capping cold reads at 866 MB/s. The *relative* ordering above is sound — every config met the same wall — but the absolute cold figures are not representative, and a first cold reader sweep was invalidated outright by it (1 reader tied 12 purely because the device, not the reader, was the constraint).

Suites at v0.15.57: **366/366 default, 499/499 extensive, 285 passed + 70 skipped CPU-only.**

---

## v0.15.55 — `--adapt` persisted regime verdicts it never measured

Found by the first deliberate `--adapt` cold-start run on a Gen3 workstation, where the profile had genuinely never existed.

**The ramp and the save floor are the same three seconds.** `RAMP_SEC` is 3.0 s of warmup before the governor will classify anything, and `adapt_save_min_ns()` is a 3 s floor below which a run is considered too trivial to teach the per-machine profile. Because those numbers are equal, the threshold that qualifies a run to persist its verdict sits *precisely where measurement begins* — so every run in roughly the 3–6 s band saved a conclusion drawn from a sliver of evidence. `dominant_regime()` returned whichever non-warmup bucket owned the most time with **no minimum at all**; its only test was greater-than-zero, so 0.1 s won uncontested.

Measured on a 24-core PCIe Gen3 box, same corpus, same flags, two sizes:

| | 8 GiB (3.2 s) | 64 GiB (32 s) |
|---|---|---|
| warmup share | **96%** | 10% |
| classified | 0.1 s (4%) compute-bound | 26.6 s (85%) sink-bound |
| **persisted regime** | **`compute-bound`** | **`sink-bound`** |

The short run's verdict was not merely thin, it was **backwards**, and contradicted by two other signals in the same run: the profile's own recorded rates (`cpu_gibs` 7.14 vs `sink_gibs` 2.17 — the CPU was 3.3x faster than the sink) and the `[WRITER]` verdict (*output device saturated — the sink is the bottleneck*). A modest compress on a fast box is the common case, not a corner case, so this was the default outcome rather than an edge one.

`dominant_regime()` now carries its state table in the code and requires both an absolute floor (2 s classified) and a share (25% of the run) before it will name a regime; below either it returns `unclassified`, which the profile writer already handled by simply recording no regime. Rates are still recorded either way — those *are* measured. Both terms are load-bearing: the floor rejects the short-run band, the share rejects a long run that spent almost all of itself ramping. The floor is deliberately **not** derived from `adapt_save_min_ns()`, since collapsing "worth recording at all" into "measured what it claims" is what caused the bug.

A rejected classification now says so at `-v` rather than failing silently, because the shares line showing a regime while the profile records none otherwise looks like an inconsistency. A **forced** regime (`GZSTD_DEBUG_ADAPT_REGIME`) bypasses the gate at any duration — it is a deliberate test assertion, not a measurement, and suite determinism depends on it.

**What was already correct:** cold-start creation writes schema epoch 2 with a full fingerprint, and convergence is clean — `overall_gibs` over five identical runs went 2.3945 → 2.4771 → 2.5245 → 2.5517 → 2.5627, deltas roughly halving each time, settling by run 4–5.

The regression test asserts the *invariant* (a sub-threshold classification must leave the profile's regime unset) rather than a fixed timing, so it holds at any machine speed instead of pinning the boundary this box happens to land on. It is also the only test in that section that runs a qualifying run, so it clears the profile afterwards — the tests that follow assert the profile is absent, and on first writing it this test failed its *neighbour* rather than itself.

Suites at v0.15.55: **366/366 default, 499/499 extensive, 285 passed + 70 skipped CPU-only.**

---

## v0.15.50–54 — the v0.15.48 unwind guard never ran, and everything behind it was broken

Five defects in one mechanism, found by running the suite on a PCIe Gen3 box for the first time and then chasing a `-Wunused-function` warning into the teardown code. They are reported together because **each one strictly hides the next**: nothing downstream of an unreachable code path can be observed, so fixing each defect is what made the following one appear. The last two were found only *because* the earlier fixes landed.

**1. The guard was inert (v0.15.52).** v0.15.48 added an RAII `ThreadGuard` to `compress_cpu_mt` so an exception escaping the reader region would unblock and join ~6 live threads instead of destroying them joinable and hitting `std::terminate`. But **there was no `catch` anywhere in the program.** With no handler, `[except.terminate]` leaves unwinding implementation-defined and gcc calls `std::terminate` *at the throw point* — no stack unwinding, so no destructor between the throw and `main` ever runs, `ThreadGuard` included. Verified with a five-line program: the destructor does not run. So the guard could never have done what its comment claimed, and the changelog entry that said an unexpected throw would surface *as a normal error instead of an abort* was wrong — it aborted either way. `main` is now a thin exception boundary over `gzstd_main`, mapping an unexpected throw to `EXIT_ERROR` with a message. `die_*()` is still `std::exit`, so the deliberate fatal paths are untouched.

**2. Once unwinding happens, it was a use-after-free (v0.15.50).** `MmapRegion` and `DirectReadPool` were declared *below* the guard, and destruction runs in reverse declaration order — so unwinding freed `dr_pool` and `munmap`ed the region **while the workers were still live and still holding zero-copy views into exactly that memory**. `~DirectReadPool` is a bare `free()`: it neither sets `done_` nor waits for outstanding slots, and `g_direct_read_pool` still pointed at the freed object. Both violated invariants were already written down and held everywhere else — the declaration comment claimed the pool *outlives the workers*, and the normal path is explicit (join, clear `g_direct_read_pool`, then leave scope). The fix makes lifetime enforce what the comments promised: the two objects are declared above the guard, and the global is cleared inside it once no worker can call `release()` again.

Demonstrated rather than argued, under ASAN with the injection hook below:

| build | result |
|---|---|
| old order, no handler | no unwind at all; `std::terminate` at the throw point |
| old order + handler | **`AddressSanitizer: SEGV`, READ access, in `ZSTD_XXH64_update` ← `ZSTD_compress_frameChunk`, on a worker thread** |
| new order + handler | clean, `EXIT_ERROR`, no sanitizer error |

**3. The unwind path is now reachable and testable (v0.15.51).** `GZSTD_DEBUG_THROW_READER` throws out of the producer after its frames are queued but *before* `set_done()`, so the workers are provably mid-compress when the stack unwinds — the only way the suite can reach this path, since a real escaping exception there is a `bad_alloc` or an unexpected library throw and is not reproducible on demand. That absence is precisely why a guard written for this case shipped without ever running. One suite test now asserts the run exits diagnosably and neither segfaults (139) nor aborts (134).

**4. The writer blamed itself for the producer's failure (v0.15.53).** With unwinding finally happening, `ThreadGuard` marked `workers_done` but never set `producer_done`/`total_tasks` — so the writer woke into what looked like a completed run with a hole in it, fired its stuck watchdog, and `die()`d with `internal error: writer stuck  workers_done but frame 0 of 0 missing`. Exit was non-zero and memory-safe, but it named the wrong component and pre-empted the real error. The GPU-fault path already had the right shape for this (`g_gpu_aborted` tells the writer that missing frames are *expected* and the output is discarded), so the concept is now generic: `g_run_abandoned` carries the same meaning for an unwind, and `run_abandoned()` is what the writer actually tests. It is set *first* in the guard, before anything that could wake the writer, so there is no window where `workers_done` is visible without it.

**5. …and that fix exposed a deadlock the old `die()` had been hiding (v0.15.54).** Because the writer had always `std::exit`ed at defect 4, the guard's joins had *never once executed*. Letting the writer leave cleanly reached them for the first time — and hung. Backtraces showed the main thread inside `compress_cpu_mt [clone .cold]` blocked in `std::thread::join`, with every CPU worker parked in `acquire_out_buf` on the output-pool drain wait. The pool is bounded and refilled by the writer; the writer had exited, so no slot could ever come back, and the guard deadlocked waiting for workers that were waiting for it. The escapes for this already existed and were checked against `g_gpu_aborted` alone — the comment even says *the writer has stopped draining, so no slot will ever free*. They test `run_abandoned()` now, so a worker parked for a slot leaves on either cause. Verified: injection now exits 1 with `gzstd: fatal: …` from the top-level handler, ASAN-clean, and a normal round-trip is byte-identical.

**First full suite run on the Gen3 workstation.** v0.15.x had never executed there at all — this is `RELEASING.md` item 4, and it is what surfaced the whole chain above. Final tallies at v0.15.54: **365/365 default, 498/498 extensive, 284 passed + 70 skipped CPU-only** (the CPU-only drift note is the documented one — GPU-gated sections compile out entirely). Both build configurations compile; the CPU-only config had never been built on this box before and emits four `-Wunused-function` warnings whose call sites are all legitimately inside `HAVE_NVCOMP`.

**A `//` comment ended in a line-continuation backslash** (v0.15.49), splicing the following line into it and warning `-Wcomment` on every build. Nothing was actually lost, because the swallowed line was itself a comment; the reason to fix it is that any code later inserted at that point would have disappeared silently.

**The GPU decode-pool test asserted nothing on Gen4 and failed outright on Gen3.** `parallel-extract: GZSTD_POOL_GPU adds GPU decoders to the pool` invoked `-d --tar` with no backend flag, so asymmetric mode defaulted decompress to `--cpu-only` on PCIe Gen<4 — which clears `gpu_capable` and makes `GZSTD_POOL_GPU` inert. It had been passing on Gen4+ because the fabric happened not to trigger that default, not because it verified engagement. It now passes `--hybrid` explicitly and asserts the same property on both fabrics. Output was byte-identical throughout; only the assertion was wrong.

---

## v0.15.48 — the open-items ledger, closed

Eight items that had been carried as "open, low priority" — some since v0.15.35 — plus the `--gpu-only` `--tar` path that had never been run.

**`--adapt` correctness cluster.** Three items in one subsystem:

- **A sub-floor probe left no trace at all**, not even `runs`, so the predictor re-selected the same exploration forever. A deliberate exploration is now admitted under the floor to record its *attempt stamp only* — no rate, because a run that short measured nothing worth keeping.
- **`g_adapt_gpu_engaged` was set at worker SPAWN**, so a worker that started and then failed made an all-CPU run file its rate under `hybrid`. It now fires at the first frame a GPU actually *delivers*, on both the compress and decompress paths. Care was needed picking the site: the obvious one is the shared `writer_thread`, where a CPU-only batch would have set it too.
- **A residency bucket mixed durations.** A run too short for the GPU to engage cannot inform a cpu-vs-hybrid comparison — one side is unmeasurable at that size — yet it still filed a rate that a long, GPU-worthy run then inherited. Runs under the engagement guard no longer contribute to the backend pair; they still contribute runs, regime, read-path and reader count. Kept as a gate rather than a fourth key dimension, because the sub-guard side has no hybrid measurement to pair with.

**The `--tar` reader bucket was never written.** v0.15.45 keys the settled reader count by residency, but `--tar` create's source is a *directory* and the residency probe answers −1 for anything that is not a regular file — so the tar half had no coordinates and silently re-climbed every run. It now samples up to 24 member files (a bounded walk; a backup source can hold millions). A second bug surfaced immediately behind it: the tar label was set inside `if (priors.loaded)`, so a *fresh* profile filed tar runs in the non-tar bucket — the same mistake already made once with `g_adapt_src_cold`, and worth naming as a pattern.

**Extract pool.** The governor computed its settled size from what it *asked* for, while the supervisor clamps every round to `ewgrow_cap_` — so a cap-clamped final round persisted a pool size that never existed and the next run seeded from a fiction. And `q_max_bytes_` was sized once from the *base* pool width, so a grown pool kept a queue budget for a pool half its size and the new writers starved on an empty queue, which is the opposite of what growing was for.

**The decompress size estimator opened and pread up to 1 MiB per input** — O(inputs) of startup before any work, noticeable on a shell glob over thousands of archives. Only the first four inputs are sampled now; the rest are scaled by the ratio those established. The consumer is a coarse "is there enough work for a GPU" gate, so a wrong guess costs a suboptimal backend choice, never correctness.

**`std::terminate` on an exception escaping the compress reader region.** ~6 threads are live there, and destroying a joinable `std::thread` is an immediate abort: no unwinding, no message, no actionable exit code. An RAII guard now makes the unwind path do exactly what the success path does — mark the queue done, release the throttle, mark the result store, then join — so an unexpected throw surfaces as a normal error. The deliberate fatal paths were always safe (`die_*()` is `std::exit`).

**CLI ergonomics.** `--watchdog SECS` accepts the separated form like every other valued flag (`--watchdog 60` used to arm a silent 30 s default). `-T -5` is a usage error instead of being quietly clamped away. And `-0` selects the default level, as **verified against real zstd v1.5.7** — it was a usage error here, so a script written against zstd failed for no reason. A suite test asserted the old behaviour and now asserts parity.

**xattrs and SELinux contexts on symlinks and special files are restored.** They were stored on create and silently dropped on extract, because `apply_ext` is fd-based and a symlink or device node is never opened for writing. `apply_ext_path` opens the member `O_PATH|O_NOFOLLOW` through the same secure per-component walk and uses `lsetxattr`, so the link *itself* is labelled rather than whatever it points at — following it would be a way to write attributes onto an arbitrary target. **Stated plainly: this is unverified on the development box.** Linux forbids `user.*` xattrs on symlinks and FIFOs outright, so only `security.*` and `trusted.*` can exercise the path, and both need privilege. Verified only that the regular-file and directory paths did not regress. It wants an SELinux host or a root-capable CI job.

**The `-d --tar` writer pool was steering on its own supply.** In parallel extract the Extractor's own meter is deliberately null (to avoid double-counting), and the controller's signal counted tar bytes *parsed* — but parse is decode-bound and runs ahead of the writers, which are device-bound. A pool sized from that is reacting to how fast work arrives, not to how fast the sink drains it. It now reads a dedicated write-completion counter, published where a file is actually closed.

**`--gpu-only --tar` create, validated for the first time**: byte-identical round-trip on a 16 GiB tree, and correct alongside the reader controller seeding from the profile.

---

## v0.15.47 — `--tar` create joins the controller, and four liveness bugs go with it

v0.15.46 stopped at `--tar` create because it deadlocked. It is wired now, and the cause was **neither** of the things two speculative fixes chased.

**A wait predicate becoming true wakes nobody.** The final `fetch_add` sets `next_chunk == nchunks`, satisfying the parked readers' exit condition — but the controller's only notifiers were a supervisor step and `stop()`, and `stop()` runs *after* those threads are joined. So they slept through teardown and the join never returned. Found by making the program report its own state, since `gdb` cannot attach here (yama `ptrace_scope`): a watchdog printed every chunk claimed and pushed, `k0–k11 exit`, `k12–k31 parked`, once a second, forever. The fix is a `wake_all()` on the work-exhausted path, guaranteed to run because `floor >= 1` keeps a reader unparked to make the overrun claim. The instrumentation is kept behind `GZSTD_DEBUG_TAR_RD` — it found this in one run.

An independent review (Codex CLI) of the fix returned **SAFE TO TEST** with a full stall table, and confirmed the two sibling pools do *not* share the bug: both call `stop()` before joining, so their teardown ordering supplies the notification. It also found three more defects, all fixed here.

**A pre-existing backpressure deadlock.** An explicit fixed-share hybrid request with no usable GPU falls back to the CPU pipeline still carrying `--cpu-batch`'s queue floor. CPU workers then hold out for a depth the queue's byte cap cannot reach, so the producer never reaches `set_done()` to release them and blocks pushing. Trigger: `CUDA_VISIBLE_DEVICES="" gzstd -T1 --cpu-share=0.5 --cpu-batch=64 --throttle-factor=1 --tar BIG`. The parse-time guards only covered runs that were *already* `--cpu-only`, so they missed the fallback. Verified: hung indefinitely before, completes in 14.4 s now.

**A member that becomes a FIFO between layout and assembly** made the blocking `open()` wait for a writer forever — and because that reader owned a claimed chunk, the pusher waited on it and the whole assembly wedged. Members are now opened `O_NONBLOCK` and verified still regular; anything else is treated exactly like an unreadable member (zero-filled, flagged, archive continues). Verified with a real mid-run swap: 0.2 s instead of a hang.

**Abort was not cancellation.** A GPU fault released queue backpressure but never told the tar readers to stop, so a faulted run still paid the entire input read before reaching the CPU rebuild. They now check the abort flag before claiming.

**The controller could persist a step it never validated.** If work ended mid-probe — or a tick produced zero reader bytes — the evaluation never ran and `want` stayed at the speculative value, which teardown then reported *and wrote to the profile* as though measured. It now reverts an unjudged probe at `stop()`. Verified: a short run reports the base count instead of a mid-probe one.

Also: the diagnostic watchdog now writes with `poll` rather than `fprintf`, so stderr on an undrained pipe cannot stall teardown at `join()`.

Three timeout-guarded regression tests cover the first three — a regression shows up as a timeout, not a suite that never returns.

---

## v0.15.46 — one reader controller for two of the three pools, and an honest stop at the third

The sizing controller from v0.15.44 is now shared code (`ReaderPoolCtl`) rather than living inside the compress reader, and the **decompress prefetch reader** uses it. Deliberately shared: three copies of a park/unpark mechanism is exactly how this file has produced sibling-path defects before.

**What the decompress reader had instead**, and why it went: dormant threads that woke once, at most doubling the pool, on the governor's `SOURCE_BOUND` verdict. That was inert on any box with ≥96 threads — `cap = min(n*2, 12)` with `n` already 12 leaves zero dormant readers — and its direction was not derivable anyway, since `SOURCE_BOUND` covers both a copy-bound reader wanting more threads and a saturated device wanting fewer. Measured: it climbs 12 → 18 and then starts at 18 from the profile on later runs.

Its ceiling is **RAM-clamped**, unlike the compress one: the look-ahead ring is `n_slots × BLOCK` and BLOCK is 64 MiB here, so a flat ceiling of 32 would reserve 4 GiB of ring against 1.5 GiB today. It budgets an eighth of available memory, so a small box gets a small ceiling rather than a controller that can only make things worse.

**`--tar` create is deliberately NOT wired, after trying.** Its readers *claim* a chunk and only then wait for it to come within the pusher's window, with the pusher blocking until the head chunk arrives — so parking a reader anywhere in that cycle strands work the pusher is waiting on. It deadlocked outright: a 20 GiB tree wedged with the archive 17.6 GB in, all 116 threads in `futex_wait`, while the same run without `--adapt` completed at 2.99 GiB/s. Two attempted fixes (waking parked readers on work exhaustion, then pinning the window to the ceiling) each moved the deadlock rather than removing it. The reorder buffer's invariants need working out before a controller belongs there, so it gets its own change instead of a rushed one.

**A test was asserting the removed rule.** "Source-bound wakes dormant prefetch readers" tested the mechanism this replaces, so it now asserts what the new design actually guarantees: the controller arms under `--adapt`, reports its range, settles inside it, and does not exist without `--adapt`. Whether it *steps* depends on measured rates and on the run lasting long enough to take a window — which a fixture must not assume.

---

## v0.15.45 — the settled reader count persists, keyed by the regime it was measured in

v0.15.44 re-climbed from the static start on every run, costing about 1.5 s on a 32 GiB input. The settled count now persists — but **one number per machine would have been wrong**, because the count is regime-dependent: 27 copy-bound and 6–18 device-bound on the same machine, same direction. A flat prior would be re-taught on every alternation, which is the workload-blind failure the extract writer prior already hit once.

**Keyed by the coordinates knowable BEFORE the run**, which is the constraint a seed has to live with — the governor's own regime verdict only exists once the run is over. Those coordinates are input residency and sink class, and they reproduce the three regimes actually measured:

| bucket | regime | settled here |
|---|---|---|
| `cold_file` | device-bound | 6 |
| `warm_nofile` | copy-bound | 27 |
| `warm_file` | sink-bound | masked; any value performs the same |

**No flat fallback for seeding**, and that was a real finding rather than a precaution: with one, a copy-bound run started at 6 because that was the cold bucket's number, when its own optimum is 27. Seeding from a different regime is worse than not seeding at all. An empty bucket means this machine has not measured this regime, so the run starts at the static default and goes and measures it. The flat key is still written, for diagnostics.

**The geometry is recorded with the value** (`rd_ctx_chunk_mib`, `rd_ctx_threads`). The count is a property of chunk size and consumer count, not of the box alone, so a run whose shape differs materially declines the seed rather than inheriting it.

A seed only moves the *starting* point; the controller still probes, so a machine whose media or workload changed re-measures instead of freezing on a stale verdict. Verified: cold runs climb 12 → 6 then start at 6 from the profile and stay there, while a copy-bound run in the same profile correctly ignores that bucket and starts at 12.

**Known limitation:** a fast regime produces short runs, which fall under the profile's save floor — so the fastest regimes are the least likely to persist a verdict. The only cost is re-climbing, and the floor exists for good reasons, so it is left alone.

---

## v0.15.44 — the reader pool sizes itself by measuring, because the direction is not derivable

The right number of buffered reader threads depends on which stage is the bottleneck, and that is not knowable before the run. Measured on the server, compress, median:

| regime | 3 | 6 | 12 (auto) | 16 | 24 | 32 |
|---|---|---|---|---|---|---|
| copy-bound (warm, sink removed) | 6.23 | 16.25 | 26.44 | 27.96 | — | — |
| device-bound (cold, buffered) | — | 3.94 | 3.48 | 3.38 | 3.35 | 3.28 |
| sink-bound (warm, real NVMe sink) | flat 3.58–3.74 — the count is masked entirely |

**A static formula is wrong in at least one regime whatever value it picks**, which is why the auto count stays where it is: `max(3, min(12, hw/8))` is right for the copy-bound case, the one where the lever has real leverage.

**The direction cannot be read off the regime label.** The governor's `SOURCE_BOUND` covers *both* the copy-bound case (wants more readers) and the device-saturated case (wants fewer). The pre-existing actuator grew the pool on `SOURCE_BOUND`, and cold compress classifies `SOURCE_BOUND` for 84% of its run while wanting the opposite — so a rule-based lever would have moved the wrong way on exactly the workload it was reaching for. It was also inert in practice: `cap = min(n*2, 12)` with `n` already 12 on any box with ≥96 threads means zero dormant readers.

So the controller takes a step, measures it over a window, and keeps it only on a 5% gain — reverting and reversing direction otherwise, bounded to six rounds. Parked readers wait on a condition variable and hold no pool buffer, so they cannot deadlock the ones still running. `--adapt` only; O_DIRECT (single-stream by design), a user `--read-threads` pin, and the probe passes of the read-path measurement are all excluded — the last because timing one configuration while resizing underneath it measures a moving target.

**The step is proportional, and that mattered more than expected.** A ±1 step cannot cross a 6–32 range inside a bounded round budget: measured, it settled at 14 where 24 was worth 16% more, purely because each probe costs two ticks and the run ended first. Halving/one-and-a-halving reaches the useful range in three steps. With it: **+12.8% copy-bound** (15.89 → 17.92 GiB/s, settling at 27 from a start of 12, reproducible across four runs), **+11% device-bound** (3.32 → 3.69), and **neutral when sink-bound** (8.09 → 8.11), where the count is masked and the exploration costs nothing measurable.

**`--gpu-only`'s inverted verdict, resolved.** v0.15.43 recorded that the read probe chooses *buffered* under `--gpu-only` (O_DIRECT 1.78 vs buffered 3.48 GiB/s), the opposite of every other configuration, and flagged that those are pipeline rates rather than device rates because the reader sits blocked on the pool while the GPUs consume. Checked end-to-end on a cold 32 GiB compress: **buffered 2.37 GiB/s vs forced O_DIRECT 2.16** — the probe's choice is correct, and worth about 10%. The figures still must not be quoted as device measurements, but the decision they drive is sound.

---

## v0.15.43 — the read probe reaches --gpu-only, and the compress buffer pool stops ignoring RAM

Housekeeping on the two versions below, plus one measurement worth recording.

**`--gpu-only` reaches the probe, and disagrees with everything else.** Same code path as `--hybrid`, so it inherited the probe for free — but on a cold 64 GiB compress it measured **O_DIRECT 1.78 GiB/s against buffered 3.48** and chose buffered, the opposite of every other configuration (cpu-only and hybrid both measure O_DIRECT around 4 GiB/s and take it). The reason matters more than the number: with the GPUs as consumer, the reader spends its time blocked on `pool->acquire()`, so a pass measures *the pipeline*, not the device. O_DIRECT is a single stream by design and cannot hide that latency behind other readers the way a 12-thread buffered pool can. The probe is therefore doing exactly what it claims — picking the faster path *in this configuration* — but the figure is not a device measurement and must not be quoted as one.

**The compress buffer pool now has a RAM cap.** Its only ceiling was 1,024 buffers, which at the default 16 MiB chunk is a 16 GiB pool, and O_DIRECT requests transparent huge pages so much of it can become resident. The preflight RAM check does not account for it. It now bounds itself to at most a quarter of available memory, never below 8 buffers — the same rule `compress_nvcomp`'s pool has always used. Found by review, not by a failure.

**Help text corrected.** `--direct-read` still described itself as "mainly a benchmarking / one-pass tool" and, after v0.15.41, as something `--adapt` might enable. Both were false: gzstd now measures and adopts it on any cold compress with no `--adapt` involved. Both the `-h` summary and the long entry say what actually happens, including every condition under which it declines.

---

## v0.15.42 — the cold-input read probe now actually runs, reads one file, and cannot mistake a short read for the end of it

An independent review of v0.15.41 (Codex CLI, gpt-5.6-sol, high effort) returned **NOT SAFE TO TEST**. Every finding was reproduced before being fixed.

**The probe never ran on the default path.** Compress sets `opt.hybrid`, so a plain `gzstd -f -o out.zst FILE` goes to `compress_nvcomp` — which had no probe gate at all. Only `compress_cpu_mt` did, and every manual test of v0.15.41 had passed `--cpu-only`, which is precisely how it was missed. Adding the gate to the GPU producer was not enough either: **mmap is decided first there and always won**, so eligibility is now computed *before* the mmap branch and mmap stands aside when the probe applies. If the probe then declines internally, the pooled buffered reader takes over — itself faster than mmap on a cold input (19.95 s vs 25.12 s). Default cold 64 GiB compress: **25.12 s → 14.06 s**.

**A short read is no longer treated as end-of-file.** `pread` may legally return less than asked for while data remains. The reader treated any short read as the last chunk, dropping the rest of that chunk *and every chunk after it*, then reporting success — a silently truncated archive. Pre-existing, and present in v0.15.40; fixed rather than inherited. The chunk is filled to capacity, `EINTR` retries, and only a zero-length read means EOF. Under O_DIRECT a short read that cannot resume on an aligned boundary now fails loudly instead of truncating.

**One file identity for the whole read.** Each pass reopened the input by pathname, so replacing the path between passes spliced a prefix of one inode onto a suffix of another and still reported success. O_DIRECT is a property of the open file description and cannot be shared with a buffered pass, so both descriptions are opened up front, checked to be the same inode, and held for the entire read; a mismatch declines the probe rather than guessing.

**Two gates did not say what they meant.** `adapt_path_residency` returns −1 when it cannot tell, and −1 < 0.95 — so inputs of *unknown* residency were admitted to a branch justified entirely by the input being cold. Residency must now be known and cold. Separately, the cold-rate label was set only once a profile had loaded, so the first run on a machine recorded no cold rate and the pair needed an extra run to become comparable; it is now gated on `--adapt` itself.

Also: `ReadWindow::eof` was a plain `bool` written by every reader thread and is now atomic, and the shared pool is sized for the largest reader count any pass will use rather than the first pass's single O_DIRECT reader. Measurement put that second bias within run-to-run noise, so the rationale originally given for it was wrong.

---

## v0.15.41 — a cold compress input is read with O_DIRECT, chosen by measuring both paths on the real data

Measured on the server, 64 GiB single-file compress, median of 5:

| | mmap | pooled pread | O_DIRECT |
|---|---|---|---|
| cold | 2.77 | 3.94 | **4.88 GiB/s** |
| warm | 7.99 | **10.04** | 4.87 GiB/s |

**O_DIRECT is residency-independent** — it bypasses the page cache and reads at device speed either way — while mmap collapses when nothing is cached. So the best read path *inverts* with residency, and no single default is right for both.

**Why measure rather than rule.** `--direct-read` regresses compress 20–40% on the low-core workstation, and neither PCIe generation nor "is it NVMe" separates that machine from the server — both are Gen4+ NVMe. PCIe generation describes the *GPU* link and says nothing about storage. The rate is the only signal that distinguishes them, so a cold compress of a regular file ≥ 4 GiB reads its first window with O_DIRECT and its second with the buffered pool, times both, and finishes with whichever won. The probe bytes are real work — nothing read twice, nothing discarded — so the only cost is one window on the losing path: **13.35 s against 12.95 s with O_DIRECT pinned**, about 1.8%, versus 22.9 s for the mmap default.

**Windows are bounded by time, not bytes.** At the default 16 MiB chunk a fixed 32 MiB window would be *two reads* — enough to measure device-queue ramp and thread-pool startup and nothing else — and it would shrink to a single read if `--chunk-size` were raised. The buffered pass additionally runs at least three full rounds across its reader threads, or it is timed while still spinning its pool up and hands the comparison to O_DIRECT for the wrong reason.

**`--adapt` keeps a learned verdict** so later runs skip the measurement, recording read-path rates under a cold-only key dual-written alongside the existing flat keys. A cold key absent from an older profile reads back as zero, meaning untried — additive, so no schema epoch bump and no host profile reset.

**A note on the regime classifier.** The `--adapt` probe is deliberately *not* gated on `SOURCE_BOUND`, unlike its decompress sibling. Gating it that way was tried first and never fired once: a 64 GiB cold compress that is plainly read-limited still classifies as **compute-bound**, because under mmap the read is paid as page faults *inside* the compression workers, so they look busy and the governor cannot see the stall. mmap disguises I/O as compute, closing the source-bound gate on precisely the case this exists for.

---

## v0.15.40 — the --adapt profile knows which format it was written in

The profile is a per-machine cache that accumulates measurements, and every format change so far has raised the same question by hand: do the deployed hosts need their `~/.cache/gzstd/profile.json` cleared? It now answers itself.

**Two fields, two different jobs.** Every write stamps `gzstd_version` — which build last touched this file — purely as information. Every write also stamps `gzstd_profile`, a **schema epoch** bumped by hand only when a format change makes previously recorded values unsafe to reuse: a key whose meaning changed, a value that may have been written under a since-fixed bug, or a new key the old code could not populate. A run that finds a different epoch discards the whole file and measures fresh, saying so at default verbosity, because a machine that had converged suddenly exploring again would otherwise read as a regression.

**The version is deliberately not the trigger.** `GZSTD_VERSION` bumps on every executable build here — several times a day during an arc like this one — so keying the reset on it would throw away everything each machine had learned, constantly. Verified: a profile carrying epoch 2 and version `0.9.50` is preserved intact (`runs` 99 → 100, rates untouched) while its version stamp is refreshed.

**Epoch 2 is set now**, which resolves the open deployment question for this tag without anyone touching a cache file. Profiles predating it may hold a workload-class value written into the wrong bucket, or a `_run` stamp that suppressed exploration — both fixed during v0.15.35–39, but the recorded values are already wrong. Those hosts will discard and re-measure on their first `--adapt` run.

A schema mismatch is also distinguished from a corrupt file in the `-v` note: "written under an older schema; starting fresh" versus "unreadable; rewriting". They had the same message before, which made a routine upgrade look like corruption.

Verified: an epoch-1 profile with 99 runs of history is discarded and rewritten at epoch 2 with the current version; the next run does not reset again; a truncated file still reports unreadable rather than stale.

## v0.15.39 — two fixes that were only half-fixes, and a flag that got sticky

A third bounded pass by the independent reviewer, this time asked to confirm or reject each of its own previous findings as fixed rather than to review freely. Four of six confirmed; two were incomplete in ways the original triggers did not cover, and one fix introduced a small regression of its own.

**Driver invalidation skipped entries whose driver was never recorded.** v0.15.38 cleared stale GPU-derived keys when the stored driver differed — but guarded on `!prev_drv.empty()`, so an entry with a missing or empty `driver` field (an older profile, or a run where the version probe failed transiently) skipped the clear entirely, then had the new driver stamped on top, and the next run read those stale values as valid. The mirror case was just as wrong: a non-empty driver becoming empty cleared the keys but left the old string stored, so invalidation re-triggered on every subsequent run. Now only an exact match skips the clear, and the driver is persisted unconditionally — empty included — so the stored state cannot drift from what was actually probed.

**The stdout sentinel still collided where it mattered.** v0.15.38 stopped `-o stdout` from bypassing the `-c`/`-o` conflict check, but that was the parser half. Execution still inferred the destination from the output *string*: any output equal to `"stdout"` went to fd 1 regardless, so `gzstd -o stdout FILE` wrote compressed bytes to the terminal instead of creating a file named `stdout` — and because `to_stdout` remained false, it also bypassed the write-to-a-terminal guard. `opt.to_stdout` is now authoritative; `parse_args` already sets it for `-o -` and the no-args stdin case, so the flag alone suffices. A file named `stdout` is now reachable, which it had not been.

**And the fix for that made `-o` sticky.** Marking a named output set the flag but never cleared it, so `gzstd -o out.zst -o - FILE` reported a `-c`/`-o` conflict for a command line containing no `-c`. The flag is now reassigned on every occurrence, including to false for `-`.

Confirmed fixed and not re-examined: the clamp/load ordering, the two byte domains, absence outranking a policy decline, and the bounce-buffer ceiling — the reviewer verified the writer expression matches the supervisor in all four branches (pinned, no prior, seeded prior, no `--adapt`) and that the size is finalised before any writer starts.

It also corrected three of my verification claims: the driver check proved only the non-empty-to-different case, the four `-c`/`-o` forms proved the parser check rather than the file-opening path, and zero-as-absent is genuinely equivalent for every consumer — that one I had asserted without checking all of them.

**A fourth pass found all three of those fixes still incomplete**, each for a reason the original trigger did not reach — which is the argument for reviewing a fix against the *class* of bug rather than the instance.

- **`gstr()` cannot tell a missing key from an empty string.** A missing, null, or numeric `driver` all read as `""`, so on a host whose GPU enumerates but whose version probe returns `""`, the stored and current values compared equal: the loader called the GPU priors valid *and* the save skipped clearing them. Both sides now require the key to be present *and* a string before equality may skip the clear. Verified across missing, `null`, numeric, and genuinely-empty stored values.
- **The stdout fix broke multi-input routing.** With several inputs and no `-o`, each destination is derived in the loop, and `derive_output("-")` yields the stdout sentinel without setting the flag — so `cat data | gzstd FILE -` wrote a disk file named `stdout` instead of streaming. The decision is now made per file from the *input* (`-` with a derived destination) rather than from the output string, which keeps a literal `-o stdout` reachable without losing stdin routing.
- **`-o` was order-dependent in the other direction.** `-o - -o FILE` left `to_stdout` set from the first occurrence, so it reported a phantom conflict too. Destination provenance is now separate from the resolved effect: `stdout_flag` records an explicit `-c`, each `-o` *replaces* both `output_named` and `output_dash`, and `to_stdout` is resolved once after parsing. The conflict check keys on the explicit `-c`, so neither order of two `-o` flags can fabricate one.

**Two order-dependence bugs in multi-input handling, found by the same pass and fixed rather than filed.** Neither was introduced by this arc; the first has been there as long as the multi-input loop.

`gzstd - a.bin` sent **both** inputs to stdout and never created `a.bin.zst`. The "reading stdin implies writing stdout" default is decided from `inputs[0]` alone but was applied run-wide, so the per-file output derivation was skipped for every input and the two archives were silently concatenated on fd 1. The same two arguments reversed — `gzstd a.bin -` — behaved correctly, which is the clearest statement of the defect: identical inputs, different results by position. The default is now taken only for a single input; with several, each destination is derived per file and the `-` among them is routed to fd 1 there. Verified against a build of the previous version: dash-first now matches dash-second.

Fixing that exposed a second one. The per-file loop set `opt.keep = true` when a file's destination was stdout — mutating state that persists across the loop, so one stdout-destined input suppressed `--rm` for every input after it. `--rm - a.bin` left `a.bin` in place while `--rm a.bin -` removed it. The decision is now a per-file local. Verified in all six combinations: `--rm` deletes the named source in either order, the default keeps in either order, `--rm -c` keeps (nothing replaced the source), and stdin is never a deletion target.

An earlier draft of this entry claimed the first fix would change deletion behaviour and needed a semantics decision. That was wrong: `keep` defaults to true, so deletion only ever happens under an explicit `--rm`, and that path already excluded `-`.

One cost it flagged and I am leaving: the size estimator opens and preads up to 1 MiB per decompression input, so a very long input list pays that at startup. It predates this change and is not worsened by it.

Six regression assertions were added for the two order-dependence bugs, after the reviewer pointed out that neither was encoded anywhere — the existing multi-file tests use two named files and the stdout tests only cover single-input `-c`, so both defects could have returned silently. They discriminate: run against a build of the previous version they fail exactly the two dash-first cases and pass the dash-second ones, which is the bug's signature. Getting them to run took two attempts, both instructive — first they went inside the `$EXTENSIVE` block and silently did not execute, then into a section whose conditional nesting I had misread. A test that does not run looks identical to a test that passes; only the `EXPECTED_TESTS` drift note caught it.

Suites: 360/0 default, 493/0 extensive, 279/0 CPU-only. Both configurations built.

## v0.15.38 — the independent reviewer finds a bug I shipped, and three of my own fixes overclaiming

A bounded second pass by Codex CLI / GPT-5.6-sol over the v0.15.37 diff. Seven findings, two of them high severity, and two cases where the previous changelog entry claimed more than the code delivered. The most important is a defect introduced by v0.15.37 itself.

**The adversarial clamps compared every stamp against a value that had not been loaded yet.** `D.runs` was read *seven lines after* the clamp block that validates against it, so it was still zero: `if (v > D.runs) v = 0` zeroed every legitimate positive stamp on every profile load. The consequences were exactly what the clamps were added alongside — `stale_run` became 0, so any profile with `runs >= 20` fired the recheck on *every run*, and the fault-attempt stamp introduced in the same version was erased, re-enabling the repeated fault-and-rebuild cycle it was meant to bound. `runs` is now loaded and validated first, and the 2^53 rebase test is `>=` rather than `>`, since at exactly 2^53 adding one already fails to count.

**The driver-change invalidation only worked on load.** v0.15.37 masked stale hybrid rates when the stored driver differed — but the *save* path reloads the JSON, stamps the new driver onto the entry, and merges only the current direction, leaving those values in the file where the next run reads them as valid for a driver that never produced them. Because `driver` is entry-wide, a compress run re-blessed the decompress keys too. The save path now zeroes every GPU-derived key in **both** directions before the driver is overwritten.

**Splitting one variable across two byte domains moved the bug rather than fixing it.** v0.15.37 corrected the size gate by scaling `known` to uncompressed bytes — but the duration predictor divides that same `known` by `input_gibs`, an input-domain rate. An archive expanding 16× then predicted 16× its real duration and admitted probes that finish under the save floor, recording nothing. There are now two variables: `known_work` for the batch gate, `known_input` for duration.

**Two process-wide booleans could not represent "declined, then confirmed absent."** In a multi-input run a short first file can decline by policy before device detection ever runs, and a later file can then confirm there is no GPU at all — with nothing to clear the earlier flag, an all-CPU aggregate on a GPU-less host was filed as hybrid. Confirmed absence now outranks a policy decline.

**The bounce budget sized itself against the wrong ceiling.** It ignored profile-seeded extras while the supervisor's actual cap includes them, so a foreign prior of 32 writers on an eight-core host sized for 16 threads while the live pool reached 32. It now matches the supervisor's own expression. The 1 MiB floor is documented as what it is: past 256 writers the floor wins and the aggregate grows again, accepted because smaller O_DIRECT writes would cost more than the memory saves.

**`-o stdout` slipped past the new `-c`/`-o` conflict check**, because "stdout" is also the internal sentinel for the stdout destination. A named-output flag now distinguishes a real path from the sentinel; `-o -` remains exempt, since it requests the same destination `-c` does.

Also recorded: the reviewer's verdict on the v0.15.37 attribution change. It accepted the semantics — a key meaning "performance of choosing hybrid" legitimately includes hybrid's decision not to initialise CUDA — while rejecting the implementation, both objections being the findings above. It notes one residual limitation, now on the open list: a residency bucket still mixes short workloads where hybrid declines with longer ones where it would engage, so a correction learned on a short run can be inherited by a longer one until the recheck cadence fires.

Two claims in the v0.15.37 entry were wrong and are corrected here: the driver-change closure, and that the predictor had been made domain-correct.

Suites: 354/0 default, 487/0 extensive, 273/0 CPU-only. Both configurations built.

## v0.15.37 — a second model reviews the keyed prior, and finds the fixes incomplete

The v0.15.36 keying was reviewed independently by a different model (Codex CLI / GPT-5.6-sol) rather than another instance of the one that wrote it — the review agents used until now share the author's weights, and therefore its blind spots. It produced the full state-transition table and seven findings, including incomplete fixes for three of the four defects v0.15.36 claimed to close. Three are fixed here; the rest are recorded in `AGENTS.md` rather than rushed in ahead of a tag.

**A faulted probe never recorded its attempt.** `adapt_merge_dir` increments `runs` and returns early on a GPU fault, and that return sits *above* the attempt-stamp block v0.15.36 added. So a probe that faulted left its `_run` stamp untouched and was selected again on the very next run — a machine with failing hardware re-paying the fault and the CPU rebuild forever. The stamp is now written on the fault path too, unconditionally: a fault means no backend rate was accepted, even when the run was attributed to the backend it was probing.

**The exploration predictor mixed byte domains.** `explorable` divides a stat'd input size by `overall_gibs` — but `overall_gibs` is a *payload* rate, which for decompress means uncompressed output bytes. Dividing compressed bytes by an uncompressed rate understates a decompress run's duration by the compression ratio, so a 1 GiB archive expanding to 8 GiB predicted 0.5 s against a 3 s floor and was refused exploration entirely. Exactly the high-ratio archives that most deserve a probe were the ones it silently skipped. A companion `input_gibs` key now records the same wall clock against the input domain, and duration prediction uses it.

**A driver change invalidated the GPU rate but not the hybrid one.** The profile loader already drops `gpu_gibs` and `settled_batch` when the driver string changes. The end-to-end `overall_gibs_hybrid` rates came from that same GPU stack and are just as stale, yet loaded unconditionally — so a pre-change measurement could select hybrid immediately on a driver the machine had never actually measured. Those rates and their stamps are now dropped too, so the next run explores afresh.

### Closing the remaining known issues

With a tag imminent, the rest of the open ledger was worked through rather than carried.

**The `--adapt` prior no longer learns from runs that measured something else.** `--keep-going` (damaged-archive recovery: single reader, one-shot decode, deliberately slow) and `--sliding-window` (a compatibility constraint) both force cpu-only for reasons unrelated to speed, and were filing that as a cpu-only rate. **`--cold` and `--direct-read` leave the run unkeyed**: both bypass the page cache, so a resident input would be classed *warm* and then measured at device speed, teaching the warm bucket a cold number. **Multi-input runs without `-o`** no longer assume a regular sink — the destinations are derived later in the loop and any of them could be an existing FIFO or device, which would make the run sink-bound while the guard reported otherwise.

**A stale hybrid rate can now be corrected.** The deeper half of the frozen-winner problem was that a run which *selects* hybrid and then has the GPU decline was filed as a cpu measurement — so the stale hybrid number survived and won again, forever. The fix is to distinguish *declined by policy* from *absent*: when the lazy-engagement guard or the drained-queue check keeps the GPU out, that is the hybrid backend correctly deciding the GPU would not pay for itself, and the resulting end-to-end rate is hybrid's honest rate for that workload. When the GPU is simply absent or hidden, the run measured the CPU and is still not called hybrid.

**Compress gained the drained-queue guard decompress always had.** If the CPU pool finishes the whole input while `cuInit` runs, spawning GPU workers creates contexts for no work and only delays exit. This is the case that came up repeatedly in round-trip testing.

**A hand-edited profile can no longer disable exploration.** A `_run` stamp greater than `runs` made the recheck condition false forever, silently retiring that side; a `runs` value past 2^53 stops incrementing and freezes the cadence entirely. Both are now clamped on load.

**The O_DIRECT staging footprint is bounded.** Each writer lazily allocates an aligned bounce buffer for large-file part writes. At a fixed 16 MiB ceiling with a pool that `--adapt` may grow to the machine's thread count, the aggregate reached ~4 GiB on a 256-thread box — harmless here, an OOM on a small host. The per-thread size is now derived from an aggregate budget, so total staging stays around one pipeline's worth however wide the pool grows. Measured on a large-file archive with a 192-thread writer pool, median of three: **1.07 s → 0.98 s**, so the smaller chunks cost nothing.

Also removed dead `peak_decoders`, and corrected the decode-pool comments and `-v` string, which still described the earlier "inline until writers starve" design rather than the START-HIGH/CONTRACT one that actually ships.

### Pre-tag validation

Three archive shapes round-tripped byte-identical (2.3 GB of huge files, 20 000 small files, 2.3 GB compressible). More usefully, the GPU path was finally exercised end-to-end — every earlier attempt had the GPU decline to engage, which turned out to be the feature working rather than a test failure.

Forcing the question produced the clearest measurement yet of the v0.15.31 lazy-engagement guard. On a 2.3 GB compressible input at `-T 1`, hybrid with the guard disabled ran **5.88 s** with 8 device workers online; with the guard active it declined the GPU and finished in **3.24 s**. The sample it acted on: *8% done at 253 ms, 2.48 s remaining, guard 4.0 s.* The guard was right — engaging would have cost ~4 s of `cuInit` to save ~2 s of work. Decompress declines for a different documented reason: 96 CPU threads drain a 1.7 GB archive before `cuInit` returns.

With the guard overridden, hybrid compress (8 workers), `--gpu-only` compress with GPU verify, and `--gpu-only` decompress (8 devices) all round-trip byte-identical.

## v0.15.36 — the backend prior is keyed by input residency, and the state machine that keying broke

The six-angle review left exactly one finding unfixed: a decided backend prior `return`s before the warm-input residency rule, so `--adapt` with a populated profile disabled the v0.15.2 warm/cold distinction. Choosing which rule wins looked like a policy call — but that was the wrong question, because the two rules answer different ones.


The review left one finding deliberately unfixed: a decided backend prior `return`s before the warm-input residency rule, so `--adapt` with a populated profile disabled the v0.15.2 warm/cold distinction entirely. Choosing which rule wins looked like a policy call needing measurement — but framed that way it was the wrong question, because the two rules answer different ones. The residency probe measures **this input** (≥95% page-cache resident → compute-bound → cpu-only; cold → disk-bound → hybrid). The prior measures **end-to-end outcome per backend, averaged over every run on the machine**.

So neither has to lose: **residency now SORTS the runs and the end-to-end measurement still DECIDES within a bucket.** The rate pair is recorded under `overall_gibs_{cpu,hybrid}_{warm,cold}` alongside the flat keys, and the prior prefers the bucket matching the current input.

This is the third time this codebase has hit the same defect and the third time keying has been the answer — after the read-path prior (keyed by path) and the v0.15.28 writer prior (keyed by archive shape, motivated by exactly this: one prior per machine was workload-blind, so alternating shapes kept re-teaching one number). A single blended EMA is not merely imprecise here, it is wrong for *both* regimes on any box that mixes them. Measured on the development box with a 1.4 GiB archive, the same file decompressed warm and cold: **2.28 GiB/s warm, 1.55 cold — and the flat key sat at 1.91, a number that describes neither.**

Details that matter:

- **Decompress only.** Compress returns before the residency rule, so its keys are untouched.
- **No new constant** — the existing 0.95 residency threshold is the bucket boundary.
- **Empty bucket falls back to the flat pair**, so a profile written by an older build decides exactly as it does today until the buckets fill (the same fallback contract as `tar_write_threads_<class>`). Verified against a flat-only profile.
- **Once a bucket holds any measurement it is used exclusively.** A missing side reads as untried and gets explored, which fills the pair honestly rather than comparing a warm number against a cold one.
- **The probe runs above the `backend_user_set` return**, because it is needed for *recording*, not only for deciding: a run with an explicit `--hybrid` still produces a measurement that belongs in a bucket, and filing those flat-only would mean a user who always names a backend could never fill both sides. (This was a real gap in the first cut of the change, caught by watching a `--hybrid` run fail to update its bucket.)
- **The probe happens once** and the warm-input rule reuses the value instead of sweeping mincore a second time.
- **The prior's message now names the bucket** — `profile prior (measured, warm input 100% resident: cpu-only 8.00 vs hybrid 2.00 GiB/s end-to-end)`. A deciding prior returns before the residency notice, so without this the user loses the only explanation of why a resident input went cpu-only.

Verified end to end on one machine and one file: with the warm bucket favouring cpu-only and the cold bucket favouring hybrid, the identical command chooses opposite backends purely on the residency of its input. What is *not* yet measured is whether the split changes real throughput on the Gen<4 workstation — its Gen<4 branch takes cpu-only before residency is ever consulted, so the keyed prior may be a fast-fabric-only win. Worth confirming there before treating it as settled.


### Four more defects, in the keying itself

Keying a state machine changed which states are reachable, and four assumptions that had been true stopped being true. Three were found by review of the new code, one by hand before it.

**The hybrid side of a bucket was never explored.** The inherited reasoning — "hybrid untried, so letting the static rules run IS the exploration" — holds only where the static rule picks hybrid. Inside a warm bucket it picks cpu-only, so the pair never completed and the prior could never decide anything. The same was already true on any Gen<4 box, making this a latent v0.15.33 bug the earlier angle missed by trusting the comment.

**The first fix for that explored on every run, forever.** A hybrid probe whose lazy GPU-engagement guard declines is recorded as a CPU measurement, so the hybrid side stayed untried and the probe repeated indefinitely. Fixed by stamping the *attempt* — the `_run` key with no rate — which holds the next probe off for the recheck interval.

**The recheck branch never terminated, because it never stamped.** The stamp was gated on a string sniff (`why[0] == 'e'`, for "exploring…"), which silently excluded `"re-measuring the alternative"` — an `'r'`. Two reachable states re-fired forever: workloads that record no backend rate by design (`-t`, `-d --tar`), and any run in the band where the save floor and the GPU-decline guard overlap. Each such run was forced onto the measured-slower backend. Control flow no longer depends on message text; an explicit flag is set at each decision site.

**The recheck was blind to a frozen winner.** It keyed on the loser's staleness alone. When hybrid wins but its guard declines the GPU, every run files as cpu, so the cpu stamp advances forever while the hybrid number driving the decision is frozen with nothing able to re-measure it — precisely the latch the recheck exists to break. It now triggers on the staler of the two sides.

**Multi-file runs were filed under the first file's residency.** `gzstd -d warm.zst cold1.zst cold2.zst` is one process with one recorded rate, dominated by cold reads, taught to the *warm* bucket. This is the poisoning `g_adapt_tar_wclass_mixed` already prevents for archive shapes; the residency key had no equivalent. Every input is now probed and a disagreement falls back to the flat pair.

The full state table — `{cpu measured} × {hybrid measured} × {stamped-but-unmeasured} × {what the static rules would do}`, with the next-run outcome for each cell — is now a comment in the code. Every defect in this mechanism has been a missing row, so the table is the artifact worth maintaining, not the prose.

## v0.15.35 — reviewing the data-writing paths: a resize that could race the live writer, and priors that were never read

A risk-ordered review of the v0.15.21–34 arc, run one angle at a time because the arc kept surfacing bugs at its tail end. Angles 1 and 2 cover the two paths that write user data.

### Review of the compress bringup restructure (v0.15.31)

The v0.15.31 change moved GPU detection, device selection, per-device sizing and worker spawning onto a background thread while the main thread became the producer. Two findings, both in `compress_nvcomp`.

**A `ResultStore::slots` resize could race the running writer.** `init_slots` resizes the slot vector and then fills it; `writer_thread` iterates that same vector in `drain_slots_locked` on every wakeup. Before v0.15.31 the resize ran microseconds after the CPU pool started, with the reader not yet going — no frames existed, so the writer was never awake to see it. The restructure moved it onto the bringup thread, where it now fires seconds into a run whose CPU workers are waking the writer thousands of times a second. The writer could observe a torn range, or a slot after the `resize` but before its `make_unique`, and dereference a null `unique_ptr` mid-archive. `decompress_nvcomp` has taken `results.m` around this call all along, with a comment naming the hazard; compress was simply left behind by the restructure and now does the same.

**The `--tar` size gate ignored an explicit backend choice.** The gate that skips GPU bringup for an archive smaller than one GPU batch is a mirror of the one in `apply_backend_defaults` — but that one returns early on `backend_user_set` ("this only ever moves a DEFAULT"), while the `--tar` copy checked only `--gpu-only`. So `--tar --hybrid` on a small tree was silently downgraded to cpu-only, as was any run using `--cpu-share`/`--gpu-batch`/`--gpu-streams` (which imply an explicit backend). Now both gates test the same condition.

### Review of the extract writer pool and its workload class (v0.15.27/28)

Second angle, same method: the `-d --tar` writer pool, its `--adapt` contraction supervisor, and the workload-class prior. The pool mechanics audited clean — retirement reads its flag inside the CV predicate under the queue lock so a retiring worker exits *before* popping and can never drop a write, base writers are never retirable so the pool cannot shrink below what is needed to drain, and `stop_pool` joins the supervisor before every writer, retired ones included. All three findings were instead **state leaking across boundaries** — across archives, and across build configurations.

**Every `--adapt` prior was dead in the CPU-only build.** `adapt_load_priors()` has exactly one call site, and it sat inside `apply_backend_defaults`'s `#ifdef HAVE_NVCOMP`. The *save* side did not: profile writes are guarded only by `#ifndef _WIN32`. So a `USE_NVCOMP=OFF` build wrote `tar_write_threads`, the class buckets, `overall_gibs*` and the read-path rates on every qualifying run, and never read a single one back — the writer pool re-probed from the default forever and the class buckets accumulated write-only. Most of what the profile carries (read path, writer probe, extract pool size) has nothing to do with whether the binary has a GPU backend, so the load and the backend-agnostic priors now live outside `HAVE_NVCOMP`; only the GPU batch start and the scheduler EMA seeds stay behind it. This is the third defect in the configuration nobody compiles.

**The extra-writer target was not reset between archives.** The governor zeroes `g_adapt_ewgrow_target` once per operation, but `extract_tar` loops over every archive in `tar_sources` with a fresh pool and a fresh supervisor each time. A fresh supervisor starts at `active = 0` and its first wait predicate compares against the global — which still held the previous archive's target. Publishing the seed was skipped when an archive wanted no extras, so archive N+1 was force-grown to archive N's size the instant its supervisor started, before its own class or prior could say otherwise. The worst case is exactly the one the workload classes exist to prevent: a small-file archive settling at ~44 extras, then a huge-file archive — whose own prior is 0 — inheriting all 44. The seed is now published unconditionally, including zero.

**The learned pool size could be filed under the wrong workload class.** `g_adapt_tar_wclass` is published per archive, but the settled size is a single per-operation number and the governor's sizing geometry latches on the first archive it sees. In `gzstd -d --tar --adapt small-files.tzst huge-files.tzst`, archive 1 converges and archive 2 publishes its class — so the profile recorded the small-file answer in the large-file bucket, poisoning the seed for every later run of that shape. A mixed-shape run is now detected when a second, different class is published; it keeps the flat key (already documented as "most recent, whatever the shape") and skips the class bucket.

Five further items are recorded as suspected-but-unproven rather than fixed, the most substantive being that the governor never reads the supervisor's `ewgrow_cap_`, so a cap-clamped final grow round could persist a size the pool never actually reached.

### Review of the GPU engagement guards (v0.15.28–31)

Third angle: the guards that decide, on every run, whether the process pays for CUDA at all. They changed default behavior on every machine, so a wrong answer here is invisible — the run simply uses the slower engine and still exits 0. Four findings, and three of them made the guards *too eager to skip the GPU*.

**Decompress cut its "is this still worth it?" sample short when the READER finished, not when the job did.** This is the exact trap the compress side documents avoiding: `compress_nvcomp` ends its sample on `comp_progress() >= 0.98`, while `decompress_nvcomp` ended it on a flag set immediately after the last frame was enqueued. The decompress progress signal itself is honest — `read_bytes` is incremented by the workers, not the reader — but the reader never blocks (the queue caps sit far above the frame count of a typical archive), so on a high-ratio archive it finishes in a fraction of a second with the workers a few percent in. The sampler read that as "the run ended during the sample" and skipped GPU bringup on precisely the archives GPU decompress exists for: 800 MiB compressed expanding to 24 GiB, where the honest extrapolation says seconds of work remain. The stop predicate is now job-tied like compress, and the flag means only a genuine abort; teardown wakes the sampler instead of stopping it.

**`--adapt` could inflate the size gate's bound to gigabytes and silently disable the GPU default.** The gate asks "can this input fill even one GPU batch?", computing the bound as `chunk_mib × gpu_batch_cap` — but the `--adapt` prior overwrites `gpu_batch_cap` with the profile's settled batch *earlier in the same function*. On a box that settled at 256, the bound became 16 MiB × 256 = 4 GiB, so a 3 GiB archive holding 48 full batches of work was routed to the CPU on the grounds that it could not fill one. The gate now reasons about the batch geometry `parse_args` derived, snapshotted before the prior touches it.

**`--chunk-size` inflated the bound without being able to inflate the actual GPU chunk.** `compress_nvcomp` hard-clamps the chunk to 16 MiB (a larger host chunk would overflow the device input slot), so `--chunk-size=256` gives the GPU 16 MiB frames regardless — but the gate sized itself off the unclamped request, making the bound 16× the geometry the run would actually use. The gate now applies the same clamp.

**The comment describing the decompress bound stated the opposite of what the code does.** It claimed that comparing against the compressed size "errs toward keeping the GPU, which is the safe direction." The test is `known < one_batch → cpu-only`, so understating the work pushes toward *dropping* the GPU. The direction is now documented correctly, with the error growing as the compression ratio does. Left as behavior: correcting it means assuming a ratio, and tuned constants do not belong in defaults — the honest fix is to use the frame count where it is already known.

Audited clean here: the bound is `uint64` throughout and cannot reach zero except through the debug hook (which fails open, as documented); both env hooks reject malformed, empty and negative values and fall back; the `stat()` loop correctly refuses to size symlink targets it cannot resolve, non-regular files and stdin, so `--tar` creation from a tree opts out on its own; `-d --tar ARCHIVE MEMBER…` sums only the archive, not the members; every uncertain branch of the worth-it sampler falls toward engaging; and the v0.15.29 walk reorder is safe because `build_layout` reads none of the chunk-size state resolved after it.

### Review of the backend prior (v0.15.33)

Fourth angle, and the one that mattered most: these defects corrupt the *profile*, so unlike a bad run they persist and compound. The profile I/O itself audited clean — the JSON parser rejects `1e999` and every malformed token, all four double→integer casts are clamp-guarded, the pid-temp + rename covers disk-full and an unwritable cache dir, and a faulted run merges no rate at all. The four findings are all in what gets recorded, and under which key.

**The run recorded the backend it was *asked* for, not the one it *used*.** `adapt_backend_used`'s own comment claims it reads "the POST-defaults opt, i.e. what really ran" — true only for the startup size gate, which mutates `opt`. Three paths decide at runtime not to use the GPU and cannot: the `--tar` size recheck hands a `cpu_only` **copy** to `compress_cpu_mt`, "no devices found" falls back in place, and the lazy-engagement guard declines before the first CUDA call. Each measures the CPU while still calling itself hybrid, pulling `overall_gibs_hybrid` toward the cpu-only rate — corrupting the one comparison this feature exists to make. The cleanest trigger is a box whose GPUs are busy or hidden (`CUDA_VISIBLE_DEVICES=`), which still writes to the same profile entry because the fingerprint enumerates GPUs from `/proc/driver/nvidia`. A `g_adapt_gpu_engaged` flag, set where GPU workers are actually spawned, now decides the key.

**`-d --tar` extract and `-t` filed their rates in the same two keys as plain `-d`.** Both blocks clear `src_path` with a comment stating the rates are not comparable, then record the same rate under `overall_gibs_cpu`/`_hybrid` regardless. An extract is device-write-bound at the box's write ceiling and barely moves with the backend; `-t` has no sink at all and is systematically the fastest of the four decompress shapes. Since the backend is chosen by PCIe generation, the misattribution is systematic per box rather than random — on a Gen4 host every extract landed in `hybrid` at the write ceiling while `cpu` collected warm plain-`-d` runs, and a single sample carries half the EMA weight. All three paths now record no backend.

**Explore-once could never terminate.** When only hybrid has been measured, cpu-only is explored once so the pair becomes comparable — but nothing checked that the exploratory run would last long enough to be *recorded*. Under the 3 s save floor it records nothing, cpu-only reads as untried forever, and every subsequent run explores again. The read-path prior guards exactly this, with a comment naming the failure ("a probe that can never record would pay the alternative path indefinitely without closing the loop"); the backend prior now uses the same predicate.

**The verdict could latch permanently on the wrong backend.** Once both keys exist only the winner ever runs again, so only the winner's EMA moves while the loser's number stays frozen at whatever it was. The EMA weights the newest sample at 50%, so two unrepresentative winner runs are enough to walk it under the stale loser and invert the default — with no path back, because nothing measures the loser again. With this project's own numbers: cpu 4.24 / hybrid 1.87, then two `-19` compresses (the level is not part of the key) leave cpu at 1.17, hybrid wins on 1.87, and the wrong default is permanent on a box where cpu-only is 2.3× faster. Each key now carries the run index it was last measured at, and the prior re-measures the side it is not choosing on a bounded cadence — one run in `ADAPT_BACKEND_RECHECK_RUNS`, costing nothing where the verdict is stable.

Recorded but deliberately not changed: a decided prior returns before the warm-input residency rule, so `--adapt` with a populated profile disables the v0.15.2 warm/cold distinction and replaces a per-run measurement of *this input* with a per-machine EMA that blends warm and cold runs. Which measurement should win is a policy question that needs data from both test machines, not a review-time edit.

### Review of the decoupled extract pipeline (v0.15.22–26)

Fifth angle, over what these notes call the densest concurrency in the tree. The concurrency itself held up: slot ownership makes concurrent writes to a reorder slot impossible, the GPU rescue path detects failure per frame rather than per batch and cannot race the batch's own write, teardown orders its five stage shutdowns correctly with every predicate re-checked under its lock, and the retirement floors make retiring the last worker of a stage unreachable. Both findings are on paths the design did not cover.

**A budget-permit leak on an aborted parse could hang the extraction.** A permit spans dispatch→consume on the parse thread, and the prefetch is deliberately greedy, so one partition routinely holds several permits and can hold all of them. But `parse()` does not always run to the end of its slice: tar-level corruption — a bad header checksum, a truncated long-name or pax record — calls `fail_data()` and returns mid-slice, abandoning every prefetched permit. Every other parse thread then blocks forever in `budget.acquire()`, the join never returns, and a corrupt archive **hangs instead of exiting 4**, which is the one outcome this case should produce. The contrast is instructive: zstd-frame corruption was always safe because it goes through `die_data` → `exit(4)`; only the tar-level path returns. Partial leaks degrade to a permanently narrowed pipeline and accumulate toward the same deadlock. Fixed by reclaiming the unconsumed range after `parse()` returns — guaranteed to terminate, because readers and decoders are only shut down after this join.

**The parallel extract path was copying every file byte.** `read_segs` decided "no owning frames, fall back to a copy" by testing `!src`, but `src` is set only by the sink constructor; `run_parallel` builds its reader with the *producer* constructor, which sets `prod` instead. The test dates from v0.14.72, before producer mode existed in v0.14.86, and was never taught about it — so every regular file and every large-file part job in the parallel path took the copy branch, reintroducing exactly the per-byte memcpy v0.14.72 was written to remove. It stayed invisible because the copy is now spread across N partitions rather than concentrated on one serial thread, and because the CHANGELOG asserted the parallel path reused the zero-copy writes unchanged. It did not.

The obvious risk in fixing it is memory: queued jobs now pin whole frames instead of exact-size copies. Measured rather than assumed, on a 1.13 GiB archive of 40 × 30 MB files plus 3000 small ones, median of three runs — **peak RSS fell 27%, from 4.28 GB to 3.11 GB**, with wall time flat to slightly better (0.91 s → 0.87 s), because the copy branch had been allocating a duplicate buffer per file. Extraction verified byte-identical on both the pool and the default path. No throughput claim is made from one box: this machine is device-write-bound, so the removed memcpy has little room to show up here and would matter more on a decode-bound or CPU-poor host.

### Review of argument parsing and CPU-only flag policy (v0.15.34)

Final angle, over the smallest surface — and the one that produced the sharpest result, because it is small enough to check exhaustively. The v0.15.34 "identical by construction" claim was verified rather than trusted: all seven GPU options are matched by the same helper with the same name string in both build configurations, both `--opt=value` and `--opt value` forms consume identically, and no prefix-matching catch-all survives. That bug class is genuinely closed **for the options inside the GPU `#ifdef` block**. Everything below is an option parsed *outside* it, which the rewrite never reached.

**`--cpu-batch` could hang a CPU-only build outright.** The guard that neutralizes this hybrid-only knob reads `if (opt.cpu_only && ...)` — but in a `USE_NVCOMP=OFF` build nothing ever sets `opt.cpu_only`, since `apply_backend_defaults` is compiled out and `main` dispatches to the CPU paths unconditionally. So the guard never fired and the knob reached the workers' non-hybrid branch, which is compiled in both configs. Past the documented stop-and-go pathology, that deadlocks: workers wait for queue depth ≥ `cpu_queue_min` while the producer is blocked in `TaskQueue::push` on the byte cap, so when the cap admits fewer frames than the requested batch, neither side can advance and the done flag is never set. Confirmed by running it — `gzstd -d -T 2 --cpu-batch=64` on a 1.4 GiB incompressible archive hangs indefinitely on the CPU-only build, while the identical command on the GPU build completes, because there the guard resets the knob. In a CPU-only build every run is effectively `--cpu-only`, so the guard now applies unconditionally.

**`parse_u64_value` accepted negative numbers.** `std::stoull` is *defined* to negate and wrap for a leading `-`, consuming the whole string and throwing nothing, so `--chunk-size=-1` passed every check and arrived as 18446744073709551615 — which the RAM clamp then reported back to the user as "reduced --chunk-size from 18446744073709551615". Every `size_t` option was affected: `--read-threads`, `--write-threads`, `--chunk-size`, `--cpu-batch`, `--gpu-batch`, `--gpu-streams`, `-M`. The sign is now rejected explicitly. The `int` and `double` helpers were already correct.

**`--verify-retries` inverted its own meaning on overflow.** It was the one integer flag that hand-rolled `strtol` without an `ERANGE` check: an out-of-range value clamped to `LONG_MAX`, `(int)LONG_MAX` is −1, and the `> 0` test then read that as "0 = unlimited". A request to *bound* the retries produced unbounded retries. It now goes through the same range-checked helper as every other integer flag.

**Thread counts had no upper bound**, so `--write-threads=1000000` spawned until `std::system_error` escaped an uncatching path and aborted the process instead of reporting a usage error. `--write-threads`, `-T` and `--read-threads` are now rejected at parse time with a message that says why.

**Four GPU options and one GPU demand were swallowed in silence** in CPU-only builds: `--cpu-share`, `--hybrid-floor`, `--hybrid-floor-factor` and `--watchdog` were consumed, validated and ignored with no note at any verbosity; `--hybrid` printed its note only on the compress path, so `-d`/`-t` said nothing; and `--verify-engine=gpu` — a demand for the GPU, which the GPU build warns about when it cannot honour it — produced no diagnostic at all. All now raise the same "GPU tuning flags are ignored in this CPU-only build" note that v0.15.34 introduced.

Audited clean: conflict checks are order-independent and cannot be bypassed by an implied backend; all four `parse_*_arg` helpers reject a missing or empty value; the `--verify-engine`/`--verify-retries` substring arithmetic is off-by-one-free; short-flag bundling correctly excludes `-vv`/`-qq`, digits and value-taking flags; `--` is handled consistently in both the bundling pre-pass and the main loop, so the v0.14.80 `--tar --` crash stays fixed; `-` reaches stdin; an empty argv cannot null-deref.

The rest of the v0.15.31 restructure audited clean, including the parts flagged as risky when it was written: both CPU fallbacks return before `std::move(pre_layout)`, the bringup thread is joined before `workers` is iterated and no early return or `die()` sits between spawn and join, worker-referenced state (`per_dev`, `json_sink`, `fatal_msgs`) outlives every join, the zero-GPU-worker path completes correctly, and the worth-it sampler's CV predicates are re-checked with every uncertain outcome falling toward engage.

## v0.15.34 — CPU-only builds treat GPU flags coherently (and stop mangling their arguments)

The `USE_NVCOMP=OFF` build failed 7 suite tests, and had done for long enough that they read as background noise (verified identical at v0.15.26). They were pointing at a real policy inversion, plus a latent bug none of them tested.

**The policy was backwards in both directions.** A *demand* for the GPU was swallowed silently; a *hint* about how to use one was a fatal error.

- `--gpu-only` matched a catch-all `--gpu-*` branch that did nothing, so `opt.gpu_only` was never set. A script asking for GPU compression got CPU compression and **exit 0** while believing otherwise. It also meant the conflict checks could never fire: `--gpu-only --cpu-only` and `--sliding-window --gpu-only` both succeeded, because the flag they guard against was never recorded.
- `--pinned` / `--no-pinned` weren't in that branch at all, so they died as *unknown option* — the opposite treatment for flags that are pure tuning.

Now: a demand fails loudly (`--gpu-only` is a usage error naming the cause and the fix), a hint is accepted and ignored, with a `-v` note so the ignoring isn't silent.

**The latent bug the tests never caught.** The catch-all matched by prefix and did not consume values, so the *separated* form `--gpu-batch 8` left `8` behind as a positional — and the user got `-o/--output cannot be used with multiple input files`, an error with no visible relationship to what they typed. Every GPU option in the CPU-only branch is now matched by the same `parse_*_arg` helper the GPU build uses, so argument consumption is identical **by construction** rather than by a second implementation that can drift.

**Result:** CPU-only build goes from **266 passed / 7 failed** to **273 / 0**; GPU build unchanged at 354/0 default and 487/0 extensive.

That these had been failing for many versions is the argument for building both configurations routinely — this is the second defect in two days found only by compiling the config nobody compiles (v0.15.32 fixed one that didn't build at all).

## v0.15.33 — the backend prior compares end-to-end rates, not per-engine ones

**`--adapt` was choosing the GPU for compress on a box where the CPU is 2.3× faster, and the prior that exists to prevent exactly that was firing the wrong way.** Found by the `--adapt-warm` benchmark: across all five 20 GiB profiles, `adapt` tracked hybrid at 1.86 GiB/s while `cpu-only` did 4.24 — a 2.1–2.5× loss on every single profile, with warm priors making no difference.

The prior compared `cpu_gibs` against `gpu_gibs` (the hybrid scheduler's per-engine EMAs) and chose cpu-only when the CPU "dominated" by 1.5×. On this machine those record **cpu 1.88 / gpu 4.06** — concluding the GPU is twice as fast, when an actual benchmark of the same workload has hybrid 2.3× *slower*. Not a threshold that needed tuning: an invalid comparison.

**Why those numbers can't answer the question.** Three independent reasons, any one of which would be enough:

- **Duty-cycle bias.** Each EMA samples only ticks where that engine moved bytes — *"idle ticks shouldn't collapse the EMA"* — so a **bursty** engine is measured only while it bursts. The GPU works in batches; the CPU runs continuously. The tell-tale: the pair sums to ~3× the rate the run actually achieved, so they are plainly not shares of one clock.
- **cuInit is invisible.** Seconds of bringup never appear in a 0.5 s tick window.
- **The CPU is held back in hybrid** by the queue floor, so its measured rate is not what cpu-only would achieve with the whole machine.

**The replacement** is the pattern already proven twice here — the read-path prior, and v0.15.28's workload-class writer prior: record the *same* end-to-end number under a key for the thing that produced it, try the untried alternative once, then let the better measurement win with a margin. `overall_gibs_cpu` and `overall_gibs_hybrid` are filed per direction by the backend that actually ran; with both present the faster wins with a 5% anti-flap margin; with only hybrid measured, cpu-only is explored once so the pair can be compared next run; within the margin the static rules keep their say. This prices cuInit and coordination overhead by construction instead of modelling them.

The per-engine keys stay exactly as they were — the scheduler pre-warms its rate EMAs from them, which is a fair use of precisely what they measure. Only the *backend choice* moved off them.

**Measured convergence** (20 GiB, cold, `--direct-read`):

| run | compress | what it did |
|---|---|---|
| 1 | 9.87 s | no prior → hybrid, records `hybrid 2.19` |
| 2 | 4.43 s | explores cpu-only, records `cpu 4.62` |
| 3 | 4.48 s | both measured → picks cpu-only |
| 4 | 4.45 s | stable |

**2.2× faster, converged in two runs, self-measured.** Decompress converges the same way on this box (8.97 → 6.76 s, cpu-only 2.96 vs hybrid 2.45). Note this is not a hardcoded "CPU wins" — on a box where hybrid is genuinely faster the same two runs select hybrid, which is the entire point.

**The suite test for this changed contract, and got better for it.** The old test seeded `cpu_gibs: 10 / gpu_gibs: 2` and asserted cpu-only — it could only ever check one direction, and it failed here exactly as it should have. It now seeds the end-to-end pair and covers three cases: cpu faster → cpu-only, **hybrid faster → hybrid**, and hybrid-only → explore cpu-only. The middle one matters most: without it this would be indistinguishable from a hardcoded "CPU wins" wearing a measurement. The crafted profile deliberately leaves the per-engine numbers pointing the *other* way, so if anything ever wires them back into the choice, the test catches it.

**Validated:** default suite **354/0**, extensive **487/0**, no drift note; convergence measured end-to-end on 20 GiB corpora in both directions.

## v0.15.32 — fix a broken CPU-only build, and pre-deployment checks

**The `USE_NVCOMP=OFF` build did not compile.** `gpu_min_useful_bytes()` (added in v0.15.28) reads `opt.gpu_batch_cap`, which lives inside the `#ifdef HAVE_NVCOMP` block of `Options` — so every CPU-only build since v0.15.28 has been broken. It went unnoticed because nothing in this stretch of work rebuilt that configuration. Both GPU-engagement helpers are now `#ifdef HAVE_NVCOMP`-guarded; neither has any meaning without a GPU.

This was found while checking whether the tree was safe to tag, which is the honest argument for running that check: the answer was no, for a reason nobody would have seen until the release build ran.

**Pre-deployment checks now passing**, all on one machine by varying `CUDA_VISIBLE_DEVICES`:

| configuration | result |
|---|---|
| 8 GPUs | 352/0 suite, round-trips |
| 2 GPUs | workers spawn, round-trips |
| 1 GPU | workers spawn, round-trips |
| no GPU visible, forced engage | `no devices found; hybrid running CPU-only`, round-trips |
| `USE_NVCOMP=OFF` build | compiles, 266/7 (see below) |

The 1- and 2-GPU cases matter beyond the obvious: they are the first exercise of the **provisional-count over-estimate** path, where the throttle is sized for 8 devices but only 1–2 exist. On the 8-GPU box the provisional and real counts coincide, so that path had never run.

**Pre-existing, not from this work:** the CPU-only build fails 7 tests (`--gpu-only no GPU`, the four `--pinned` forms, `conflicting flags`, `--sliding-window --gpu-only rejection`) — GPU flags are accepted rather than rejected when nvCOMP is compiled out. Verified against v0.15.26, built the same way: **identical 7 failures, identical 266 passes.** Worth fixing, but it is an older gap and not a regression.

## Follow-up validation (2026-07-27) — two open questions closed, no code change

Both were recorded as untested rather than measured; both now have numbers, and neither needed a fix.

**The `prefer_inline` threshold (`comp_ratio > 0.90`) was untested in the 0.80–0.95 band.** Built three 12 GiB archives of twelve ~1 GiB files at ratios 0.850 / 0.920 / 0.960 and measured both decode paths directly (`--cpu-only` throughout, to isolate the decode path and leave the GPUs alone):

| ratio | inline | pool | gate chose | |
|---|---|---|---|---|
| 0.850 | 6.10 s | **5.90 s** | pool | picked the faster |
| 0.920 | 6.31 s | 6.32 s | inline | tie |
| 0.960 | 6.28 s | 6.50 s | inline | correct |

The gate is right at every point, and the wider finding is that **inside the grey band the two paths are within ~3%** — the exact threshold placement barely matters there. The −28% divergence that motivated the gate lives at ratio 1.00, well clear of the band. So 0.90 is safe and there is nothing to re-tune.

**The `md` writer-class bucket had its optimum untested** (only `sm` and `lg` were measured). Swept `--write-threads` over a 15 GiB / 15 000 × 1 MiB archive:

```
4 → 4.08 s    8 → 4.04 s    16 → 3.20/4.03 s    32 → 4.05 s    60 → 4.12 s
```

Flat from 4 to 60. Unlike `lg` — where 60 writers cost ~9% against 16 — the middle bucket simply does not care, so there is no optimum to miss. `--adapt` converges 60 → 19 → 16 across three runs and persists `tar_write_threads_md: 16`, which lands on the plateau. The three buckets are now characterised: `sm` wants many writers, `lg` wants few and is punished for too many, `md` is indifferent.

**Not done: the engagement guard is still blind to a *busy* GPU.** Left deliberately — see the note at the end of v0.15.31.

## v0.15.31 — compress gets it too: 512 MB went from 5.51 s to 0.25 s

v0.15.30 gave decompress a measured "is the GPU still worth starting" test and recorded compress as owed. This is that debt paid.

| compress | before | after | `--cpu-only` |
|---|---|---|---|
| 512 MB | 5.51 s | **0.25 s** | 0.26 s |
| 5.3 GB | 5.79 s | **1.52 s** | 1.42 s |

`compress_nvcomp` called `cudaGetDeviceCount` as its very first act, so ~5.4 s of cuInit was charged before the reader or the CPU pool had done anything. Detection now moves to a background bringup thread that owns the whole sequence — decide, detect, select, size the per-device state, spawn the workers — while the main thread goes on to be the producer. It has to own the *spawn*, not just the detection: the decision needs the pipeline to have made measurable progress, and in compress the reader **is** the main thread, so joining any earlier would reintroduce exactly the stall being removed.

**Adaptive hybrid only.** Fixed-share (`--cpu-share`) keeps the synchronous bringup — the GPU must be warm before the reader starts or a small input drains entirely to CPU and the explicit split is silently ignored (the v0.13.11 regression). `--gpu-only` also stays synchronous: it has no CPU pool to overlap with, and "no devices" should stay a clean error rather than a new failure path threaded through the archive writer.

### The bug this shook out: read progress is the wrong signal for compress

The first working version skipped the GPU on *every* compress, including jobs long enough to want it. The trace said `0% read at 0 ms` — the sample was being cut short instantly.

The cause was not the plumbing. The sampler was reusing decompress's progress signal, bytes read from the source, and tying its stop to teardown. Neither survives contact with compress: **with an mmap'd input the reader reaches the end almost immediately** — the pipeline's own note is "producer_done fires at t≈0 with mmap" — so the main thread arrives at teardown while the CPU pool still has seconds of work, and read progress reads ~100% when nearly all the compression is still to do.

So compress now measures **work completed** — frames handed to the writer, scaled back to input bytes — and stops on that rather than on the reader. The verdicts are right in both directions and visible at `-vv`:

- 512 MB → `100% done at 82 ms` → skip
- 5.3 GB at `-T 2` → `8% done at 252 ms → 2.61 s remaining` → skip (correctly: less than cuInit)
- 5.3 GB at `-T 1` → `4% done at 252 ms → 5.13 s remaining` → **engage**, 8 device workers online

That last case round-trips byte-identically, which is the case that matters — it is the one with GPU-produced frames interleaved with CPU ones.

**Structural note.** `DevStats` and `StatsSink` each hold a `std::mutex`, so they are neither movable nor resizable and cannot be sized from a provisional count and fixed up later. They are now built once, inside the bringup, from the real device count — held by `unique_ptr` at function scope so they outlive the workers holding pointers into them. The throttle and scheduler still size from a provisional count (RAM-capped, so over-estimating is harmless — decompress's own justification). Running with **zero** GPU workers is a supported state, verified rather than assumed: `HybridSched`'s queue floor is `streams × batch`, so with nothing registered it is 0 and the CPU pool competes freely.

**Validated:** default suite 352/0, extensive 485/0. GPU-engaged compress round-trips byte-identically at 5.3 GB; `--gpu-only` and `--cpu-share` still take the synchronous path (confirmed by the absence of a bringup sample and 8 workers online).

**Known gap, deliberately not closed: the guard is blind to a *busy* GPU.** It asks only whether the job outlasts cuInit, not whether the device is available — observed live, with three GPUs at 78–98% and all eight holding another tenant's VRAM, it would still engage. Not fixed speculatively, for two reasons. `select_best_gpus` already ranks devices by utilisation and free VRAM, so contention is accounted for in *which* device is chosen, just not in an all-or-nothing abstain; and `HybridSched` already rebalances on measured rate, so a slow GPU sheds work on its own. The residual cost is cuInit plus a few slow batches before that rebalance — real, but not obviously worth a threshold that could refuse a briefly-busy GPU. It needs evidence it bites in practice and a contended machine to validate against, and running GPU benchmarks on someone else's saturated devices is its own problem.

## v0.15.30 — the GPU has to prove it can still pay for itself

### The measured test

The size gate rules out inputs that cannot fill one GPU batch, but that is only a *necessary* condition — it says nothing about whether the job outlasts cuInit. On a fast many-core box it under-shoots badly. Measured, identical output both ways:

| operation | `--cpu-only` | `--hybrid` (before) |
|---|---|---|
| compress 512 MB | 0.22 s | 5.51 s |
| compress 5.3 GB | 2.93 s | 5.79 s |
| decompress 5.3 GB | 1.88 s | 3.09 s |

All of it bringup the job never recoups. The honest test is the one the `-d --tar` decode pool already applies before spawning GPU decoders: *does enough work remain to outlast the init?*

That cannot be answered up front — it needs a rate, and any rate model is wrong here (this box compresses at ~2.4 GiB/s aggregate whether it has 96 cores or 256: the pool is memory-bound long before it is core-bound). So it is **measured**: the deferred bringup thread watches what the pipeline actually achieves for ~0.25 s and extrapolates, *before* the first CUDA call — so a "no" costs nothing. No rate constants and no core-count thresholds; it reads the box it is running on.

**Measure a windowed rate, not cumulative progress.** The first attempt extrapolated from bytes-since-zero and got it backwards: on the 5.3 GB decompress it saw 2% of the source in 151 ms and predicted **7.4 s** remaining for a job that had ~1.4 s left — because cumulative progress folds in the startup ramp. Discarding a 0.10 s warm-up and measuring the rate over the next 0.15 s gives 11% read → 1.26 s remaining → skip. (Same shape as v0.15.26's integrated-rate fix, for the same reason.) The verdict is visible at `-vv` as `[GPU] bringup sample:` rather than being a black box.

Decompressing 5.3 GB: **3.09 s → 1.55 s**. Conservative in every uncertain direction — an unknown size, an unreadable meter, or a job too slow to sample all engage the GPU, and `--gpu-only` always engages since it has no CPU pool to fall back on.

**This is wired into decompress only.** Compress needs the same treatment but a larger restructure first — see below.

**Validated:** default suite **352/0**, extensive **485/0**, no drift note. The new test asserts both directions on one fixture — guard 60 s ⇒ skip, guard 0 ⇒ engage — so it proves the *guard* is what decides rather than merely that something was skipped. A deliberately slow decompress (`-T 1`, 4.66 s remaining) still engages, and `GZSTD_DEBUG_GPU_GUARD_SEC=0` brings all 8 devices online.

That test needed `--hybrid` passed explicitly, which is worth recording: its fixture is small *and* warm, so the pre-existing residency rule already defaults it to cpu-only, `decompress_nvcomp` is never entered, and the code under test never runs. The first version of the test passed no backend and silently exercised nothing.

**Benchmark:** `gzstd-benchmark.sh` gains an `adapt` row (`--adapt --no-profile`, so a `/dev/null`-sink run cannot write bogus calibration into the machine's real profile) and an opt-in `--adapt-warm` row that lets the governor keep what it learns, primed by one untimed run per file so every timed iteration is genuinely warm. `XDG_CACHE_HOME` is redirected into the run's temp dir, which is what makes the warm row safe. The pair brackets `--adapt`: floor and steady state. First result on a 20 GB corpus is itself the argument for the compress work below — `--adapt` tracked hybrid exactly at 2.03 GiB/s while cpu-only did **4.48**.

### Owed: the same guard on compress

`compress_nvcomp` still calls `cudaGetDeviceCount` synchronously as its first act, so compress has neither the deferral nor the sample — the 512 MB / 24× case above is **not yet fixed**. The reason it is not a small change: in compress the *main thread is the producer*, and GPU workers are spawned before the reader runs, so nothing has made progress at the point a sample would be taken. Getting a measurement requires what decompress already does — a bringup thread that owns detection *and* worker spawn, so the main thread can proceed to the reader meanwhile. That means moving `per_dev`/`json_sink`/`fatal_msgs` sizing and the watchdog setup off the main thread onto that thread, with a provisional device count for the throttle and scheduler sizing (RAM-capped, so over-estimating is harmless — the same justification decompress uses). Deliberately left as its own change rather than rushed onto the archive-writing path.

## v0.15.29 — `--tar` creation gets the small-input gate too: walk before cuInit

v0.15.28 stopped small inputs paying for a GPU they never use, but explicitly left one case out: **`--tar` creation from a directory tree**. The startup gate can only `stat()` what it is handed, and a tree's real size is not known until it has been walked — so creating an archive from a small tree still paid the full init.

**Creating a 4 MiB archive from 200 files took 4.78 s. It now takes 0.039 s.**

The fix is a reorder, not a wider gate. `compress_nvcomp` called `cudaGetDeviceCount` as its very first act and did not build the tar layout until ~150 lines later; the walk now happens first, and the same `gpu_min_useful_bytes()` bound is applied to the real total. **Nothing is lost by the reorder** — that detection is synchronous, so it never overlapped the walk anyway. It simply ran first and charged cuInit before anyone knew whether it was worth paying.

The layout is then handed to whichever path runs, so **the tree is walked exactly once either way** — `compress_cpu_mt` takes an optional prebuilt layout, used both by this gate and by the pre-existing "no CUDA devices found" fallback, which would otherwise have re-walked the whole source. The suite asserts the walk count is 1, along with the round-trip.

The threshold itself moved into a shared `gpu_min_useful_bytes()` helper, so the startup gate and this deferred one cannot drift apart — they were duplicating the same `chunk_mib × gpu_batch_cap` computation and the same `GZSTD_DEBUG_GPU_MIN_BYTES` override.

**Still not covered:** compressing from a pipe or stdin, where the size is genuinely unknowable up front. That is inherent, not an oversight.

**Also:** the `-v` gate message now says it is *overriding the announced backend*. The `[STARTUP]` banner is printed before the walk runs, so it still announces hybrid; the following line now makes the relationship explicit rather than appearing to contradict it.

## v0.15.28 — small inputs no longer pay for a GPU they never use; writer prior keyed by archive shape

### Compressing 200 KB took 5.6 seconds

The largest find of this round, and it had nothing to do with `--adapt`. Bringing up CUDA costs a fixed cuInit — measured **2.3 s** before a decompress and **5.6 s** before a compress on this 8-GPU host — and it is charged before any work begins. On a small input the CPU pool finishes the entire job in ~20 ms and the process then sits waiting to join a GPU bringup that had nothing to do:

| operation | before | after |
|---|---|---|
| compress a 200 KB file | 5.59 s | **0.018 s** |
| decompress a 200 KB file | 2.27 s | **0.022 s** |
| `-d --tar` a tiny archive | 3.47 s | **0.024 s** |

There was already a guard for "the CPU drained everything during init — skip spawning GPU workers", but by the time it fires the init has been paid. The only real fix is not to start it.

The threshold is the GPU's **own batch geometry**, not a tuned constant: one batch on one device is `gpu_batch_cap` frames of `chunk_mib` each — 8 × 16 MiB = 128 MiB by default. Below that the GPU cannot fill a single batch, so no amount of GPU throughput could repay its initialization, and the bound moves correctly by construction if `--gpu-batch` or `--chunk-size` change it. For decompress it is compared against the *compressed* size, which understates the work and so errs toward keeping the GPU — the safe direction. It applies only where the size is genuinely known (regular files); stdin, FIFOs, and `--tar` creation from a directory tree keep today's behaviour, and an explicit `--hybrid`/`--gpu-only` is untouched because a user-set backend returns before this runs.

Verified at the boundary: a 200 MB input still brings up all 8 devices, a 95 MiB one goes cpu-only and says why at `-v`, and a 24 GiB archive still defaults to hybrid.

This matters most for the small-file case that motivated it — a script invoking gzstd per file was paying multiple seconds of setup for milliseconds of work.

**Not covered:** `--tar` creation from a directory tree still pays the init, because the backend default is chosen at startup and the tree's size is not known until the walk runs. Fixing that means deferring the backend decision past the walk rather than widening this gate, so it is left alone; the compress path already overlaps GPU bringup with a background thread, which hides most of it once the tree is more than trivially small.

### The writer prior is per (machine, archive shape)

**v0.15.27 closed the `--adapt` extract gap but left the writer prior workload-blind — one persisted number per machine, so alternating archive shapes kept re-teaching it.** That is now keyed by shape. Along the way, the same NVML startup cost fixed there turned out to be paid a *second* time, on a path that has nothing to do with `--adapt`.


The optimal writer count depends on the archive, not just the box: 390 K small files run 92.8% busy / 0.6% starved at 60 writers (metadata- and syscall-bound — the parallelism is real), while 13 huge files run 11.2% busy / 84.0% starved at the same 60 (bandwidth-bound — concurrent O_DIRECT streams contend). v0.15.27 could *correct* a wrong prior mid-run, but the correction was thrown away: whichever shape ran last overwrote the single stored number.

The archive's geometry is known before the pool starts — the parallel extractor already has per-entry boundaries and the frame table — so classify by **mean bytes per entry** and persist one settled size per class (`tar_write_threads_sm` / `_md` / `_lg`, buckets at 64 KiB and 4 MiB). That ratio is the physically meaningful one: it *is* bytes-of-work per open/close, which is exactly the metadata-versus-bandwidth tradeoff being sized for. Three buckets is a compromise — enough to separate the extremes that genuinely differ, few enough that each converges in a couple of runs. A class with no prior yet falls back to the flat key, then to the plain default, and probes from there; the flat key is still written, so profiles from older builds keep working and the streaming extract paths (which cannot know the geometry up front) still have something to seed from.

Measured, alternating the two shapes on one profile — the run that used to be poisoned is run 5:

| run | archive | pool start | settled |
|-----|---------|-----------|---------|
| 1 | 13 huge files | 60 (flat prior) | 16 |
| 2 | 13 huge files | 17 (class `lg`) | 16 |
| 3 | 390 K small files | 16 (flat prior) | 32 |
| 4 | 390 K small files | 32 (class `sm`) | 48 |
| 5 | 13 huge files | **16 (class `lg`)** — not 48 | 16 |

All three classes verified end-to-end, including a 15 GiB / 15 000-file `md` archive (60 → 19 → 16 across three runs).

### A second 300 ms NVML probe, on every GPU-capable run

`detect_min_pcie_gen()` — which picks the decompress backend default and the `--direct` default, on **every** run, `--adapt` or not — opened NVML, enumerated every device, and shut it down: the same ~300 ms measured for the fingerprint in v0.15.27. It already had a sysfs fallback sitting directly beneath it, unused whenever NVML worked.

Sysfs is now the primary path and NVML the fallback. The one thing NVML did better is preserved: its `MaxPcieLinkGeneration` is slot-aware (a Gen4 card in a Gen3 slot correctly reads Gen3), whereas a device's own `max_link_speed` is what the *card* supports. sysfs exposes the upstream port as the parent directory in the PCI tree, so taking `min(device, upstream port)` reproduces the same answer — verified against NVML on an 8-GPU host. The scan now also filters to display/3D controllers (class `0x03xxxx`); the old fallback matched any NVIDIA function, so an audio function or bridge could have dragged the MIN below the GPU's real link.

### Two bugs found while reviewing v0.15.27's own code

- **The "three consecutive ticks" surplus gate wasn't consecutive.** The counter was incremented inside the action's condition chain, so it only counted ticks that had already passed every other gate — an intermittent surplus could accumulate to the threshold across a long gap and act on evidence that was never contiguous. The run-length is now evaluated every tick, independent of the action.
- **The sizing rounds could wrap across a `Meter` reset.** `wrate` and the new busy/starved deltas are unsigned differences; a reset (the GPU-fault rebuild path) would turn a negative delta into an enormous positive rate and force a false keep. They now re-baseline instead, the same convention the regime snapshot already used.

### `--write-threads` is honoured again

The flag documents itself as "number of parallel file-writer threads" and its own comment claimed the user pin "always wins" — but it only won for the *base* pool, leaving `--adapt`'s probe free to spawn writers on top of it. A user who pins a number now gets that number: the sizing supervisor is not armed at all when `--write-threads` is set. This matters precisely for the hand-tuning the flag exists for ("the optimum is hardware-dependent — tune it on the target box").

**Also:** `gzstd-test.sh` had `EXPECTED_TESTS=345` while 346 ran, so every run ended with a drift note. Corrected.

**Validated:** default suite **350/0** and extensive **483/0**, both with no drift note; byte-identical extraction on all four test archives (incompressible 24 GiB, 390 K-file metadata, 15 GiB mid-size, decode-bound base64).

Extract, `--adapt` vs the plain default, each run starting from a pristine profile (the worst case for `--adapt`, since it has to re-derive the pool every time):

| archive | default | `--adapt` at v0.15.26 | `--adapt` now |
|---|---|---|---|
| incompressible 24 GiB (write-bound) | 7.73 s | 8.36 s (**+7% slower**) | 7.50 s |
| decode-bound base64 | 4.60 s | 3.20 s | **2.46 s (47% faster)** |
| 390 K-file metadata | 14.8 s | 14.9 s | 14.9 s (parity) |

## v0.15.27 — --adapt: close the residual extract gap — procfs fingerprint, and a writer pool that contracts

**v0.15.26 left `--adapt` extraction measurably SLOWER than the plain default on a trivial-decode archive and called the residual "warmup + writer-probe overhead, orthogonal". It was neither orthogonal nor warmup.** Measured on a 24 GiB incompressible archive (13 huge files, read from one NVMe and written to another): default **7.79 s**, `--adapt` **8.36 s** — a **7% regression** on exactly the workload where `--adapt` has the least to contribute. Two independent causes, both now fixed.

### 1. A 300 ms NVML probe on the startup path

`--adapt` builds a hardware fingerprint to key its per-machine profile. The GPU half of that fingerprint came from `nvmlInit_v2` + per-device enumerate + `nvmlShutdown`, which on an 8-GPU host measured **300 ms** — paid before any work begins, every run. It was the *entire* fixed cost of the flag: `--adapt --no-profile` (0.913 s on a trivial extract) matched the plain default (0.920 s) exactly, while `--adapt` took 1.159 s.

The NVIDIA kernel module already publishes what the fingerprint needs as plain procfs text — per-GPU model names under `/proc/driver/nvidia/gpus/<pci-bdf>/information`, the driver version in `/proc/driver/nvidia/version`. Reading those costs microseconds. The directory names are PCI bus addresses, so iterating them sorted reproduces NVML's index order: the output is **byte-for-byte identical** to the NVML path (verified against a live 8-GPU entry with mixed PCIe/NVL models — same fingerprint hash, `b622a1e77b0f5f3a`, so no machine silently orphans the priors it has learned). NVML remains the fallback for hosts exposing the device nodes without procfs. The fingerprint is also memoized now, since it is hardware and cannot change within a process.

**Fixed `--adapt` overhead on a trivial extract: 247 ms → 5 ms.**

### 2. A writer-pool prior that could only ever grow

The profile persists one settled `tar_write_threads` per machine — but the optimal writer count is **workload**-dependent, not just machine-dependent. This box measured 60 writers on a 390 K-small-file extract (metadata/syscall-bound, where parallelism genuinely pays: 60 writers run **92.8% busy / 0.6% starved**). Replay that prior on a few-huge-file extract and it is badly wrong: concurrent O_DIRECT streams contend, so the same 60 writers run **11.2% busy / 84.0% starved** and the extract is ~9% slower than at 16. Measured writer sweep on the incompressible archive (medians): 2 → 8.55 s, **4–16 → 7.6–7.8 s (a wide flat plateau)**, 60 → 8.47 s.

Nothing could walk that back. The prior was baked into the *base* pool, which has no retire path, and the only corrective actuator (action 5c) grows and is gated on `SINK_BOUND` — a classification an over-provisioned extract never reaches, because the per-thread busy average is dragged under the sink threshold by the very surplus that needs correcting. So `--adapt` loaded a bad number and then had no mechanism to question it.

- **The prior is now seeded as retirable extras** above the proven `min(cpu,16)` base, so the pool still *starts* where the profile says but can be contracted back.
- **Action 5c is bidirectional.** The contract direction is driven by the writers' own busy/starved split — a starved writer is by definition one the pipeline doesn't need — and is deliberately *not* gated on `SINK_BOUND`, for the reason above. It fires only on an unambiguous surplus (>50% starved, EMA-smoothed, three consecutive ticks), which is far clear of both measured regimes (0.01 metadata-bound vs 0.84 over-provisioned).
- **Every contraction is keep-or-revert on integrated rate**, reusing the v0.15.26 lesson: measure over ~0.8 s (the sink drains in bursts; a single window false-reverts on a gap), keep unless throughput dropped >15%, and on a drop spawn the writers back and lock the floor. Steps are geometric — half the distance to the floor, not the reader/decoder side's third, because each round costs ~0.9 s and a /3 walk from a large prior doesn't converge inside a short extract.
- **Grow never fires into a measured surplus.** Without that gate the two directions fight: a sink-bound classification arriving mid-count grows the pool and resets the surplus counter, undoing the trim it was about to confirm (observed: 60 → 36, then straight back to 44).

The controller walks **60 → 38 → 27 → 22 → 19 → 18 → 17 → 16 with zero false reverts**, and the settled size persists, so the next run on that workload starts at the answer.

**Note on `--adapt`'s single per-machine writer prior:** it is workload-blind by construction, so alternating between a metadata-bound and a bandwidth-bound archive will keep re-teaching it. That is now self-healing rather than permanent — the contraction corrects it within a run, and fully across one or two — but a workload-keyed prior (archive geometry is known before `start_pool`, like `comp_ratio`) is the real answer if this proves annoying in practice.

**Validated (real NVMe, all byte-identical, suite 346/0):**
- Incompressible 24 GiB (ratio 1.00, write-bound): default 7.79 s → `--adapt` **7.87 s first run, 7.24 s at steady state** (was 8.36 s). Profile converges 60 → 27 → 16 over three runs.
- 390 K-file metadata-bound (ratio 0.96): 14.82 s vs 14.92 s — parity. **No wrong trim** (0.4% starved, well under the bar) and the grow probe still fires (60 → 68 → 76).
- Decode-bound base64 (ratio 0.75, pool path): default 4.82 s → `--adapt` **2.92 s (39% faster)**, up from 33%. A contraction attempt here correctly **reverted** ("load-bearing") — the safety net firing on the one case that needed it.
- The `-v` `[WRITER]` line now reports the settled pool size alongside the peak, so a contracted run no longer reports only a high-water mark it spent little time at.

## v0.15.26 — -d --tar: fix a -28% incompressible-extract regression — inline trivial decode, size the pool by measured throughput

**The v0.15.25 controller contracted the reader/decoder pool on queue depth alone, which collapses it to 1 reader + 1 decoder on a barely-compressible archive and starves the writers — a measured 28% regression versus not using the pool at all.** Found while building the "keep readers high" case that v0.15.25's entry said was still owed: extract a 30 GiB incompressible archive from one NVMe to another. `--adapt` ran at **2.3 GiB/s** while the plain inline path ran at **3.2**; the writers sat **57% starved**.

**Why queue depth is the wrong signal.** In a write-limited pipe every *intermediate* queue equilibrates near-empty at steady state — the reader's input drains as fast as the writer consumes, whatever the thread counts. So "input queue empty ⇒ over-provisioned" can't distinguish real surplus from a stage that merely matches a throttled downstream. It retires readers and decoders down to the floor, and 1 + 1 (≈1.8 GiB/s here) can't feed a 3.2 GiB/s sink.

**Two fixes, each matched to a real property:**

- **Trivial decode ⇒ inline (a data property, like the <2% D2H rule).** When the archive stores at >90% of source (already-compressed media, encrypted backups), decode is ~free, so the decoupled read→queue→decode pipeline only adds per-frame handoff cost. The fused inline path — each of the ≥2 parse partitions reads *and* decodes its own frames — has no handoff and its per-partition parallelism is enough when reading is the only real work. Route those extracts to inline and never spawn the pool. Restores the writer/disk ceiling (~2.9–3.0 GiB/s).

- **Compression-heavy ⇒ pool sized by measured throughput.** The pool still starts high and contracts, but each contraction is now **keep-or-revert**: measure end-to-end rate after retiring a step, and if throughput dropped, spawn the workers back and lock that stage's floor. This keeps readers/decoders high enough to feed the writers on a write-limited compressible archive, while still contracting to 1 + 1 on a genuinely sink-bound or CPU-constrained run (there, contracting reduces oversubscription so the rate holds — no revert). Reading and decoding are sized independently: an incompressible-but-pooled run keeps ~all readers (reads are the work) and trims decoders to a handful.

**Signal notes (the two dead ends, so they aren't re-tried).** The writer-*starved* ratio is unusable as the keep-or-revert signal: with more writers than a fast sink needs, each writes a frame then waits, so the ratio pins ~85%+ regardless of reader/decoder count — blind to the very thing it would control. End-to-end **rate** is the direct outcome instead. But it must be **integrated over ~0.8 s**, not sampled per 120 ms window: the sink drains in bursts, so a single window's rate swings 0→8 GiB/s and a contraction landing next to a gap false-reverts. Integrating over several burst/gap cycles removes that.

**Validated (real NVMe, all byte-identical):**
- Incompressible 30 GiB (ratio 1.00) → inline, **2.9–3.0 GiB/s** (was 2.3); the residual gap to the raw inline default is `--adapt`'s own warmup + writer-probe overhead, not the decode path.
- 2-core decode-bound (base64, ratio 0.75; `taskset -c 0,1 -T2`) → pool, contracts cleanly to **1R + 1D, 0 false reverts**, ~25% faster than inline.
- Sink-bound 390 K-file archive (ratio 0.96) → inline, 16.4 s (no regression); the writer-growth actuator still fires (16 → 24).
- Full suite 346; `USE_NVCOMP=OFF` builds. Test hooks: `GZSTD_DEBUG_POOL_CTL` (per-window controller trace — so the little AI is never a black box again), `GZSTD_NO_PREFER_INLINE` (run the pool on trivial decode for A/B).

**Honest caveat.** On this 256-core box the throughput is a wide flat plateau (any R/D from ~4 to 96 hits the writer ceiling), so the pool's landing is imprecise — but functionally irrelevant, since every point on the plateau is the same speed. The signal is clean and the convergence precise exactly where it matters (the constrained/decode-bound box); it is only noisy where it doesn't (the plateau). A few-huge-file incompressible archive (Np≈2) still gets inline (~2.3) rather than the pool's ~2.9 — the one case a reader-count lever would add value, left for the workstation-class box.

## v0.15.25 — -d --tar: unified extract controller — start-high, contract to the bottleneck (Phase 3, part 2)

**With reading, decoding, and writing now independent stages (v0.15.24), this makes the parallel extractor allocate CPU by starting each stage OVER-provisioned and contracting the ones that aren't needed — instead of growing reactively from zero.**

**Why start-high, not grow.** Growing from zero can't adapt in time for short jobs — the extract is over before the controller ramps, so it runs the whole time under-provisioned. And the reader is *first* in the pipeline: if it's slow off the draw, everything downstream idles at the start. Since idle pool workers just block on their empty input queue (≈ zero CPU), over-provisioning is cheap and under-provisioning is what hurts. So the pool comes up fully provisioned at t=0 and the controller retires what the workload doesn't use. This also **structurally eliminates the bootstrap problem** the reactive approach had (a decoupled pipeline needs ≥1 reader *and* ≥1 decoder to flow; growing them one at a time could stall the pipe and kill the very signal the controller watches — start-high means it flows from the first frame).

**Asymmetric by each stage's physics:**
- **Reader / decoder:** start at `n_dec_max`, contract. Idle = free; more never *hurts* (readers block on I/O, decoders are pure-CPU with no shared-resource contention).
- **Writer:** unchanged — starts moderate (base/profiled) and grows via the keep-or-revert probe (action 5c). Writers are *not* started high: concurrent `open/write/close` thrash ext4's journal/dentry locks past ~16, so maxing W would make a short job slow from its first write — the opposite of the goal.
- **GPU:** unchanged (lazy) — its ~3 s cuInit + VRAM are never speculative.

**Contract signal + emergent budget.** A stage whose *input queue* sits empty for a few 120 ms windows is over-provisioned → retire a step of its workers (via per-worker retire flags, mirroring the writer supervisor), keeping ≥1; a queue that backs up resets the counter (that stage is at its right size). No explicit `N+R+D+W ≤ cores` accounting is needed: every stage blocks on its input queue, so only the *bottleneck* stage stays CPU-runnable while the others sit blocked — the budget is emergent. The GPU now engages on a clean decode-bound signal (`ddq` stays backed up while decoders were never contracted), which by construction can't fire on a sink-bound run.

Validated (real NVMe): on a sink-bound 390 K-file archive the pool **started 96 readers + 96 decoders and settled to 1 + 1** (the sink is the limit — the writer probe took W to 24), byte-identical to the serial walk with no hang; force-pool byte-identical; full suite 346. `USE_NVCOMP=OFF` builds.

**Perf validation still owed on a CPU-poor box.** This 256-core server is so CPU-rich that essentially every archive is sink-bound at 96-way decode, so R/D correctly contract to minimal here — which validates the *contraction* but not the "keep readers/decoders high because they're the bottleneck" win. That needs a workstation-class box (few cores, ideally a slower disk to reach read-bound); tracked for the incoming test hardware.

## v0.15.24 — -d --tar: decouple reading from decoding in parallel extract + uncap the writer-sink probe (Phase 3, part 1)

**Groundwork for a bottleneck-aware extract controller: make reading, decoding, and writing independently scalable so spare CPU can flow to whichever stage is actually the limiter.** Two of the three stages needed work; the third (writers) needed its ceiling lifted.

**Reader ↔ decoder decouple in `run_parallel`.** Every other decode path already separates I/O from compute — `decompress_cpu_mt`/`decompress_nvcomp` stream via a reader→TaskQueue→worker pipeline, and tar-*create*'s `assemble` runs `--read-threads` readers feeding the compress workers. Only `run_parallel` (the seek-based parallel extractor) fused them: `decode_seek_frame` did `pread` + `ZSTD_decompressDCtx` on one thread, so read concurrency and decode concurrency were the same knob. This splits it:
```
parse dispatch → dq → READER pool (pread only) → ddq → { CPU decoders, GPU workers } → psync → parse consume
```
`decode_seek_frame` is now `pread_seek_frame` (I/O) + `decompress_seek_frame` (verify + decompress); the fused form remains for the inline (non-offload) path and `seek_feed`. The **reader pool is the sole I/O stage** — the GPU pool decoder was converted to H2D straight from the reader's owned buffers (dropping its own `pread` and its staging copy; a rescued frame decompresses the buffer already in hand, no re-read), and read-byte accounting is now the reader's alone (counted once). In-flight frames stay bounded by the same permit budget (a permit spans dispatch→consume across *both* queues, so `ddq` cannot outgrow it); deadlock-freedom holds by the same argument extended one stage. This is the structural step — readers and decoders currently scale in tandem (R=D); the unified controller (part 2) will budget them independently.

Byte-identical to the serial walk on real NVMe across serial / forced pipeline / `--adapt` / forced+GPU on a 390 K-file archive, and on a 6 GiB decode-heavy archive where the GPU decoded 250 frames through the new `ddq` route (0 rescued). Full suite 346, including the four `parallel-extract` metadata tests (symlink/hardlink tree byte-for-byte vs tar).

**Writer-sink probe uncapped.** The `--adapt` writer-grow probe (action 5c) was hard-limited to `2×base` (≈32 on a many-core box), so a CPU/metadata-bound sink left most cores idle — on a real-NVMe 390 K-file extract the pool sat at 16 writers with the sink 79 % busy / 5 % starved and ~84 of 96 cores idle. The supervisor ceiling is now the machine's usable thread count and the probe does as many keep-or-revert rounds as needed; it still only grows while each `+step` round pays ≥10 %, so a device-bandwidth-bound sink settles low and low-core boxes keep the old ~`2×base` behavior.

**Writer-probe latch no longer sticks across media.** A converged verdict (`tar_wt_converged`) used to disable the probe permanently, keyed only to the hardware fingerprint — but the writer-sink optimum is media- and workload-dependent. Testing on a tmpfs (bandwidth-infinite sink → extra writers never pay → the probe reverts at base and latched off) poisoned the per-machine profile so it then refused to probe on real disk. The settled size still seeds the start, but a stale convergence no longer disables re-probing (the probe is cheap and self-limiting, ~one confirming round when nothing has changed). **Hosts that ran tmpfs-based tests may have a `~/.cache/gzstd/profile.json` throttling extract writers to a low count — worth clearing.**

Suite: 346 (no arg-parse change; the new env/structure is exercised by the existing `parallel-extract` + `GZSTD_FORCE_POOL`/`GZSTD_POOL_GPU` tests).

## v0.15.23 — -d --tar: adaptive (auto) GPU engagement in the decode pool + VRAM budget dimension (Phase 2)

**Phase 1 (v0.15.22) added GPU-stream decoders to the parallel decode pool but only spawned them eagerly via `GZSTD_POOL_GPU`. This makes them engage AUTOMATICALLY under `--adapt` — but only on the box that actually benefits (CPU-poor / GPU-rich, decode-bound), never speculatively — and fixes the frame budget that was starving the GPU on exactly that box.**

**Adaptive (lazy) engagement.** Under `--adapt` the decode pool's controller already grows the CPU decoder pool while the writers are decode-starved. It now has one more move: when the CPU pool has grown to its cap AND the writers are STILL decode-starved for a couple of windows, the CPU alone cannot feed the sink, so the controller brings the GPU streams in — spawning them lazily (the ~cuInit cost and the VRAM are paid only at that point, never up front). This is the decode-side analogue of the compress hybrid scheduler's CPU/GPU work-sharing.

**Remaining-time guard (why this doesn't regress the common case).** Lazy engagement fires only if the estimated remaining extract time (from the live progress meter) exceeds cuInit by a margin (~4 s). A fast/short run whose CPU pool briefly reads as starved — e.g. extracting to tmpfs or a very fast NVMe on a many-core box, where even a full CPU pool cannot outrun a RAM-speed sink — finishes before the GPU could come online, so the guard skips it and never does the speculative multi-GB VRAM grab (which would be antisocial on a shared box). The slow CPU-poor box this targets always has plenty of work left, so it engages there. No core-count or machine-specific threshold is used — the separation falls out of "will the GPU be ready in time to help." `GZSTD_POOL_GPU` remains an eager force (spawn up front, for tests / when the user knows they want it).

**VRAM dimension of the frame budget.** The in-flight frame budget was `2·D` in CPU decoders (capped by the seek_feed 4 GiB host ceiling) — sized for the CPU pool, so on a CPU-poor box (small `D`) it was tiny (e.g. 4–16 frames) and starved the GPU streams, which want batches of tens of frames. When GPU decoders engage, the budget now grows to also cover `ndev·gpu_batch` frames in flight, still capped by the same 4 GiB host ceiling so RAM stays bounded. The `Budget` semaphore gained a grow-only `grow()` (add permits + wake blocked acquirers) — monotonic, so it cannot break the pool's deadlock-freedom argument (more permits is strictly more concurrency).

Validation (256-core + 8×H100, all byte-identical to the serial walk): under plain `--adapt` with the CPU pool throttled small and a decode-bound extract, the controller auto-engages the GPU and the streams decode real frames (a 10 GiB archive: budget grown 4→128, dozens of frames GPU-decoded, 0 rescued); the fast-tmpfs full-CPU-pool run correctly SKIPS the GPU via the remaining-time guard (no VRAM grab); the eager `GZSTD_POOL_GPU` path still engages and grows the budget; and the Phase 1 matrix (symlink/hardlink metadata, oversize-frame CPU routing, `GZSTD_DEBUG_OFFLOAD_FLAP` stress, no-GPU fallback) all still round-trip.

**Still deferred:** the perf win itself needs a genuine CPU-poor / GPU-rich box to demonstrate (a CPU-rich box's pool clears the starvation before the GPU is worth engaging); Phase 3 (D2H-cost-aware routing — keep trivially-compressed / small frames on CPU per the existing <2% rule, decode-heavy frames on GPU) is unbuilt.

Suite: 346 normal (the Phase 1 GPU-gated test now also exercises the budget-grow path; lazy engagement is validated manually — it is timing-dependent by construction and a deterministic suite test would need a multi-second extract).

## v0.15.22 — -d --tar: GPU-stream decoders in the parallel decode pool (opt-in, Phase 1: correctness)

**The v0.15.20 decode pool is CPU-only: its one decode primitive is `decode_seek_frame` (pread + `ZSTD_decompressDCtx`). On a CPU-poor / GPU-rich box doing a clean full extract, that leaves the GPU streams idle while a handful of CPU decoders drain the pool. This adds GPU-stream decoders that batch-drain the SAME shared queue alongside the CPU decoders — the decode-side mirror of the compress hybrid scheduler. Niche by design (only helps CPU-poor + GPU-rich + decode-bound), so it stays opt-in and never touches the default or CPU-rich path.**

The pool already decoupled decode from parse (a shared frame queue feeding per-partition reorder buffers, both engine-agnostic), so a GPU worker slots in cleanly: it batch-pops `(partition, frame)` items, preads their compressed bytes, `nvcompBatchedZstdDecompressAsync` on its own CUDA stream, copies each result back, and scatters each into the same `psync[pi].ready` reorder buffer the CPU decoders use. One GPU worker per selected device (honoring `--gpu-devices`), each a lean self-contained `PoolGpuDecoder` (its own stream + VRAM-fit-clamped buffers; no auto-tuner / HybridSched / ResultStore / pinned budget — those serve the streaming `decompress_nvcomp` path).

**Correctness never depends on the GPU:**
- A frame larger than `GPU_SUBCHUNK_MAX` (16 MiB, nvCOMP's per-chunk cap) and any per-frame nvCOMP status/size failure route to the CPU rescue path — the same `decode_seek_frame` the pool already trusts, which succeeds on a transient glitch and dies with a clean data error on genuine corruption, exactly matching CPU-only and stock zstd. A whole-stream CUDA fault rescues the in-hand batch on CPU and retires that GPU worker (a wedged GPU stays wedged — the Turing fault history).
- The CPU decoders are a guaranteed backstop: the adaptive controller grows the CPU pool in the same branch that turns offload on, and `GZSTD_FORCE_POOL` maxes it up front — so even if every GPU init fails (or none is present) the queue still drains. GPU workers are purely additive.
- Read-byte accounting stays exactly-once: the GPU decoder never touches the Meter; the worker counts a GPU-decoded frame's input once, and a CPU-rescued frame is counted by `decode_seek_frame` — a frame preaded on the GPU then rescued on CPU is still counted once.

**Gating (opt-in):** GPU decoders engage only when `GZSTD_POOL_GPU` is set AND the run is not `--cpu-only` AND a GPU is present AND the pool is active (`--adapt` or `GZSTD_FORCE_POOL`). With the variable unset the path is inert — the default and every existing decompress mode are byte-for-byte unchanged. When built `USE_NVCOMP=OFF` the toggle is a harmless no-op.

Correctness (all byte-identical to the serial walk): a 4.4 GiB archive with the CPU pool throttled so the GPU streams decode 100+ frames (repeated 3× to shake out the async-H2D staging race that an early single-buffer version had); symlink + hardlink metadata preserved (same inode); mixed archives where some frames exceed 16 MiB (routed to CPU) and where none are GPU-eligible (GPU spawn skipped entirely); the `GZSTD_DEBUG_OFFLOAD_FLAP` stress with GPU workers present (offload toggling every 2 ms while 8 streams decode); real `--adapt`; and the no-GPU fallback (`CUDA_VISIBLE_DEVICES=`). New `-v` lines: `[TAR] decode pool: GPU decode on N device(s)` and a per-run `N GPU stream(s), G frame(s) GPU-decoded, R rescued to CPU`.

**Scope — this is Phase 1 (correctness + opt-in).** On a CPU-rich box the large CPU pool usually claims the frames before the GPUs finish `cuInit`, so the frame split there is ~0 (harmless); the real win needs a CPU-poor + GPU-rich box, where perf validation is still owed. Phase 2 (adaptive GPU engagement off the writer-starvation signal, plus a VRAM dimension on the frame budget) and Phase 3 (D2H-cost-aware CPU/GPU routing — keep trivially-compressed/small frames on CPU) are deferred.

Suite: 346 normal (1 new, GPU-gated: `GZSTD_POOL_GPU` adds GPU decoders to the parallel pool and matches tar byte-for-byte; skipped when no GPU).

## v0.15.20 — -d --tar: adaptive hybrid decode pool (decode offloads to a pool only when writers starve)

**Parallel full extraction (`run_parallel`) decoded each partition's frames inline on its one parse thread, so a huge file trapped in a single byte-balanced partition decoded on ONE core — starving the 16-writer pool whenever decode (not the disk) is the bottleneck. This adds a shared decoder pool so any partition's frames can decode across all cores, engaged adaptively so it never touches the write-bound fast path.**

Measured behavior (both on genuine `run_parallel` archives — confirmed by the `[TAR] parallel-extract` line; see the methodology note):
- **Decode-bound** (cmptest: 17 large files → tmpfs, writes ~free): inline left the writer pool **71–90% starved** (one core per file cannot feed 16 writers); the pool un-starved it to ~8% and cut the extract phase **~1.8×** (4.88 → 2.70 s). A handful of decoders saturates the sink — D=8 and D=96 were within noise.
- **Write-bound** (usr: 814 K tiny files → disk): the pool is **free** — forced-always-on 26.5 s vs inline 25.5 s (within noise; both ~75% writer-busy / ~0.3% starved, ABCCBA-averaged). Decoupling decode adds no measurable cost when decode is not the limiter.

So there is **no measured downside** and the always-on pool would be safe; extraction nonetheless engages it **adaptively** — zero-cost *by construction* on the write-bound path, not merely empirically. Under `--adapt` (no `--adapt` = pure inline, unchanged): each parse thread decodes inline by default; a controller samples the live writer starved-fraction (`Meter::extract_starved_ns`/`extract_busy_ns` deltas, 120 ms windows — the v0.15.11 signal) and flips a shared `offload_active` atomic. Starved >20% turns offload on and grows the pool one step (`n_dec_max/8`) with a settle-cooldown between grows (so growth does not lap the ~50 ms writer-flush lag and overshoot); starved <5% for a few windows turns it back off. The producer is hybrid: frames already handed to the pool always drain from it, and only *undispatched* frames pick inline-vs-pool live, with the dispatch cursor kept monotonic (`next_dispatch = k+1` after an inline decode) so no frame decodes twice across a transition. On cmptest it engages (peak ~36 decoders); on usr it engages only on brief write-bound transients (peak 12–24), harmlessly (the pool is free there).

Deadlock-free by construction (a partition only does a blocking budget-acquire while holding zero permits); the frame budget scales with the live decoder count (`2·D`, capped by the seek_feed 4 GiB RAM ceiling) so an idle pool buffers nothing. Diagnostic env toggles kept in the binary: `GZSTD_FORCE_POOL` (offload forced on, `--adapt`-free, = the always-pool path for A/B), `GZSTD_POOL_DECODERS=N` (decoder cap), `GZSTD_DEBUG_OFFLOAD_FLAP` (toggle offload every 2 ms to stress the inline↔pool transitions).

Correctness: the pooled / rapidly-flapped / forced paths are byte-identical to the serial walk across repeats (including a 3 GiB two-huge-file archive that exercises budget backpressure), and the full rpfrancis extraction (9,387,810 entries / 9,173,439 regular files) is byte-for-byte identical to GNU tar.

**Measurement methodology (learned the hard way):** the `[WRITER]`/`[ADAPT]` diagnostics print on **both** the serial `run_sink` and the parallel `run_parallel` paths (both share the writer pool), so they do not identify which ran. `run_parallel` engages only when `build_full_parallel_plan` succeeds — an archive with a duplicate-normalized-name or a leaf/dir path collision (common in huge real trees) correctly falls back to serial. Confirm with the `[TAR] parallel-extract` line before attributing any extract measurement to the decode pool. An initial round of rpfrancis measurements did not: rpfrancis-native falls back to `run_sink`, so those numbers reflected the serial path plus shared-box contention, not the pool — which briefly produced a phantom "+7.6% write-bound cost" that the usr test (real `run_parallel`) disproved.

Suite: 345 normal (1 new: `GZSTD_FORCE_POOL` routes the parallel path through the decode pool + reorder buffers and matches tar byte-for-byte).

## v0.15.14 — fix: non-root extraction of read-only files >4 MiB silently dropped them (data loss)

**A restore-time data-loss bug, surfaced while regime-measuring a real `/usr` archive (403 files silently lost).** Extracting a read-only file (a mode with no owner-write bit, e.g. 0444 shared libraries) larger than 4 MiB, as a non-root user, on the O_DIRECT extract path (auto-enabled on Gen4+, or forced with `--direct`), failed with EACCES and skipped the file.

Root cause: the big-file windowed-part path creates the file with its archived read-only mode, then the **O_DIRECT sub-4K tail reopens it `O_WRONLY` with no `O_CREAT`** — which rechecks write permission and is denied for the non-root owner of a 0444 file. Root never hit it (it bypasses the permission check); small files never hit it (they write through the create fd, no reopen); non-direct never hit it (writes go through the shared fd). Deterministic, not a race — the same files failed identically with and without `--adapt`.

Fix (the pattern `make_dir` already uses for restrictive directory modes): `dispatch_large` creates the file with a scratch **owner-write** mode (`e.mode | 0200`), and `finalize_big` reapplies the TRUE stored mode via `set_meta_fd` once every part is written — and it runs on all completion paths (success, part failure, truncation), so the scratch bit is never the final mode. Verified against the real `/usr` repro: 403 write failures → 0, formerly-failing 0444 libraries now extract byte-identical with mode 444 restored.

Suite: 344 normal (1 new: `--direct` extract of a read-only >4 MiB file round-trips with mode 444 restored; skipped when run as root or on a non-O_DIRECT fs).

## v0.15.13 — test-suite runtime: −270s off the default tier (measured, not guessed)

**A per-section wall-time profile drove a set of targeted cuts to the default suite, none of which lose default-tier correctness coverage.** The measured baseline was ~950s; two sections alone were 27% of it.

- **`--adapt persistent profile` (127s → ~15s):** the cost was structural, not the fixture — five level-19 single-thread runs (9.8s each) to clear the 3s profile-save gate, plus two ~25s `--calibrate` corpus passes. Two **debug-only** env hooks fix it: `GZSTD_DEBUG_ADAPT_SAVE_MIN_MS` lowers the save-gate so a sub-second default-level run qualifies, and `GZSTD_DEBUG_CALIBRATE_BYTES` shrinks the calibrate corpus (1 GiB → 8 MiB). Both are read-once, never consulted in a real run (verified: the default 3s gate still blocks a fast run). The one test that asserts the *real* gate ("sub-3s run writes nothing") deliberately sets neither hook.
- **Demoted to `-e` (−16 tests from the default tier):** Bounded-queue pooled-reader regressions (~47s — a stable guard for a fixed past bug in well-worn code) and the `--cpu-share` / hybrid-GPU-bringup sections (~108s, perf-shaped and GPU-gated). They still run under `-e`; normal 359→343, extensive stays 476.
- **Parallelized the upfront fixture generation** — the independent fixtures build concurrently (medium.txt's ~400 subshell text lines dominated); the directory tree waits on its three inputs.

**Honest non-result:** GPU acceleration (129s) and native tar extraction (80s) were investigated for fixture shrinking and left alone — both are operation-count-bound, not data-bound (medium.txt is 1 MiB; the tar fixtures are 5 MB and *deliberately* >4 MiB to exercise the large-file extract path). Shrinking would have saved ~nothing and risked silently dropping that path's coverage.

Also fixes a pre-existing count drift (`EXPECTED_TESTS` was 357 while 358 ran).

## v0.15.12 — --adapt × --tar: the extract writer-pool ACTUATOR (ROADMAP 2.5 #1, action 5c)

**v0.15.11 made SINK_BOUND classify on `-d --tar`; this makes it ACT.** On a write-bound extract the governor now grows the parallel file-writer pool, measures the sink rate, keeps the gain or reverts, and persists the settled pool size so the next run starts there. It is the extract twin of the plain writer probe (5b), but the actuator is different: the plain probe adds a second drain thread to the fixed-function `DirectWriter`; here the pool is a shared job queue, so the lever is *how many writers pull from it*.

**Scoping honesty (measured against the architecture, not assumed):** the "four levers" the sweep imagined — writer threads, decompress threads, read threads, sink budget — are not uniform. Only the writer pool is a clean in-run lever. Decompress concurrency in `run_parallel` is the partition count `N`, **fixed at start**; the `run_sink` parse is **serial by design**; the extract read is done *by* those same `N` partition workers (no separate read pool); the FrameSink budget is documented tar-exempt. So this change governs the one lever that in-run probing fits, and the others remain start-size/cross-run follow-ups.

**Mechanism (event-driven, C++17, no polling, no governor→worker pointer):** the governor moves `g_adapt_ewgrow_target` and notifies a global CV; the Extractor arms a **supervisor thread** (only under `--adapt`) that reconciles the live extra-writer count to that target — spawning fresh extras to grow, setting per-extra retire flags + waking the queue to shrink. Extras share `run_writer_loop` with the base pool (identical `-v` and live-governor accounting); a retired extra stops taking work at once. All spawned extras (retired or not) join at `stop_pool`. The probe steps by **half the base pool** (a 16-pool +1 would never clear the 10% keep bar), caps at 2 rounds and at a doubled pool, and — because a +50% step is a smaller relative signal than the plain probe's 1→2 — takes its keep/revert baseline from an **EMA of the write rate**, not a single noisy tick (an early build kept a spurious round on a 0.47 GiB/s lull; the EMA fixed it). Persists `decompress.tar_write_threads` (settled size, seeds next run's `start_pool`) + `tar_wt_converged` (latches the probe off once it finds no further gain, like 5b's negative verdict). `--write-threads` (user pin) always wins.

Verified live on the 256-core box: a write-bound extract (25 GiB compressible) probed 16→24 **KEPT (+26%, 3.74→4.72 GiB/s)** then 24→32 **reverted** (no gain) and settled at 24 — a genuine measured optimum above the static 16, with the 25 GiB tree round-trip byte-identical *through* the live 16→24→32 resize; a second run read the persisted size and started there with the probe latched off. (Run-to-run device variance is real — another run measured no gain and settled at 16; latest-wins persistence absorbs it.) Also fixed: the `-v [WRITER]` per-thread average now divides by the peak concurrent pool (base + peak extras), not the base alone — extras' busy time was reading as 138%.

Suite: 359 normal (section rewritten: extract sink-bound engages the writer-pool grow + round-trips exact; inert without `--adapt`).

## v0.15.11 — --adapt × --tar: the extract sink becomes visible to the governor (ROADMAP 2.5 #1)

**The `-d --tar` writer pool now feeds the governor its busy/starved time, so `SINK_BOUND` classifies on extract — previously structurally impossible.** The parallel file-writer pool kept its own busy counters (`wpool_busy_ns_`), never `Meter::writer_disk_ns`, so the classifier's sink term read ~0 on the one path we've long measured as device-write-bound; the governor only ever saw the source/compute split there. Two problems had to be fixed, not one — the ROADMAP framing ("wire the existing counters into the Meter") under-described it:

1. **Gating:** the pool timing was accumulated only under `-v` (`measure = verbosity >= V_VERBOSE`), so `--adapt` without `-v` saw nothing. Now `measure = verbose || adapt`.
2. **Shape:** the pool flushed its counters *once at thread exit* (a deliberate anti-contention choice — 16 writers must not fetch_add per job). A governor that ticks every 100 ms on deltas would read zero all run, then everything at once. Fixed with a coarse ~50 ms live flush of thread-local deltas into a *separate* set of Meter fields (`extract_busy_ns`/`extract_starved_ns`/`writer_pool_threads`), bounding atomic traffic to ~20/s/thread — the no-contention intent preserved. The fields are deliberately kept distinct from `writer_disk_ns` (whose `-v` `[WRITER]` line and the auto-tuner's sink-freeze read it with their own semantics; reuse would have silently collided). Feed is via a dedicated `Extractor::sink_meter_` pointer, separate from the deliberately-null `m_` (which stays null to avoid double-counting bytes) — it carries only sink timing, never bytes. The classifier takes `max(disk-term, extract-term / writer_pool_threads)`, the divisor because pool busy sums across N writers.

**A gate the change made necessary:** unlocking `SINK_BOUND` on extract also unlocked the writer-parallelism probe (M4 action 5b) on a path with **no actuator** — the probe's only consumer is the plain `DirectWriter`'s second drain thread; the extract pool doesn't read `g_adapt_writer_probe`. Left ungated it "measured" a phantom action (wrote_bytes variance over its 4-tick window), reported a false `+64%` KEPT, set `ADAPT_ACT_WRITER_PROBE`, and would have **persisted a bogus `writer_par` verdict** to the profile. Gated off extract (`is_extract_`), both the real-tick controller and the forced-hook raise. Action 4 (throttle-grow) was already naturally inert on extract (its `last_sink_bursty_` reads `writer_disk_ns`/`writer_starved_ns`, both ~0 there). So extract `SINK_BOUND` is now **classify-and-report only**; the extract-side actuator (governing `--write-threads`) remains future work.

Verified live on the 256-core box: a write-bound extract (43 GiB compressible, 95:1) classifies `warmup -> sink-bound` (75% share, writer 92.6% busy) with `actions: none`; a genuinely upstream-bound cold extract (incompressible, writer 15.5% busy) correctly stays `compute-bound` — the signal isn't blindly flipping to sink.

Suite: 358 normal (1 new: forced sink-bound on `-d --tar` classifies, round-trips exact, and shows no writer probe — the `is_extract_` gate).

## v0.15.10 — M5 close-out: docs, ROADMAP reconciliation, extensive suite, A/B

**Docs:** the `--adapt` help no longer claims observe-only — it describes the shipped actions (latches, ranked dispatch, probes, deadlines) and the always-wins rule for explicit flags; CLAUDE.md gains `AdaptGovernor` in the key-classes table plus the profile path and the three deterministic test hooks. **ROADMAP reconciled:** 1.3 (rate-matched dispatch), 2.1–2.3 (profile/calibrate/auto-update), 3.1–3.2 (pipe-aware/unknown-size scheduling) marked SUBSUMED by their v0.15.x replacements; 1.11's premise narrowed by the residency default; 4.2 re-opened per-machine by the writer probe.

**Extensive suite: 474/474** (`-e`, full zstd-CLI-compat sections included).

**A/B on the 256-core Gen4+ box (mixed profile, 19.5 GiB, 3 runs each; the second box's matrix is still owed):** cold compress is the headline — the first `--adapt` run seeds the profile at parity, runs 2–3 use the accumulated prior and hit **1.94–1.96 vs 1.47 GiB/s no-adapt (+33%)**: the calibrate-seeded backend prior flipping compress to cpu-only, exactly the M3 design intent. Warm compress: parity within noise (2.02 vs 2.09 median). Decompress currently pays adapt's exploration cost on this box (warm 6.9 vs 9.9, cold 4.2 vs 5.0 GiB/s): the read-path probe tries `--direct-read`, which this box measures slower, and the run sequence sat at the edge of the shared box's documented ±30% noise band — the probe-once-then-remember design converges after the verdict lands, but these numbers need a quiet-box rerun before firm claims. Honest accounting either way: on a box whose static defaults were already measured-optimal, adapt's value is the floor it puts under *other* hardware, at the price of bounded exploration here.

**The --adapt chapter (v0.15.0–v0.15.10) is complete** and remains opt-in; the default-flip decision stays at v1.0. Carried follow-ups: quiet-box + second-box A/B matrices; profile decompress `cpu_gibs` tap is compressed-unit (skews the 1.5× backend prior); CPU worker-pool grow on the sink signal; tar-create reader scale-up; mmap queue-starvation classifier fallback.

## v0.15.9 — --adapt M4 action 6: GPU deadline demote + escalate (M4 complete)

**The last governor action: a device whose oldest in-flight batch blows its deadline is demoted (no new intake for the run), and past a second larger deadline the wedge is treated as a fault and takes the existing proven abort → CPU-only rebuild path** — NOT duplicate-dispatch, per the design red-team that killed the re-enqueue idea twice (compress frees inputs post-H2D; `ResultStore` has no dedupe, so a late duplicate completion is a FrameThrottle permit leak — the v0.13.34 deadlock class). Mechanics: the workers publish each device's oldest in-flight submit time (front of the drain FIFO, maintained under the FIFO's own mutex) and an EMA batch duration; the governor tick compares age against max(4× EMA, 2 s) → demote, and max(2× that, 6 s) → escalate (compress only — decompress keeps its per-frame CPU rescue, which already covers erroring frames). The demote is consumed at the same refusal spot as ranked dispatch; **escalation is thrown by the device's *worker* thread — the drain thread is the one wedged in `cudaEventSynchronize` — into the same catch as a CUDA error.** One subtlety found live: a demoted worker parks in `wait_for_gpu_yield` and only re-evaluates its park predicate, so escalation makes that predicate return true — the worker unparks straight into the throw. Accepted residual (per plan): a truly never-returning CUDA call can still block the drain join at teardown, exactly as today for any wedged call.

Fault-injection hook (plan-mandated): `GZSTD_DEBUG_ADAPT_STALL=-1:<secs>` stalls the first drain thread to pop a batch (wildcard device — a fixed id can miss on boxes where that device gets no work). Verified live on the 8-GPU box: 3 s stall → `[ADAPT] gpu6 deadline: batch in flight 2.1 s (limit 2.0 s) — demoted`, run completes exact; 8 s stall → escalation at 6.1 s → **rebuild banner printed before the stall even cleared** (the abort preempts the wedge) → exit 0 with exact output; inert without `--adapt`.

Dedicated adversarial review (plan-mandated): FIX-FIRST, 2 highs + 1 medium + 3 lows, all banked. **HIGH-1:** during a *real* permanent wedge the worker is parked in its function-local stream-acquisition wait (its one stream never frees), whose CV the governor couldn't reach — escalation only fired for stalls that eventually cleared. Fix: an escalation wake registry (workers RAII-register their `{mutex, cv}` pair; the governor locks each pair's mutex before notifying — no lost-wakeup window — and re-notifies every tick while the escalation is pending), plus the stream-wait predicate and post-wait throw. **HIGH-2:** the demote/escalate consumption sat in `should_gpu_take`, but the park predicate calls `should_gpu_take_at` directly — moved there (also fixing a 64-vs-32 bound mismatch). **MEDIUM:** the escalation suite test could pass on the governor's *announcement* alone (finite stall + single-batch input completes normally) — it now uses a 256 MiB corpus with a pinned small batch and asserts the rebuild banner, pinning that the abort executed. **LOW (fixed):** an ordinary drain-error abort now clears the deadline tap so the log can't misattribute the fault. Accepted per review: front-of-FIFO is approximate toward *newer* (demote can only be late, never early); the cold-start 2 s floor can demote a pathologically slow first batch (plan constants); a truly never-returning CUDA call still blocks the drain join at teardown, as today.

**M4 is complete.** All six governor actions shipped: source latch (v0.15.3), ranked-engine overflow dispatch (v0.15.4), reader scale-up (v0.15.5), sink budget grow (v0.15.6), read-path priors + writer-parallelism probe (v0.15.7/8), deadline demote/escalate (v0.15.9). M5 close-out (docs, ROADMAP reconciliation, A/B benchmarks on both boxes, `-e` suite) is next.

Suite: 357 normal / 474 extensive (3 new: demote fires on a stalled batch, escalation takes the abort-and-rebuild path, inert without --adapt).

## v0.15.8 — --adapt M4 action 5 (second half): the writer-parallelism probe

**The general "try it, measure it, keep or revert, persist the verdict" mechanism lands on the writer: a SINK_BOUND O_DIRECT run probes +1 parallel drain thread, keeps it on ≥ 10% sustained sink-rate gain, reverts otherwise (≤ 2 rounds per run), and the verdict persists so the next run starts at the answer.** This re-opens ROADMAP 4.2 as a per-machine question, exactly as the plan framed it — "tested negative for buffered" was *our* boxes; on ext4-class NVMe this will probe-negative once and the profile remembers, at the cost of one probe per fingerprint instead of a regression every run.

The enabling refactor: **`DirectWriter`'s drain is now fully positional.** Every queued op carries the logical offset it was enqueued at, writes go through `pwrite` at explicit offsets (order-independent), sparse holes punch by the op's own offset (no `lseek` chaining), and the sub-ALIGN tail goes through a lazily-opened plain fd — the old in-place `fcntl` O_DIRECT flag-flip would race a concurrent aligned `pwrite` on the other thread. With ops order-free, the probe is just a second drain thread pulling from the same queue: the governor raises a flag on SINK_BOUND (real runs: the tick controller with a 4-tick measurement window against the pre-probe rate; forced hook: raised at start for deterministic tests); the primary drain spawns the second thread on first sight; a revert parks it (it stops stealing ops but stays joinable, re-engaging if a later round re-probes). `enqueue` notifies all — with two drain waiters on one CV, a `notify_one` swallowed by the parked probe thread would strand the op. Verdict → `writer_par` in the profile (latest-wins, so hardware changes can flip it back); a recorded negative blocks all future probing on that fingerprint.

Verified live: 5/5 concurrent-drain round-trips exact on 256 MiB random, sparse (zero-heavy) output exact through the positional punch path, engagement token in the summary (`writer-drain2(probed)` / `writer-probe(kept)`), never spawns without `--adapt`.

Suite: 354 normal / 471 extensive (3 new: dual drain engages with exact round-trip, negative writer_par verdict blocks the probe, inert without --adapt).

## v0.15.7 — --adapt M4 action 5 (first half): read-path priors — try the alternative next run

**A machine whose runs classify SOURCE_BOUND on read path P now gets the alternative path tried on the next `--adapt` run, and thereafter the better *measured* payload rate wins (5% margin against flapping).** This is the plan's "settled designs stay respected as starting points — the profile can only move a box off them by measuring better on that box": mmap-on-6.4+ and the pooled pread reader remain the defaults everywhere; only a source-bound verdict plus a measured win on *this* fingerprint switches anything. Mechanics: the reader entry points tap which path actually engaged (`mmap` | `pread` | `direct` — the engaged path, not the requested one, so fallbacks record honestly); the profile records `src_path` and a per-path end-to-end payload rate (`path_<p>_gibs`, EMA-merged); at startup a source-bound history with an untried alternative flips to it once (`[ADAPT] … probing --direct-read this run`), and with both measured picks the winner. Pairs: compress mmap ↔ pooled pread; decompress/test pooled pread ↔ `--direct-read` (regular-file inputs only). User pins (`--mmap`/`--no-mmap`, `--direct-read`) are never overridden; `--tar` and `-l` keep their own paths. Verified live: a crafted pread-only source-bound profile flips a decompress run onto the real O_DIRECT reader (`[DIRECT-READ]` engaged, output exact); a worse-measured alternative stays put.

Review angle: FIX-FIRST, all findings banked. **CRITICAL:** the decompress `--direct-read` probe run recorded itself as *pread* (the O_DIRECT branch lives inside the streaming reader whose entry tap says pread) — the probe would have re-fired forever, poisoning `path_pread_gibs`, with the comparison branch dead for organic profiles; fixed by re-tagging the tap at the `use_direct` engagement point, then verified end-to-end with a real ≥3 s run: probe → `path_direct_gibs` recorded → next run does not re-probe and picks the measured winner. **MEDIUM:** the comparison was regime-gated, producing a 2-cycle when the winning path changes the regime (win → compute-bound → revert → source-bound → re-flip); the comparison is now regime-free — the rates are the settled verdict, the regime only ever the reason to explore. **MED-LOW:** the probe now skips runs the profile's own overall rate predicts will finish under the 3 s save gate (a probe that can never record would pay the alternative path indefinitely). **LOW:** tar runs no longer record into the `path_<p>_gibs` keys (their write-bound payload rates aren't comparable with plain-decompress reads).

The second half of action 5 — the mid-run writer-parallelism probe (try +1 pwrite writer when SINK_BOUND on O_DIRECT output, keep on ≥10% sustained gain, revert otherwise, verdict persisted) — is the next slice: it carries real concurrency surface and gets its own review angle.

Suite: 351 normal / 468 extensive (3 new: source-bound prior probes the alternative path, worse-measured alternative never flips, explicit --mmap pin wins).

## v0.15.6 — --adapt M4 action 4: sink budget grow (bursty sink-bound)

**Under `--adapt`, a SINK_BOUND regime showing alternating starved/busy bursts grows the `FrameThrottle` budget** — the v0.14.74 smoothing insight generalized: when the tick window shows BOTH substantial device-busy AND substantial writer-starved time (each ≥ 20% of the window), a deeper in-flight budget can smooth the alternation; a purely-saturated sink cannot be helped by buffering more, so it never grows. Mechanics: the governor raises a one-per-tick request; the operation's `FrameThrottle` consumes it on its next `release()` and grows one bounded step (+25% of current max, ≥ 1 frame) — consumption-side wiring, so no governor→throttle pointer exists across their unrelated lifetimes. Growth is `{lock; max_ += n; permits_ += n;}` + n wakes, which never disturbs the FIFO deadlock-freedom argument (permits only ever get more plentiful). Ceiling per operation: min(available RAM / 2, 32 GiB hard cap, `--memlimit`) / frame size, armed at each of the four throttle construction sites. Guards: user-pinned `--throttle-frames` never grows, `--memlimit` stays authoritative, `--tar` extract keeps its deliberate 16 GiB cap (its FrameSink has its own v0.14.74 grow), disabled throttles stay disabled. One `[ADAPT]` line per step at `-v`; `sink-grow(throttle)` in the summary. Observed live: 384 → 480 frames on the first bursty tick, output exact.

Honest limit (review finding, ship-with-follow-up): the CPU workers' per-worker output-buffer pools are sized once at spawn from the *initial* budget and deliberately never grow (the bounded-pool invariant is load-bearing history), so on cpu-only paths permit growth moves the block point from `throttle.acquire` to the pool's `wait_for_drain` rather than deepening realized in-flight — the GPU paths, whose buffers are per-batch, get the full benefit. Growing the pools on the same signal is the follow-up.

Suite: 348 normal / 465 extensive (3 new: bursty sink-bound grows the budget with exact output, pinned --throttle-frames never grows, inert without --adapt).

## v0.15.5 — --adapt M4 action 3: reader scale-up (source-bound, io-dominant)

**The decompress parallel-prefetch reader can now grow under `--adapt`: dormant reader threads spawn up front (2× the initial count, ≤ 12) and wake when the governor classifies SOURCE_BOUND with the *io* state dominating the reader's 4-state split** — a copy-bound reader gains nothing from more prefetch threads (the per-frame copy sits on the single consumer), so io-dominance is the gate, computed as the io share within the reader's own time each tick. Mechanics per the plan: the slot ring's modulo geometry is frozen at spawn, so the ring is sized for the *cap* up front, while slot buffers allocate lazily on first claim AND the prefetch look-ahead is bounded by 2× the *active* reader count — lazy alloc alone wouldn't cap memory, since a lagging consumer lets the initial readers run ahead and touch every slot (review finding); together they keep the un-scaled run's resident footprint exactly what it was before the ring grew (a cap-sized ring at 64 MiB/slot would otherwise commit ~1.5 GiB). Work claiming is already dynamic (`next_block.fetch_add`), so woken readers integrate for free. Dormant threads park on a CV — no polling — and the wake flag is written under the same mutex every waiter re-checks under, so the wakeup cannot be lost (the v0.15.4 MAJOR-1 lesson applied at design time). User-pinned `--read-threads` never scales; `--direct-read` is exempt (O_DIRECT is single-stream by settled design). One `[ADAPT]` line on wake; `reader-scaleup` in the summary's action list. The forced-regime test hook asserts the io-dominance gate along with the verdict, making the suite tests deterministic.

Deferred within action 3, explicitly: the tar-create member-reader pool gets the same treatment in a later slice (same mechanism, separate pool), and the mmap queue-starvation classifier fallback (mmap leaves all four reader counters at zero, so SOURCE_BOUND is invisible there) lands when a consumer for it exists — it needs queue-depth taps the governor doesn't have yet.

Review angle: COMMIT-READY, one minor acted on (the active-bounded look-ahead above — the original lazy-alloc-only claim was inaccurate) plus exit-safety hardening: the scale CV/mutex are deliberately leaked heap objects, since `die()`/`std::exit` can fire while dormant readers are parked and destroying a CV with waiters is UB. Verified clean: both wake conditions written under the waiters' mutex, all joins dominate every exit path, forced-hook determinism, io-dominance consistent with the SOURCE_BOUND thresholds.

Suite: 345 normal / 462 extensive (2 new: forced source-bound wakes the dormant readers 3→6 with exact output, inert without --adapt).

## v0.15.4 — --adapt M4 action 2: ranked-engine overflow dispatch (the heart of the chapter)

**Every engine is now ranked individually under `--adapt` — the CPU pool as one engine, each GPU device as its own — and dispatch generalizes the proven tail-yield inequality to every engine, all run long:** engine E takes work only when the queue depth exceeds what all strictly-faster engines will drain within E's own batch window (`(depth − batch) / faster_sum ≥ 1.3 · batch · streams_E / rate_E`). The fastest engine has an empty faster-set and always feeds; slower engines are overflow; an engine slower than everyone combined naturally starves to zero intake. With one GPU engine and a faster CPU this is byte-for-byte the v0.13.58 tail inequality — on the measured boxes it degenerates to today's cpu-primary hybrid; on a fast-GPU/weak-CPU box it inverts automatically, same formula, no box assumptions.

Mechanics, all `--adapt`-gated (non-adapt dispatch is byte-for-byte unchanged):
- **Per-device rate table** in `HybridSched` (fixed 32-slot, indexed by raw CUDA device id so `--gpu-devices` subsetting can't alias), fed by per-device byte windows on the existing tick. **Payload-unit CPU EMA added for the comparisons:** the aggregate CPU counter reports *compressed* bytes on decompress while the GPU side reports decompressed — units that never mattered while decompress never declined, but the ranked inequality compares engines head-to-head, so it gets its own consistently-payload EMA (frames are uniform in payload size, making byte rates a faithful frames/sec proxy).
- **Ranking on the tick, inequality in the hot path:** the tick computes each device's `faster_sum`; the take-site just applies the inequality. Faster-set *membership* swaps only after 3 agreeing ticks (rank hysteresis); the *sum* refreshes every tick — rate drift stays live inside a stable ranking (observed live: injected 50 GiB/s prior drifting to the real 7.7 while the decline held).
- **Intake stays a live decision** — declined workers park in the existing `wait_for_gpu_yield` spot (no permits held, no `gpus_waiting_` increment, drain/abort exits intact) and re-evaluate on queue events, so a transient *shallow queue* can't permanently kill a good device (a transient *rate dip* right before parking can, since a parked device's EMA freezes — the between-runs profile promotion is the recovery path for that, by design). **The one-way latch is the floor quiesce:** a device declining for 5 consecutive ticks releases its share of the queue-floor reservation (`[ADAPT] gpuN ranked slow … floor reservation released`), computed by summing streams over non-quiesced devices at the floor's single choke point — no subtraction bookkeeping, so a quiesce and a stream exit can interleave freely. Promotion is between runs via the profile, per the plan's no-flapping rule.
- **Decompress declining is NEW behavior** (`should_gpu_take` hard-returned true for decompress), reached only under `--adapt`; the trivial-frame CPU routing stays safe because the CPU pool never latches off (its lockout remains the existing floor-factor mechanism). When at the current depth *every* live device would decline (evaluated exactly, per device), the aggregate `tail_yield_` latches and zeroes the floor — same rule as the aggregate tail yield.
- **`RateMatchState` deleted** (~90 lines + plumbing through 6 signatures): its CPU frame allowance was computed but read by nothing since the semaphore scheduler landed.
- **Test hook** `GZSTD_DEBUG_ADAPT_RATES=cpu=50,gpu0=0.01,…` injects engine rates into the ranker (unit-level dispatch-math testing without exotic hardware, per plan), publishing an initial ranking at construction so sub-tick runs exercise the decline path.

Verified live on the 8-GPU box: injected slow ranking → GPU batch intake drops from 37 batches (control, throttled CPU) to **0**, output integrity intact, all 8 devices quiesce with the `[ADAPT]` line and the summary reports `device-quiesce(ranked)`; fixed `--cpu-share` and non-adapt runs untouched.

Dedicated adversarial review angle (per plan): 2 majors + 5 minors, majors fixed pre-commit. **MAJOR-1:** the `tail_yield_` latch's CPU wakeup could be lost when the latch fired from the loop-top gate (no queue lock held — a CPU between its floor-predicate check and its CV wait misses the only notification that will ever come; pre-existing in narrower form on the aggregate path, but this change armed the window all run in both directions). Fix: the latch may only fire from a park-predicate caller, which holds `TaskQueue::m_` — an unlocked decline parks and latches at the predicate microseconds later; applied to both the ranked and aggregate latch sites. **MAJOR-2:** the ranked all-decline latch could fire before `producer_done_` (pre-done, a shallow queue measures the reader), permanently stripping the floor from devices still initializing — a 1-GPU pipe run would have latched at its first decline. Fix: the aggregate latch keeps its tail semantics (`producer_done_` gate); per-device parking stays armed all run. Minors fixed: the quiesce evaluation now mirrors the hot path's `--cpu-queue-min` drain-of-last-resort short-circuit (a last-resort-draining device no longer counts as declining), `dev_` null guard on the ctor injection block, test-hook injection extended to the full 32-slot table so >16-GPU boxes can't false-fail. Accepted trades flagged by review: quiesce decides on tick-instant depth snapshots; parked-device EMA freeze (above).

Suite: 343 normal / 460 extensive (5 new: injection zeroes compress intake, declined-run round-trip, injection zeroes decompress intake, inert without --adapt, inert under fixed --cpu-share).

## v0.15.3 — --adapt M4 opens: the governor acts — source-bound GPU-batch growth latch

**First acting consumer of the regime classifier (M4 action 1, risk-ascending order): when the governor says SOURCE_BOUND, the GPU batch auto-tuner latches.** The reader is the faucet — a bigger batch can never be filled faster than the source feeds it, so growth is pure pop-latency. The mechanism is deliberately "one more producer for an existing latch": the same `frozen` flag the v0.14.52 sink-freeze uses, now with a `freeze_reason` code (sink | source). Differences from the sink freeze, both principled: **no down-clamp** (there is no writer head-of-line wave to shrink — the latch lands in place at the tuner's current best), and **no desync jitter** (`gpu_desync_batch` is now gated on the *sink* reason specifically: its randomization exists to de-correlate completion waves at the in-order writer, which a starved queue doesn't produce).

Wiring keeps the governor decoupled from worker lifetime: `AdaptGovernor` publishes its regime through a global atomic (`g_adapt_regime`, reset at start, stored on every transition); the two GPU tuner sites (compress + decompress) read it at the top of their existing tune block — **before the `MIN_BATCHES` window gate, because a starved queue may never accumulate a full measurement window** — and latch under the tuner mutex (try-lock, retried next iteration). Actions record themselves in an action-flags bitmask; the end-of-run `[ADAPT]` summary now prints the real action list instead of "(observe-only stage)".

Test hook: `GZSTD_DEBUG_ADAPT_REGIME=source-bound|sink-bound|compute-bound` forces the classifier's verdict from t=0 (skipping ramp + hysteresis), same family as `GZSTD_DEBUG_GPU_CORRUPT` — the plan's sanctioned pattern for making sub-second suite runs exercise governor actions deterministically.

Suite: 338 normal / 455 extensive (5 new: forced regime prints the transition, cpu-only reports actions none, source-bound latches the tuner, summary lists the latch, latched-run output round-trips).

## v0.15.2 — --adapt priors + the residency-informed decompress default (unconditional)

**The Gen4+ decompress default is no longer a blanket hybrid — it reads the input's page-cache residency first (unconditional, no `--adapt` needed).** The blanket choice was measurably wrong for warm inputs: a ~fully-resident input feeds at memory speed, the run is compute-bound, and cpu-only is the fast engine on the fast-fabric boxes. Measured this build (256-core Gen4+ 8-GPU box, warm `-t`, median-of-3): high-compressibility 20 GiB corpus **16.3 vs 4.9 GiB/s** (cpu-only vs hybrid, 3.4×), mixed profile **7.4 vs 4.9** (+50%), and a realistic ~50%-ratio 130 GiB tar archive **4.8 vs 4.7** (parity) — warm cpu-only wins or ties, never loses, and frees the GPUs; cold stays hybrid (measured parity there too: 2.7 both, disk-bound). Now `apply_backend_defaults` probes the first input with its own `stat` + `open` + `mmap(PROT_READ)` + `residency_fraction` + `munmap` (microseconds; `mincore` never faults pages in; **never `MAP_POPULATE`** per the fault-storm rule): ≥ 95% resident → cpu-only with a default-verbosity notice mirroring the Gen<4 one; cold or unknown → hybrid exactly as before, with the residency shown in the `-v` `[ASYMMETRIC]` line. Guards: regular-file output only (a warm input piped to a slow consumer is sink-bound, not compute-bound — piped stdout and `-o` to an existing FIFO/device keep today's default), pipes/unreadable inputs skip the probe (a named FIFO is stat-rejected without ever being opened), `--tar` probes the archive from `tar_sources` (not the synthesized stdin input), `-l` stays silent. One measured subtlety: Gen4+ auto-`--direct` means a *freshly written* archive is NOT warm (O_DIRECT bypasses the cache on the way out) — the warm path engages for genuinely re-read data, which is exactly the case it exists for.

**Under `--adapt`, the profile's measured priors now drive the initial configuration** (each behind its own user-flag guard, logged as `[ADAPT]` lines):
- **Backend per direction:** with both engine rates on record, cpu-only when the CPU engine dominates outright (> 1.5× the GPU engine — hybrid coordination can't win back a gap that wide; the measured story of the fabric boxes), else hybrid, announced at default verbosity. Never gpu-ONLY by default (hybrid contains it, minus the single-engine failure mode). On this Gen4 8-GPU box the calibrate-seeded prior flips compress to cpu-only — the correct measured answer a blanket "hybrid for compress" has never given it.
- **GPU batch start point:** the tuner starts at the cap (`shared_tune.batch_size = gpu_batch_cap`), so seeding the cap with the profile's settled batch replaces the exploration ramp while leaving the tuner free to move.
- **Scheduler EMA seeds:** the `HybridSched` constructor pre-warms `cpu/gpu_rate_ema_` from the profile with sample counters at 1, NOT the refusal-arming 2 — one live tick per side is required before any refusal can latch `tail_yield_` off a stale prior (the design-review F5 rule).
- **Verify-engine auto:** an observed regime replaces the PCIe-gen guess — GPU verify wins exactly when the GPU is the idle resource, so any recorded regime other than compute-bound picks it (the lever the old comment always said `--adapt` would pull).
- **Driver-mismatch quarantine:** a changed driver version invalidates only the GPU rates and settled batch; CPU and device-rate priors survive the bump. `-vv` prints the fingerprint hash + driver so users (and the suite) can see the profile key.

Priors review angle: 6 findings, all fixed pre-commit — the big one caught empirically before the agent reported it: **`--tar` decompress/test probed stdin's residency instead of the archive's** (tar mode keeps its archive in `tar_sources` and synthesizes `inputs = "-"`), so the warm-archive case never engaged and a warm redirected stdin could steer a cold extraction to cpu-only. Also: the profile's `settled_batch` double→size_t cast was UB on foreign values (now clamped to `HARD_BATCH_CAP` pre-cast, mirroring the emitter's magnitude guard); fingerprint/driver tests now skip on non-nvCOMP builds; `-o` to an existing non-regular sink defeats the warm default as intended; the notice no longer recommends `--gpu-only` when PCIe gen is undetectable.

Suite: 333 normal / 450 extensive (7 new: warm-input default announced, piped-output skip, fingerprint at -vv, crafted-profile prior flips compress to cpu-only, explicit --hybrid beats the prior, driver-mismatch invalidation, --tar probes the archive not stdin).

## v0.15.1 — --adapt persistence: per-machine profile, hardware fingerprint, --calibrate

**The measured-calibration cache lands: a clean `--adapt` run of ≥ 3 s records what it learned to `${XDG_CACHE_HOME:-~/.cache}/gzstd/profile.json`, keyed by a HARDWARE-ONLY fingerprint** (CPU model + logical cores, GPU name list via two new dlopen-shim NVML calls, kernel release — FNV-1a hashed; the raw string stored for debugging). The GPU driver version deliberately lives INSIDE the entry, not the key: a driver bump must not orphan a machine's priors — the prior layer (next stage) invalidates only the GPU rates on mismatch. Per direction the entry carries: overall payload GiB/s, the hybrid scheduler's per-engine EMAs (tapped from `HybridSched::tick`), source/sink device rates (reader-io / writer-disk time, so an mmap run simply leaves source unmeasured), the GPU tuner's last-used batch, the governor's dominant regime, run/fault counts, timestamp. Numeric fields merge as EMA (old·0.5 + new·0.5) so hardware/driver drift converges in a few runs; a GPU-fault → CPU-rebuild run merges NOTHING measured (its clock was rebased mid-run and its GPU numbers are garbage) — it bumps the fault count only, order-verified in review. Writes are per-pid tmp + atomic same-dir rename, happen only on exit 0, and every profile I/O failure is a `-v` note, never fatal.

**The profile is parsed by a deliberately strict JSON subset** (objects/strings/numbers/bools, depth-capped, 1 MiB cap, no arrays — the schema never needs them): anything unmodeled rejects the WHOLE file, which is then discarded and rewritten — for a regenerable cache the failure mode of strictness is a one-run recalibration, never a wrong prior. Discard-and-rewrite is total (a partial parse can never be merged into), and emitted output must survive its own re-parse: keys escape exactly like values (a foreign key with a quote must not poison other machines' entries on the next load) and the integer-emit magnitude guard runs BEFORE the long-long cast (a hand-edited `1e300` is legal input; casting it is UB).

**`--calibrate` measures this machine's engine rates explicitly** and records both directions like qualifying runs: a generated corpus (half text-like, half xorshift-random; 1 GiB, RAM-clamped) rides a **memfd exposed as `/proc/self/fd/N`** — a RAM-backed REGULAR file, so the pipeline's real readers engage; fmemopen looked like a pipe and measured the single-reader stdin path (1.0 vs 2.85 GiB/s, same corpus, same box). Each engine gets an untimed warmup pass (cuInit, contexts, pools, thread spinup outside the clock) that doubles as the compressed-corpus capture, and timed passes run over 4 concatenated corpus copies so per-call setup stays under the noise floor — the first cut's sub-second passes disagreed with themselves by 5×; the final form reproduces this Gen4 8-GPU box's known ground truth (CPU wins BOTH directions) with ≤ 13% run-to-run drift. Optional sink row only with `-o NEWFILE`: buffered write + fsync folded into the measured time (page-cache-only timing overstated the device), the target must NOT pre-exist and is removed — a pre-existing target is a usage error (exit 2), because a user reading "-o FILE" as "write the report here" must never lose that file. A faulting GPU discards its calibrate rows rather than recording garbage.

`--no-profile` suppresses read + write (benchmark honesty; `gzstd-benchmark.sh` should adopt it when --adapt benches land). Persistence review angle ran to a verdict: 7 findings (the `-o` data-loss footgun HIGH; emit-cast UB; unescaped keys; page-cache sink timing; cpu-row-keyed save; relative XDG; a latent null-guard) — all fixed before commit; the clean-list confirmed parser robustness against malformed input, atomicity, fingerprint stability on ARM/no-GPU/no-NVML boxes, and multi-file accounting.

Suite: 326 normal / 443 extensive (10 new: profile write/merge/corrupt-tolerance/no-profile/sub-3s/failing-run/read-only-dir gates, --calibrate records both directions, --calibrate --no-profile records nothing, --calibrate -o existing-target refused with the file intact).

## v0.15.0 — the --adapt chapter opens: AdaptGovernor skeleton (regime classifier, observe-only)

**`--adapt` lands as the sensing half of the adaptive governor — classify the run's bottleneck, act on nothing yet.** The 0.15.x line builds measured-regime self-tuning per the reviewed design (ranked per-device engine dispatch, measured keep-or-revert I/O probes, persistent per-machine calibration); this first stage is the skeleton every later action hangs off. A per-operation `AdaptGovernor` runs a condition-variable-stoppable 100 ms tick thread (same cadence as the hybrid scheduler's tick loop; exists only under `--adapt`) that samples the Meter's reader/writer state counters as WINDOWED deltas — not cumulative fractions, so a regime change at t=30 s isn't diluted by the first 30 s — and classifies each window: **sink-bound** (write-path busy ≥ 0.55, the proven v0.14.52 sink-freeze threshold, checked first), **source-bound** (per-thread reader io+parse+copy ≥ 0.85 with blocked-downstream ≈ 0, the `[READER]` verdict thresholds), else **compute-bound**. Transitions need 5 consecutive agreeing windows (0.5 s hysteresis) after a 3 s ramp guard; `Meter::reset()` mid-run (GPU-fault CPU rebuild) makes a delta go negative and the governor re-baselines and skips the window. `[ADAPT]` lines print at `-v`: each transition live, and an end-of-run regime-share summary alongside `[READER]`/`[WRITER]` (a sub-ramp run reports "run shorter than the 3 s ramp" rather than pretending). Wired across plain compress/decompress/test AND the tar entry points; two documented sensing gaps carry to the action stages: the extractor's writer pool keeps its own busy counters (extract sink-bound invisible for now) and the mmap zero-copy reader leaves all four reader counters at zero (a source-starved mmap run classifies compute-bound until the queue-starvation fallback lands).

**`--adapt` deliberately takes over zstd's same-named flag** (zstd's varies the compression LEVEL with I/O conditions; it was a warn-level compat no-op here since the compat layer landed). Bare `--adapt` is now the real governor flag; zstd's value form `--adapt=min#,max#` also enables the governor but warns that the level bounds are ignored — gzstd adapts the pipeline, not the level. Documented in the long help; the compat suite asserts all three behaviors (bare = real + silent, value form = runs + warns, `-q` still suppresses the warning).

`--no-adapt` is parsed today so scripts keep working when the default flips (decision at v1.0). Explicitly gated observe-only: compressed output is byte-identical with and without `--adapt` (suite-asserted). A concurrency/lifecycle review angle ran clean (per-iteration governor+Meter share scope; stop covered on every loop exit including TEST's early `continue`; die()/std::exit safe with the thread alive since main's frame isn't unwound); its one hardening note — stop() is single-caller-thread only — is documented at the declaration. The 100 ms tick is a flagged deliberate exception to the no-fixed-waits rule: telemetry cadence, not a scheduling wait, and no correctness path ever waits on it.

Suite: 316 normal / 433 extensive (6 new adapt-section tests: --adapt/--no-adapt accepted, round-trip under the governor, byte-parity with/without the flag, [ADAPT] present at -v, silent without -v/--adapt; compat section reworked: bare --adapt real + silent, --adapt=min#,max# runs + warns, -q suppression re-anchored to the value form).

## v0.14.95 — v0.14.xx close-out review: two pooled-reader deadlocks, malformed-tar abort, exit-code fidelity, help/doc accuracy

**A three-angle sequential review (concurrency hangs → data correctness → help accuracy) to close out the v0.14.xx line before v0.15.0.** Each angle ran to a verdict and its fixes were banked before the next started. Findings, most severe first:

**1. GPU-fault abort wedged the buffered pooled reader (deadlock in the exact path that exists to save the data).** The abort protocol woke the queue, throttle, results, and idle CVs — but never `DirectReadPool`, whose `acquire()` had no escape predicate. Pool-as-backpressure is the designed steady state, so at fault time the producer is very likely parked in `acquire()`; with the workers gone, no `release()` was ever coming, and `TaskQueue::push`'s drop-on-done path leaked each dropped view task's pool slot on top (dropping a `Task` never ran `release_input()`). The reader thread — which is the driver thread — hung forever and the fault→discard→CPU-only-rebuild path was never reached. Never observed in the field because the GPU boxes that reproduce faults take the mmap reader (kernel ≥ 6.4), which has no pool; the pooled reader engages for redirected-stdin/blockdev input, `--no-mmap`, or pre-6.4 kernels. Fixes: `DirectReadPool::set_done()` (acquire returns −1, reader exits), wired into both abort sites (GPU worker catch and drain-thread catch); `push`'s drop path now releases the task's input; the compress emit callback stops the readers on abort instead of streaming the rest of the file into drops; `g_direct_read_pool` is atomic (worker threads read it during abort while the producer writes it).

**2. gpu-only compress could wedge with NO fault when the per-stream batch exceeds the pool size.** The pooled reader parks in `acquire()` at `pool_n` queued tasks (its designed backpressure); a stream in `wait_for_batch_or_cap(pop_n)` with `pop_n > pool_n` waits for a queue depth that can never exist — compress queues set no depth cap, view tasks count zero bytes against the byte cap, and in gpu-only there is no CPU pop to break the standoff. Reachable today: `--gpu-only --gpu-batch=512 < bigfile` where the RAM clamp yields a smaller pool. Fix: mirror the pool ceiling onto the queue (`set_max_depth(pool_n)`) so the existing capacity escape fires and the v0.14.58 partial-batch pop handles the rest. `push()` can never actually block on this cap — a queued view task holds a pool slot, so `acquire()` gates the producer strictly earlier (the bounded-queue `space_cv_` discipline is untouched).

**3. A malformed tar member header aborted the process instead of reporting a data error.** `parse()` read the `L`/`K` longname and pax `x`/`g` sizes from the untrusted header (base-256 admits ~2^63) and allocated unchecked — `bad_alloc` in the parse thread → `std::terminate` → SIGABRT (exit 134) on all of `-d`/`-t`/`-l --tar`. Crafted-archive-reproduced on all three paths. Fix: 64 MiB extension-header cap (≈3 orders of magnitude above anything a real archiver emits; the foreign-scan walk already had its own tighter cap and bails — this parser is authoritative, so it fails cleanly). All three paths now exit 4.

**4. Extract exit-code fidelity: corrupt archive now exits 4, matching the documented contract and the `-t`/`-l` paths.** The extractor collapsed structural damage and per-member restore failures into one `had_error_` flag, so `-d --tar` on a corrupt archive exited 1. Structural sites (bad checksum, truncated stream/long-name/pax/file-data/sparse-extension, implausible header sizes, bad PAX sparse map) now also set a `data_error_` flag; extraction maps it to exit 4, per-member restore failures stay exit 1.

**5. The rename-fallback finalize could install a truncated output, delete the source, and exit 0.** When the atomic rename fails (near-dead code — the tmp is created adjacent to the target), the `rdbuf()` copy's stream state was never checked: a mid-copy ENOSPC/EIO silently set badbit, the good tmp was removed, and the no-keep unlink deleted the input — the same silent-loss class the v0.14.94 `ferror`/`fclose` gate closed on the primary path. Now: check the copy, keep the complete tmp (cleanup disarmed, path named in the error), die exit 3 before the tmp removal and the source unlink.

**6. Help/parse audit (every flag in both help texts checked against `parse_args` and behavior).** Parse fixes: `--verify-retries` and `--verify-engine` accepted only the `=VAL` form while their help showed the space form — both now take both, like every other value flag; `--cpu-share` documented `[0..1]` but validated nothing (1.7 ran silently, skewing the fixed-share scheduler) — now a usage error, parity with `--hybrid-floor-factor`. Help fixes: `--verify-engine` had NO long-help entry (added, under `--verify`); the EXIT CODES table omitted 6/7 (documented only inside `--keep-going`); `-T 0` conflated "all cores uncapped" with the no-flag default's 96 cap; `--read-threads` claimed "ignored for stdin" but a seekable stdin redirect uses the multi-reader pool (both helps); short `-l` column list missed `Skips`; `--keep-going` entry now states the forced single reader; `--verify-engine E` short-help said "auto=by GPU speed" — it picks by PCIe generation; NVML/Xid expanded at first use per house style; two mid-sentence line breaks mended; `-M` moved from HYBRID SCHEDULER to CPU TUNING (it is an all-modes flag). Deliberately-undocumented zstd-compat flags and aliases were confirmed intentional and left undocumented.

**Also:** stale architecture-comment claim fixed (decompression has streamed through bounded queues since v0.13.x, not "read entirely into memory"); stale `read_threads` field comment (auto is clamp(threads/8, 3, 12), not 3); machine-specific hostnames scrubbed from CHANGELOG/ROADMAP/code comments in favor of spec descriptors; CLAUDE.md refreshed (line count, exit codes 6/7, extensive-suite note).

Suite: 310 normal / 426 extensive, all passing (crafted malformed-header archives verified exit 4 on `-d`/`-t`/`-l`; space-form verify flags and `--cpu-share` bounds verified live).

## v0.14.94 — disk-full: fix the O_DIRECT hang AND the buffered silent-success data loss

**Field report (0.14.89, a workstation running backups): gzstd hangs when the output partition fills.** Reproduced unprivileged with `RLIMIT_FSIZE` + `SIGXFSZ` ignored (writes fail mid-stream exactly like ENOSPC) — and the investigation found TWO bugs, the second one worse than the reported hang:

**1. The `--direct` hang (what the field hit — Gen4+ boxes auto-enable `--direct`).** On a write failure, `AsyncWritePool::worker_fn` set `error_` and quietly returned — but that worker holds the only hand that releases `FrameThrottle` permits. The compress workers starved on permits, the writer thread never collected another batch, `submit()` was never called again, and the `had_error()` checks there were unreachable: the process hung forever (same permit-starvation family as the v0.14.58 GPU deadlock). Fix: the worker now dies (`die_io`, exit 3) at the failure point — it is the sole writer of the output stream, so it holds no stdio lock another thread needs during exit; the error flag + notify stay for the `flush()`/destructor paths.

**2. Buffered output reported SUCCESS over a truncated archive — and then deleted the input.** With buffered (non-`--direct`) output, strace showed **8655 kernel-rejected writes and exit 0**: glibc's `fwrite` can report a full count while merely buffering the bytes after a failed flush — the failure is sticky in `ferror()`, which nothing checked; `fclose()`'s return (the final flush) was ignored too. The truncated tmp was then atomically renamed over the target, and — for plain no-`-k` compress — the INPUT was unlinked. A full disk could destroy both the backup target and the source. `--verify` cannot catch it (frames are verified from RAM before the write). Fixes: `robust_fwrite` now fails on sticky `ferror` (any earlier write failure means the stream is already corrupt); main's finalize checks `ferror` + `fclose` and dies (exit 3) BEFORE the rename and BEFORE the no-keep unlink; the stdout path checks its final `fflush`/`ferror` the same way. Verified across buffered/`--direct`/`--verify` tar creates and plain no-keep compress: all exit 3 with a clear error, the truncated tmp is cleaned up, the input survives.

**3. And the fix's own regression, caught by the new suite test at a tighter limit: concurrent `die()` calls segfaulted.** With the failure landing mid-submit, the write worker's new `die_io` raced the writer thread's (now-reachable) `had_error()` `die_io` — two threads in `std::exit()` at once race the runtime's exit handlers (deterministic SIGSEGV, interleaved error lines). `die()` now has a first-dier-wins latch; a concurrent dier parks in a sleep loop (the no-fixed-waits rule governs scheduling paths, not the death path — a doomed thread sleeping costs nothing). 5/5 clean exit-3 runs at the limit that previously segfaulted 3/3.

Suite: 310 normal / 426 extensive (2 new: buffered disk-full → exit 3 + input kept + no output; `--direct` disk-full → exit 3, no hang). If a 0.14.89–0.14.93 process is currently hung on a full partition: it is safe to kill; check for a leftover `*.tmp` output and free space before rerunning.

## v0.14.93 — warm/cold-adaptive `-l` fallback walk (3× faster on cached files, zstd-class)

**The `-l` fallback frame walk now picks its I/O strategy by sampling page-cache residency.** Measured on a 65 GiB single-frame archive (Gen4+ server): warm, zstd's `read()`-based walk beat our mmap walk 3× (1.1 s vs 3.0 s) because every touched header page cost a fault — 560k of them; cold, our `MADV_SEQUENTIAL` stream beat zstd's queue-depth-1 strided reads (38.7 s vs 42.4 s). Neither strategy dominates, so `list_zst` now runs both, choosing per file: `residency_fraction` samples up to 64 spread `mincore(2)` windows over the existing mapping (microseconds, non-perturbing — querying never faults pages in); ≥95% resident routes to a new **buffered pread walk** (`buffered_frame_walk` — page-cache lookups instead of faults), anything colder keeps the mmap+`MADV_SEQUENTIAL` walk, which also stops hinting when the buffered walk runs. The threshold is lenient by design: the penalty is asymmetric (a wrong warm guess costs ~10% cold; a right one wins 3× warm). Result: warm 65 GiB single frame **3.0 s → 1.29 s** (faults 560k → 4.4k), zstd-class; cold unchanged (~40 s, correctly dispatched at ~2% residency); `-v` logs the route either way.

**The buffered walk is authoritative when it succeeds, so it validates strictly and can never print a wrong summary.** It parses RFC 8878 frame headers itself (including the FCS 2-byte +256 offset and the XXH64 tail) and bails — returning the reset counters to the always-correct mmap+library walk — on anything it doesn't fully model: legacy v0.x frames, reserved block types, any size running past EOF, short reads. Verified: counts byte-identical to the mmap walk and `zstd -l` across multi-frame, 3M-tiny-frame (sizeless FCS → `?`), and single-frame fixtures; a truncated file bails to the mmap walk's corrupt verdict (exit 4); the meter ticks through both walks. The remaining cold-side idea (batched `posix_fadvise(WILLNEED)` prefetch to beat the sequential stream too) stays on the ROADMAP.

Also: `scripts/drop_cache [-c] FILE...` — the diagnostic these measurements used: evicts a file's pages via `POSIX_FADV_DONTNEED` (fdatasync first, so fresh writes drop too) or, with `-c`, reports residency via the same `mincore` mechanism. Unprivileged, per-file, no umount dance (inode/dentry-cold tests still need one).

Suite: 308 normal / 424 extensive (1 new: warm un-tabled file routes to the buffered walk and matches `zstd -l` counts).

## v0.14.92 — one pax record-grammar walk: `for_each_pax_record` (the v0.14.91 review's deferred hoist)

**The PAX extended-header record grammar — `"<len> <key>=<value>\n"` with the length counting its own digits — now lives in exactly one place.** The extract parser (`Extractor::parse`) and the seek scanner (`foreign_scan_entries`) each carried a verbatim copy of the walk, including the subtle `-2` that drops the separator space and trailing newline from the record's self-counted length; the v0.14.91 review flagged that arithmetic as the one genuinely drift-prone piece of pax handling (the same class of duplication whose divergence corrupted archives in v0.14.76, and the same one-source rule that hoisted `match_tar_member` in v0.14.82). New `for_each_pax_record(rec, fn)` owns only the grammar walk; each caller keeps its own key dispatch in a callback, so their different surrounding state stays untangled. `fn` returning false aborts the walk — that is the scanner's GNU-sparse bail (now hoisted to the top of its dispatch, behavior identical); the extract parser always returns true. Malformed records (no space, non-positive length, missing `'='`) skip or end the walk benignly, exactly as both originals did, and values remain binary-safe (delimited by the length prefix, not any terminator). Deliberately NOT hoisted: the eight common key handlers each caller duplicates — consolidating those would rewire working extractor state into a shared struct for zero runtime gain (the review's own "premature abstraction" verdict).

No behavior change intended, verified at the nasty edges on both rewired callers: foreign header-hop listing still byte-identical to `tar -tvf` (including the nonzero-size dir header case), crafted `RHT.security.selinux` extraction unchanged, GNU-tar PAX GNU.sparse.1.0 archives still extract with holes restored (16 blocks vs 12288 apparent), and a pax-sparse FOREIGN seekable archive still bails the scanner to the walk via the callback-abort path. Suite: 304 normal / 420 extensive, all passing.

**Also in this release: PLAIN (non-tar) compress output now ends with the standard zstd seekable seek table.** Investigating the `-l` meter (below) revealed only `--tar` create appended the table — plain gzstd output missed the v0.14.88 O(1) `-l` fast path and foreign seekable readers couldn't random-access it. Now every compress path emits it: the writer thread's csize recording is no longer tar-gated, each compress entry point pins its resolved chunk geometry, the serial single-thread path records inline (it bypasses the writer thread), and `finish_tar_index_frame` emits a table-only trailer when there's no GZIDX body (whose absence used to double as the "already appended" latch — the geometry is now consumed explicitly, or main()'s second append site would emit a duplicate). Deliberately self-validating: the frame count comes from what the writer actually wrote and the stream size from the meter, and the total must land inside the last frame's range for the pinned fixed chunking or no table is emitted — an unexpected frame split can only cost the fast path, never produce a wrong table (readers independently validate tiling too, so `cat a.zst b.zst` still falls back to the walk with correct counts — verified). Excluded: `--sliding-window` (one streamed frame, no chunk geometry) and `--no-index` (opts out, same flag as tar). Verified across cpu-only/gpu-only/hybrid and stdin-pipe input: footer magic at EOF, `-l` frame/skip counts identical to `zstd -l`, `zstd -t`/`zstd -d` accept the output, round-trips byte-identical.

**And the plain `-l` fallback frame walk shows a progress meter — with intra-frame ticks.** When an archive has no usable table (plain zstd files, pre-v0.14.83/foreign archives, `--no-index`), the walk streamed the whole file silently — 41 s on the 54 GiB reference archive. It now shows `[LIST] scanning frames NN% X / Y` on stderr, throttled to 10 updates/s with the first only after 100 ms (a warm cached walk never flashes one — verified both ways: 50-frame warm file shows nothing, 3M-frame file shows and clears it). Visibility follows `progress_loop`'s exact rules: default/`-v` on a terminal, `--progress` forces, `--no-progress`/`-q` silence (all verified byte-clean). Single-frame archives (plain `zstd`/`-T0` output — the walk is otherwise ONE opaque `ZSTD_findFrameCompressedSize` call) get a byte counter via `hop_frame_blocks`: an advisory hop over the frame's RFC 8878 block headers that ticks the meter and does the cold page reads with progress, while the authoritative size still comes from the library call immediately after, over now-warm pages (~free) — the listing's counts never depend on the hop, and any anomaly (reserved block type, size past EOF) just stops it. Verified cold (`posix_fadvise DONTNEED`): a single-frame 800 MiB file ticks 28%→93% during its walk; a tabled gzstd file stays instant with no meter.

Suite: 307 normal / 423 extensive (3 new: seekable footer on plain output, `zstd -t` accepts it, `--no-index` omits it). One pre-existing test needed its assertion honest-ified: `--single-thread compresses` grepped for a literal `100.00%` ratio, and the ~33-byte table on a 1 MiB incompressible input legitimately makes that `100.01%` — it now asserts the summary line's `=>` shape instead of an exact ratio.

## v0.14.91 — `--selinux` (GNU-tar-compatible contexts) + header-hop `-l` for foreign seekable archives

**`--selinux` stores and restores SELinux security contexts, closing the third leg of the extended-metadata item (xattrs v0.14.3, ACLs v0.14.3, contexts now).** Contexts travel as PAX `RHT.security.selinux` records — GNU tar's own key, so archives interoperate in both directions. gzstd reads and restores them directly through the `security.selinux` xattr (`lgetxattr`/`fsetxattr`), deliberately avoiding a libselinux dependency (same reasoning as the NVML dlopen shim: never add a hard link-dep for an optional capability): on a host without SELinux labeling, create simply finds no contexts, and extract's restore is best-effort — `EPERM`/`ENOTSUP` reported at `-v`, never fatal — riding the existing `apply_ext` xattr path. Gating matches `--xattrs`/`--acls` exactly: give the flag on BOTH create and extract, records are parsed-and-ignored without it. With `--selinux --xattrs` together the context rides only in the RHT record, never duplicated into `SCHILY.xattr.*` (same dedupe pattern as the ACL records). Verified: a crafted PAX archive carrying the record extracts cleanly with the apply attempted (exit 0, warning at `-v` on this unlabeled host), the record is ignored without the flag, and `--selinux` create on an unlabeled tree emits no records. **Honest limitation:** create-side record *emission* with a real context is verifiable only on an SELinux-labeled host (this box cannot set `security.selinux` unprivileged) — the emission shares `pax_record`/`apply_extended_metadata` with the well-tested SCHILY records, but end-to-end SELinux round-trip should be spot-checked if a labeled host becomes available.

**`-l --tar` on a FOREIGN zstd-seekable archive (t2sz-style: seek table, no gzstd index) now lists via header-hop instead of decompressing the whole archive** — the follow-up left open by v0.14.83's seek interop. `foreign_scan_entries` (the verified scanner behind foreign selective extraction and parallel full extraction) gained an optional listing-metadata collector: the same hopped headers already contain everything `tar -tvf` prints, so it now also captures mode/uid/gid/uname/gname/mtime/size/linkname (with `'K'` long-link and pax `linkpath`/`mtime`/`uid`/`gid`/`uname`/`gname` overrides, previously ignored as extract-irrelevant) and `list_tar` feeds the result to the same `list_entries` the index route uses — one formatter, so the two routes cannot drift. Listing verified byte-identical to `tar -tvf` on a foreign archive (dirs, symlink, multi-frame file, restrictive modes), member filtering included; any scan bail (GNU sparse, pax globals, bad checksum, truncated/corrupt table) falls back to the unchanged decompress walk, re-verified by corrupting the seek-table tail. gzstd-indexed archives still use the instant index route; `--no-index` archives carry no seek table and walk as before.

A third review angle (efficiency/reuse over the changed code only) recommended two items and endorsed leaving the rest: `gather_selinux` now does ONE `lgetxattr` into a stack buffer for the ~100% case (contexts are 30–60 bytes; the gather runs per member in the parallel pool) with an ERANGE re-probe loop that also closes the probe-to-fetch race where a growing context was still silently dropped; the second recommendation — hoisting the duplicated pax record-grammar walk into a shared `for_each_pax_record` helper (the `match_tar_member`-style one-source hoist; the length arithmetic is the drift-prone piece) — is deliberately DEFERRED rather than applied mid-release: it mechanically rewires `Extractor::parse`'s working dispatch, and both copies are currently pinned by adversarial tests. Tracked on the ROADMAP.

A second pre-commit review angle over the `--selinux` paths confirmed the pax-record encoding injection-safe (length-prefixed records make binary values inert on both gzstd's and GNU tar's parsers), the hardlink/dedupe/engage-condition handling clean, and surfaced two findings: (1) fixed — `gather_selinux` used a fixed 1 KiB buffer and silently dropped larger contexts (big MCS category sets); with `--xattrs --selinux` together the SCHILY dedupe would then lose the context entirely, strictly worse than `--xattrs` alone; now uses `gather_xattrs`' size-probe two-call pattern. (2) documented, not fixed — xattrs/contexts on SYMLINKS and device nodes are stored on create but silently not reapplied on extract (`apply_ext` is fd-based; symlinks need `lsetxattr` through the secure parent-fd walk). This gap is inherited from `--xattrs` (pre-existing, all the way back to v0.14.3) and GNU tar does restore these — now stated plainly in both flags' `--help` and tracked on the ROADMAP rather than left implied.

A pre-commit adversarial review of the scanner changes confirmed the new parsing memory-safe and the entry state machine leak-free, and surfaced two real issues, both fixed before commit: (1) the scan accumulates entries in RAM (the walk streams bounded), so a crafted archive of RLE-bomb header frames could OOM the new `-l` route — a 1 GiB accumulation cap now bails to the always-correct walk (also shields the pre-existing `-d` foreign paths); (2) the hop listed size 0 for a dir/link header carrying a nonzero size field where the walk and GNU tar print the stored size — the listing now captures the size before the stream-geometry zeroing (regression-tested with a crafted nonzero-size dir header).

Suite: 304 normal / 420 extensive (4 new: crafted-RHT extract with `--selinux`, record ignored without the flag, unlabeled-tree create emits nothing, foreign header-hop `-l` parity + route check).

## v0.14.90 — tar input ergonomics: `--exclude-from`/`-X`, `--exclude-vcs`, `--files-from`, `-P`

**Four GNU-tar creation conveniences, closing the ROADMAP's "more tar input ergonomics" item.** All are creation-only (like the existing `--exclude`) and rejected with a clear usage error on `-d`/`-t` or without `--tar`:

- **`--exclude-from FILE` / `-X FILE`** reads exclude patterns from FILE, one per line (`-` = standard input). Patterns land in the same list as `--exclude`, so anchoring and subtree-drop semantics are identical by construction. Lines are literal (no comment syntax); empty lines skipped; an unreadable FILE is a usage error (exit 2), matching GNU tar's fatal exit.
- **`--exclude-vcs`** appends GNU tar's version-control exclusion table (CVS/RCS/SCCS, .svn, .git + .gitignore/.gitattributes/.gitmodules, .cvsignore, GNU Arch, Bazaar, Mercurial, darcs). Each entry behaves like a bare-name `--exclude` (matches anywhere, directory match drops the subtree). Member listing verified identical to GNU `tar --exclude-vcs` on a tree with .git/.svn dirs. Expanded once after parsing, so repeating the flag doesn't duplicate patterns.
- **`--files-from FILE`** reads the paths to archive, one per line (`-` = standard input), each honoring the `-C` in effect at the flag's position exactly like a positional source (it feeds the same `push_positional` path, so the `tar_source_dest` pairing can't desync). Deliberate divergence from GNU tar, documented in `--help`: every line is a literal path — a line starting with `-` is never parsed as an option (GNU tar's in-file `-C` handling is a known footgun). Long-form only: GNU tar's `-T` short form is taken by threads (zstd CLI compatibility).
- **`-P` / `--absolute-names`** keeps the leading `/` on stored member names at creation (verified listing parity with GNU `tar -P`), including the `src == "/"` root case (children stored as `/proc`, not `proc`). Extraction always strips leading slashes and contains output inside `-C` (`norm_member_name` + the `openat`/`O_NOFOLLOW` walk), so `-P` on extract is refused rather than silently ignored — refusing loudly beats implying GNU tar's unconfined absolute-path restore, which gzstd deliberately does not offer.

Suite: 300 normal / 416 extensive, all passing (6 new: exclude-from file + `-X -` stdin, exclude-vcs, files-from with positional `-C`, `-P` create parity, `-P` extract refusal). Extensive (zstd-CLI compat) run since this touches `parse_args`. `EXPECTED_TESTS` also absorbs 4 tests the two prior releases added without bumping it (2 parallel-extract audit tests in v0.14.87, 2 sparse-format tests in v0.14.89) — verified against the git diffs, so the drift note is silenced honestly, not papered over.

## v0.14.89 — `--sparse` create now emits PAX GNU.sparse.1.0 (with `--format` to select OLDGNU)

**`--tar --sparse` now stores sparse files as PAX GNU.sparse.1.0 by default instead of OLDGNU `'S'`.** Both formats round-trip through gzstd, GNU tar, bsdtar, and Python, but they degrade very differently for a sparse-*unaware* reader: OLDGNU `'S'` with >4 segments puts extension blocks between the header and data, so such a reader reads them as file content and **loses archive alignment — corrupting every member after the sparse file**; PAX 1.0 keeps the member a plain regular `'0'` file whose data is `[map text block][segments]`, so an unaware reader mangles only that one file and the rest of the archive stays intact (both verified with a naive-POSIX-walk harness: OLDGNU → lost alignment, PAX → clean EOF). PAX 1.0 is also the format modern GNU tar writes for `--sparse --format=posix`.

**`--format=FMT` selects the sparse encoding** (it affects only sparse files — the rest of the archive is always ustar + GNU extensions): `posix`/`pax` → PAX GNU.sparse.1.0 (default, so `--format=posix` is effectively a no-op you can pass for GNU-tar-command parity); `gnu`/`oldgnu` → legacy OLDGNU `'S'` (there is no dedicated GNU tar flag for OLDGNU sparse either — it's `--format=oldgnu`). The zstd/gzip output-format values that share the `--format=` prefix stay accepted for zstd-CLI compatibility (`--format=zstd` silent; `--format=gzip`/`xz`/`lzma`/`lz4` warn-and-ignore, since gzstd only emits zstd) — the new tar-format parsing folds them in rather than rejecting them. Any other value is a clean usage error.

**Implementation.** The map block lives in the member's data region (not the header), so the fragile layout offset-math is untouched: `sparse_map_text()` is the single source of truth for the `"<n>\n<off>\n<len>…"` map (shared by the sizing, the reader's synthesis, and matching what `read_pax_sparse_map` already parses), the member's `size` is `map_padded + Σ segment lengths`, and the `GNU.sparse.major/minor/name/realsize` pax records are injected in `apply_extended_metadata` (which now also runs for PAX sparse without `--xattrs`/`--acls`). The `assemble` reader synthesizes the map bytes for the leading region and reads the segments after it.

**Two GNU-tar-interop subtleties, both diagnosed against GNU tar 1.35's own output and fixed:** (1) a plain `size` pax record short-circuits GNU tar's sparse detection (it then extracts the raw map+data as a regular file), so gzstd carries the stored size **only** in the ustar header, like GNU tar; (2) GNU tar reconstructs the holes only when the member's headers use the **POSIX** ustar magic (`"ustar\0"`+`"00"`), not gzstd's usual OLDGNU magic (`"ustar  "`) — so PAX sparse members (their pax block + ustar header) now emit POSIX magic. With both fixes, GNU tar / bsdtar / Python / gzstd all extract gzstd's PAX sparse archives with holes preserved, no `--sparse` flag needed. Verified across a multi-segment file and a 10 GiB-logical file (data past the 8 GiB base-256 boundary); `--format=oldgnu` still emits `'S'` and round-trips; `-l --tar` reports the logical size; `-t --tar` verifies clean.

## v0.14.88 — `gzstd -l` is O(1) on any seekable archive + consistent sparse-hole detection on extract

**Sparse-hole detection on extraction is now consistent across all write paths.** The buffered writer's `pwrite_sparse` scanned for all-zero 4 KiB blocks starting at each frame *segment's* offset rather than the *file's* offset. A regular file's data begins at a 512- but rarely 4 KiB-aligned tar offset and arrives as several frame segments, so the zero-scan windows straddled the file's real 4 KiB blocks — a fully-zero block sharing a window with a neighbouring segment's non-zero bytes was written out (allocated) instead of left a hole. Result: the same file could restore with slightly different physical block counts depending on the write path — buffered extraction allocated a few extra blocks where the O_DIRECT writer (which scans file-4 KiB-aligned blocks in its bounce buffer) correctly left holes; content was always byte-identical, only the sparse layout differed. `pwrite_sparse` now aligns each zero-scan window to the file's 4 KiB grid (first short window to the next boundary, then full blocks), so holes are found regardless of segment boundaries and the sparse layout is identical across buffered vs O_DIRECT and serial vs parallel extraction. Verified: a set of large files with a 2 MiB interior zero region now restores to the exact same block count under all four (serial/parallel × buffered/O_DIRECT) paths (previously serial+buffered was the outlier). No hot-path cost — the alignment is a couple of integer ops per segment, and the aligned-start common case is unchanged.

**Plain `gzstd -l ARCHIVE.zst` no longer walks the whole file to print the frame summary.** The zstd-style summary (Frames / Skips / Compressed / Uncompressed / Ratio / Check) previously walked every frame + block header from offset 0 to EOF via `ZSTD_findFrameCompressedSize`; over an mmap with `MADV_SEQUENTIAL` that streams essentially the entire archive, so on a real 54 GiB / 6631-frame backup it took ~41 s (disk-read-bound), not the near-instant the `-l --tar` member listing already delivers. Now `list_zst` first reads the standard zstd seekable-format **seek table** at EOF (which gzstd appends by default, and which t2sz / libzstd-seekable also produce): every frame's compressed and decompressed size is right there, so Frames, Skips, and Uncompressed come from ~two tail reads plus one frame-header peek for the XXH64 flag — **O(1), independent of archive size.** The 54 GiB summary that took 41 s is now effectively instant.

Correctness is preserved exactly: the fast path's numbers match the walk and ground truth (`decompress | wc -c`) — verified on an indexed archive (data-frame count identical, uncompressed byte-exact), with the two extra Skips correctly accounting for the GZIDX index and the seek-table frame itself (which never self-lists). It is guarded by a strict integrity gate — the entries' compressed sizes must tile the file exactly up to the seek-table frame, and a 1 GiB frame-count cap bounds a forged footer — so a truncated, concatenated, or forged file degrades to the full frame walk rather than printing wrong numbers. Anything without a valid table (plain single/multi-frame zstd, `--no-index` archives, stdin, corrupt footer) uses the unchanged walk, and `MADV_SEQUENTIAL` is now applied only on that fallback (the fast path touches just the tail + one header, so it shouldn't hint a whole-file scan). `-l --tar` (the member listing) was already index-based and is unchanged.

**Plain `gzstd -l ARCHIVE.zst` no longer walks the whole file to print the frame summary.** The zstd-style summary (Frames / Skips / Compressed / Uncompressed / Ratio / Check) previously walked every frame + block header from offset 0 to EOF via `ZSTD_findFrameCompressedSize`; over an mmap with `MADV_SEQUENTIAL` that streams essentially the entire archive, so on a real 54 GiB / 6631-frame backup it took ~41 s (disk-read-bound), not the near-instant the `-l --tar` member listing already delivers. Now `list_zst` first reads the standard zstd seekable-format **seek table** at EOF (which gzstd appends by default, and which t2sz / libzstd-seekable also produce): every frame's compressed and decompressed size is right there, so Frames, Skips, and Uncompressed come from ~two tail reads plus one frame-header peek for the XXH64 flag — **O(1), independent of archive size.** The 54 GiB summary that took 41 s is now effectively instant.

Correctness is preserved exactly: the fast path's numbers match the walk and ground truth (`decompress | wc -c`) — verified on an indexed archive (data-frame count identical, uncompressed byte-exact), with the two extra Skips correctly accounting for the GZIDX index and the seek-table frame itself (which never self-lists). It is guarded by a strict integrity gate — the entries' compressed sizes must tile the file exactly up to the seek-table frame, and a 1 GiB frame-count cap bounds a forged footer — so a truncated, concatenated, or forged file degrades to the full frame walk rather than printing wrong numbers. Anything without a valid table (plain single/multi-frame zstd, `--no-index` archives, stdin, corrupt footer) uses the unchanged walk, and `MADV_SEQUENTIAL` is now applied only on that fallback (the fast path touches just the tail + one header, so it shouldn't hint a whole-file scan). `-l --tar` (the member listing) was already index-based and is unchanged.

## v0.14.87 — parallel-extract audit follow-ups: determinism on malformed archives + honest summary figures

Post-merge multi-angle audit of v0.14.86 found no correctness, race, or security defect on legitimate archives (parallel extraction is byte- and metadata-identical to the serial walk), but surfaced two polish items, both fixed here.

**Leaf/directory path collisions now fall back to the serial walk.** On a *malformed* archive where one path is both a non-directory entry (file/symlink/special) and also used as a directory — e.g. a file `x` plus a file `x/y` — the serial walk resolves the contradiction deterministically by archive order, but two parallel workers could race the shared path with a nondeterministic winner (no security escape: the `openat`/`O_NOFOLLOW` walk still contains it, and `unlinkat` can't replace a directory — only the file-vs-refused outcome differed run-to-run). `build_full_parallel_plan` now rejects any leaf whose path is an explicit directory entry or a parent component of another entry (`tar_leaf_dir_collision`), forcing serial so parallel == serial even on adversarial input. Legitimate archives (a real directory `d` with children `d/a`, `d/e/f`) are unaffected — verified they still engage parallel. The foreign header-hop scan now records each entry's typeflag to feed the same check.

**Extraction summary figures now match the serial walk.** The parallel path's reported output under-counted by the trailing end-of-archive zero blocks, and its reported input double-counted the boundary frame that two adjacent partitions each `pread`. Fixed: the trailing stream bytes are added once after join so output equals the full decompressed stream, and the later partition skips read-accounting for its shared leading frame (`decode_seek_frame(..., count_read=false)`) so input reflects the archive's logical compressed size, not the physical re-read. A parallel and a serial extraction of the same content now print an identical `in => out` and ratio. Extracted data was never affected — these were reporting-only counters.

## v0.14.86 — parallel-dispatch full `-d --tar` extraction (the last serial extract stage, parallelized)

**Full-archive `-d --tar` extraction now parses and dispatches in parallel instead of on a single serial thread.** Extraction already decompressed in parallel and wrote in parallel (the v0.14.72 writer pool), but one thread still walked the whole decompressed tar stream in order — parse a header, dispatch its file, advance to the next — because tar has no inline index and each header's position depends on the previous member's size. This release breaks that dependency using the frame table + entry boundaries shipped for selective extraction (v0.14.80/82/83): the entries are split into N contiguous partitions, and each worker `pread`s + decompresses only *its* frames (the verified `seek_feed` decode path, now shared as `decode_seek_frame`) and runs the standard `parse`/`handle_entry` over its byte slice, dispatching file writes to the one shared writer pool. It closes the last open item on the `--tar` roadmap for the 0.14.x line (route 2, no-selection case).

**The design reuses the existing extractor wholesale rather than forking a second one.** A `StreamReader` gained a producer-callback source with a partial-slice byte limit, so a worker sees a clean EOF exactly at its partition's final entry boundary (the tail of its last frame, which belongs to the next partition, is never delivered); `parse`/`handle_entry` gained a parallel mode that routes hardlink and directory-metadata collection into per-worker lists (merged in archive order after join, so `finish_deferred` still creates hardlinks after their targets exist and applies directory metadata in reverse order). Everything else is unchanged code: zero-copy `DataSeg` file writes, large-file windowed part jobs, symlinks/specials/sparse handled by the one worker that owns the entry, the `openat`/`O_NOFOLLOW` path-traversal security walk, and the `map_owner` name→id cache (now mutex-guarded for the workers). The index is only a source of partition boundaries + the frame map — it does **not** carry xattrs/ACLs/sparse maps, so each worker re-parses its slice for full-fidelity metadata rather than trusting the index blob.

**Engaged by default with automatic fallback; no format change.** The parallel path runs when there's no member selection, the archive is seekable (not stdin), it isn't `--keep-going` (whose damage recovery needs the whole serial stream), a valid frame table + contiguous entry list is recoverable, there are no duplicate normalized entry names (tar's last-writer-wins can't be preserved across partitions), and it's worth splitting (≥2 partitions after capping N by CPU count, 16, entry count, **and frame count** — adjacent partitions share their boundary frame, so more partitions than frames would just multiply that redundant decode). Anything else — foreign archives the scan can't model, `--no-index`, huge single-frame chunks, stdin, `--keep-going`, tiny archives — falls back to the unchanged serial `FrameSink` walk. Both **our indexed archives and foreign zstd-seekable archives** (t2sz-style, via the header-hop scan refactored into a shared `foreign_scan_entries`) are supported. Corrupt frames still `die_data` (exit 4) like every non-`keep_going` decode path.

**Honest scope (per ROADMAP):** on today's single-stream-ceiling NVMe the observable win is bounded — extraction is write-bound, the writer pool already saturates the device, and the serial parse it replaces was a small fraction of the wall (`-v [EXTRACT]` measured it near-zero on the write-bound archives here). The value is **future-proofing** for write fabric that scales with concurrent streams (multi-controller / CXL / next-gen NVMe), where parallel parse+dispatch feeding a scaling writer pool becomes the dominant lever; the serial path stays as the always-correct fallback. Verification: parallel extraction is byte-for-byte identical to the serial walk and to the source tree across a mixed tree (nested/empty dirs, relative symlinks, cross-partition hardlinks, a read-only directory, a large multi-frame file, unicode+long names); foreign seekable full extraction matches too; single-byte frame corruption yields exit 4; idempotent re-extraction over an existing read-only tree is clean; and the whole path is **ThreadSanitizer-clean** (zero data races) across 2–16 partitions.

## v0.14.85 — `-t --tar` reports the true stream size (consistent with create and `zstd -l`)

**The `-t --tar` verify summary printed the file-content-only byte total, so its `compressed => uncompressed (ratio)` disagreed with the create summary on the same archive.** The create line reports the full tar-stream size it compressed (file data plus every 512-byte header, per-file padding, and pax ACL/xattr / long-name record); `-t` reported `validated_bytes` — the sum of member content only, which omits all that structure. On a real 171k-entry home backup with `--acls --xattrs` the two differed by ~310 MiB (3%): create said 10.31 GiB, `-t` said 10.00 GiB, and the ratios disagreed (43.7% vs 45.0%) — mildly alarming on a backup-then-delete tool even though no data was lost (the gap is pure tar overhead, and the archive verified structurally valid). `-t --tar` now reports the total decompressed tar-stream size (captured from the parser's `consumed_total`), so its size and ratio match the create summary and `zstd -l`/`gzstd -d | wc -c` exactly. `validated_bytes` (the file-content total) is unchanged and still drives the `-l --tar` listing footer, where "N files, X" legitimately means content bytes. New regression test builds a many-small-file archive (where the two definitions differ ~50x) and asserts the `-t` figure matches `gzstd -d | wc -c` within rounding. Suite: 290 normal / 406 extensive.

## v0.14.84 — pre-tag audit of the v0.14.82/83 seek code (rounds 1–4)

**Round 4 — resource hygiene and error paths.** The suspected ZSTD_DCtx leak in the foreign scanner turned out not to exist (the early returns are inside the range-reader lambdas; the scan body funnels through the single free), and the rest traced clean: both table parsers rewind on every exit, failed planners are reset by the caller, seek_feed joins its pool on all non-fatal paths, and the Extractor's drain guarantees the feeder can never hang on a dead consumer. Two real gaps, both fixed: the two new decode sites (foreign scanner, seek_feed workers) neither checked `ZSTD_createDCtx()` for NULL (dereferenced under memory pressure → segfault instead of a clean error) nor applied `apply_mem_limit_to_dctx` — meaning a user's `--memlimit` window cap, enforced by every other decode path, was silently bypassed by seek extraction. Scanner OOM now falls back to the walk; a worker OOM dies cleanly; both honor the memory limit.



**Round 3 — semantic parity between the seek planners and the walk.** Most equivalences hold by construction and were verified as such: selector and entry-name normalization are the same shared functions on both paths, duplicate member names keep tar's last-wins ordering (every match's range is included in archive order and the Extractor replays them), positional -C root binding is literally the same first-matching-selector function, and a scanner misparse is always checksum-caught into the walk — never wrong output. One real divergence found and fixed: the foreign header scanner zeroed the data size for typeflags '1'–'5', but the Extractor's actual alignment rule consumes zero data only for links and directories ('1','2','5') while devices/fifos/labels/unknown types consume `size` bytes. On a foreign tar carrying a nonzero size on a device entry (pathological but legal-ish), the scanner desynced — checksum-caught, so it only cost the seek path, but a selected member after the mismatch could fail where the walk succeeds. The scanner now mirrors the parser's rule exactly, verified with a crafted archive (device entry with bogus size + data blocks, then a selected member: seek path engages and extracts).

**Round 2 — concurrency.** Reviewed the seek_feed worker pool (claim gate, reorder window, condition-variable discipline), its interaction with FrameSink backpressure and worker-side die(), and the writer-thread frame recording across passes. The pool's core is sound — the smallest outstanding claim always passes its gate so the reorder window cannot deadlock, predicates are re-checked under the mutex so wakeups cannot be lost, the slicer never holds the pool mutex while blocked on the sink, and die() from a worker is lock-free. Two real findings, both fixed:

1. **GPU-fault rebuild silently dropped the seek table.** A GPU fault discards the output and reruns the whole compress pass — including assemble() and the writer — but the recorded per-frame sizes were never cleared between passes, so the retry appended to the first pass's entries, the completeness check failed, and finish_tar_index_frame quietly omitted the table (archive valid, seek-extract and outside random access silently gone). Now cleared at each assemble(). Verified with GZSTD_DEBUG_FAIL_GPU_AFTER on --gpu-only: the rebuilt archive seek-extracts.
2. **seek_feed's reorder window counted frames, not bytes.** Every in-flight frame holds compressed + decompressed buffers, so a huge --chunk-size archive could pin ~3×threads×chunk of RAM where the normal decompress paths are bounded by the 4 GiB FrameThrottle. The window is now sized from a 4 GiB byte budget over the plan's largest frame (and the pool spawns no more workers than the window admits).

**Round 1 — untrusted-input robustness.** Reviewed the new parsers that consume attacker-controllable bytes (`-l`/`-d` run them on any archive): the seekable-footer probes, the GZIDX cross-validation, the foreign header scanner, and `seek_feed`. Three findings, all fixed and regression-tested with real crafted archives (suite test #12):

1. **OOM crash (both footer parsers):** the seek-table allocation was bounded only by the file size, so a forged frame count (e.g. 500M) on a large archive demanded an allocation up to the whole file → `bad_alloc` → terminate, from a plain `-l --tar`. Now capped at 1 GiB (>100M frames — the index blob's cap philosophy); verified prompt failure under a 3 GB address-space limit.
2. **Infinite loop (foreign header scan):** a forged base-256 size field near 2^64 wrapped `tar_round_up`, letting the entry end land *behind* the cursor while still passing the `<= tar_size` check — the scan walked backward forever. The size is now bounded by the remaining stream before the round-up, and `eend <= cursor` bails to the walk; verified prompt exit on a crafted archive that previously hung.
3. **Unbounded per-worker allocation (seek_feed, minor):** a forged u32 dsize in the table let each worker allocate up to 4 GiB before decompression corroborated it (content-size-less GPU frames skip the earlier check); now capped at 2 GiB/frame like the foreign scanner.

Suite: 289 normal / 405 extensive.

## v0.14.83 — indexed archives adopt the standard zstd seekable format (t2sz/ratarmount interop)

**The frame table moved out of our private format into a spec-conformant zstd seekable-format seek table (contrib/seekable_format — what t2sz produces and indexed_zstd/ratarmount-class readers consume), appended after the GZIDX index so its footer is the file's last bytes where seekable readers look.** Any seekable-format tool can now random-access gzstd archives with zero gzstd knowledge — verified with a format-blind reader (python struct parse of the footer + plain `zstd` decompressing a single frame extracted by `dd`, byte-identical to the stream slice; now suite test #10). The table lists every frame per spec, including the GZIDX skippable itself as a decompressed-size-0 entry, so it tiles the whole file for strict readers. gzstd's own seek-extract now consumes the same standard table (v0.14.82's private "GZFT" section is gone — one wheel, the standard one), generalized from uniform-chunk division to a prefix-sum offset map (binary search per range), which is exactly what reading *foreign* seekable archives will need (ROADMAP: zstd-ecosystem seek interop, part 2). Index format bumped to `GZIDX002` (dev-phase break, deliberate): readers locate the index trailer immediately before the seek-table frame, falling back to the trailer-at-EOF probe when the seek table is absent (a frame overflowing the format's u32 size fields — huge `--chunk-size`); v0.14.80–82 archives fall back to the decompress walk. Validation hardened accordingly: csize prefix sum must land exactly on the index frame start AND dsize sum on the tar-stream size, data frames must precede skippable entries, so truncated/concatenated/forged files still can never validate. pzstd's inline tags are deliberately NOT emitted: legacy format, superseded by `zstd -T`, and its tags benefit only pzstd's own decompressor. No behavior change for GNU tar / zstd / pzstd consumers (skippable frames ignored by spec; `tar --zstd -x`, `zstd -t`, byte-identical `-d` all re-verified).

**And the interop runs the other way too: selective extraction now works on FOREIGN zstd-seekable archives (t2sz-style) that gzstd didn't create.** They carry the frame map but no entry index, so `build_foreign_seek_plan` header-hops: it walks tar headers decompressing only the frames header (and pax/longname) bytes land in — file data is skipped by pure offset arithmetic — building (name, hdr_off, entry_end) per member, then selects and feeds the same seek_feed → Extractor pipeline. Verified on synthetic t2sz-style archives with deliberately block-UNALIGNED frame boundaries (GNU and pax formats, long names, multi-member selections): extracting one small file read 1 of 11 frames (9.9% of the archive). The scan is deliberately paranoid, falling back to the full walk on anything its grammar doesn't fully model — GNU sparse ('S'/`GNU.sparse.*`), pax globals ('g'), bad header checksums, extension records > 1 MiB, mid-stream skippable frames — and a scan miss can only yield "Not found in archive", never corruption: the Extractor re-parses the sliced stream with the real parser, checksums and all. Foreign tables are validated like our own (entries must tile the file exactly, first bytes must be a zstd data frame, per-frame content sizes cross-checked at decompress). pzstd inline-tag reading was considered and deferred: pzstd's chunks may lack the content-size header the map needs, the tool is legacy, and none were available to test against — ROADMAP notes it. Suite: 288 normal / 404 extensive (new: foreign header-hop test with a python-built spec archive).

## v0.14.82 — seek-based selective extraction: pull one file from a 100 GiB backup in milliseconds

**`-d --tar ARCHIVE MEMBER...` now extracts a selection by SEEKING to its frames instead of decompressing the whole archive.** The v0.14.80 index already recorded each entry's tar-stream offsets; this release adds the missing half — a **frame table** mapping those offsets to compressed file offsets — and the extract-side consumer. On create, the writer thread records every data frame's compressed size in write order (CPU, GPU, and hybrid paths all funnel through it), and `append_tar_index` folds a `"GZFT" | chunk_size | frame_count | csizes` section into the index body after the entry records; prefix sums give the seek offsets. On extract with MEMBER selection, a planner matches the selectors against the index records (the *same* `match_tar_member`/`norm_member_name` code the stream walk uses, hoisted so semantics can't drift), coalesces the selected entries' `[hdr_off, entry_end)` ranges, maps them to frames, and a small pread+decompress pool feeds the sliced tar stream — plus the end-of-archive zero blocks — into the unchanged FrameSink → Extractor pipeline, so member matching, path-traversal security (`openat`/`O_NOFOLLOW`), metadata/hardlink/deferred-dir handling, and "Not found in archive" reporting are all the existing code. A frame fully covered by the selection is moved into the sink without a copy (the common case: a big member's interior frames). Measured on a 9.76 GiB / 39 GiB GPU-written archive: **extracting one small file takes 0.58 s wall / 0.006 s user, reading 0.1% of the archive, vs 6.7 s wall / 70 s CPU** for the v0.14.81 full walk — and the seek time is O(selection), not O(archive), so the gap grows linearly with archive size. Extracting a 19.5 GiB member via seek runs at the same ~2.7 GiB/s write-bound rate as a full extract (1251 of 2501 frames read, zero-copy path).

**Backward and forward compatible in both directions, deliberately.** The trailer magic stays `GZIDX001`: the v0.14.80/81 parser reads exactly `entry count` records and ignores trailing body bytes, so **older binaries keep the instant `-l` on new archives** (verified against a v0.14.81 build: index listing used, byte-identical) — a version bump would have silently degraded them to the walk. New binaries treat the table as strictly optional and validate it independently: frame count must match the tar-size/chunk geometry, and the csize prefix sum must land EXACTLY on the index frame's own start, so truncated, appended-to, or concatenated archives (`cat a.tar.zst b.tar.zst`) can never validate — they fall back to the walk, as do foreign/`--no-index` archives, stdin, and `--keep-going` (whose damage recovery needs the whole stream). The offsets come from the (untrusted) archive, so the planner also requires the entry table to describe a well-formed contiguous stream (entries contiguous from 0, header ≥ 1 block, data inside the stream) before slicing on it, and each decompressed frame must match the index geometry exactly or the extract dies with a data error. Frames written by nvCOMP may omit the content-size header; the feeder validates via the decompressed size instead (H100-created archives verified end-to-end). Suite: 286 normal / 402 extensive — 9 new tests cover the seek path (engagement + fraction read, multi-frame members, dir selectors with deferred mtimes, GNU long names, hardlink-without-target parity with the walk) and every fallback. Closes ROADMAP's parallel-dispatch extract route 2 for the selective case; the no-selection parallel dispatch remains future work (today's hardware is write-bound anyway, see the roadmap caveat).

Also fixed in passing: **v0.14.81 broke `USE_NVCOMP=OFF` builds** — the `--verify` pool sizing references `GPU_SUBCHUNK_MAX` from CPU-visible code (a dead branch when `cpu_only`, but it must compile) and the constant was declared inside `#ifdef HAVE_NVCOMP`. Moved it out; CPU-only build compiles again and passes the seek-extraction tests.

## v0.14.81 — pre-release multi-agent audit of v0.14.77–80: two crashes and eight hardening fixes

An 8-angle agent review of everything since v0.14.76 (member selection, positional -C, ownership mapping, listing parity, sizeless streaming, member index, progress meter), run before tagging the first downloadable build. Ten findings, all fixed:

**Two reproducible crashes.** (1) `gzstd --tar -- SRC` (and `-d/-l --tar ARCHIVE -- MEMBER`) SEGFAULTED: the `--` end-of-options branch still used the pre-v0.14.77 single push, so `tar_sources` and the parallel `tar_source_dest` (-C bindings) desynced and consumers indexed out of bounds. Both push sites now go through one `push_positional` helper, and both consumers bounds-check as belt-and-braces. (2) A LEADING zstd skippable frame — pzstd emits exactly that — defeated the v0.14.79 sizeless-frame detection (`ZSTD_getFrameContentSize` reports 0 for skippables), sending the archive down the batch path that slurps the whole file into RAM: reproduced at 3.0 GB peak RSS for a 600 MB archive. `peek_first_frame_decomp_size` now hops over leading skippable frames (bounded at 16) to read the first DATA frame; same file now lists in 47 MB.

**Correctness/robustness.** Name-first ownership (v0.14.78) silently fell back to numeric ids for any group whose NSS record exceeded 4 KB — `getgrnam_r` ERANGE was indistinguishable from "no such name" and cached; now retried with a doubling buffer (to 1 MiB). `read_tar_index` could be made to allocate 16 GiB by a forged trailer; the compressed blob's embedded content size must now corroborate the trailer BEFORE the allocation, and the sanity cap dropped to 1 GiB (forged-trailer cost measured at 32 MB, clean fallback to the walk). An unmatched member selector now exits 1 on `-l` and `-d` alike (was 4 = "corrupt data" on `-l` — off the documented exit-code table, and inconsistent). The index-append paths got the file's standard write robustness (`robust_fwrite` for EINTR/short writes; `write()==0` no-progress guards in `append_plain_fd` AND the pre-existing gap in `pwrite_all`) and were consolidated into one `append_tar_index` helper so a future output route can't half-forget the index.

**Drift-proofing and the transient test failure.** The index and walk listing routes shared only a promise of byte-identity; now they share the code — `filter_entry`/`list_count` serve both `handle_entry` and `list_entries`, and `list_tar` builds ONE Extractor with one footer/exit-code epilogue for both routes (the same de-duplication lesson as v0.14.76's `entry_header_len`). The one-off extensive-suite flake was root-caused to the RSS test asserting below the 256 MiB FrameSink budget — sink occupancy between the streaming producer and the header-skipping consumer is scheduling-dependent, so one consumer stall crossed the 146 MiB threshold; the test now uses a 400 MB archive with a 500 MB bound (slurp failure mode ≈ 850 MB+, streaming worst case ≈ 320 MB). Dead `Extractor::dest_fd_` removed (roots_[0] is the root). Suite: 277 normal / 393 extensive, all green, including new regressions for both crashes and the exit codes.

## v0.14.80 — member index in a skippable frame: instant -l --tar (~0.2 s at any archive size)

**`--tar` create now appends a member index; `-l --tar` reads it and lists without decompressing anything.** The index is a zstd SKIPPABLE frame after the tar data — part of the zstd format spec, ignored by every decoder — so the file stays a standard `.tar.zst`: GNU `tar --zstd -x` extracts it unchanged and `gzstd -d` produces a byte-identical tar stream with or without it (both verified). The frame carries one record per member (type, mode, owner ids + names, sizes, mtime, device numbers, and each entry's tar-stream offsets for future seek-based selective extraction) zstd-compressed with a self-locating 24-byte trailer at EOF, so the lister finds it with two seeks. Measured: **9 GiB / 20k-entry archive lists in 0.24 s vs 4.97 s** for the decompress walk — and the indexed time is O(index), not O(archive), so a 40+ GiB backup lists in ~0.2 s where the walk takes minutes on NVMe-limited hardware. Cost: ~6 bytes per member (119 KB for 20k entries) and no measurable create or extract time (alternating create runs land within noise; extraction skips the frame by spec). The index path reuses the walk's exact filter/format code, so listings — including MEMBER selection and `Not found in archive` errors — are byte-identical to the walk and to `tar -tvf`; uname/gname are stored 31-byte-truncated like the ustar fields to keep that guarantee. Robust fallbacks: no trailer (foreign/older archives), damaged/truncated trailer, data appended after the index, or a corrupt blob all fall back silently to the decompress walk (bounds-checked parse, 16 GiB decompressed-size sanity cap, and the skippable-frame header is verified to sit exactly where the trailer claims). `--no-index` opts out on create; `-t --tar` deliberately never trusts the index and always verifies the real stream. Groundwork for ROADMAP's parallel-dispatch extract route 2.

Two smaller `-l` fixes landed with this: plain `-l` now counts skippable frames in the `Frames` column like `zstd -l` does (the two only ever agreed before because gzstd archives had no skippable frames), and `-l --tar`'s decompress walk now shows the standard progress meter on stderr — like `-t`/`-d --tar` always did — when the listing is piped/redirected or `--progress` is given (never on the instant index path, and never interleaved with a listing that shares the terminal). Walk listings have streamed entries incrementally since v0.14.79; the meter now shows the decompress running underneath them.

## v0.14.79 — stream sizeless single-frame archives (tar --zstd): instant -l, constant memory

**Archives made by `tar --zstd` / `tar -I zstd` / piped `zstd` were pathologically slow to list, extract, or verify — fixed.** Those tools emit ONE zstd frame with NO content-size header (zstd can't know the size when reading a pipe). gzstd's router only sent frames with a *known-large* size to the incremental streaming decoder; a sizeless frame fell through to the parallel batch path, whose reader cannot size its output buffer, bails, and slurps the ENTIRE compressed file into RAM before decoding a byte. User-visible symptoms on a 40+ GiB backup: `-l --tar` printed nothing for many minutes (`tar -tvf` starts instantly) while RSS climbed toward the archive size. Measured on a 1.75 GiB reproduction: first listing line at 6.4 s with 4.3 GB peak RSS before; **0.4 s to first line and 47 MB RSS after** (12 GiB archive: first line still ~0.4 s). Fix: a shared `needs_stream_decode()` gate — seekable inputs whose first frame is huge *or sizeless* now route to `decompress_stream_from_file()` (4 MiB-chunk incremental decode, constant memory) in all four consumers: `-d --tar`, `-l --tar`, `-t --tar`, and plain `-d` (which also stops feeding the bogus −1 size to the progress meter and preallocator on sizeless input). gzstd's own multi-frame archives keep the parallel batch path (66 GiB archive still lists in ~9 s on the server). Verified: single-frame extract tree identical to GNU tar's, plain `-d` output byte-identical to `zstd -d`, structure `-t` valid.

## v0.14.78 — GNU-parity ownership restore (name-first chown) and byte-identical -l --tar listing

**Extraction as root now restores ownership by NAME first, GNU tar's default.** Previously the extractor always `fchown`ed the archive's numeric uid/gid — behaving as if `--numeric-owner` were permanently on — which puts files on the wrong accounts when restoring across hosts whose uids differ (the classic `postgres` is 118 here, 999 there). Now the stored uname/gname is looked up in the local passwd/group database (cached per name; archives repeat a handful of owners) and wins when the account exists; the numeric ids remain the fallback, and `--numeric-owner` on extract disables the lookup — exactly GNU tar's contract, now documented under `--numeric-owner` in `--help`. Also parses the POSIX pax `uid`/`gid`/`uname`/`gname` records (pax-format archives store ids > 2097151 and names > 32 bytes there; GNU tar honors them, gzstd previously ignored them). Verified with a `GZSTD_DEBUG_FAKE_ROOT` test hook (same pattern as the GPU fault-injection env): an archive carrying the runner's own user/supplementary-group names with bogus numeric ids lands on the correct local ids name-first, and on the raw ids under `--numeric-owner`.

**`-l --tar` output is now byte-identical to `tar -tvf`.** The old listing used fixed-width columns; GNU tar's `simple_print_header` does not — it keeps a sticky "user/group size" column that starts at 19 and grows to the widest line seen (all later lines shift), prints `major,minor` in the size column for device nodes, uses `h`/`C`/`V` type letters (with regulars whose stored name ends in `/` listed as `d`), appends ` -> `/` link to ` suffixes, and escape-quotes control bytes (`\n`, `\t`, `\303` …). Reimplemented all of it, plus the `%Y-%m-%d %H:%M` local mtime and numeric-owner fallbacks. Byte-diffed against real `tar -tvf` on: a local tree (setuid/sticky/fifo/hard+symlinks), crafted GNU and PAX archives with mid-listing column growth, device nodes, control-char names, >32-byte pax owner names, and a fragmented `--sparse` member (logical size shown, like tar). All identical; `--help` now states the guarantee.

Suite: 283/283 green (4 new parity tests; listing-format tests skip gracefully without python3).

## v0.14.77 — tar-compatible member selection and positional -C (extract, list, create); review cleanups

**`-d --tar ARCHIVE MEMBER...` and `-l --tar ARCHIVE MEMBER...` now select members, with GNU tar's exact semantics.** A member name picks the entry stored under that name; a directory picks its whole recursive subtree; trailing slashes are ignored; a selector that matches nothing is reported (`Not found in archive`) with a non-zero exit while the matching members still extract. `-C` between members is positional like tar's: it redirects the members that FOLLOW it (`gzstd -d --tar b.tar.zst -C /a src/x -C /b src/y`), and a RELATIVE second `-C` chains off the previous one exactly like GNU tar (which physically chdirs at each `-C` — verified side-by-side). Implementation: positionals after the archive on `-d`/`-l` become selectors, each remembering the `-C` in effect at its command-line position; the Extractor filters entries against the selector list (non-matching entries' data is skipped, keeping the parser aligned) and every path-resolving helper takes a root index so deferred work — writer-pool jobs, large-file parts, hardlinks, dirmeta — replays under the right `-C` root. All verified against GNU tar on the same trees: subtree selection, positional and chained `-C`, unmatched-name errors, hardlink-without-target failure, sparse member with holes intact, and `-l` filtering match entry-for-entry. Note the (previously undocumented) ability to pass multiple archives to `-d`/`-l --tar` is gone — the tail positionals are selectors now, as in tar; `-t --tar` keeps its multi-archive meaning.

**`--tar` create takes positional `-C` too, matching `tar -c`.** `gzstd -o my.tar.zst --tar --numeric-owner --acls --xattrs -C /dir1 user-data -C /dir2 sys1 sys2 sys3` reads each RELATIVE source from the `-C` in effect at its position but stores the member under the name as typed (`user-data/`, `sys1/`, … — no path prefixes), so one archive can gather trees from several roots. Absolute sources ignore `-C`, and a relative second `-C` chains off the previous one — both exactly as tar behaves (member listings verified identical to `tar -cf` with the same arguments, xattrs intact through a round-trip). Implementation rides the same parse-time mechanism as extraction's selectors: each post-`--tar` positional remembers the `-C` in effect, and the create driver hands `enumerate()` the `-C`-prefixed filesystem path with the unprefixed member name — the walker always supported split fspath/member naming, so no layout changes. Also added a `--tar` compatibility block to `--help` EXAMPLES (multi-root create, selective extract with positional `-C`, member-filtered listing).

**Review cleanups from the v0.14.76 audit follow-up.** (1) The OLDGNU sparse extension-block size lives in one place now — `entry_header_len()` — called by both `LayoutBuilder::add()` and `apply_extended_metadata()`'s recompute, so the two header-length computations can never drift again (the drift was the root cause of v0.14.76's corruption bug); verified byte-identical archives before/after. (2) `make_dir()` also widens a PRE-EXISTING directory lacking owner rwx (fd-based chmod under the O_NOFOLLOW walk), so re-extracting an archive over a populated destination is now idempotent even for read-only directories — GNU tar fails that re-extract; `finish_deferred()` still restores the exact stored mode. (3) Fixed a stale help note claiming extraction wasn't built in.

Full suite green after each change (268/268 before the features, 279/279 with the new selective-extraction and multi-root-create tests).

## v0.14.76 — two extraction/creation correctness fixes found in a pre-release audit

Swept the codebase before tagging a release, focused on `-d --tar` restore fidelity and `--tar` archive-creation correctness. Two confirmed bugs, both found by direct reproduction (not just reading):

**Directories with restrictive owner permissions broke extraction.** `Extractor::make_dir()` created a directory with its final stored mode immediately, so a directory lacking owner write/execute (read-only package caches are the common real-world trigger — e.g. Go's module cache) blocked every child underneath it: the leaf `mkdirat` succeeded, but the following `openat(O_CREAT)` for a file inside it (or `mkdirat` for a subdirectory) failed EACCES. Reported per-file (non-zero exit, named path) rather than silent, but left whole subtrees of a restore missing. Fix: create directories with owner rwx forced on (`mode | 0700`); `finish_deferred()` already reapplies the true stored mode/ACLs once the tree is fully populated, matching GNU tar's own create-permissive-then-fix-later approach.

**`--sparse` combined with `--acls`/`--xattrs` corrupted the archive for fragmented sparse files.** `LayoutBuilder::add()` correctly grows a sparse entry's `hdr_len` for the OLDGNU extension blocks needed once a file's sparse map exceeds 4 segments. But `apply_extended_metadata()` — the second pass that recomputes every entry's header offsets once PAX xattr/ACL records are known — rebuilt `hdr_len` from scratch and dropped that extension-block term, undercounting the header size for any sufficiently fragmented sparse file. The declared header length then disagreed with what `emit_header()` actually wrote, corrupting the stream: `-t --tar` reported CORRUPT / truncated tar stream, and extraction produced truncated output. Reproduced with a 6-segment sparse file plus `--xattrs`; fixed by adding the same extension-block term to the recompute loop. Verified byte-identical (`cmp`) after the fix, holes and xattr intact.

Full suite (268/268) green after each fix.

## v0.14.75 — cap the --tar decompress throttle (found by v0.14.74's clamp)

v0.14.74's batch clamp shrank the CPU queue floor (gpu_batch × streams), which
had been silently locking the 96 hybrid CPU threads out of the tail of the
queue — freed, decode finished ~60% through the run and raced the disk by the
FULL pipeline throttle budget (134 GiB in flight on the 8-GPU box), which the
writer then visibly drained for 44 s after the last worker exited ("[WRITER]
join took 44095 ms" on a quiet box).  Wall time was unchanged — the disk is
the clock either way — but holding 100+ GiB of decompressed frames to feed a
~3 GiB/s sink is pure RAM waste.

**`--tar` decompress now caps the auto throttle budget at 16 GiB of frames,
never below the GPU batch-floor deadlock guardrail** (`source=tar-extract` in
the -v line; the guardrail lands at 2048 frames / 32 GiB on the 8-GPU box, so
the same run holds ≤32 GiB and the drain tail drops proportionally).  cpu-only
budgets were already below the cap and are unchanged; user-pinned
--throttle-frames is untouched.  The cap must NOT go below the guardrail:
FrameThrottle::acquire is greedy — a stream holds a partial permit set while
waiting for the rest, and with delivered-but-unwritten frames pinning the
remainder the head-of-line frame can sit unpopped forever (all-or-nothing
acquire semantics would be needed to go lower).

## v0.14.74 — smooth GPU completion bursts at the in-order writer; quieter startup

The v0.14.73 server rerun proved the extract stage now saturates the device
(writer pool 86–95% busy) but GPU modes still trailed cpu-only ~10%: writers
starved 9.2% of the gpu-only run vs 1.0% cpu-only.  Cause: GPU completion is
BATCH-granular — a frozen batch of 64 lands 1 GiB of frames per stream at once,
then the strictly-ordered writer waits on the next head-of-line batch while the
sub-second downstream buffering (256 MiB sink + job queue) drains dry.  Two
mitigations, both compress and decompress:

- **Sink-limited freeze now clamps the batch DOWN, not just in place**:
  `min(best, max(16, best/4))` — sink-limited means GPU throughput has headroom,
  and batch size is purely the completion granularity at that point.  64 → 16
  (256 MiB bursts instead of 1 GiB), 256 → 64.  `gpu_desync_batch` keeps
  jittering below the clamp.  Applies to both the compress (writer-disk signal)
  and decompress (writer-disk or extract-backlog signal) tuners.
- **The `--tar` FrameSink deepens for GPU/hybrid decompress**: 256 MiB → up to
  4 GiB (clamped to a quarter of available RAM), enough to absorb one
  completion wave so the extractor keeps feeding the pool through head-of-line
  gaps.  cpu-only keeps 256 MiB (its arrival is near-continuous); the deque
  only holds frames when the consumer lags, so unused budget costs nothing.

Ceiling note: these bring GPU modes TOWARD cpu-only on a device-bound extract;
they cannot pass it (a GPU adds no speed to a disk-bound stage).  cpu-only
remains the `-d --tar` default.

Also: the per-device "[GPUn] ready, semaphore scheduling active" lines are
demoted to -vv — `[GPU] N device(s) online` already says it at -v, and eight of
them drowned the interesting startup output.

## v0.14.73 — parallel large-file extraction: windowed part jobs on the writer pool

v0.14.72's `[EXTRACT]` line immediately paid off on the server rerun: with the
small-file copies gone, the serial parse thread was 83–92% in the LARGE-file path —
byte-wise the archive is dominated by >4 MiB files, and that path was a single
parse-thread fill/zero-scan feeding a single persistent O_DIRECT writer (~one device
stream), with the 16-thread writer pool sitting 84% starved beside it.

Large files now go through the SAME writer pool as small ones, as 64 MiB windowed
zero-copy part jobs:

- The serial thread only creates + pre-truncates the file and hands out `DataSeg`
  views per window; all byte work (assemble, zero-scan, pwrite) runs on the pool
  threads for many windows and files CONCURRENTLY — parallel memcpy, parallel scan,
  and real device queue depth instead of one stream.
- One shared write fd per file (O_DIRECT under `--direct`); parts `pwrite` their
  absolute offsets — thread-safe, no per-part opens.  Pre-truncating to the full size
  makes zero-skips holes and the final length exact with no FINALIZE step; whichever
  part completes last applies metadata and closes the fd.
- O_DIRECT parts assemble ≤16 MiB chunks into a lazily-allocated per-writer-thread
  aligned bounce buffer and write coalesced non-zero 4 KiB runs; the file's sub-4K
  tail goes through a separate buffered fd (O_DIRECT can't write unaligned lengths).
  Buffered mode (no `--direct`) writes straight from the frame segments — no copy at
  all.  O_DIRECT-less filesystems (tmpfs) fall back to buffered parts per file.
- RAM stays bounded: the job queue budget scales to `writers × 2 windows` (256 MiB
  floor), so a huge file pins only its in-queue windows' frames, never the whole file.
- The entire single-lane `ld_*` pipeline (persistent writer thread, DJob queue,
  6×16 MiB staging pool, `[UNTAR-LARGE]` stats) is deleted; `[WRITER]` busy/starved
  now covers ALL file bytes, and the `[EXTRACT]` bucket becomes `large-dispatch`
  (view-building + create/truncate only).

Verified: round-trips byte-identical across buffered / `--direct` / `--no-sparse` /
GPU-hybrid on a large-file-heavy tree including multi-part files (up to 1 GiB),
window-edge sizes (64 MiB ± 1, +4096, +1234), interior zero runs, a 512 MiB
mostly-hole file (extracted with identical block count — holes preserved through the
O_DIRECT path), and a hardlink to a multi-part file.

## v0.14.72 — zero-copy tar extractor; extract-aware tuner freeze; honest extract diagnostics

Server profiling of a 319 GiB / 505k-file archive showed every `-d --tar` backend
converging on ~2–3 GiB/s with the writer pool 84% STARVED — the bottleneck was the one
serial parse thread memcpy'ing the entire uncompressed stream into writer-pool jobs,
while the `[WRITER]` verdict wrongly blamed the device.

**Zero-copy extractor.**  `Job::data` (a per-file `std::vector<char>` copy) is replaced
by `DataSeg` views into the decompressed frames: refcounted spans (`FrameBuf` +
pointer + length), one per frame touched (a file crossing a frame boundary gets two).
The serial thread now only parses headers and hands out views; the writer pool writes
straight from the frames (per-segment `pwrite_sparse` at a running offset — sparse-hole
detection is offset-based, so per-segment stays correct on freshly created files).
Frame lifetime: a frame is freed when the last job referencing it is written; with the
job queue bounded at 256 MiB of logical bytes the pinned overhang is ~one frame per
queued job.  The fd-mode StreamReader (no frames) falls back to one owned, copied
segment behind the same interface.

**Extract-aware sink-limited freeze (GPU decompress tuner).**  The freeze keyed solely
on `writer_disk_ns`, which `--tar` extract never feeds (the untar pool tracks time in
its own counters) — so on extract-bound runs the tuner saw "writer 0% busy" and grew
batches to 512 against an in-order sink (deep head-of-line, 22 s end-of-run join).
The freeze now also fires on the in-order writer's time blocked pushing into the
extract FrameSink (`producer_wait_ns`), reported as "extract backlog N%".

**Honest diagnostics.**  New `-v` `[EXTRACT]` line breaks down the serial parse
thread's time: sink-wait (decompress behind) | dispatch-blocked (writer pool/device
behind) | inline-meta | large-file | parse+dispatch (its own irreducible work).  The
`[WRITER]` verdict now picks between three evidence-backed attributions instead of
always blaming the device: decompress-bound / write-side-bound (pool busy or parse
thread dispatch-blocked) / SERIAL-parse-thread-bound (pool idle, device had headroom)
— each with the numbers inline.

## v0.14.71 — remove dead `busy` machinery from the decompress GPU worker; test-count constants

Cleanup fallout from v0.14.70's analysis.  The decompress GPU worker's `busy` flag was
**write-only false**: every batch completes inline (H2D → kernel → sync → D2H → deliver)
within one loop iteration, so no stream is ever in flight at intake time.  The flag and
everything guarding on it were dead code inherited from the compress worker's old poll
design:

- `DecompStreamCtx::busy`, the `if (C.busy) continue` skip, the `any_busy` tail-yield
  guard, the `self_busy` non-blocking intake branch, and the `all_idle` termination scan
  are deleted; decompress intake now always uses the blocking permit-free wait (which is
  what actually executed all along).
- `TaskQueue::ready_for_batch_or_cap` and `FrameThrottle::try_acquire` had exactly one
  caller each — that dead branch — and are deleted with it.  No behavior change; the
  compiled hot path is identical to what really ran before.
- `gzstd-test.sh`: EXPECTED_TESTS drift-check constants updated (257 → 268 default,
  355 → 366 extensive) — the suite had grown 11 tests since the constants were last bumped.

## v0.14.70 — event-driven GPU completion (ROADMAP 1.10): drain thread replaces the poll loop

The compress GPU worker's completion poll (`cudaStreamQuery` + yield, plus the
synchronous-drain fallback) is gone.  Each device now runs TWO threads:

- **Worker (intake/submit):** waits for an idle stream, pops a batch (wait
  without permits → acquire → non-blocking pop, unchanged v0.14.58 invariant),
  submits H2D + kernel (+ in-VRAM verify), records a new per-stream `ev_done`
  event (`cudaEventBlockingSync`), and pushes the stream onto a FIFO.
- **Drain:** pops the FIFO in submit order, parks in
  `cudaEventSynchronize(ev_done)` — an OS-level block, not a spin — then does
  readback + delivery (`gpu_drain_batch`) and returns the stream to the idle set.

Why this shape:

- **No spin anywhere.**  Every wait is a CV or a blocking-sync CUDA event.  The
  poll loop and its `yield()` are deleted.
- **The v0.14.60 starvation class is structurally gone.**  Blocking in intake
  cannot starve completion because completion runs on its own thread; the
  self-busy non-blocking intake special case is deleted with it.
- **FIFO drain is writer-optimal and strengthens the deadlock-freedom
  argument:** submit order is pop order is seq order, so the in-order writer's
  head-of-line frame is always at the front of some device's drain FIFO.
- **One completion path.**  The async-poll and sync-drain twins (whose perf
  recording had to be kept in sync by hand) collapse into `gpu_drain_batch`.
- **Events, not host callbacks:** `cudaLaunchHostFunc` is documented to NOT run
  when the batch faults — every GPU fault would become a hang.  A drain thread
  synchronizing on the event sees the error and routes it to the abort path.
- **Intake and D2H readback now overlap** (they serialized on one thread before).

Error handling: the drain thread runs the full abort protocol itself (raise
`g_gpu_aborted`, `set_done` the queue and throttle, wake the writer) so a worker
blocked in intake always wakes; the worker rethrows the drain error into its
existing catch, which joins the drain thread before freeing stream buffers.
Two hardening fixes that fell out of the analysis: `wait_for_gpu_yield` now
bails on `g_gpu_aborted` (a parked worker could previously sleep through an
abort forever since CPU workers exit without draining the queue), and
`StreamCtx::acquire_out_buf` returns immediately on abort instead of waiting
for writer drains that will never come.

Watchdog: worker phases gain `wait_idle_stream`; the drain thread heartbeats
separately (`drain_phase`/`drain_heartbeat`/`drain_state` in the dump), and the
per-stream `last_cuda_query` field is now `last_event_sync`.

Decompress is untouched: its GPU worker synchronizes inline per batch (required
by `GetTempSizeSync`) and has no poll loop to replace.

Verified on the workstation: 20 GiB round-trips byte-identical across
gpu-only/hybrid/cpu-only; pinned `--gpu-batch`/`--gpu-streams` deadlock shapes
clean under `--watchdog`; simulated GPU fault (`GZSTD_DEBUG_FAIL_GPU_AFTER`,
which now throws on the drain thread) still aborts + rebuilds CPU-only,
byte-identical.  (`--verify-engine=gpu` cannot initialize on the 11 GiB
workstation cards — pre-existing VRAM limit, confirmed identical on v0.14.69.)

## v0.14.69 — `--watchdog=SECS`; strip the now-obsolete test hooks

Follow-up cleanup to v0.14.68.

- `--watchdog` now takes an optional timeout: `--watchdog` = 30 s (unchanged),
  `--watchdog=SECS` sets the stall timeout (0 disables the check).  Only the `=` form
  takes a value, so it can never swallow a following filename.  This replaces the
  `GZSTD_WATCHDOG_SECS` env var, which is removed.
- Removed the two temporary reproduction hooks whose jobs are done: `GZSTD_DEBUG_FREEZE_
  WRITER_AFTER` (freeze the writer to exercise the watchdog without a real hang) and
  `GZSTD_DEBUG_SLOW_PRODUCER_US` (throttle the producer to reproduce the v0.14.60
  completion-poll starvation).  Both deadlocks are fixed; the hooks were only ever manual
  diagnostics and are trivial to reconstruct if a future wedge needs them.
- The `g_debug_freeze_writer_after` global and its writer freeze-point are gone.
- Test hooks that remain are the ones the suite actually uses: `GZSTD_DEBUG_CORRUPT_FRAME`
  / `_PERSIST` (verify tests) and `GZSTD_DEBUG_FAIL_GPU_AFTER` (GPU-fault rebuild).

## v0.14.68 — demote the deadlock watchdog to opt-in `--watchdog` (diagnostic)

The GPU deadlock watchdog (v0.14.59) that captured the v0.14.60 hybrid hang was still
**armed by default** on every hybrid/GPU compress run.  With the deadlocks fixed
(v0.14.58/60) it no longer needs to run unconditionally.  It is kept — it is the fastest
way to turn a future wedge (marginal hardware, a new nvCOMP, an unfamiliar box) into a
JSON evidence dump — but is now **off unless you pass `--watchdog`**, and mentioned only
in `--help` alongside `--cold` as a diagnostic.

- New `--watchdog` flag (default off): arms the stall detector for the run.  On no
  writer progress for the timeout (30 s default, `GZSTD_WATCHDOG_SECS` to tune) while the
  run is not done, it dumps the per-worker / per-stream / queue / throttle / NVML /
  kernel-Xid snapshot and hard-exits.  Does not recover — it is evidence capture.
- The `wd_*` instrumentation stays inline in the GPU workers but is a cheap no-op while
  `g_wd` is null (the default), so the normal path is unaffected.
- `GZSTD_DEBUG_FREEZE_WRITER_AFTER` (exercise the watchdog without a real hang) and
  `GZSTD_DEBUG_SLOW_PRODUCER_US` remain as undocumented test hooks.

## v0.14.67 — honest `[VERIFY]` verdict: stop crying "fell behind" when the sink hid it

v0.14.66 validated on the 8-GPU box: `[WRITER] starved 0.0%`, `write-path busy 83.1%`,
verdict "output device saturated — optimal", and `4.03 GiB/s` — i.e. **at the no-verify
disk ceiling**.  Yet the `[VERIFY]` line still said `fell behind: throttled compression
6635x for 6.81 s`.  Both can't be the headline: if the writer never starved, the sink
never idled, so those 6.81 s of compress-side back-pressure were fully overlapped by disk
writes and cost **zero** wall-time.  (The submit-burst and full queue persist because the
writer pushes to the verify queue at memory-copy speed, always faster than any number of
decompress threads — so the queue caps during a burst regardless.  Harmless when the
writer never starves.)

Fix: the verify verdict now consults the writer's starvation fraction (`Meter::
writer_starved_ns`, the same signal behind the `[WRITER]` line):

- writer never starved (< 2%): `applied back-pressure Nx (T s) but the writer never
  starved — overlapped disk writes, not the end-to-end limiter`.
- writer starved ≥ 2%: `fell behind … starving the sink X% — verify capped throughput`
  (the honest "verify is the real bottleneck" case — a fast-disk / slow-CPU box).
- no meter / no stalls: unchanged.

No pipeline change; this only corrects the reporting so `[VERIFY]` and `[WRITER]` agree.

## v0.14.66 — feed --verify AFTER the write, not before (kill the verify burst at its source)

v0.14.65's de-sync **falsified** the theory it was built on: breaking GPU-completion
lockstep dropped `[WRITER] head-of-line` from 4.2% to 0.1% but left `max submit-burst`
pinned at ~2062 against the 2048 queue.  So the burst was never synchronized GPU
delivery.  Tracing it to `writer_thread`: the writer drains **all** consecutive ready
frames from the ResultStore into one batch, then submitted **every** frame to the verify
pool in a tight loop *before* writing any of them.  On a disk-bound run a big contiguous
run accumulates while the previous batch is on the (slow) sink, so that pre-write dump
overflowed the bounded verify queue and the writer ping-ponged against it (12069 stalls,
14.4 s) — **idling the sink while verify caught up.**  The measurements also show verify
was never the bottleneck: `drain 2.76 / 0.85 per-thread` ⇒ only ~3 of 17 threads busy on
average; its real capacity (~14 GiB/s) dwarfs any single device's sink rate.  A
delivery-shape defect, not a verify-speed one — and it only appears when the sink is the
bottleneck, so the fix is general, not box-specific.

Fix (Option A, "pace the feed"): submit each drained batch to verify **after** handing it
to the write backend.  `AsyncWritePool` is double-buffered — `submit()` blocks until the
*previous* batch is written — so by the time the writer feeds verify, that stage runs
**concurrently with the current batch's disk write**.  Verify drains the batch with huge
margin (14 GiB/s vs a ~3 GiB/s sink) and the queue is empty again before the next batch,
so the pre-write burst never forms and the sink is never starved.

- **Root fix, zero extra memory:** no deeper queue, no tunable cap, no box-specific
  constant.  The verify queue depth is unchanged.
- **Correctness unchanged:** the pool holds a `shared_ptr`, so a frame outlives its write
  regardless of submit order, and a verify failure discards and rebuilds the whole output
  either way.  The test-only corruption hook still fires before the write.
- **Degrades correctly on slow-CPU (verify-bound) boxes:** `submit()` still blocks as
  real back-pressure, but the sink already holds the batch to write, so it does not idle.
- Verify with `-v --verify`: `max submit-burst` should now be small and the
  `throttled compression … x` count should collapse toward zero on a disk-bound run.

## v0.14.65 — de-sync sink-limited GPU batch completions (fixes writer head-of-line)

NOTE: this was aimed at the `--verify` burst and **missed** — v0.14.66 found and fixed the
real cause.  It stays in because it fixes a *different*, real problem: writer head-of-line
stalls.  On a multi-GPU box, identical GPUs running the same frozen batch size finish in
**lockstep**, so the in-order writer stalls on one straggler while a contiguous run of
frames buffers.  Measured effect: `[WRITER] head-of-line` dropped from 4.2% (avg 1281
frames stuck) to 0.1% (avg 30).  It did NOT move `max submit-burst` (still ~2062) — which
is what falsified the "synchronized delivery causes the verify burst" theory and pointed
v0.14.66 at the writer's pre-write submit loop instead.

Fix: `gpu_desync_batch()`.  Once the writer is **sink-limited** (the tuner's `frozen`
latch), batch size no longer moves throughput, so every batch is jittered to a random
size in `[batch/2, batch]`.  Randomizing per batch — rather than giving each device a
fixed offset — matters: fixed distinct periods drift back into a beat pattern and
re-synchronize, whereas random jitter breaks lockstep permanently.  Completions then
scatter into a steady stream instead of a wave, so the writer flushes small contiguous
runs (the head-of-line win above).

- **Head-of-line-safe:** the jitter only ever *shrinks* a batch (upper bound is the
  frozen size), never grows it, and never exceeds the per-stream buffer.
- **Output-deterministic:** batch size only controls how many independent frames a launch
  grabs; frame contents and the archive bytes are identical regardless.
- **General, not machine-tuned:** gated purely on the sink-limited signal — while the
  auto-tuner is still probing it measures clean same-size batches, and compute-bound runs
  (where batch size still governs GPU throughput) keep the uniform throughput-optimal
  batch.  No box-specific constants; a phase de-correlator that helps any multi-GPU box
  (or even multi-stream single-GPU) whose writer is the bottleneck.
- Applies symmetrically to the compress and decompress GPU worker paths.

## v0.14.64 — correct the verify "memory-bandwidth" misdiagnosis; add a burst probe

Two things.

- **Corrected wording.**  v0.14.60–v0.14.63 described the verify-thread plateau as a
  "memory-bandwidth plateau."  That was wrong, and the numbers refute it: verify at
  ~2.7 GiB/s moves ~10 GiB/s of memory traffic, i.e. **~2–3% of a 256-core box's
  300–600 GB/s** — RAM is nowhere near the wall.  And the `[VERIFY]` line's own figures
  (`16 thread(s), drain 2.70 / 0.87 per-thread`) show verify did the work of **~3 busy
  threads** with ~13 idle — 5× spare capacity, so it is not resource-bound at all.
  `helped` halts thread growth simply because the drain rate already equals production;
  more threads can't lift a production-limited rate.  Comments and CHANGELOG fixed.

- **Burst probe.**  The real reason hybrid `--verify` back-pressures (and cpu-only does
  not) appears to be *bursty* delivery: if the 8 GPUs finish batches in lockstep, the
  in-order writer drains a wave and dumps it on verify at once, briefly overflowing the
  queue — even though verify has spare capacity to drain it.  The `[VERIFY]` line now
  reports `max submit-burst N frames` (longest run of submits <100 µs apart).  A large N
  on the 8-GPU box confirms synchronized GPU delivery (then the lever is staggering the
  GPU batch completions / varying per-GPU batch size); a small N means the queue fills
  for another reason.  Measure before building the de-sync.

Practical note: on a disk-bound workload the GPUs buy no throughput, so `--verify
--cpu-only` is the fast path (smooth delivery, no back-pressure).  Diagnostic + wording
only; no effect on output.

---

## v0.14.63 — deeper verify queue to absorb hybrid write-bursts

The v0.14.62 hybrid thread-cap raise confirmed the diagnosis rather than fixing it:
on the 8-GPU box verify grew from 16 to **17** threads and then `helped` halted it
(a *drain* plateau — `drain` stayed 2.75 GiB/s because verify already matches
production; more threads can't lift a production-limited rate — **not** memory
bandwidth: verify uses ~2–3% of the box's RAM bandwidth), while `peak backlog` stayed
**pinned at 1024/1024**.  So the limiter is the **verify queue depth**, not the thread
count: the queue held only 16 GiB (1024 frames) while the hybrid throttle allows 134 GiB
in flight, so the disk's write-bursts filled it and back-pressured the writer
(`throttled compression 13020x for 13.18 s`) even though verify's aggregate rate nearly
matched the disk.

Raised the verify-queue sizing from ~6%/16 GiB to ~12%/32 GiB of available RAM (still
RAM-proportional, still honoring `--memlimit`), which absorbs the bursts.  This is the
"queue lever" flagged in v0.14.62.  The verify queue is post-write RAM held separately
from the throttle, so the sizing stays proportional to keep small-RAM boxes safe.  No
effect on output or on what gets verified.

---

## v0.14.62 — `--verify` polish: hybrid thread cap, output-line fixes

Follow-ons to v0.14.61, from a 319 GiB real-data run.

- **Hybrid verify cap.** On the 8-GPU box, verify still fell behind: it hit the
  cpu-only ceiling of 16 threads (`peak backlog 1024/1024, throttled compression
  14274x`) because the hybrid producer is much faster than cpu-only.  In hybrid the
  GPUs carry the compression, so the CPU pool has spare cores — raised verify's
  ceiling there to `cores/8` (≤32) vs `cores/16` (≤16) for cpu-only.  The rate-matched
  `helped` guard in `maybe_grow` still halts growth once an added thread stops raising
  the drain rate, so the higher ceiling only spends threads that actually help.
  (v0.14.63 note: this raise confirmed rather than fixed it — the limiter turned out to
  be bursty GPU delivery overflowing the verify queue, not thread count.)
- **`[VERIFY]` on its own line.** It was being appended to the live progress bar's
  last in-place (`\r`) update; now printed on a fresh line.
- **Correct source name in the summary.** `--tar` synthesizes its input, so the
  compress summary labelled it `(stdin)` even when reading a directory tree — it now
  shows the actual source path(s), matching the extract summary.

Cosmetic + tuning only; no effect on output or on what gets verified.

---

## v0.14.61 — `--verify` no longer serializes the pipeline to one thread

`--verify` on 300+ GiB of real data ran ~3.5× slower than the same compress without it
(4m03s vs 1m10s), yet the `[VERIFY]` line claimed `kept up (compression never waited on
verify)`.  Both the slowdown and the false "kept up" traced to the same wrong reference
point.

The verify tap holds a `shared_ptr` copy of each finished frame until it is checked, so
the compressor's output buffer isn't recyclable until then — verify back-pressures
*compression*.  The `VerifyPool` decides to add a thread only when `backlog >
max_queue/2` (here 512), but the frame throttle caps in-flight frames at ~384 (peak
backlog 373) — **below** that mark.  So the grow trigger was **unreachable**, the pool
sat on its **first thread**, and verify silently capped a 4.6 GiB/s pipeline at one
thread's ~0.9 GiB/s (243 s ≈ 221 GiB ÷ 0.91).  The `[VERIFY]` verdict compounded it:
`kept up` was gated on whether the *writer* blocked on a *full verify queue* (it never
did — the queue is bounded from below by the throttle), which says nothing about the
back-pressure on compression.

- **Scaling fix:** grow while verify is *saturated* — `backlog > nthreads*4` (reachable
  under back-pressure) instead of `> max_queue/2`.  The existing `helped` guard (a new
  thread must raise the drain rate >10%) still halts growth once an added thread stops
  raising drain, so it scales to match the producer and no further.  Removed the now-dead
  `high_water_`.
- **Honest verdict:** the summary now judges by worker saturation (`verify_ns /
  (wall × threads)`), not the writer-stall count — `verify-bound: … N% busy` when verify
  capped throughput, `kept up (workers N% busy; not the bottleneck)` otherwise.

Measured (24-core box, cap 2, 1.8 GiB incompressible → RAM): the pool now grows to 2
threads and `--verify` costs ~5% (0.94 s vs 0.89 s) instead of stalling on one thread;
verdict reads honestly.  On the 256-core box `vmax=16`, so it will scale to the handful
of threads that match the disk.  No effect on output or on what gets verified.

---

## v0.14.60 — fix the real hybrid deadlock: completion-poll starvation

The v0.14.59 watchdog caught the intermittent hang in the act on the 8-GPU box and
named the root cause unambiguously.  The dump: **all 8 GPU workers BLOCKED** (heartbeat
frozen) in `intake_wait_batch`, **several streams busy 7–32 s** holding the writer's
head-of-line frames, the throttle only **~14% used** (`in_flight 350 / 2432`,
`block_count 0`), GPUs **idle at 0% util, no Xid**.  So it was never a GPU fault and
never the permit deadlock — it was our own pipeline.

**Root cause.** One worker thread serves all of a device's streams, looping
*intake → completion-poll*.  With a slow producer (the trigger was `--tar --xattrs
--acls` over a huge small-file home dir, which the parallel assembler feeds slowly) and
a pinned `--gpu-batch`, the queue sits below the batch size, so the worker **blocks in
`wait_for_batch_or_cap` for an idle stream — and never reaches the completion poll for
its *other*, busy streams.**  Their kernels finish but are never drained, so the
in-order writer wedges behind a finished-but-undelivered batch and the run freezes.
Long-standing: the pre-refactor `pop_batch_greedy` blocked the same way under a pinned
batch; v0.14.58 fixed the orthogonal permit-hoarding but not this.

**Fix.** A worker must never block in intake while it still has an in-flight stream to
drain.  Both GPU worker loops now check `self_busy` (any of this device's streams in
flight); if so they intake **non-blockingly** (`ready_for_batch_or_cap` +
`try_acquire`) and otherwise fall straight through to the completion poll — they only
take the blocking `wait_for_batch_or_cap` when nothing of theirs is in flight (so there
is no poll to starve).  This guarantees in-flight batches are always drained promptly,
so the writer can't wedge behind one.  New non-blocking primitives: `TaskQueue::
ready_for_batch_or_cap` and `FrameThrottle::try_acquire`.

**Deterministically A/B-proven.**  A temporary env hook `GZSTD_DEBUG_SLOW_PRODUCER_US`
(µs/frame throttle in the tar assembler) reproduces the slow-producer starvation on a
warm-cache box.  With `--gpu-only --chunk-size 1 --gpu-batch=16 --gpu-streams=8` at
150 ms/frame, the **pre-fix** build wedges on the first run with the exact production
signature — worker `BLOCKED` in `intake_wait_batch`, streams busy in a staircase
(s0 13 s … s5 1 s, all unpolled), throttle only 16% used — while the **fixed** build
completes cleanly (gpu-only and hybrid) under identical conditions.

The v0.14.59 diagnostic watchdog is **retained** for now so the fix can be confirmed on
the box that actually triggers the hang (if it ever fires again, it dumps a fresh
snapshot); it will be removed once validated.  No effect on output.

---

## v0.14.59 — TEMPORARY deadlock-diagnostic watchdog (compress GPU path)

A `-vv` trace of a fresh hang on a **quiet** 8×H100 box (gzstd the only GPU user,
no Xid, GPUs clean afterward) showed the v0.14.58 permit fix did **not** cover this
failure: it's a different mode.  One GPU stream took a low-seq batch
(`[GPU1/S0] take batch=32 seq=[6131..6162]`) and **never completed** — the worker
kept polling `cudaStreamQuery` (NotReady), the in-order writer wedged behind those
frames, the CPU raced ahead then stopped on the `queue_floor` reservation, and the
run froze (`[HYBRID] tick … 0/0`).  The throttle was only ~5% used, so it is **not**
the permit deadlock.  The existing GPU-fault→CPU-rebuild path can't catch it because a
*hang* never returns a CUDA error.

To stop guessing, this adds a **temporary, diagnostic** watchdog (no recovery — it is
not a fault/rebuild).  When the writer makes no progress for `GZSTD_WATCHDOG_SECS`
(default 30; 0 disables) **while work is still pending** (frames in flight or queue
not drained — so end-of-run fsync/rename never false-fires), it dumps a JSON snapshot
to `./gzstd-deadlock-<pid>-<time>.json` and hard-exits.  The snapshot is built to
*definitively* separate a real GPU stall from a software wedge:

- per-worker **heartbeat** sampled 500 ms apart → `SPINNING` (a stream that never
  completes) vs `BLOCKED` (parked in our own CV/lock), plus the worker's current phase;
- per-stream `busy` + last `cudaStreamQuery` result **and its age** → kernel genuinely
  NotReady (running/hung) vs finished-but-stuck-downstream;
- throttle (in-flight/max), queue (size/drained), NVML per-GPU util+memory, and a scan
  of the kernel ring buffer for **Xid/NVRM** lines (`journalctl -k`, dmesg fallback).

`GZSTD_DEBUG_FREEZE_WRITER_AFTER=N` freezes the writer after N frames to exercise the
watchdog without a real hang (validated: fires + dumps; a healthy run does not).  This
code is explicitly temporary — it exists to capture the real wedge so it can be root-
caused, after which it (and the planned CUDA host-callback/event redesign that removes
the completion-poll yield) replace it.  Diagnostic only; no effect on output.

---

## v0.14.58 — structural fix for the intermittent GPU permit-hoarding deadlock

The long-standing intermittent hang on pinned `--gpu-batch`/`--gpu-streams` runs is
fixed at the root.

**Mechanism.** A GPU stream acquired its *whole* batch of throttle permits up front
(`acquire(pop_n)` *before* popping), and `FrameThrottle::acquire` is incremental
("take what's available, wait for the rest") — so a stream grabs partial permits and
**blocks holding them**. In "locked" mode (set whenever you pin `--gpu-batch`) the
stream then **slept in the batch pop waiting for a full batch while still holding
`pop_n` permits**. The throttle budget is sized to exactly the aggregate GPU demand
(`devices×streams×batch`) with no headroom, so when every stream hold-and-waited at
once, all permits were sequestered and the in-order writer wedged behind a head-of-
line frame no one could pop. It was intermittent because it needs a transient where
all streams wait at once with the bounded queue below the aggregate threshold — a
function of producer rate, VRAM-reduced batch sizes, and startup timing, i.e.
load/contention.

The previous patches were point fixes that each covered one `(mode × reader)` combo
and left holes — the compress hybrid guard only armed for the pooled reader (its
`queue_depth_cap` is unset for the **mmap** and **`--tar` assembler** producers, and
absent in **gpu-only** where there's no scheduler), and the decompress gpu-only
soft-min was a parallel special case. So pinned-batch `--tar` compress and gpu-only
compress in particular could still wedge.

**Fix.** Both GPU worker loops (compress and decompress) now **wait for the batch
without holding any permits, then acquire and non-blocking-pop**: `wait_for_batch_or_cap(pop_n)`
→ `acquire(pop_n)` → `try_pop_batch_signal(pop_n)` → release excess. A stream never
sleeps while holding permits, so it can't sequester the budget. This is sound because
the producer never touches the throttle (it's bounded only by the queue), so a
permit-free waiter is always eventually fed; the writer drains the frames whose
holders *do* have permits, releasing them. `wait_for_batch_or_cap` also returns at the
queue's bounded capacity, so a stream never waits for a batch larger than the queue
can supply (`--gpu-batch` stays the pop size when the queue can fill it, and is
silently capped when it can't). This removes the need for the conditional full-batch-
wait guards and the gpu-only soft-min special case — both deleted; mirrors the CPU
worker's proven wait→acquire→try-pop→release-on-miss pattern, now batched.  The
now-unused `pop_batch` / `pop_batch_greedy` queue methods are removed (the latter's
`while (size < min_n) cv_.wait` — wait-for-a-full-batch-while-the-caller-holds-permits
— was the buggy model this replaces).  All waits remain pure CV/predicate, no polling
or fixed sleeps; the new path holds permits for a *shorter* window (acquire→pop is
immediate, no batch-ready wait while holding), which slightly improves throttle
utilization rather than costing anything.

Validated: the exact reproducer configs (`--hybrid`/`--gpu-only` compress and
decompress, plain and `--tar`, with pinned `--gpu-batch`/`--gpu-streams` and the tight
`--throttle-frames=1`) all complete with byte-identical round-trips, including a 15×
repeat of the worst case; full suite green.

---

## v0.14.57 — `[READER]`/`[WRITER]` bottleneck diagnosis for `-d --tar` extraction

`-d --tar` extraction returns early via `extract_tar()`, bypassing the per-file loop
that prints the `-v` `[READER]`/`[WRITER]` bottleneck diagnosis — so unlike plain `-d`,
a tar extract gave no definitive answer to "what capped throughput?".  Added the same
class of diagnosis, adapted to the extract pipeline (decompressors → in-memory
`FrameSink` → parallel `Extractor` write pool):

- **`[READER]`** — factored the plain-`-d` reader breakdown into a shared
  `print_reader_diag()` and call it from both paths.  The shared decompressor
  populates the same reader counters, so this is identical: io / parse / task-copy /
  blocked-downstream, per thread.

- **`[WRITER]` busy/starved** — the tar writer is the `Extractor`'s pool (default 16
  threads), not the single in-order `DirectWriter`, so its counters aren't in the
  meter.  Instrumented the pool's `writer_loop` with **per-thread local accumulators**
  (busy = time in `write_small`; starved = time blocked waiting for a job), flushed
  with one atomic add at thread exit — no shared-counter contention across the 16
  writers.  The persistent large-file O_DIRECT writer's existing `write`/`jobwait`
  timing folds in when large files were written.  Reported as a per-thread average,
  like `[READER]`.  Gated to `-v`: with no `-v`, the timing `now_ns()` calls are
  skipped entirely (zero cost).

- **`[WRITER]` verdict** — read straight off the sink seam: the producer
  (decompressors) blocking on a full sink ⇒ extract/write-bound; the consumer
  (`Extractor`) blocking on an empty sink ⇒ decompress-bound (the engines, not the
  device, capped output).  This is the same producer/consumer-wait signal previously
  shown only at `-vv` in the `[SINK]` line, promoted to a `-v` verdict.

The writer busy% and the sink verdict cross-check each other (idle pool + consumer-
wait ⇒ decompress-bound; saturated pool + producer-wait ⇒ write-bound), giving a
definitive read.  Profiling-only; no effect on output or throughput.

---

## v0.14.56 — fix bogus D2H/total times on the `-vv` GPU batch line; `-vvv` per-chunk `in=`; human-readable byte totals

At `-vv` (`V_DEBUG`) the per-batch `[GPU/S] done batch=…` line reported a wild `d2h=`
(and therefore `tot=`) — e.g. `d2h=736848192.52ms`, which is the machine's whole
monotonic clock, not a transfer time.  Cause: the async-polling completion path only
captured the D2H start timestamp when the `-vvv` perf breakdown was active
(`d2h_t0 = g_perf ? now_ns() : 0`), so at `-vv` it stayed 0 and `now_ns() - d2h_t0`
returned the full clock.  Batches that happened to finish via the sync-drain path
(which sets the timestamp unconditionally) looked fine, hence the mix of sane and
garbage lines.  Now the start is captured whenever either `g_perf` **or** `-vv` is on,
and the elapsed computation is guarded (`d2h_t0 > 0 ? … : 0`) — mirroring the
decompress path, which already did this.  The bogus value also fed the per-stream
`total … time=` accumulator, so that line is corrected too.

Same root shape, separate spot: the `-vvv` per-chunk trace (`[GPU/S] chunk seq=…`)
printed `in=0.00 B` for every chunk.  It read the input size from `C.batch[i].len()`,
but by completion the host input buffer has been moved out (only the `seq` metadata
survives), so the length reads 0.  It now uses the saved `C.h_in_sizes[i]` array — the
same source the batch `in_sum` already used — so `in=` shows the real chunk size.

Also switched the `[GPU/S] total …` line's `in=`/`out=` from raw byte counts
(`in=8589934592B`) to human-readable units (`in=8.00 GiB`), matching the per-batch
line.  Profiling-only (`-vv`/`-vvv`); no effect on output or throughput.

---

## v0.14.55 — native Blackwell cubins for the verify kernel; Reader line counts `--tar` reads

Two follow-ons to v0.14.53.

**Native Blackwell cubins (toolkit-gated).**  Added `sm_100` (Blackwell datacenter)
and `sm_120` (Blackwell consumer) to the verify kernel's native SASS set, so those
cards skip the one-time PTX JIT.  These cubins can only be emitted by nvcc ≥ 12.8, so
they are gated on the detected compiler version — on an older toolkit they are dropped
and Blackwell still runs via the `compute_75` PTX, so the build never fails and
coverage is unchanged.  Mechanically, the architecture list moved from a
`CMAKE_CUDA_ARCHITECTURES` set *before* `enable_language(CUDA)` to a
`CUDA_ARCHITECTURES` **target property** set after it — the only point where
`CMAKE_CUDA_COMPILER_VERSION` is known, which the toolkit gate needs.  A user
`-DCMAKE_CUDA_ARCHITECTURES=…` override is still honored (detected before
`enable_language`, before CMake seeds its own default).  The portable release build
(CUDA 12.6 container) gates the Blackwell cubins out but keeps full coverage via PTX
JIT; bump that container to CUDA ≥ 12.8 to bake them in — a startup-JIT optimization,
not a coverage change.

**Reader line now counts `--tar` create member reads.**  The `-vvv` PERFORMANCE
BREAKDOWN reported `Reader: 0.000 s (0.00 GiB)` for a `--tar` create, which looked
like the reads were free.  They weren't measured: `--tar` create uses the parallel
assembler (`tarx::assemble`) as its producer and returns before the
streaming/mmap/pooled reader block that owns the `read_ns` / `read_bytes_total`
counters — so the member `pread`s in `read_seg` never touched them.  Now the
member-read loop accumulates into those counters.  The figure is aggregate across the
assembler's parallel reader threads (so it can exceed wall time and the reported rate
is the per-thread average) — the same convention as the `CPU compute` line.  Only real
file bytes are counted; the synthesized tar headers and padding are not disk I/O.
Profiling-only (`-vvv`); no effect on output or throughput.

---

## v0.14.53 — GPU-verify kernel runs on every supported GPU, with a CPU fallback

The `--verify-engine=gpu` byte-compare kernel (`gpuverify.cu`) was compiled for only
two architectures — `sm_75;sm_90`, exactly the two test boxes — and with **no PTX**.
On any other card (Ampere `sm_80/86`, Ada `sm_89`, in-between parts like `sm_87`, or
any future Blackwell+) the kernel launch would fail with no-image-for-device, so
`--verify-engine=gpu` aborted the whole compress on hardware it had never been built
for.  Two fixes make it broadly portable:

- **Broadened the compiled image set** to native SASS for Turing→Hopper
  (`75/80/86/89/90-real`) plus a low virtual-arch PTX (`75-virtual`).  PTX is
  forward-compatible: a `compute_75` image JITs onto **any** device with capability
  ≥ 7.5, so the kernel now runs on every GPU nvCOMP itself supports — including
  architectures released after this binary — without re-listing each one.  The real
  cubins are just a JIT-skip optimization for the common cards.  (Also fixed a latent
  CMake bug: the arch list was `set()` *after* `enable_language(CUDA)`, which seeds a
  cached `sm_52` default — so the intended defaults had silently never applied.  It is
  now set before `enable_language`.)

- **Graceful CPU fallback.**  Before committing to GPU verify, a one-shot probe
  (`gzv_kernel_available`) does a trivial kernel launch on the device; if there is no
  compatible image it quietly demotes to the CPU `VerifyPool` (which covers every
  frame) and warns, instead of aborting mid-run.  Defense-in-depth — with the PTX
  fallback in place this should essentially never fire, but a verify path must fail
  safe, not hard.

No release-pipeline change: the portable build inherits the new architecture default
(it passes no `-DCMAKE_CUDA_ARCHITECTURES`), and the CUDA 12.6 build container
compiles all listed real arches and the `compute_75` PTX.

---

## v0.14.52 — GPU batch auto-tuner freezes once the writer is the bottleneck

At 130 GiB scale on the 8-GPU box, the GPU batch auto-tuner was hunting a moving
GPU-throughput target (e.g. "settled at batch=64 @ 6.11 GiB/s") while the run only
delivered ~4.6 GiB/s — because the **output device**, not GPU compute, was the cap
(`[WRITER]` verdict: near sink-limited, ~66–71% busy across configs).  In that
regime GPU batch size can't lift end-to-end throughput, and a *larger* batch only
hurts: it raises latency to the in-order writer, deepening head-of-line blocking
(a forced `--gpu-batch=256` flipped the pipeline to upstream-bound with ~1800–2190
frames stuck behind the missing in-sequence frame).

The tuner now reads the same writer-busy fraction the `[WRITER]` line reports.  Once
the writer is sink-limited (busy ≥ 55%, after a 3 s ramp-up guard so initial
exploration still runs), it **latches frozen** at the best batch found and stops
probing — no more churn chasing a number that doesn't move the run, no wandering
into the head-of-line cliff.  The gate is sticky per run and applies to both the
compress and decompress GPU tuners.

Deliberately one-directional and regime-gated: on a GPU-**compute**-bound box (slow
GPU, idle disk — writer busy ~1%, starved ~99%) the freeze never fires and the
tuner keeps exploring, exactly where batch size still matters.  This is the first
piece of `[WRITER]`-verdict-driven control wired into a live decision — the
groundwork a future `--adapt` mode generalizes.

---

## v0.14.48–0.14.51 — GPU-side `--verify`, and a `--verify-engine` flag that picks by bottleneck

`--verify` on a `--gpu-only` compress had to decompress-check on the CPU, which on
a fast many-GPU box competes with the GPU pipeline's host-side staging for CPU and
memory bandwidth (residual ~+5.8 s on 8×H100 after the v0.14.47 fixes).  But those
GPUs are **disk/sink-bound** — ~half-idle waiting on the writer.  So verify on the
GPU instead.

After each batch compresses, the GPU worker decompresses it back in VRAM (nvCOMP,
same stream) and **raw-byte-compares** it against the still-resident original
input — no host traffic, no CPU.  A custom compare kernel (`gpuverify.cu`, the only
kernel gzstd ships; CMake enables the CUDA language only under `HAVE_NVCOMP`) sets a
device mismatch flag; any decode failure or mismatch throws → the existing
gpu_worker catch → abort → CPU-only rebuild.  Raw compare, not XXH64: absolute, no
collision window.  A `[GPU-VERIFY]` line at `-v` reports frames/bytes checked and
aggregate GPU verify time.

**Measured both ways on both boxes — and the winner flips with the bottleneck:**

| box | bottleneck | idle resource | CPU verify | GPU verify |
|-----|-----------|---------------|-----------|-----------|
| 8×H100 (Gen5) | disk (sink) | GPU compute | +5.8 s | **+1.4 s** |
| 2×2080 Ti (Gen3) | GPU compute | CPU | **+0.07 s** | +0.80 s |

GPU verify wins only when the GPU is the idle resource (disk-bound, fast Gen4+ card
with VRAM to spare); when the GPU is compute-bound, CPU verify rides the idle CPU
and is far cheaper.  So `--verify-engine=cpu|gpu|auto` (default **auto**) picks by
PCIe gen: **GPU on Gen4+, CPU otherwise** — CPU being the safe fallback (no VRAM
cost, never terrible).  GPU verify only applies to `--gpu-only` (every frame flows
through the GPU worker); hybrid/cpu-only always use the CPU `VerifyPool`.  VRAM: per
stream adds a decompressed-output buffer (= input size) + an nvCOMP decompress temp
(~= the compress temp), so on a small-VRAM card the default config may not fit and
it falls back as usual.

(The static PCIe-gen heuristic stands in for a true runtime read of the bottleneck
— the `[WRITER]` sink-vs-upstream verdict — the lever a future `--adapt` mode would
steer this with.)

## v0.14.44–0.14.47 — `--verify` observability at `-v`, rate-matched + RAM-sized verify pool; drop a redundant O_DIRECT line

`--verify` ran silently — no way to see whether it was keeping up or how much CPU
it cost, which matters on `--gpu-only` runs where it competes with the otherwise
near-idle CPU.  The `VerifyPool` now tracks per-frame counts, active decompress
time, peak queue backlog, and stall time (how long a full verify queue blocked the
writer = verification back-pressuring the pipeline).  After the pass, `-v` prints
one line, e.g.:

```
[VERIFY] 1 thread(s) (cap 24), 32 frames (512.02 MiB) decompress-checked @ 3.34 GiB/s, peak backlog 13/256 frames; kept up (compression never waited on verify)
```

Threads (vs. the cap), intrinsic decompress rate (bytes ÷ active verify time, so it
reflects how fast verification works rather than how long it sat idle), peak
backlog, and a verdict: "kept up" or "fell behind: throttled compression Nx for Ts".
On hardware that produces frames faster than one verify thread can check (e.g.
Gen5 + H100), this shows the pool growing toward the cap and, if it saturates, the
back-pressure it applies.

Also removed the duplicate `[O_DIRECT] using O_DIRECT for output` line — the Gen4+
auto-default notice (`PCIe GenN detected; defaulting output to --direct`) already
says it.

The new `[VERIFY]` line immediately exposed a bug it had been hiding: on an 8×H100
Gen5 box, `--verify --gpu-only` cost ~5 s, and the line showed why — the pool had
ballooned to **256 threads (the full core count), peak backlog pegged at 256/256**,
yet was doing only ~2 threads' worth of work (65 GiB ÷ the reported per-thread rate
≈ 2 thread-seconds active).  Two causes: (1) the cap was `hardware_concurrency`, and
(2) growth spawned a thread on *every* `submit()` while backlog was high, so a GPU
burst rocketed it to the cap and it never shrank.  The extra ~254 threads did not
verify any faster — this data is ~half incompressible, where "decompress" is
`memcpy` + XXH64, i.e. **memory-bandwidth-bound** — they just contended with the
GPU pipeline's host-side staging (8× H2D/D2H, 12 readers, the O_DIRECT writer) for
CPU and bandwidth, dropping output 4.35 → 3.75 GiB/s.

**Fix (v0.14.46):** cap the verify pool at `max(2, cores/8)` (leave the cores for
the pipeline) and grow **rate-matched** instead of per-submit — every 50 ms it
measures the drain rate and adds a thread only while the previous one raised that
rate by >10%.  Once an added thread stops helping (bandwidth/CPU plateau, or it has
matched the producer) it holds, so the pool settles at the few threads it actually
uses.  The `-v` line now reports the aggregate **drain** rate (compare to the
compress out-rate for keep-up) alongside the per-thread rate.

That cut the `--gpu-only` penalty from ~5 s to ~1–2 s, with the pool settling at
8–9 threads (gpu-only) / 13–15 (hybrid) instead of 256.  The residual was *not* a
capacity deficit — verify keeps up on average (`drain ≈ out`, ~1 thread's worth) —
but the 8 GPUs deliver in bursts that briefly overflowed the fixed 256-frame queue,
and since the verify tap lives in the writer, a full queue stalls the writer for a
moment (×~5000, ~2.8 s total).

**Dynamic sizing (v0.14.47):** make both knobs scale to the machine instead of
fixed constants.  The queue depth is now sized from available RAM
(`compute_verify_queue_depth`: ~6 % of free RAM, capped at 16 GiB and honoring
`--memlimit`, divided by the worst-case frame size, clamped to 32..8192 frames) —
so a big box gets a deep queue (~1024 frames) that absorbs the GPU bursts without
back-pressuring the writer, while a limited-RAM box gets a shallow one and never
risks OOM (the old fixed 256 frames was up to 4 GiB regardless of RAM).  And the
thread cap drops to `clamp(cores/16, 2, 16)` — the pool only ever needs a handful,
and a deeper queue means it grows to even fewer (bursts are buffered, not drained
by spinning up threads).

## v0.14.43 — delete the compress CPU-rescue machinery; clean GPU-fault abort

With the v0.14.38 posture (a faulting GPU is abandoned and the archive rebuilt
CPU-only), the old mid-run rescue had nothing left to do: the rescue queue, the
`C.delivered`-watermark re-enqueue, and `gpu_only_cpu_fallback`'s "finish the tail
on CPU" all produced output the rebuild then discarded — a pure ~2× CPU drain on
the fault path.

Replaced with a true abort.  A compress GPU fault sets `g_gpu_aborted` (kept
separate from `g_gpu_failed_restart`, which `--verify` also sets after a pass):
the producer's `TaskQueue::push` drops further frames, the CPU/GPU workers exit
without draining the queue, and the writer bails its in-order wait without the
"writer stuck" watchdog firing.  The driver rebuilds CPU-only as before.
`g_gpu_aborted` is cleared before every pass (the rebuild included) — a missed
reset made the first rebuild emit empty output, caught in fault-injection testing.

Deleted: `cpu_worker_rescue`, the compress rescue pool and its `RescueQueue`, the
`gpu_worker` rescue parameter, and the "keep inputs alive for rescue" retention
(GPU-worker inputs now release at H2D always, freeing pooled-reader slots / mmap
views sooner in hybrid too).

Kept deliberately, because it is NOT vestigial: the hybrid CPU pool in normal
operation; the startup VRAM-skip CPU dispatch (nothing written yet); and the whole
DECOMPRESS rescue path — a faulted GPU on decompress is finished on CPU and the
output is kept and correct (decompression is deterministic and checksum-verified).

## v0.14.41–0.14.42 — `--keep-going`: recover a damaged archive on decompress

Read-side counterpart to `--verify`.  Normally a frame whose XXH64 checksum fails
aborts `-d` (exit 4).  With `--keep-going` gzstd recovers what it can and reports
what is damaged.  Implies `--cpu-only` and the single-threaded reader (so each
frame's output offset is a simple prefix sum); decompress-only.

Each frame is one-shot-decoded into a full-size buffer: the one-shot decoder
writes the whole frame before validating the trailing checksum, so a
checksum-mismatch frame is recovered in full and keeps its exact length — nothing
downstream (in particular `--tar` member boundaries) desyncs.  A hard decode error
keeps the valid prefix plus zeros (the buffer is zeroed first so no stale pool data
leaks).  Classified by `ZSTD_getErrorCode`: checksum mismatch → exit 6
(unverified); any other error → exit 7 (incomplete).

Reporting: plain `.zst` lists damaged output byte ranges; `--tar` names the
affected member files (the extractor records each member's data range via
`StreamReader.consumed_total` and intersects it with the damaged frames).  Without
the flag the decompress error messages now suggest `--keep-going`.

Framing break (v0.14.42): corruption that destroys frame boundaries (corrupt
header, declared-size overrun, truncation) cannot be skipped without a magic-scan
resync, which gzstd does not do.  The reader keeps the frames parsed so far, flags
the break, and stops cleanly — the driver prints "RECOVERY STOPPED after N
frame(s)" and exits 7 — instead of an abrupt die.

## v0.14.39–0.14.40 — `--verify`: background decompress-verify while compressing

zstd (and gzstd) only WRITE the per-frame XXH64 checksum; it is validated at read
time, never at creation.  That is fine for a CPU compressor, but the GPU is an
untrusted producer — a fault can emit wrong bytes that surface only at restore,
possibly after the source is gone.  `--verify` closes the gap: a background pool
independently decompress-verifies every finished frame (the `gzstd -t` round-trip)
while compressing, and on any mismatch discards the output and rebuilds CPU-only,
retrying until it verifies clean (`--verify-retries=N` caps it; 0 = unlimited, the
default for an unattended backup).  Off by default, compress-only.

The pool starts at one thread and grows on backlog, capped at hardware concurrency
— its threads contend with the compress workers, so verification steals CPU from
compression (backpressure in the right place), and on a GPU run the otherwise-idle
CPU absorbs it nearly free.  A mismatch over a pipe (cannot rewind to rebuild) is a
fatal data error.  The rebuild reuses the GPU-fault discard-and-rebuild path; a
verify mismatch warning is loud enough to survive `-q`.

## v0.14.38 — GPU fault → full CPU-only rebuild; GPU subchunk overflow guard

Background: a `--gpu-only` compress on a faulting GPU produced a corrupt archive.
A CUDA `illegal memory access` is asynchronous and sticky — the faulting kernel
corrupts its output while the error only surfaces at a later sync, so the
completion path saw `cudaStreamQuery == cudaSuccess` and `nvcompSuccess` and
delivered an already-corrupt frame.  The old rescue trusted any frame past its
`delivered` watermark, so the corrupt frame stayed in the output and the run
exited 0.  (An earlier attempt verified every GPU frame by CPU-decompressing it
before commit; that worked but cost ~30% throughput on every run to defend
against a rare event, and it treated the symptom, not the cause.)

New approach — a faulting GPU is an unreliable narrator, so don't try to salvage
its work:

1. Subchunk overflow guard.  GPU input/output slots are sized to
   `gpu_chunk = min(host_chunk, GPU_SUBCHUNK_MAX=16 MiB)` and the worker does not
   split a Task into subchunks, so a host chunk larger than 16 MiB (e.g.
   `--chunk-size 32`, or the hybrid ultra-window auto-bump) overflowed the device
   slot on the H2D copy — itself an illegal memory access.  The compress driver
   now clamps the host chunk to GPU_SUBCHUNK_MAX whenever a GPU is in play
   (warning if it had to), and the H2D loop hard-guards the slot bound.  This
   removes one concrete way we could provoke the fault ourselves.

2. Fault → discard → CPU rebuild.  On ANY GPU fault during compression, the whole
   GPU/hybrid attempt is abandoned: nothing the GPU produced is trusted, the
   partial output is discarded (the O_DIRECT writer is reopened with O_TRUNC, or
   a buffered output is truncated to 0), and the entire archive is rebuilt
   CPU-only from the original input — which is always still present (and for
   `--tar`, the layout is rebuilt from the source paths).  Speed is secondary to
   a correct archive.  If the input/output is not seekable (stdin/pipe) the
   rebuild is impossible and we die with a clear message to rerun `--cpu-only`.
   This replaces the old per-frame verify and the partial CPU "finish the tail"
   fallback for the fault case.

   The pipe case is unrecoverable by construction: stdin is already consumed and
   any partial (untrusted) output has already streamed to the downstream reader
   and cannot be recalled.  So a GPU fault while `--gpu-only`/`--hybrid` reads or
   writes a pipe dies loudly with EXIT_GPU_FAIL (5) and a message stating the
   already-emitted bytes are incomplete/corrupt and to rerun `--cpu-only` (or use
   regular files so a fault can be recovered).  `--tar` re-reads its source paths,
   so only a piped OUTPUT blocks its rebuild.

Tested deterministically (real GPU faults are intermittent): a debug-only hook,
$GZSTD_DEBUG_FAIL_GPU_AFTER=N, makes a GPU worker throw a simulated fault after
N delivered frames (disabled otherwise — a single almost-always-false branch).
The suite asserts that a fault triggers a CPU-only rebuild whose output is a
valid archive that round-trips exactly, for both a plain file and `--tar`.

---

## v0.14.34 — document -l/--list in --help

The `-l`/`--list` mode was listed in the short `-h` summary but had no entry in
the full `--help` OPERATION section.  Added one (after `-t`) describing both
forms: plain `.zst` frame summary (`zstd -l` style, header-only so it stays fast
on huge archives) and `-l --tar` archive listing (`tar -tvf` style).  No code
change.

---

## v0.14.33 — default to no preallocation (fallocate hurts O_DIRECT)

Output preallocation (`fallocate` to the expected size up front) is now OFF by
default.  It was added back in v0.11.x for extent-stall-free dense writes and was
very likely a genuine win then — the write path was synchronous and buffered, so
reserving the extents avoided a per-`fwrite` allocation stall (journal commit on
ext4) on every extent boundary.  Two things changed since: O_DIRECT became the
Gen4+ default, and the write path was restructured into the async, pipelined
DirectWriter.  In that world `fallocate` backfires — it creates *unwritten*
extents, and each O_DIRECT write then pays an unwritten→written conversion that
costs more than the allocation it was meant to avoid.  Measured ~4-6% slower with
preallocation, on both compress and decompress, reproduced on two different
machines (Gen5 and Gen3, both ext4/NVMe).  It's also why `-d --tar` extract
(which never preallocated) was the faster path all along.

`preallocate_output` defaults false; `--preallocate` opts back in (for
filesystems/workloads where reserving space helps, or for early out-of-space
detection).  Preallocation only ever applied to the O_DIRECT paths where the
size is known, so this is purely an O_DIRECT-path change.  Bonus: with no
preallocated blocks, the sparse path's seeks create holes naturally — the
preallocate+punch-hole hybrid is no longer exercised by the default.

Also moved the deep write-path diagnostics added while chasing this (the
`[WRITER] of busy:` ::write/bounce-copy/zero-scan split, `[SINK]` producer/
consumer waits, `[UNTAR-LARGE]` fill/scan/bufwait/write/jobwait) from `-v` to
`-vv` — they're per-batch internals.  `-v` keeps the high-level regime lines
(`[READER]` state, `[WRITER]` busy/verdict).

## v0.14.32 — offload the sparse zero-scan off the write path

After v0.14.31 the writer breakdown still showed ~15-19% of the write thread's
busy time in the per-block zero-scan (is_all_zero), serial ahead of the
O_DIRECT write — the last gap to extract parity.  Moved that scan upstream into
AsyncWritePool::submit(), which runs on the writer thread (it has slack — it
just hands batches to the write worker), BEFORE the backpressure wait, so it
overlaps the worker draining the previous batch.

submit() now computes a per-frame "dense" flag (frame_dense: true when the frame
has no all-zero 4 KiB block).  The write worker writes dense frames whole with no
scan; only frames that actually contain a hole run the full write_sparse.  This
is behavior-identical (a dense frame is exactly one write_sparse would emit as a
single run) — it just relocates the scan off the write-critical-path.  Gated to
sparse + O_DIRECT; buffered/--no-sparse/compress paths unchanged.  Genuinely
sparse frames early-exit the dense check cheaply and take the existing path.

Verified byte-identical with O_DIRECT engaged: dense/incompressible (zero-scan
now 0% on the write worker, confirmed via -v), holey/sparse (holes preserved),
--no-sparse, compress.  Full suite green.  Closes the remaining gap so plain -d
matches -d --tar extract's write rate.

## v0.14.31 — asynchronous DirectWriter: pipeline the O_DIRECT write off the copy/scan

The v0.14.29/30 breakdown nailed it: on a cpu-only 130 GiB -d, the single
AsyncWritePool thread split ~69.5% O_DIRECT ::write, ~15.1% bounce-copy (the
memcpy into DirectWriter's aligned buffer), ~15.5% zero-scan — all SERIAL on one
thread.  -d --tar extract hits the device ceiling because it does the copy+scan
on its parse thread, overlapped with the pwrite on a separate writer thread.
Plain -d did all three in series, so it trailed extract (~2.7 vs ~3.7 GiB/s).

DirectWriter is now internally pipelined.  It keeps a small pool of aligned
buffers (4 × 16 MiB) and a dedicated write thread that drains an ordered op queue
(writes + sparse seeks), so the caller thread's bounce-copy and zero-scan overlap
the O_DIRECT ::write instead of serializing.  The public API is unchanged
(write/seek_forward/finalize/total_bytes), so both compress and decompress get
it.  Mechanics: only full 16 MiB buffers flush mid-stream (no unaligned tail
except at finalize); sparse seeks flush the current buffer then enqueue an
ordered SEEK (hole-punch + lseek run on the write thread, after the buffered
data); total_bytes() reports the caller's logical position (not the lagging
physical one) so write_sparse's alignment guard still holds; finalize drains and
joins the thread before truncating preallocation slack; write errors set a flag
the caller checks.  Backpressure: the caller blocks for a free buffer when the
write thread falls behind (bounded at 64 MiB in flight).

Verified byte-identical with O_DIRECT actually engaged (random, holey/sparse with
holes preserved, --no-sparse, compress, multi-file, tiny/empty); full suite green.
Expected to bring plain -d to extract's write rate (~4 GiB/s) — server timing to
confirm.

## v0.14.30 — fix the writer breakdown so the O_DIRECT ::write split actually reports

v0.14.29's split read the ::write time from g_direct_writer at summary time, but
that pointer is already torn down by then, so an O_DIRECT run mislabeled itself
"buffered write" and lumped bounce-copy + write together.  Moved the ::write
timing to a file-scope counter (g_odirect_write_ns) that survives the
DirectWriter's destruction, with a g_odirect_used flag set in open(), so the
O_DIRECT ::write / bounce-copy / zero-scan split prints correctly.  First good
server reading: on a 130 GiB cpu-only -d, zero-scan is ~12.7% of the write
thread's busy time (not the ~1.5% the buffered local box showed — the fast
O_DIRECT write makes the serial CPU scan a much larger slice), confirming the
single-threaded scan + bounce-copy on the write thread is the gap vs extract
(which pipelines them).  Still measurement only.

## v0.14.29 — writer-busy breakdown (-v): split the write thread's time three ways

Investigating why single-file -d trails -d --tar extract on the same data (plain
~2.6-2.9 vs extract ~3.7 GiB/s, both write 130 GiB O_DIRECT).  The existing
[WRITER] line said "write-path busy 97%" and concluded "device saturated", but
that busy time lumps together the CPU zero-scan, the bounce-copy into the aligned
O_DIRECT buffer, and the actual write syscall — all serial on the single
AsyncWritePool thread.  Added a breakdown line at -v:

  [WRITER]   of busy: O_DIRECT ::write X% | bounce-copy→aligned Y% | zero-scan Z%

writer_iowrite_ns times the write/seek calls inside write_sparse (vs the scan),
and DirectWriter now reports the time inside the ::write syscall alone, so the
bounce-copy (DirectWriter::write memcpy's every byte into its aligned buffer
before flushing) is isolated as iowrite − write.  This makes visible that plain
-d serializes copy + write on one thread where extract pipelines the copy (parse
thread) against the pwrite (writer thread).  Measurement only — no behavior
change (counters are -v-gated; ruled out the zero-scan, ~1.5%, as the cause).

## v0.14.28 — no-scan fast path when sparse output is disabled

write_sparse scanned every 4 KiB block (is_all_zero) even with sparse disabled —
the run-coalescing loop calls is_all_zero regardless of the sparse_ flag, so
--no-sparse still paid a full single-threaded zero-scan on the write thread's
critical path.  Added a sparse_-off fast path that writes the whole buffer in one
call with no scanning.  Speeds up --no-sparse decompression, and isolates the
scan cost while we investigate why plain -d (single-file decompress) trails
-d --tar extract: extract pipelines the zero-scan (parse thread) against the
O_DIRECT write (writer thread), whereas AsyncWritePool's single write thread does
scan + write serially.  Behavior unchanged for the default (sparse auto-on);
round-trips verified byte-identical (random + holey) with --no-sparse.

## v0.14.27 — MADV_SEQUENTIAL on the -l frame walk (cold-file speed)

Plain -l mmaps the file and walks frame + block headers with
ZSTD_findFrameCompressedSize, which skips block content by pointer math — so it
touches only the block-header bytes, roughly one per 128 KiB.  On a cold file
that is ~1M scattered, serial, synchronous page faults (a 65 GiB / 8343-frame
archive took ~42s: ~19s iowait on random reads + ~23s fault handling).  zstd -l
instead streams the file sequentially, so it's readahead-bound.

Added madvise(MADV_SEQUENTIAL) on the mapping.  The walk's accesses are
forward-only, so the kernel now prefetches in large sequential reads instead of
faulting one header page at a time — turning the cold path from random-IO-bound
into sequential-read-bound, matching zstd's profile.  Warm (page-cache-resident)
runs were already fast; this fixes the cold case.  Note the cold floor is still
the time to stream the file once (both tools read it); a header-only variant
would need parallel/async scatter reads and is left for later.

## v0.14.26 — GPU and hybrid compress now write XXH64 content checksums

The CPU compressor sets ZSTD_c_checksumFlag, so CPU-compressed frames carry a
per-frame XXH64 content checksum (self-verifying; -t and any zstd decoder catch
silent bit-rot).  nvCOMP's batched zstd compressor does NOT add it, so GPU- and
hybrid-compressed archives shipped without that integrity check — gzstd -l (and
zstd -l) showed Check: None.  An oversight: GPU/hybrid output was less protected
than CPU output.

Fixed by stitching the checksum onto each GPU-produced frame.  zstd's content
checksum is the low 32 bits of XXH64(uncompressed_frame, 0); we compute it at H2D
staging (while the uncompressed chunk is still on the host) using ZSTD_XXH64
(exported by libzstd, which builds xxHash under the ZSTD_ namespace — no new
dependency), then after D2H set the frame header descriptor's content-checksum
bit and append the 4 bytes (little-endian).  Done at both gpu_worker delivery
sites (async + sync drain); rescued chunks already get the checksum from the CPU
recompress path, so every frame ends up self-verifying with no duplication.  The
append lands in the heap FrameBuf, avoiding the fixed GPU output slot's overflow.

Verified: GPU- and hybrid-compressed archives now show Check: XXH64, a stock
zstd -t validates the checksum, round-trips are byte-identical, and a flipped
byte is caught as a checksum mismatch by both zstd and gzstd.  Small per-frame
host-side XXH64 cost on the GPU path, accepted for the integrity guarantee.

## v0.14.25 — silence a -Wnonnull false positive in the sink-mode decompress helpers

`decompress_from_buffer` / `decompress_stream_from_file` take `out` and, since
v0.14.19/20, are also called with `out=nullptr` when routing to the in-memory tar
sink (verify/extract/list).  GCC inlined the `out=nullptr` constant and flagged
the `robust_fwrite(..., out)` path as a possible `fwrite(NULL)` — but that path is
unreachable whenever `out` is null (the `g_tar_decomp_sink` branch handles output
first).  Guarded the `fwrite` with `if (out)` to make that explicit; behavior is
unchanged (out is always valid on that path at runtime).  Build is warning-clean.

## v0.14.24 — `-l` / `--list`: zstd-style frame info, and `-l --tar` archive contents

`-l` was a no-op stub that pointed users at `zstd --list`.  It now lists for real,
in two modes:

- **`-l file.zst`** — a `zstd -l`-style summary: `Frames  Skips  Compressed
  Uncompressed  Ratio  Check  Filename`.  It mmaps the file and walks the frame +
  block headers WITHOUT decompressing — `ZSTD_findFrameCompressedSize` skips block
  content, so over the mapping only header pages fault in; uncompressed is summed
  from each frame's content-size header and `Check` is read from the frame
  descriptor's checksum bit (`XXH64`/`None`).  Fast even on a huge multi-frame
  file.  Works on any zstd file (ours or foreign); frame count matches `zstd -l`.

- **`-l --tar archive.tar.zst`** — lists the archive's contents in `tar -tvf`
  style (perms, owner/group, size, mtime, name; `-> target` for symlinks), then a
  `N files, <size>` footer.  It reuses the verify fast path — decompress through
  the in-memory `FrameSink` and the `Extractor` in a new **list mode** that prints
  each entry and skips the data — so it's decompress-bound, not pipe-bound, and
  works on foreign `.tar.zst` too.  The tar header parser now also reads the
  ustar `uname`/`gname` fields so owner/group show as names when present.

Entries print to stdout; the footer and any notices go to stderr, so the listing
pipes cleanly.  The Gen3 "defaulting decompress to --cpu-only" notice is
suppressed under `-l` (a listing shouldn't announce a backend; plain `-l` doesn't
even decompress).

## v0.14.23 — persistent cross-file O_DIRECT writer for -d --tar extract

The large-file extract writer (`stream_large_direct`) was spawned **and joined
per file**: at every large-file boundary the parse thread blocked until that
file's tail writes drained before it could even read the next entry's header.
That per-file join is a structural bubble independent of hardware, and it scales
badly with file count (an archive of thousands of medium-large members pays it
thousands of times).

The writer is now **persistent across all large files**.  At a file boundary the
parse thread enqueues a `FINALIZE` job (ftruncate to the exact size + metadata +
close, performed on the writer) and moves straight to the next entry, so a file's
tail writes overlap the next file's parse/read.  The parse thread remains the
sole sink reader; FIFO job ordering guarantees each file's `WRITE` jobs all
complete before its `FINALIZE`, and that earlier files finish before later ones.
The single O_DIRECT write stream is unchanged (concurrent streams contend — see
ROADMAP); this removes the *pipeline* bubble, not the device ceiling, so the gain
shows wherever writes aren't the sole wall (slower disks, faster future arrays,
many-medium-file archives).

Reliability: fd lifecycle is single-owner (parse opens, `FINALIZE` closes — no
double-close; fallbacks close their own fd); on a write error the writer keeps
draining so pending `FINALIZE`s still close their fds (no leak), and the parse
thread drains the current file's remaining stream bytes so later entries stay
aligned; the writer is joined in `stop_pool` before `finish_deferred`
(hardlinks/dir metadata), with a defensive join in the destructor.  Verified
byte-identical across many large files (cross-file pipelining), mixed
small/dir/symlink/holey content, multi-frame, and foreign `tar|zstd` archives;
full suite 251/251.

## v0.14.22 — extract writer buffer tuning + accurate large-file counter label

- **Bigger / deeper O_DIRECT write-ahead buffers** for large-file extract:
  `DBUF_CAP` 8→16 MiB (larger transfers sit closer to the device's single-stream
  ceiling) and `NDBUF` 3→6 (a deeper queue so the writer doesn't starve when the
  parse thread briefly stalls — measured `jobwait` was several seconds of the
  wall under CPU contention).  96 MiB peak, one large file extracts at a time.
- **`[UNTAR-LARGE]` label fix:** it counts only files that took the large-file
  path (`>4 MiB`), so it now reads "N large file(s) >4 MiB … (small files + dirs
  go to the writer pool, not counted here)" instead of looking like a total.

## v0.14.21 — -t --tar line colors, smoother extract progress, extract bottleneck counters

Three small changes, the last two prompted by `-d --tar` extract behavior on a
high-core Gen5 server (96 threads, 8 GPUs):

- **Colorized the `-t --tar` result line** to match the plain `-t` summary: input
  size cyan, output size green, ratio bold, rate green, `OK` bold-bright-green,
  entry count bold.  Cosmetic; same fields, same numbers.

- **Smoothed the extract/verify progress bar.**  The decompress writer accounted
  `wrote_bytes` once per *batch* after pushing the whole batch into the in-memory
  `FrameSink`.  When the (write-bound) Extractor drains slowly the sink fills and
  the push blocks for seconds, so a multi-GiB batch made the `out:` counter sit,
  then jump.  It now counts each frame as it is accepted into the sink (and
  releases the throttle permit per frame, so workers refill as the batch drains),
  matching how the old aio path updated every 16 MiB.  NOTE: for extract the bar
  still tracks bytes *handed to* the Extractor (sink intake), which leads the
  actual file writes by the sink + Extractor buffers — accurate "bytes written"
  accounting is folded into the parallel-dispatch work (see ROADMAP).

- **Added `-v` extract bottleneck counters.**  Large-file extract (`>4 MiB`,
  `stream_large_direct`) runs on the single tar-parse thread feeding one O_DIRECT
  writer, while the 16-thread writer pool only handles small files — so a
  large-file-heavy archive is single-parse-thread bound.  Two new verbose lines
  pinpoint where the time goes before we try to open it up:
  - `[UNTAR-LARGE] … parse: fill=…/scan=…/bufwait=… | writer: write=…/jobwait=…`
    — splits the parse thread's sink-read+memcpy (`fill`), zero-detect (`scan`),
    and wait-for-a-free-buffer (`bufwait`, = write-bound) from the writer's
    `pwrite` time and its wait-for-data (`jobwait`, = parse-bound).
  - `[SINK] producer waited … | consumer waited …` — whether decompress (producer)
    or extraction (consumer) is the bottleneck across the `FrameSink` hand-off.
  Measurement only; no behavior change.  Parallel-dispatch extraction (the actual
  fix) remains the ROADMAP item.

## v0.14.20 — -d --tar extract uses the same in-memory frame hand-off

Applies the v0.14.19 `FrameSink` change to `-d --tar` extraction: `extract_tar`
no longer creates a kernel pipe + an `Extractor` thread draining it.  The
decompressors push finished frames straight to the in-process `Extractor` (now
via `run_sink`), which reads each member's data out of the in-RAM frames and
writes it.  Same parser, same security and tar-format handling — only the byte
source changed (pipe → in-memory `FrameSink`), so there are no new
compatibility concerns.  The `g_tar_verify_sink` global is renamed
`g_tar_decomp_sink` since it now serves both `-t` and `-d`.

Unlike verify (which became decompress-bound), extract still copies and writes
every byte to disk and its tar parse is still serial, so this is a tidy
efficiency win — it removes the pipe's per-byte kernel copy, syscalls, and a
context-switch hop, lowering system time and memory-bandwidth contention — not a
parallelization.  The big extract lever (parallel parse + dispatch across entry
ranges, with directory-ordering and single-O_DIRECT-stream contention to solve)
is documented in ROADMAP as future work.  Peak RSS is unchanged (the hand-off
queue is byte-bounded; when the write-bound Extractor lags, the bound throttles
the decompressors).  Round-trip content verified byte-identical for gzstd,
multi-frame (`--chunk-size 1`), and foreign `tar|zstd` archives; full suite
251/251, extensive/compat 349/349.

## v0.14.19 — parallel -t --tar verify via in-memory random access

`-t --tar` barely beat `tar --zstd -t` while every other tar path is fast.  The
cause was structural, not algorithmic: `verify_tar` ran the full parallel
decompressor but then funneled the *entire* decompressed tar stream through a
single 1 MiB kernel pipe into one `Extractor` thread.  Tar headers are cheap to
check (one 512-byte checksum per file), but to reach the next header the
validator calls `r.skip(e.size)` on a pipe-backed `StreamReader` — and a pipe
can't seek, so `skip` *drained every data byte* through the kernel.  Verify was
therefore single-pipe-drain bound, doing O(bytes) serial work on top of the
parallel decompress.

That treated the decompressed stream like a forward-only tape.  But it is already
in RAM and in original order (`ResultStore` → `writer_thread`).  Verify now hands
those in-order frames to the **same** reused `Extractor::parse` *in memory*, so
`skip(data_size)` becomes pointer arithmetic — advance a cursor, drop the
already-checksum-validated frame — touching only the ~512-byte headers (O(files))
and never copying file data a second time.

Mechanism:

- **`FrameSink`** — a byte-bounded, in-order `FrameBuf` hand-off queue, replacing
  the kernel pipe.
- **`StreamReader`** gains a sink source alongside the fd source; in sink mode the
  read cursor points straight into the current frame (no copy) and `skip()`
  advances past whole frames without touching their bytes.  `Extractor::parse` /
  `handle_entry` / the `validate_only` path are reused verbatim.
- **`writer_thread`** routes finished frames to the sink on the common parallel
  path (zero-copy), releasing the throttle permit per frame in place of the aio
  write that would otherwise release it.  The streaming/single-frame helpers
  (`decompress_from_buffer`, `decompress_stream_from_file`) push a copy to the
  sink so foreign single-frame `.tar.zst` (e.g. `tar -I zstd`) still verify
  correctly.

No archive format change — this works on every existing gzstd archive and on
foreign `.tar.zst`.  Verify is now decompress-bound rather than pipe-bound: on a
96-thread dual-socket server a 65 GiB / 130 GiB archive verifies at 14.1 GiB/s,
matching plain `-t` (13.6 GiB/s) — roughly 5× the old pipe-bound verify, with
lower system time (no kernel pipe).  On a 24-core Gen3 workstation `-t --tar`
runs at 4.05 GiB/s vs plain `-t` 4.32 GiB/s.  Peak RSS is unchanged (the in-memory
hand-off queue is byte-bounded and the validator drains it as fast as the
decompressors fill it, so it stays near-empty).  `-d --tar` extract is unchanged
(still pipe + writing `Extractor`).  Integrity unchanged: a corrupt zstd frame
still fails the per-frame checksum (exit 4), a truncated tar is still caught by
the parser's zero-block/short-read checks.  Full suite 251/251, extensive/compat
349/349.

## v0.14.18 — decompression rate + compression ratio in the -t --tar result line

The live progress bar during `-t --tar` reports the verify rate, but it gets
overwritten by the final per-archive result line, which only showed entry count
and total size — no speed and no ratio (unlike plain `-t`).  Each archive's
decompress+validate span is now timed (`steady_clock` around the decompress
dispatch through `Extractor::run` join), and the result line gains both a
`@ <rate>/s` field and a `<comp> => <decomp> (ratio: X%)` field, matching the
plain-`-t` summary:

```
archive.tar.zst : OK, 534532 entries, 51.8 GiB => 103.6 GiB (ratio: 50.0%) @ 2.21 GiB/s — tar structure valid
```

Rate is decompressed bytes over wall time; ratio is compressed file size over
decompressed bytes (same convention as plain `-t`).  Ratio is omitted for stdin
input (compressed size unknown).  Cosmetic only — no change to verify semantics
or exit codes.

## v0.14.17 — 1 MiB pipe for the -d --tar / -t --tar stream

The decompressor feeds the tar Extractor through a pipe with a single reader (the
parse thread is inherently serial — tar headers must be read in order).  At the
default 64 KiB pipe size, a 130 GiB stream is ~2M full/empty cycles and syscalls
on that one bottleneck thread.  `extract_tar` and `verify_tar` now request a
**1 MiB pipe** (`F_SETPIPE_SZ`, best-effort) — ~16× fewer cycles on the reader.
Capped at 1 MiB (the default unprivileged `pipe-max-size`) and best-effort, so it
either helps or silently no-ops; never fails the run.  No effect on small
archives (a bigger pipe is just unused capacity, ~1 MiB kernel buffer per
archive); large streams get the syscall reduction.

## v0.14.16 — pipeline the O_DIRECT large-file extract writer

v0.14.15 wrote large files with O_DIRECT on a single thread that read a chunk
from the decompressor pipe, then wrote it, then read the next — serial, so the
pwrite and the pipe-read never overlapped.  That capped big-file extract well
below the device's single-stream O_DIRECT ceiling (~1.1 vs ~4.5 GiB/s observed).

`stream_large_direct` is now a producer/consumer: the parse thread fills one of
a small pool of 4 KiB-aligned buffers (pipe read + zero-detect, building the
write plan) while a single dedicated writer thread O_DIRECTs another.  Read and
write now overlap.  It deliberately keeps exactly **one** O_DIRECT writer —
concurrent O_DIRECT streams contend on NVMe (prior finding: 1 stream 4.5 GB/s, 4
streams ~3.0 aggregate), so the win is pipelining the single stream, not
parallelizing it.  Sparse-awareness is preserved (zero blocks skipped as holes,
non-zero runs coalesced), the unaligned final tail is handled (O_DIRECT cleared),
and buffers are reused across files.  `blk_zero` is now word-wise (uint64_t).

Content-verified across multi-buffer (>24 MiB) files, unaligned tails, interior
holes, and multiple large files in one archive; full suite 249/249.  (Throughput
gain to be confirmed on an idle box; correctness is independent of that.)

## v0.14.15 — O_DIRECT large-file extract; `-t --tar` progress bar; help fixes

**O_DIRECT on `-d --tar` large-file extract.** `-d --tar` ignored `--direct`
entirely: the Extractor's per-file writes were always buffered, even on Gen4+
where `--direct` is auto-on (the `[O_DIRECT] defaulting to --direct` banner was
misleading there).  Buffered writeback throttling capped big-file extract — the
parallel decompressor outran the disk through the page cache.  Files >4 MiB
(`stream_large`) now write with O_DIRECT when `--direct` is active, so the
producer streams straight to disk.  It's **sparse-aware**: all-zero 4 KiB blocks
are skipped as holes (when sparse is on), runs of non-zero blocks are coalesced
into one aligned O_DIRECT write, and the final sub-block tail is written with
O_DIRECT cleared.  Small files (≤4 MiB) stay buffered on purpose — O_DIRECT
backfires on many tiny, mostly-unaligned, metadata-bound writes.  Falls back to
buffered if the filesystem rejects O_DIRECT (e.g. tmpfs).  Content-verified
across `--direct` / `--no-direct` / `--direct --no-sparse`; reuses a single
reused 4 KiB-aligned bounce buffer (the parse thread writes large files serially).

**`-t --tar` progress bar.** `gzstd -t --tar` (verify) showed no progress bar —
`verify_tar` never spawned `progress_loop` (same gap as the v0.14.2 extract fix).
It now shows the bar (labeled `verified:` via the existing TEST-mode variant), and
the per-archive OK/CORRUPT result lines print after the bar stops so they don't
collide.

**Help text.** `--[no-]sparse` now documents its extract behavior precisely:
`--no-sparse` only disables the scan-for-zeros (makes *normal* entries fully
allocated) and does NOT affect entries the archive declared sparse (those always
restore their recorded holes, like GNU tar) — "don't go looking for holes," not
"materialize holes the archive declared."  Added the create-side `--sparse` entry
under `--tar` (it was undocumented).

## v0.14.14 — silence -Wmissing-field-initializers in the tar layout walk

v0.14.13 added a `sparse_map` field to `LayoutBuilder::Pending` but left the two
`pending_.push_back({...})` aggregate-initializers partial, so `-Wmissing-field-
initializers` fired (harmless — the field default-constructs empty).  Gave the
Pass-B `Pending` members explicit `{}` default initializers and trimmed the
push_backs to the fields actually set, so partial aggregate-init is clean.  No
behavior change; warning-free build.

## v0.14.13 — create sparse archives (`--tar --sparse`, GNU-interoperable)

Completes sparse support: gzstd can now *create* sparse `.tar.zst` archives, not
just read/restore them.  `--tar --sparse` detects each regular file's holes via
`SEEK_DATA`/`SEEK_HOLE` (filesystem extent metadata — **reads none of the hole
bytes**) and stores it in OLDGNU `'S'` format: only the data segments + the
segment map.  Output is **byte-identical to `tar --format=gnu --sparse`**, so GNU
tar extracts it and restores the holes — closing the last interop cell (a
gzstd-created sparse archive now restores sparse under GNU tar too).

Opt-in, matching GNU tar: **`--sparse` required on create** (default stores full
content — deterministic and maximally portable).  Extract still restores sparse
by default (`--no-sparse` to disable).  Fits the existing architecture without
cost: the hole map is built in `build_layout`'s parallel stat pass (Pass B), the
parallel chunk assembler consults the per-file map (non-sparse files are
unchanged, single linear segment), and for a big sparse file `assemble` now reads
only the real data segments — **a create-time read-speed win**, not just a size
one.  The segment map is terminated with GNU's `(realsize, 0)` marker; >4
segments spill into 512-byte extension blocks (21 each).  Verified byte-identical
to GNU tar and round-trip-clean (gzstd↔gzstd, gzstd↔GNU tar) including
multi-extension-block files.

## v0.14.12 — read PAX GNU sparse formats (0.0 / 0.1 / 1.0)

v0.14.11 added reading of OLDGNU `'S'` sparse entries and made PAX `GNU.sparse.*`
(and other unsupported types) **fail loudly**.  This completes it: gzstd now
**reads the PAX sparse formats too**, so it can extract sparse files from a
`tar --format=posix --sparse` archive instead of erroring.

All three PAX variants are handled, reusing the same segment-placement +
sparse-write path as the OLDGNU reader:
- **0.0** — segment map as repeated `GNU.sparse.offset`/`GNU.sparse.numbytes`
  PAX records.
- **0.1** — map as one `GNU.sparse.map` comma-separated record.
- **1.0** — map as a NUL-padded text block prefixing the file data (read and
  parsed from the stream, then the segments follow).

The real name (`GNU.sparse.name`) and size (`GNU.sparse.realsize`/`size`) come
from the PAX records.  Verified against real `tar` output for `--format=gnu`,
`--format=posix`, and `--sparse-version=0.0/0.1`: content-identical, restored
sparse, parser stays aligned.  Genuinely unknown entry types still fail loudly.
gzstd can now extract any zstd-wrapped tar — any GNU format, sparse or not.

## v0.14.11 — sparse files on `-d --tar`: restore holes + read GNU sparse archives

Two sparse-related fixes on extract, one of which was silent data loss:

**Restore as sparse.** `-d --tar` extracted sparse files fully-allocated — a
100 GiB-sparse / 1 GiB-real file (VM image, big DB) restored to 100 GiB on disk
(content correct, holes became real zeros).  The Extractor now writes files
sparse, leaving filesystem holes where 4 KiB blocks are all-zero (`pwrite` skips
them; `ftruncate` sets the final size).  Reuses the existing
`--sparse`/`--no-sparse` flag, so `-d --tar` matches single-file decompress:
**sparse by default for file output**, `--no-sparse` forces full allocation.

**Read GNU sparse archives (was SILENT DATA LOSS).** A `.tar.zst` created by
`tar --sparse` stores sparse files in GNU's OLDGNU `'S'` format; gzstd didn't
recognize `'S'`, hit the `default` case, and **silently dropped the file** (and
could mis-align the parser for following members).  gzstd now parses the OLDGNU
sparse header (+ extension blocks), reconstructs the file from its segment map,
and restores it sparse — content-identical, parser stays aligned.  PAX-format
GNU sparse (`GNU.sparse.*`) and any other unsupported entry type (e.g. `'D'`
incremental dumpdir, `'M'` multi-volume) now **fail loudly** (exit 4) instead of
silently skipping, so no archive is ever partially extracted without notice.
(base-256 large-number fields were already handled on read.)

The GNU `--sparse` *create* format (gzstd emitting sparse, so GNU tar also
restores holes) is intentionally not implemented: gzstd's parallel chunked
assemble makes it costly, and it's moot here — gzstd restores sparse itself,
zstd keeps the archive small, and plain entries are maximally portable.

## v0.14.10 — `-t --tar` structural verify + per-frame content checksums

Two integrity improvements, especially for backups:

- **`gzstd -t --tar ARCHIVE`** now decompresses *and* structurally validates the
  tar inside the zstd stream — every header checksum, member sizes, and that the
  archive is complete (not truncated) — writing nothing.  Plain `-t` only proves
  the zstd stream decompresses; a truncated or corrupt tar inside an intact zstd
  wrapper passed `-t` before and would only surface on a failed restore.
  Implemented as a validate-only pass through the existing `Extractor` (no file
  writes); reports `archive : OK/CORRUPT, N entries, M bytes`.  (`--tar` is no
  longer rejected with `-t`.)

- **Content checksums enabled** (`ZSTD_c_checksumFlag`) on all CPU compress
  paths.  Each frame now carries an XXH64 footer, so a bit-flip / bit-rot in the
  compressed data is **caught on decompress** — `-d`, `-t`, and `-t --tar` all
  fail with exit 4 instead of silently restoring corrupted bytes.  Previously no
  checksum was set, so silent content corruption went undetected.  Cost is ~4
  bytes/frame; output is standard zstd (any decoder verifies it).  Archives
  created from this version forward are self-verifying.

## v0.14.9 — parallelize the --tar layout walk (two-pass)

`build_layout`'s tree walk was single-threaded — it did one `lstat` per member
serially before any compression could start.  On a cold first-time backup of a
many-small-file tree this is pure dead time (the compress threads sit idle until
the layout exists): measured ~10.7s on a 1M-file / 179 GiB tree, ≈20% of cold
wall time.  (A `--read-threads` sweep first ruled out the read path — cold
small-file reads are device-bound at ~3.3 GiB/s and don't scale with more
readers; the serial walk was the real reclaimable lever.)

The walk is now three passes inside `LayoutBuilder`:
- **Pass A (enumerate, serial):** readdir-driven DFS in canonical order, using
  the dirent `d_type` so leaves need no `lstat` here; an `lstat` happens only for
  `DT_UNKNOWN` or `--one-file-system` directories (st_dev drives descent).
- **Pass B (stat, parallel):** every entry's `lstat` (+ symlink `readlink`) runs
  across N workers on disjoint slots — the cold-inode storm, concurrently.
- **Pass C (finalize, serial):** hardlink first-occurrence, owner-name
  resolution, and the offset prefix-sum, in canonical order.

Order-sensitive logic stays serial, so **archives are byte-for-byte identical**
to the old walk — verified by `cmp` against the prior binary across plain,
`--exclude`, `--numeric-owner`, `--one-file-system`, and `--acls --xattrs`, plus
the full test suite.  `-v` `[TIMING]` now reports the `enum`/`stat` split.

## v0.14.8 — don't treat special-file outputs (`/dev/null`) as clobber targets

Only regular files are a clobber risk now; `-o /dev/null` (and device nodes /
fifos) write through without `-f`, and special targets are never registered for
cleanup-unlink.

## v0.14.7 — fix -d --tar summary double-counting output size

The extract summary shared one Meter between the decompressor (tar stream →
pipe) and the Extractor (files → disk), so the reported size/rate came out ~2×.
Display only; on-disk extraction was always correct.

## v0.14.6 — add [TIMING] layout phase split to --tar

`-v` instrumentation measuring the layout walk vs the parallel ACL/xattr gather,
to drive the parallel-`build_layout` decision.

## v0.14.5 — help text tweaks for --tar

- Dropped the stale "Compression only" notes from `-h`/`--help` (and the
  matching `Options` comment): `--tar` archives are no longer compress-only —
  `-d --tar` extracts, and `--acls`/`--xattrs` apply on both create and extract.
- Added `--help` EXAMPLES for `--tar`: extraction (`-d --tar`, with `-C`),
  ACL/xattr-preserving backup + restore, and a full-system `--one-file-system`
  backup.
- Warn (instead of silently ignoring) when a flag has no effect in the given
  mode: `--direct-read` with `--tar` on create (members are read through the
  buffered parallel tar reader, so O_DIRECT input never engages), and
  `--write-threads` anywhere but `-d --tar` (it only sizes the extractor's
  file-writer pool). Both warnings are suppressed by `-q`.

## v0.14.4 — fix the CPU-only / portable (`USE_NVCOMP=OFF`) build

Two latent bugs broke any build without nvCOMP (the portable static target and
`-DUSE_NVCOMP=OFF`); they had been masked because the CUDA headers transitively
supplied what was missing:

- `INT_MAX` (RAM-frame caps) was used without `#include <climits>` — only
  resolved via a CUDA header. Added the include.
- `peek_first_frame_decomp_size()` was *declared* at file scope but *defined*
  inside the big `#ifdef HAVE_NVCOMP` block, while being called from CPU-only
  paths (the parallel reader's gate, the decompress dispatch, `extract_tar`).
  With nvCOMP off the definition vanished → link error. Moved it to file scope.

Also: `BUILD_STATIC` needs `libzstd.a`, which conda-forge's `zstd` package omits;
`BUILD.md` now documents `conda install -c conda-forge zstd-static`. The static
CPU-only binary builds and round-trips ACLs/xattrs, depending only on glibc.

## v0.14.3 — `--acls` and `--xattrs` for `--tar` (GNU-tar compatible)

`--tar` now stores POSIX ACLs and extended attributes, and `-d --tar` restores
them — both off by default and gated exactly like GNU tar: the flags must be
given on BOTH create and extract or the metadata is ignored, so the default
path pays nothing (no extra syscalls in the layout walk).

Storage matches GNU tar so archives interoperate in both directions: ACLs go in
PAX `SCHILY.acl.access`/`SCHILY.acl.default` text records, xattrs in
`SCHILY.xattr.*`.  Only non-trivial ACLs are stored (a trivial ACL just mirrors
the mode bits); when both flags are set, the `system.posix_acl_*` xattrs are
omitted to avoid duplicating the ACL records.  Verified round-trip three ways:
gzstd→gzstd, gzstd→GNU tar, and GNU tar→gzstd, including directory default ACLs
and the streamed large-file path.

Gathering is parallelized: the single-threaded layout walk stays cheap (just
the `lstat` it already did), and the expensive `acl_get_file`/`llistxattr`
calls run as a parallel pass over the entry list afterward, with a cheap serial
offset-recompute once each member's PAX block size is known.  On restore, ACLs/
xattrs are applied through the secure-by-fd write path (after permissions, so an
access ACL wins; default ACLs via `/proc/self/fd` to keep the O_NOFOLLOW
guarantee) and parallelize through the existing extractor writer pool.

Build: adds an optional libacl dependency (xattr syscalls are in glibc; only
ACLs need the library).  `BUILD_STATIC` links `libacl.a`/`libattr.a`.  Without
libacl, gzstd still builds and `--acls`/`--xattrs` are simply unavailable.

## v0.14.2 — progress bar + summary for `-d --tar` extraction

`-d --tar` showed no progress bar and no completion summary — not even with
`--progress`.  The decompress progress machinery lives in `main()`, wrapped
around the normal single-output decompress path, but `-d --tar` returns early
through a separate dispatch (`extract_tar`) that spawns the decompress workers
itself and never set up the progress thread or printed the summary.  So the
output side of `--tar` was silent while the create side (`compress_cpu_mt`
self-spawns its own progress) was not.

`extract_tar` now spawns the same `progress_loop` (input %% driven by the sum
of the compressed archive sizes) and prints a zstd-style decompress summary on
completion (`archive.tar.zst : <compressed> => <extracted>, <dir> @ <rate>`).
Single-site fix: the worker functions are unchanged, and `extract_tar` was the
only `main()` dispatch that bypassed the progress setup (compress `--tar` and
every per-file decompress already had it).

## v0.14.1 — refuse to write compressed data to a terminal

gzstd never had the guard zstd/gzip/bzip2 have against writing compressed
output to an interactive terminal.  Compressing to stdout when stdout is a TTY
(and `-f` was not given) now errors instead of spraying a binary stream at the
screen — most importantly catching `gzstd --tar ~/backup` with a forgotten
`-o`, which would otherwise dump the whole `.tar.zst` to the terminal.

Compression only (`is_stdout_tty()` check); decompressed output to a terminal
is still allowed, redirects and pipes are unaffected, and `-f`/`--force`
overrides as in zstd.

## v0.14.0 — native parallel .tar.zst creation (--tar)

`tar -cf - dir | gzstd` is limited by tar: it walks the tree and reads every
member on a single thread, so gzstd's parallel readers and CPU/GPU workers
sit starved behind one pipe — worst on directories full of small files,
where the cost is per-file open/stat/read syscall latency.

`--tar` moves the archiving into gzstd.  A single-threaded walk lstat's every
member, applies `--exclude`, detects hardlinks, and computes each member's
byte offset in the virtual tar stream (header length + data + 512-byte
padding are all known from metadata).  The stream is then split into the
usual frame-sized chunks and assembled by a pool of reader threads that pread
the member-file ranges each chunk overlaps **concurrently** — the parallel
read is the whole point.  Chunks are emitted to the existing TaskQueue in
strict sequence order via a small reorder buffer, so the FrameThrottle's FIFO
invariant (see TaskQueue::re_enqueue) is preserved and the CPU, GPU, and
hybrid pipelines are reused unchanged.  Read-ahead is gated to a window past
the push frontier, which keeps every reader busy without ever wedging the
head chunk behind later ones.

Output is a standard GNU-format tar stream (`tar --format=gnu`, the Ubuntu
default) wrapped in zstd: ustar + OLDGNU magic, `L`/`K` long-name/long-link
entries for paths over 100 bytes, and base-256 numeric fields for files over
8 GiB or high uid/gid.  Stores regular files, directories, symlinks,
hardlinks, FIFOs, and device nodes with mode/owner/mtime; sockets are skipped
with a warning, as tar does.  A member that vanishes or changes mid-read is
zero-filled for the missing bytes and reported (exit non-zero, archive still
valid) — the same outcome as GNU tar's "file changed as we read it".
Extraction is unchanged territory: any `tar --zstd -xf` or `gzstd -d | tar
-x` reads the result; listings match `tar --sort=name -cf` byte-for-byte.

List the archive sources after `--tar`; the tar-specific options
(`--exclude`, `--numeric-owner`, `--one-file-system`) may follow `--tar` too,
mixed with the sources, while `-o` and the global flags can sit anywhere.  A
source whose name begins with `-` must be written `./-name` (or placed after
a literal `--`), as with real tar.  New flags: `--exclude PATTERN`
(repeatable glob), `--numeric-owner`, `--one-file-system`.

**Extraction** (`gzstd -d --tar [-C DIR] archive.tar.zst …`) completes the
round trip.  The archive's zstd stream is decompressed on the full parallel
CPU/GPU pipeline and piped into an in-process tar extractor, so decompression
overlaps with file writes — small files are written by a worker pool, large
files stream — beating `gzstd -d | tar -x` on many-small-file archives.  It
reads any standard tar inside zstd (GNU `L`/`K` long names, base-256, ustar
`prefix`, and pax extended headers), not just gzstd's own.  Restores regular
files, directories, symlinks, hardlinks, FIFOs and device nodes with mode and
mtime (owners only as root; directory mtimes applied last so children don't
bump them).  `-C DIR` sets the extraction root (default cwd).

The extractor's file-writer pool size is `--write-threads N` (default
`min(worker threads, 16)`); past ~16 the gain plateaus on filesystem
metadata/journal contention, but the optimum is hardware-dependent — tune it
on the target box.

Extraction is **secure by construction**: every member is created through the
destination dir fd with `O_NOFOLLOW` on every path component (libarchive-style
`openat`/`mkdirat`/`symlinkat`), so a member symlink can never be traversed to
write outside the destination (symlink-escape), and any `..` component is
refused (path traversal).  A leading `/` is stripped.  Both classic tar
extraction CVEs are blocked — verified with hostile archives that leave a
sentinel outside the destination untouched.

Whole-filesystem / OS backups (`gzstd --tar --one-file-system /`) work the
same as GNU tar, and the matching was made GNU-faithful so existing exclude
lists drop in unchanged:

  - `--exclude` is matched against the filesystem path being walked (absolute
    when the source is absolute), so `--exclude=/proc`, `--exclude=/tmp`,
    `--exclude='/home/*/.cache'` all work archiving `/`.  A leading `/` is
    anchored to the source root (drops only top-level `/tmp`, not `var/tmp`);
    a bare name is unanchored (`--exclude .cache` drops every `.cache`); `*`
    spans `/`.  Output is byte-for-byte identical to GNU `tar --exclude`.
  - `--one-file-system` records a crossed mount point as an empty directory
    stub (so it exists on restore) but does not descend into it — pruning
    `/proc`, `/sys`, `/dev`, `/run` automatically.  List real partitions as
    extra sources (`--tar --one-file-system / /home /boot`) to include them.
  - An absolute source's members are stored leading-`/`-stripped with no
    `./` prefix and no synthetic root entry (`/` → `proc/`, `home/...`),
    matching GNU tar.

  - Sockets and unknown file types are *ignored* (GNU tar's "file ignored"
    class): shown only at `-v` and never an error, so the sockets scattered
    through `/run`, `/tmp`, etc. don't spam the log or force a non-zero exit.
    The default is therefore equivalent to GNU tar's
    `--warning=no-file-ignored`.  Genuine read failures (permissions, a file
    that changes mid-read) are still reported and still set a non-zero exit,
    with an accurate message (`cannot read (Permission denied)` vs `file
    changed as we read it`).

  Caveat (same as tar): run as root to read every file, write the archive
  outside the tree being walked, and snapshot (LVM/btrfs) for a consistent
  image of a live system.

POSIX only (Linux/macOS); `--tar` is rejected on Windows builds.

## v0.13.80 — spell out acronyms on first use in --help

Expanded the technical acronyms in the long `--help` on their first
occurrence so the help is self-documenting (prompted by a user not
recognizing EMA in the `--cpu-share` description):

  EMA  → exponential moving average     D2H  → device-to-host
  H2D  → host-to-device                 VRAM → GPU memory
  DMA  → direct memory access           VMA  → virtual memory area
  PCIe → PCI Express                    NVMe → NVM Express
  GC   → garbage collection             MT   → multithreaded
  OOM  → out of memory                  TTY  → terminal

Each is expanded only at its first instance per the established convention;
later uses stay bare — except EMA, which is also expanded in the
`--cpu-share` description (a reader landing there directly shouldn't have to
scroll up to `--hybrid` to learn what it means).  Universal terms
(CPU, GPU, RAM, I/O, OS, JSON, CUDA),
size units (GiB/MiB), and filesystem names (ext4, XFS, ZFS) are left as-is,
as is the deliberately terse short `-h` (its only jargon term, VRAM, kept
bare to preserve the aligned column layout).  Help text only — no behavior
change.

## v0.13.79 — remove --cpu-backlog (redundant with --cpu-batch)

Removed the `--cpu-backlog N` flag entirely.  It was a secondary, post-pop CPU
throttle that duplicated `--cpu-batch` (`cpu_queue_min`): where `--cpu-batch`
declines to pop until the queue holds ≥ N frames (checked atomically inside the
locked pop, before a permit is acquired), `--cpu-backlog` popped first and then
un-popped — pushing the frame back, releasing the permit, and waiting — making
it effectively `--cpu-batch (N+1)` with extra permit/queue churn.  The
"reserve queue depth for the GPU" job it aimed at is already handled
automatically by the adaptive scheduler's `cpu_queue_floor` (scales with
streams × batch × measured throughput, self-disables at the tail) and manually
by `--cpu-batch`.  Nothing depended on it (the only references were two
arg-parse smoke tests that passed `0` = off), and it carried a footgun — unlike
`--cpu-batch`, it was never disabled in `--cpu-only` mode despite its
`[hybrid only]` help.

Dropped: the `Options::cpu_backlog` field, its arg-parse case, the help entry,
and the per-task branch in `cpu_worker` — which removes one lock-taking
`tq->size()` check from the CPU-compress hot path.  `--cpu-backlog` now reports
`unknown option` (exit 2).  The two smoke tests were removed (default expected
test count 227 → 226; one was extensive-only, so extensive 304 → 302).

## v0.13.78 — audit cleanup: --sync-output durability, signal-safe cleanup, RAM-bound single-thread chunk

Four fixes from a full read-through of the previously un-audited code
(orchestration, output path, arg parsing, budget/sizing).  None affects a
normal round-trip; they close durability, signal-safety, and misuse-robustness
gaps.

1. **`--sync-output` now flushes the stdio buffer before fsync.**  The buffered
   (non-O_DIRECT) writer's output `FILE*` has a 1 MiB full-buffer, but
   `fsync_file()` ran `fsync(fd)` before `fclose()` flushed that buffer to the
   fd — so up to ~1 MiB of trailing output was synced only by the later
   `fclose` (which doesn't fsync) and wasn't durable across a crash, defeating
   the flag's purpose for the tail (worst on the atomic path, where the rename
   then publishes a not-yet-durable file).  `fsync_file()` now `fflush()`es
   first.  The O_DIRECT path was already correct (`finalize()` flushes before
   its own fsync).

2. **Temp-file cleanup uses `unlink()` instead of `std::remove()` in the signal
   handler.**  `cleanup_tmp_file()` runs from the SIGINT/SIGTERM handler, where
   the libc `std::remove` wrapper isn't guaranteed async-signal-safe; `unlink()`
   is.  POSIX only; Windows keeps `std::remove`.

3. **Single-threaded compress now RAM-bounds its chunk.**  `compress_cpu_mt`'s
   `-T1` early-return into `compress_cpu_stream` skipped the `check_ram_budget`
   call the multi-thread path runs, so an absurd `--chunk-size` with `-T1` threw
   an uncaught `std::bad_alloc` (→ `std::terminate`) on the input/output buffer
   allocation instead of reducing gracefully.  `compress_cpu_stream` now applies
   the same RAM cap (no-op for normal sizes; matches MT behavior).

4. **Removed dead code:** `RateMatchState::cpu_may_take()` was defined but never
   called (the live CPU gate is `should_cpu_take` + the queue floor).

## v0.13.77 — decompress integrity guard + graceful parallel-reader fallback

Three decompress-path fixes from a correctness review.

1. **GPU decompress now verifies the produced size against the frame header.**
   `gpu_decomp_worker` checked nvCOMP's per-chunk status but then trusted the
   reported output size (`h_actual[i]`) verbatim.  The CPU path gets this check
   for free — `ZSTD_decompressDCtx` rejects a frame whose output differs from
   its declared content size — so the GPU path could silently write a short or
   wrong-length frame and exit 0 on corrupt/malformed input where the CPU path
   and stock `zstd -d` both error.  It now compares `actual` to the header's
   `decomp_size` and throws on mismatch; the existing catch re-enqueues the
   undelivered tail to the main queue, where a CPU worker re-decodes it and
   either succeeds (a transient device glitch — the run completes correctly) or
   dies cleanly with a zstd data error.  No effect on valid archives, where the
   two sizes always agree; verified the guard does not false-positive on a real
   GPU round-trip.

2. **The parallel-prefetch decompress reader falls back instead of dying.**
   A mid-stream frame with no content-size header (e.g. a gzstd archive
   concatenated with a zstd-streamed segment) made the MT reader `die_data`,
   while the single-threaded reader handled the same input by streaming the
   remainder.  The MT reader now records the offending frame's file offset,
   reads `[offset, EOF)` into the caller's buffer, and signals the same
   fallback the single reader uses — the parsed frames are written, then the
   tail goes through the CPU streaming decoder.  Both readers now emit one
   shared `warning:` line (previously the only message here was a `note:`).
   Confirmed byte-identical output on cpu-only, `--hybrid`, and `--gpu-only`.

3. **Bounded the MT reader's frame-spanning re-parse cost.**
   Completing a frame that straddles a 64 MiB block re-runs
   `ZSTD_findFrameCompressedSize()` on the growing carry (zstd has no resumable
   size parser), and growing the carry by a fixed 4 MiB step made that
   re-walk quadratic for frames larger than a block (`--ultra` / huge
   `--chunk-size`: ~16 re-parses per block, each rescanning the whole carry).
   The step now grows geometrically (capped at one block), so a
   straddle-by-a-little still resolves on the first small step while a
   multi-block frame needs only ~log2(block/step) parses per block.  Overshoot
   is bounded at 2× and re-parsed in place, so output is unchanged.

Adds a regression test (`MT reader streaming-fallback`) covering the §2 path:
a > 128 MiB gzstd-archive + piped-zstd tail must round-trip, warn, and match
the single reader.

## v0.13.76 — progress bar for a redirected stdin: auto-show + real percentages

Two fixes for `gzstd -dc < big.zst`, both flowing from "we now know the
input size via fstat":

1. **Auto-show the bar.** The default-verbosity suppression treated any
   non-TTY stdin as a pipe and hid the bar, so a `< file` redirect needed
   `--progress`.  It now suppresses only a TRUE pipe — keyed on whether the
   input size is known (`total_in == 0`).  A redirect from a real file /
   block device has a known size, so the bar shows automatically; genuine
   pipes (`tar -I gzstd`, `cat | gzstd`, a terminal) stay quiet.
2. **Real percentages instead of `---`.**  The progress denominator came
   from `fs::file_size` of a named path, so stdin showed `in:--- out:---`.
   New `known_input_size()` returns the size for a named regular file OR a
   seekable stdin redirect (`fstat` S_ISREG/S_ISBLK), so `< file` now shows
   `in:NN% out:NN%`.  All five total_in sites (4 compress + decompress/test)
   route through it; named-file and true-pipe behavior is unchanged.

A side benefit: the single-threaded streaming compress paths now learn the
pledged source size on a `gzstd < big.bin` redirect too (ZSTD_CCtx_set
PledgedSrcSize), so the frame header carries the content size.

## v0.13.75 — parallel reads from a redirected stdin and block devices (fstat the fd)

The parallel readers (compress pooled + decompress prefetch) gated on a
NAMED regular-file argument, so `gzstd -d < big.zst` — stdin redirected
from a real file — fell to the single-threaded reader even though fd 0 is
fully seekable.  New `probe_preadable_input()` `fstat`s the fd instead of
trusting the path: S_ISREG or S_ISBLK + seekable → preadable, whatever the
fd's origin (named file, `< file` redirect, `< /dev/sdX`).  Pipes/FIFOs/
sockets/ttys (the `tar -I gzstd` stream, process substitution, a terminal)
are correctly rejected and stay on the sequential reader.

Both readers now take an optional borrowed fd: for a redirect they pread
the inherited stdin fd directly (positional, so it coexists with the
buffered peek and never disturbs the FILE* position) and don't close it.
Block devices are sized via SEEK_END (st_size is 0 for them).  --direct-read
is excluded (it owns its single O_DIRECT stream).

So `gzstd -d < big.zst` and `gzstd < big.bin` now get the same multi-reader
speedup as the named-file form.  What still can't: `tar -I gzstd -cf
out.tar.zst /dir` — tar generates the stream into a pipe, so the bytes
never exist as a seekable file; fstat sees the FIFO and we stay sequential
(correctly).  Validated byte-identical across named / `< redirect` / pipe /
process-substitution for cpu/gpu/hybrid, both directions.

## v0.13.74 — help: separate --direct-read and --read-threads (docs only)

The two flags were crammed together as if related.  In the short help a
`--direct-read` note ("one-pass speedup…") was stranded under
`--read-threads`; in the long help they shared one header and the
paragraph described only `--direct-read`, leaving `--read-threads`
undocumented.  They are unrelated beyond being mutually exclusive
(`--direct-read` is always a single O_DIRECT stream; `--read-threads` is
parallelism for the buffered path).  Now each has its own entry:
`--direct-read` keeps its standalone write-up (and notes it's single-stream
/ benchmarking-oriented), and `--read-threads` gets a real description
(parallel readers for the buffered input path, compress pooled reader +
decompress prefetch reader, auto `clamp(threads/8,3,12)`, 1 = single).

## v0.13.73 — progress bar: clamp out% monotonic (stop it going backwards)

The output percentage could move backwards.  Root cause: there is no
file-level "total decompressed size" or "frame count" in a zstd stream —
each FRAME header carries only its own content size, so the total is known
only after the reader has parsed every frame.  `out% = wrote_bytes /
total_out` therefore divides by a denominator that GROWS during reading;
when a burst of highly-compressible frames is discovered, total_out jumps
and the percentage dips.  (The parallel reader made it more visible by
discovering frames in fast bursts.)  Counting frames at the writer doesn't
help — `total_frames` grows during reading too.

Fix: clamp the displayed out% to be monotonically non-decreasing (track a
floor across progress samples).  Worst case is a brief forward stall if the
running estimate overshot, never a backward step; with the fast parallel
reader the estimate phase is short anyway.  Considered but rejected: an
exact compressed-domain out% (comp-bytes-written / file_size, monotonic by
construction) — it would essentially duplicate the in% bar and costs
per-frame memory proportional to frame count.  Cosmetic; affects only the
live -v/--progress bar, not the final summary.

## v0.13.72 — fix: parallel decompress reader double-counted input bytes

v0.13.71's prefetch threads added each block to `m->read_bytes`, but the
decompress workers ALSO count input per frame (the single-threaded reader
relies on them and adds nothing itself).  So `-d` reported 2× the input
("423.78 GiB =>" for a 211.89 GiB file) and the progress bar hit 100% at
the halfway point.  (`-t` was unaffected — its summary derives the input
size differently.)  Fix: the prefetch threads keep only the reader-state
timing (and the -vvv g_perf totals); the workers remain the sole counter
of `read_bytes`.  Display-only — decompressed output was always correct.

Server speedup confirmed on the 432 GiB archive: gpu-only decompress
7.37 → **14.13 GiB/s** (≈2×), cpu-only 15.12 → **18.48** (+22%), -t 18.80.

## v0.13.71 — parallel-prefetch decompress reader (multi-reader)

The v0.13.70 accounting confirmed the decompress reader is the cap on the
cpu-only path: on the server (432 GiB / 27685 frames) the single reader ran
**92% saturated** (io 52.5%, parse 2.5%, task-copy 37.0%, blocked 0.3%) and
the writer was 50% starved waiting for it, holding cpu-only decompress to
15.1 GiB/s.  (gpu-only is 7.4 — decompress, like compress, is a cpu-only
regime on that box, so the cpu-only reader is the lever.)

Compression's N-independent-reader trick can't be copied: decompress frame
boundaries are found by sequential parsing and the zstd magic can appear in
payload, so there is no safe blind resync without a frame index.  Instead,
`stream_frames_to_queue_mt`: K prefetch threads `pread` fixed 64 MiB BLOCKs
of the file (claimed in order via a shared counter) into a bounded ring, and
one consumer parses frames in-place and pushes them — a carry buffer bridges
a frame (or skippable frame) that straddles a block boundary.  This
parallelizes the dominant read I/O; the per-frame copy stays on the consumer
(a later zero-copy pass can remove it).  Reader count scales with the box
(`clamp(threads/8, 3, 12)`, `--read-threads N` overrides), matching compress.

Conservatively gated: engaged only for a seekable regular file > 128 MiB,
not `--direct-read` (O_DIRECT stays single-stream), whose first frame is a
normal known-size zstd frame.  Every other case (stdin, O_DIRECT,
unknown-size single-frame, foreign archives) stays on the single-threaded
`stream_frames_to_queue`, which still owns all the fallback paths;
mid-stream anomalies are a hard data error (as the single reader treats
truncation), never a mid-stream handoff.  Validated byte-identical against
the single reader across chunk sizes 1/4/16/64/128 MiB (frames smaller than,
equal to, and larger than a block — exercising heavy boundary-spanning and
the multi-block carry path), all consumers (cpu/gpu/hybrid), `--read-threads`
1–16, and highly-compressible input.  Server speedup to be measured.

---

## v0.13.70 — decompress reader-state accounting (-v); groundwork for parallelizing the decompress reader

Toward a "multi-reader" decompress path: first, make the decompress reader
measurable, because its architecture differs fundamentally from compress.
The compress reader slices the input at fixed byte offsets, so N threads
grab independent chunk indices trivially.  The decompress reader
(`stream_frames_to_queue`) must PARSE variable-length frame boundaries
sequentially — frame N+1's start is unknown until frame N's header is
walked (`ZSTD_findFrameCompressedSize`) — so N independent parsers aren't
safe without a frame index (the zstd magic can appear inside payload).

`stream_frames_to_queue` now feeds the same Meter reader-state counters the
compress path uses, with a new `parse` bucket: `[READER] io | parse |
task-copy | blocked-downstream` at -v.  io = read syscall; parse =
frame-boundary walk; task-copy = the per-frame `assign` out of the parse
buffer; blocked-downstream = `queue.push()` stalling on the bounded queue
(which means the GPU/CPU consumers, not the reader, are the faucet).  The
verdict distinguishes a saturated reader (and which sub-component to attack
— copy → zero-copy frame reader, parse → needs an index, io → faster
source) from a reader that's merely blocked because the consumers can't
keep up.  The `[WRITER]` three-state report (v0.13.59) already covered
decompress — it just hadn't been exercised on a decompress run.

Diagnostic-only (no behavior change).  Early local signal (tiny runs, not
representative): parse is ~0.1% — the serial spine is NOT the bottleneck,
so parallelizing it isn't the lever; copy slightly exceeds io.  The actual
optimization (likely a zero-copy frame reader if the high-core Gen4 server shows the reader is
copy-bound, or pipelined raw-read I/O if io-bound) is to be chosen from a
real-workload -v run on the server.

---

## v0.13.69 — deadlock: gpu-only DECOMPRESS with locked --gpu-batch × many streams

The v0.13.68 fix covered compress; probing its decompress analog found a
real but DIFFERENT deadlock.  Confirmed on the server: `-d --gpu-only
--gpu-batch=64 --gpu-streams=16` on a many-frame archive hangs at ~74%
(must ^C); the same flags with `--hybrid` complete cleanly.

Mechanism — classic permit / head-of-line, not the v0.13.68
pool-exhaustion: each of 128 streams (8 dev × 16) does `bp->acquire(pop_n)`
upfront, then blocks waiting for a FULL locked batch.  The throttle budget
is sized FROM `device × streams × batch`, so GPU demand can consume the
entire permit pool; the in-order writer then wedges behind a head-of-line
frame that no stream can pop.  Hybrid survives because its CPU workers are
a fine-grained relief valve (they pop one low-seq frame at a time);
gpu-only has none (rescue threads fire only on GPU failure).

Fix: in gpu-only decompress, a locked `--gpu-batch` becomes a CAP, not a
hard floor — reuse the unlocked soft minimum (`min(pop_n, 4)`), so streams
take whatever is queued and release the excess permits (the worker already
does this) instead of sequestering a full batch.  Hybrid keeps the honored
full-batch wait.  One-line predicate change (`locked_batch && !opt.gpu_only`).

Could NOT be reproduced on the 2-GPU workstation (VRAM-fit shrinks
per-stream demand on 10 GiB cards; 2 devices don't open deep enough
head-of-line gaps), so correctness was verified locally (round-trip + full
suite) and the deadlock cure is to be confirmed on the server.  Adds a
gpu-only decompress wedge canary to the bounded-queue test section.

---

## v0.13.68 — deadlock: locked --gpu-batch × many streams wedges under a bounded queue

Reproduced on the server: `--gpu-streams=16 --gpu-batch=64` hung at 45%
(^C needed).  Mechanism: a user-pinned batch makes streams wait for FULL
batches in pop_batch_greedy, and each stream acquires its FrameThrottle
permits BEFORE blocking (that order prevents a different deadlock).  128
streams × VRAM-fit ~20 frames ≈ 2,560 permits sequestered by sleeping
streams; the bounded queue (pool 1,504) can rarely present 16–25
consecutive frames past 96 CPU workers, in-flight output accumulates
until the 8,576-permit throttle exhausts, CPUs can't pop the frames the
writer needs, the writer can't release permits — circular wait.  Same
disease as the v0.13.67 floor lockout: a blocking reservation sized
against a queue that can no longer back it.

Guard: when aggregate locked demand (active streams × pop_n) exceeds
half the queue's depth ceiling, full-batch waits relax to min_n=1 with a
one-time warning — the user's batch remains the pop CAP.  Auto-tuned
(unlocked) runs already use min_n=1 and were never exposed.  Note: the
decompress path's bounded queue + locked batches has the same
theoretical hazard (its bound is TaskQueue max_depth, not the pool);
not yet plumbed — locked decompress batches there remain unguarded.

## v0.13.67 — bounded producer zeroes the AUTO queue floor; the server's compress verdict is in

The v0.13.66 cap/4 clamp (376 frames) was not enough: the GPUs' combined
appetite (16 streams × ~200 batch) exceeds the reader's ~1000 frames/s
supply, so queue depth never builds past ANY substantial floor, and the
AUTO mode is a latch under starvation (CPU <5% share ⇒ factor=4 ⇒ CPU
stays at <5%).  Server floor sweep on the 432 GiB tar:

| floor          | GiB/s | GPU share of frames |
|----------------|-------|---------------------|
| auto           | 15.74 | ~92%                |
| factor 0.5     | 15.69 | ~90%                |
| **off**        | 18.17 | ~37%                |
| (cpu-only ref) | 18.93 | 0%                  |
| (gpu-only ref) | 15.59 | 100%                |

Under a continuously-refilling bounded queue the reservation is obsolete
(GPUs pop min_n=1; the auto-tuner adapts batch size), so a bounded
producer now zeroes the AUTO floor; explicit --hybrid-floor=nominal /
--hybrid-floor-factor are honored, clamped to cap/4.

**Compress verdict for the dual-socket 8-GPU server**: CPU pool alone
18.9 GiB/s, H100 pool alone 15.6, both together 18.2 — the shared memory
fabric, not either engine, is the ceiling.  GPUs add ~nothing to compress
on this box (and with the floor fixed, no longer subtract).  The week's
totals on this workload: 5.73 → ~18–19 GiB/s (75.5 s → ~23 s), all of it
from the input path; engine mix was never the lever.

## v0.13.66 — queue floor clamped to the pooled queue's depth ceiling; gpu-only reader count fixed

The first fed-pipeline server runs falsified the "hybrid loses to
bandwidth" reading and exposed two interacting bugs:

- **Hybrid (11.63 GiB/s vs cpu-only's 18.93) wasn't contention — the CPU
  pool was locked out.**  Ratio arithmetic (gpu-only 50.23%, cpu-only
  48.70%, hybrid 50.19%) shows the GPU took ~97% of frames.  The GPU
  queue floor (streams × batch ≈ 900 frames) predates the pooled reader:
  with mmap the whole file sat in the queue and the floor was harmless,
  but the pool bounds depth at ~1500, the GPUs' batch gulps held depth
  below the floor near-permanently, and `may_take` refused the CPU pool
  forever.  Side effect: the starved CPU pool never accumulated EMA
  samples, so the tail-yield never armed.  Fix: the pooled producer
  declares its depth ceiling (`HybridSched::set_queue_depth_cap`) and
  `update_queue_floor` clamps the floor to a quarter of it.
- **gpu-only was reader-capped, not GPU-capped.**  Its 14.17 GiB/s run
  used 3 readers: the auto count divided `cpu_threads`, which is 0 in
  gpu-only mode.  Now scales from `resolve_cpu_threads` (machine
  parallelism) regardless of mode.  The H100 pool's true ceiling is
  ABOVE 14 GiB/s and still unmeasured — the auto-tune GiB/s figures are
  per-stream, not pool (a misreading that fed the earlier wrong verdict).

Corrected picture on the server: CPU pool ~19 GiB/s, GPU pool ≥14 — a
correctly-scheduled hybrid should beat both.  To be measured.

## v0.13.65 — GPU/hybrid compress gets the multi-reader pooled path (was still on fread+copy)

The first "fed pipeline" hybrid run on the server exposed the scoping gap:
the v0.13.63 multi-reader only served cpu-only compress; hybrid still ran
the single fread+assign reader and capped at 6.11 GiB/s while cpu-only did
~17 ([READER] said it directly: 1 reader, task-copy 33.8%).

Wiring it needed two things the CPU path didn't:

- **One pool buffer per Task, no refcount.**  The host-chunk→gpu_chunk
  subchunk split (one read backing many tasks) was an fread-efficiency
  artifact; the pooled reader simply preads at gpu_chunk granularity, so
  the existing single-owner slot release (`direct_buf`) works unchanged
  and seq == chunk idx stays dense.
- **Slot recycling on every GPU input lifecycle.**  Hybrid keeps batch
  inputs alive for rescue and previously never released them on success
  (owned vectors freed via destructor; pool slots would leak → reader
  starvation → hang).  Releases added at: batch fully delivered (both
  completion paths), the delivered-prefix of a mid-delivery throw (the
  rescue handoff erases those frames), and the rescue worker after
  recompression.  gpu-only keeps its existing release-after-H2D.

Pool sizing for the GPU path: cpu_threads + 32/reader + 1024 (GPU batches
hold slots from pop to delivery), clamped to file size and a quarter of
MemAvailable; the plain-pages prefault is capped at 4 GiB (beyond that,
first-touch faults amortize — MADV_NOHUGEPAGE already prevents the toxic
THP attempts).  Verified on the 2-GPU workstation: hybrid and gpu-only
--no-mmap round-trip clean through the pool (1250 frames through a ~1150-
slot pool — a single leaked slot would hang it).

## v0.13.64 — reader count scales with the worker pool; pool sized for the readers

Server sweep of `--read-threads` on the 432 GiB tar: 3 → 7.46 GiB/s,
6 → 15.61, 8 → 16.28, 12 → **18.74** (23.1 s; the same file took 75.5 s
at v0.13.59).  Two saturation signals emerged in the [READER] line: per-
thread io fell (96.5 → 75.4%) while blocked-on-pool climbed (0 → 15.3%) —
the readers were starving for pool buffers, not hitting the device.
Caveat recorded: by the 12-reader run much of the file sat in the 1.5 TB
page cache, so absolute numbers are cache-flattered; the scaling shape
and the blocked-on-pool growth are the trustworthy signals.

Auto reader count is now `clamp(threads/8, 3, 12)` — 3 on the 24-thread
workstation (measured optimal), 12 on the 96-worker server (best
measured) — and the pool gains 32 buffers per reader (512 MiB each step;
the threads+128 sizing predates multi-reader).  `--read-threads N` still
overrides.

## v0.13.63 — parallel buffered readers: fan the kernel copy out, keep the device stream sequential

The v0.13.62 analysis left one lever: the buffered pooled reader's wall is
the per-thread cold-destination copy_to_user (~3.5 GB/s node-local), not
the device (~10 GB/s buffered).  Probe on the server confirmed page-cache
reads parallelize where O_DIRECT contends: two simultaneous buffered dd
streams = 17.4 GB/s aggregate (vs 9.9 single; O_DIRECT measured 1 stream
4.5 / 4 streams 3.0 — that rule does NOT carry over).

`pooled_read_chunks` now runs N reader threads (default 3, `--read-threads
N` overrides; O_DIRECT and the scratch path stay at 1) pulling chunk
indices from a shared atomic counter against ONE fd.  Interleaved indices
— deliberately NOT partitioned file regions — keep the offset stream
near-sequential (one readahead context keeps working) and bound the
queue's seq skew to ~N.  A partitioned design would flood the ResultStore
with distant-seq frames, exhaust FrameThrottle permits, then the pool, and
starve the region the writer needs — the re_enqueue FIFO-invariant
deadlock.  EOF and abort propagate via a shared done flag; each index is
preaded by exactly one thread, so the emit set is dense and seq-exact.

With the per-thread copy wall broken, the pre-6.4 kernel gate from
v0.13.62 is removed — old-kernel large files use the multi-reader pooled
path instead of fread+assign (which remains for stdin/pipes).  Measured
(workstation, --no-mmap, 20 GiB warm, /dev/null, median of 3): 1 reader
3.11 s → 3 readers **1.67 s** — within ~15% of the mmap zero-copy path.
The `[READER]` report now prints per-thread percentages with the thread
count.  Server expectation: ~8–10 GiB/s vs the 5.7 fread floor (and 2.14
single-pooled); to be validated on the 432 GiB tar.

## v0.13.62 — buffered pooled reader gated to ≥6.4 kernels; cold-destination copy was the real culprit

v0.13.61's THP hypothesis was falsified on-box: identical 2.14 GiB/s with
MADV_NOHUGEPAGE and THP=madvise.  The controlled experiments named the
true mechanism:

- `dd` (same file, same 16 MiB buffered reads): **9.9 GB/s** — one hot
  reused buffer; the kernel copy lands in cache-warm lines.
- gzstd's pool: **2.14** — 224×16 MiB of cycling buffers means every
  pread's destination is cache-cold and recently touched by a remote
  worker core; copy_to_user has no NT stores, so cold destinations pay
  RFO + writeback (~3× the memory traffic).
- `numactl --cpunodebind=0 --membind=0`: **3.46** — removes the
  cross-socket hop (node distance 32 vs 10), recovering the NUMA share.
  `--interleave=all`: no change (spreading coldness isn't warmth).
- Old fread+assign's 5.7 decomposes cleanly: fread's kernel copy at 9.6
  (hot staging buffer, the dd pattern) + assign at ~14 (glibc switches to
  non-temporal stores for L3-sized memcpys).  Two copies, each in a fast
  regime, beat one copy in the slowest regime.

Resolution: the pooled buffered reader keeps its win where measured
(+72%, ≥6.4-kernel workstation, --no-mmap) and is gated off on pre-6.4
kernels (same per-VMA-locks proxy as the mmap gate), restoring
fread+assign's 5.7 floor on the server.  Open lever for that box:
parallel buffered readers — the O_DIRECT single-stream rule (1 stream
4.5, 4 streams 3.0 GB/s) does NOT obviously apply to page-cache reads,
and N readers would break the single-thread cold-copy wall (~3.5 GB/s
local) against the device's ~10.  Needs a dual-dd probe before building.

## v0.13.61 — pooled-reader pool takes plain pages on pre-6.4 kernels (v0.13.60 regressed 2.7× on the server)

v0.13.60's buffered pooled reader, +72% on the workstation, measured 2.14
GiB/s on the server — 2.7× WORSE than the fread path it replaced, with the
reader 99.7% saturated inside pread.  Working hypothesis (consistent with
the box's documented pre-6.4 pathologies, pending on-box validation): the
pool's MADV_HUGEPAGE + sparse one-byte-per-2 MiB prefault was built for
O_DIRECT DMA-segment merging; under buffered reads the kernel copies into
the buffer instead, and on that kernel THP never engages — so the sparse
prefault leaves 1 of 512 pages mapped and every copy page-faults into a
THP-eligible VMA, with compaction attempts on a fragmented box.

The same machinery is a WIN on modern kernels (huge-page-backed
copy_to_user: 3.10 s vs 3.85 s plain, workstation --no-mmap A/B), so it is
gated, not removed: `DirectReadPool::init(want_thp)` keeps the THP
prefault for O_DIRECT (DMA always needs it) and for buffered mode on
kernels ≥ 6.4 (per-VMA-locks check as the vintage proxy); older kernels
get MADV_NOHUGEPAGE explicitly (system THP=always must not re-enable the
pathology) plus a full memset prefault — deterministic, zero faults during
reads.  Workstation verified at parity (3.08–3.13 s); server expectation
is fread-class or better (≥ 5.7, target ~9 GiB/s), to be validated.

## v0.13.60 — buffered zero-copy reader: the fread fallback's hidden copy halved intake

Diagnosed live on the server with the v0.13.59 writer accounting plus the
-vvv breakdown: compressing a 432 GiB tar, the run reported upstream-bound
with 96 workers averaging only ~17.5 busy, and the arithmetic exposed the
reader thread as ~99% saturated — 45 s inside fread (9.6 GiB/s) plus ~30 s
of UNTIMED `t.data.assign` copying every byte a second time.  Effective
intake 1/(1/9.6 + 1/~15) ≈ 5.7 GiB/s — exactly the observed throughput.
The fread fallback is the default on that box because the pre-6.4-kernel
mmap gate declines large files.  `--direct-read` was not the answer there:
its O_DIRECT ceiling (~4.1–4.5 GB/s, matching the old dd measurement)
loses to the page cache's buffered 9.6 GiB/s with readahead.

Fix: `odirect_read_chunks` generalized to `pooled_read_chunks(o_direct)` —
the same single-stream pooled zero-copy reader, with O_DIRECT now just an
open flag.  When mmap is declined (kernel gate, --no-mmap, open failure)
and the input is a regular file, the cpu-only compress reader now preads
buffered (readahead intact, POSIX_FADV_SEQUENTIAL) straight into pooled
buffers and emits view tasks: one kernel→buffer copy instead of two, and
the pool acquire doubles as producer backpressure.  fread+assign remains
only for stdin/pipes.  Measured (workstation, --no-mmap, 20 GiB warm
input, /dev/null, median of 3 alternating): 5.34 s → **3.10 s (+72%
throughput, 3.66 → 6.30 GiB/s)**.  Path ranking per box: mmap (zero
copies) where the kernel allows; buffered-pooled (one copy) otherwise;
O_DIRECT only when the device's raw rate beats its buffered rate.

Not yet covered: the GPU/hybrid reader keeps its copy fallback — its
host-chunk-to-subchunk split means one pooled buffer would back many view
tasks, and slot recycling is single-owner (`direct_buf`); needs a
refcounted release before the pool can serve it.  Decompress readers
untouched (compressed input is 3–30× smaller; far less reader-bound).

Also: reader-state accounting (input mirror of v0.13.59's writer states,
compress readers only): `[READER] io | task-copy | blocked-on-pool` at -v
with its own verdict; task-copy > 0 IS the double-copy diagnosis.  The
nvcomp fread fallback is instrumented too, so hybrid runs on the server
will now show the copy share directly.

## v0.13.59 — writer-state accounting: every run reports whether it pegged the writer

Motivation: runs that grind below the output device's capability are
maddening to triage because the candidate causes — sink saturated, frame
stragglers stalling the in-order writer, or upstream compute/read too slow —
all look identical from the outside (disk not pegged, machine "busy").
Observed concretely on the server benchmarks: cpu-only compress pegs the
~3 GiB/s NVMe at 3.3–3.5 GiB/s while hybrid grinds at ~2.0 with strictly
more hardware in play.

The output side is now modeled as three mutually exclusive states, measured
always (two timestamps per wait segment / write call — negligible):

- **write-path busy** — inside physical write/seek calls (AsyncWritePool
  worker).  ≥85% of wall time means the sink is the bottleneck; mission
  accomplished, nothing upstream can help.
- **head-of-line** — writer idle waiting for the next in-sequence frame
  while LATER frames sit buffered: a straggler (slow GPU batch, unlucky
  frame) is capping output, the pipeline's fault.
- **starved** — writer idle with nothing buffered at all: compute/read
  simply hasn't produced; the engines are the bottleneck.

The buckets accrue on different threads (busy on the AIO worker, the waits
on the writer thread), so percentages overlap and need not sum to 100; each
is independently meaningful against run wall time.  At `-v` each run prints
the three percentages plus a one-line interpreted verdict
(`writer_verdict()`), e.g. `stragglers — writer idled waiting for the next
in-order frame while later frames sat buffered`.  These three signals are
the regime detectors a future --adapt mode would switch on (io-bound /
pipeline-bound / compute-bound).

## v0.13.58 — hybrid compress: slow GPU no longer sets the makespan (tail-aware intake)

On the workstation (slow-GPU box: GPU pool ~1.1 GiB/s vs CPU pool ~15),
hybrid compress ran 26–45% behind cpu-only.  Root cause: in adaptive mode the
GPU popped batches unconditionally, including from the near-empty queue at
the end of the run — a 1 GiB batch takes the slow GPU pool ~2 s while the
entire CPU pool sits idle waiting for it.  The damage was almost entirely a
*tail* effect: mid-run greedy intake is harmless (work conserves, both pools
stay busy), but whoever holds frames when the queue runs dry decides when the
run ends.

**Failed approach (v0.13.56, never released):** cap GPU intake at its
EMA-measured fair share of throughput per ~0.5 s scheduler window, lifting
the cap when the producer finishes so the tail drains greedily.  Falsified by
A/B on a 100 GiB page-cached input: identical to v0.13.55 within noise.  Two
design errors: (1) with mmap input (the default) the reader enqueues view
tasks for the whole file in milliseconds, so "producer done" fires at t≈0 and
the cap was disabled for effectively the entire run; (2) lifting the cap for
the tail re-creates the exact failure being fixed — the tail *is* where
greedy intake hurts.

**Shipped approach:** `should_gpu_take()` (adaptive, compress only) yields
only at the tail — the GPU starts a new batch only if the queue holds more
than ~1.3 GPU-batch-times of CPU work: `(depth − batch)/cpu_ema ≥
1.3 · batch · streams/gpu_ema` (frames are uniform, so frame counts over EMA
byte-rates compare directly).  The check arms once the producer is done
(t≈0 for mmap, so it is live the whole run; a streaming producer keeps the
queue shallow and would otherwise starve the GPU).  The first yield latches
`tail_yield_`, which zeroes `cpu_queue_floor()` and wakes sleeping CPU
workers — otherwise CPUs would refuse the very frames the floor had reserved
for the GPU that just declined them.  The GPU remains the drain of last
resort below `--cpu-queue-min` (CPUs refuse those depths; mutual yield would
hang).  A yielded GPU worker (all streams idle) parks on the queue CV via
`wait_for_gpu_yield()` — event-driven like the CPU side's `wait_for_cpu`, no
polling, no fixed sleeps — and wakes exactly when a decision input changes:
any pop (`take_front_locked` is the centralized dequeue point and notifies
when a waiter is parked; a free integer check otherwise), the queue
draining, or a scheduler tick moving the EMAs (the one input with no queue
event; `notify_gpu_yield_waiters()` covers it).  The wait predicate
evaluates `should_gpu_take_at(depth)` from the QueueState snapshot — like
wait_for_cpu's predicate, it must not call back into TaskQueue.
Fixed-share mode is unchanged (its share check oscillates per-batch by
design and must keep spinning).

Measured (workstation, 100 GiB medium-compressibility input, page-cached,
output to /dev/null, median of 3 alternating runs): hybrid 8.57 s →
**6.31 s** (−26%), variance collapsed (7.8–9.4 s → 6.2–6.3 s); cpu-only
reference 5.94 s.  Hybrid now lands within ~6% of cpu-only on this box
(≈ CUDA init cost) instead of 30–45% behind.  A fast GPU is unaffected:
its fair share is large, so the yield condition only trims the last batch.

## v0.13.55 — portable binary failed to start on machines without an NVIDIA driver

The released portable binary aborted at load time on any machine without the
NVIDIA driver:

    gzstd: error while loading shared libraries: libnvidia-ml.so.1:
    cannot open shared object file: No such file or directory

Root cause: NVML ships with the *driver* (no static archive exists), and the
build linked the CUDA toolkit's stub — which still writes a DT_NEEDED entry
for `libnvidia-ml.so.1` into the binary.  The ELF loader resolves DT_NEEDED at
startup, before `main()`, so the binary could not start at all on driver-less
machines — even though the CPU-only paths never call NVML.  The release
workflow had been masking this in its own smoke test by putting the stub on
`LD_LIBRARY_PATH`.

Fix: NVML is now loaded at **runtime** via `dlopen` (loader shim at the top of
gzstd.cpp; same function names, so call sites are unchanged).  No link-time
dependency remains.  With the driver present, behaviour is identical
(verified: NVML device ranking and PCIe-gen detection still work).  Without
it, the wrappers report failure and the existing fallbacks take over —
free-VRAM device ranking, sysfs PCIe probe — and CUDA detection already
handled the missing driver gracefully through cudart.  Verified by blocking
the driver libraries from dlopen: `--version` and full CPU round-trips work.

Build changes: CMake no longer searches for/links `nvidia-ml` (HAVE_NVML is
on whenever the GPU backend builds on UNIX; `dl` is linked explicitly).  The
release workflow's smoke test now runs with **no** stub on a driver-less
runner — the real user environment — and additionally fails the release if
`ldd` ever shows a hard `libnvidia-ml` dependency again.  BUILD.md updated:
the NVIDIA driver is now optional at runtime.

## v0.13.54 — full-file code review: 3 reproduced critical bugs + GPU-failure CPU fallback

A line-by-line review of gzstd.cpp found three serious decompress-path bugs
(all reproduced on the workstation before fixing) plus a batch of smaller
correctness and robustness issues.  Common thread: error/fallback paths written
in one era weren't revisited when later features (bounded queues v0.13.29/41,
async GPU bringup v0.13.13/15) changed the invariants they relied on.

**1. Silent data loss on concatenated zstd streams (reproduced).** When a
multi-frame input contained a frame with no content-size header *after* at
least one parseable frame (e.g. `cat a.zst b.zst` where b came from
`... | zstd` on a pipe — valid zstd), `stream_frames_to_queue` buffered the
rest of the input in `raw_data` and returned, but both decompress paths only
consumed `raw_data` when *zero* frames had parsed.  Result: the tail was
silently dropped — truncated output, exit 0.  Repro: 2 MiB concatenated input
→ 1 MiB output.  Fix: after the writer drains the parsed frames, append the
tail via the CPU streaming decoder (`decompress_from_buffer`), with a note.

**2. SIGABRT on hybrid/gpu-only decompress of streamed-zstd files
(reproduced).** The `fallback && n_frames == 0` early-return in
`decompress_nvcomp` never joined the deferred GPU-bringup thread: the joinable
`std::thread`'s destructor called `std::terminate` — an abort (exit 134) on
*every* hybrid or gpu-only decompress of a file whose first frame lacks a
content size.  It also iterated `gpu_workers` while the bringup thread could
still be appending to it (data race).  Fix: join the bringup thread first
(matches the normal path's ordering comment).  The fallback message is now a
visible warning explaining that the mode fallback is for data safety.

**3. Deadlock decompressing >64 MiB frames at low thread counts
(reproduced).** `cpu_decomp_worker`'s single-frame-file detection (v0.13.1)
blocked on `producer_done` for every frame over 64 MiB.  Once the input queue
became bounded (v0.13.29), the producer could be blocked in `push()` while
every worker sat in that wait — circular wait, hard hang.  Repro: 1 GiB of
`--chunk-size 128` frames, `-d --cpu-only -T 2` → infinite hang (now 1.2 s).
Fix (also a small speedup): only frame seq 0 can be "the single frame", and a
second pushed task disproves single-frame instantly — oversize frames in
multi-frame files now skip the wait entirely; the genuine wait is a timed
re-checking loop that can't deadlock against a blocked producer.

**4. All-GPU failure in --gpu-only now falls back to CPU instead of dying
(or hanging).** Previously: all GPUs failing at init aborted the reader and
died with EXIT_GPU_FAIL — but mid-run failures weren't counted by that check,
so with pipe input the producer blocked forever on the bounded queue (no
consumer), and with mmap input the writer's watchdog killed the run with a
misleading "internal error: writer stuck" (exit 1).  Now the last terminally
failing GPU worker runs `gpu_only_cpu_fallback`: a full CPU pool drains the
queue (maximum remaining throughput), a warning explains the fallback is for
data safety, and the run completes with exit 0.  Exit code 5 is now reserved
(documented in --help); the reader-side abort checks are gone — the queue
always has a consumer.

**5. Missing nvCOMP per-chunk status check in the compress sync-drain path
(silent corruption risk).** The async-poll completion path validated
`h_stats[i]`; the sync drain did not — a failed chunk's garbage comp_size
would have been delivered as output.  Both paths now check.

**6. GPU-decompress rescue re-enqueued empty tasks.** Inputs were released
("it's on the GPU now") *before* the kernel ran, so any failure after that
point re-enqueued zero-byte tasks — the retry was dead on arrival.  Inputs are
now released per-frame after successful delivery.

**7. Partial-batch failure accounting (compress + decompress).** A throw
mid-delivery rescued/re-enqueued the *whole* batch including frames already
pushed to the ResultStore: duplicate-seq work and permit drift.  Streams now
track `delivered` and the failure paths handle only the undelivered tail, with
exact permit release.

**Smaller fixes:**
- `--fast=abc`, `-M abc`, `--memlimit=`/`--memory` garbage, and overflowing
  `-NNN…` levels crashed with an uncaught std::stoi/stoull exception; now
  clean usage errors (exit 2).  Malformed level flags like `-5x` were silently
  swallowed (and compressed at the default level!); now "unknown option".
- Corrupt frame headers claiming absurd content sizes aborted via uncaught
  `std::bad_alloc`; now a clean data error (exit 4).
- `tasks_done` was double-counted on CPU decompress (writer + worker), skewing
  the progress bar's frame-level percentage; the writer is the sole counter.
- O_DIRECT + sparse: a sparse seek from a non-4 KiB-aligned position left the
  fd offset unaligned, making the next O_DIRECT flush fail with EINVAL
  (reported as "disk full?").  Sparse skips through the DirectWriter now only
  happen from aligned positions; unaligned zero runs are written instead
  (correct, merely less sparse — unreachable with gzstd's own 16 MiB frames).
- `DirectWriter::write_all`/`pwrite_all` looped forever on `write() == 0`;
  `robust_fwrite` could loop on a stale `EINTR` in errno.  Both bounded.
- gpu_worker counted `--direct-read` view bytes into the progress meter that
  the O_DIRECT reader had already counted (mirrors cpu_worker's guard).
- Removed the write-only `Options::remove_input` field (`--rm` works via
  `keep = false`).

## v0.13.53 — reconcile --help / -h with actual operation (docs only)

Help text had drifted from the code after a lot of churn. Audited every flag in
both screens against the parser and the runtime defaults; the flag names all
matched, but three stated defaults/details were stale. Docs follow code (no
behaviour change):

- **`--gpu-batch` default.** Both screens said `default: 16`. Actual: 8 for
  compress (`DEFAULT_GPU_BATCH_CAP`), 16 for decompress auto-scaled up by input
  size (64 above 10 GiB, 256 above 75 GiB). Updated both screens to state the
  mode-dependent default.
- **`--gpu-devices` auto.** Long help and the `Options::gpu_devices` comment said
  auto = "all GPUs for compress, 1 for decompress". Both decompress paths
  (synchronous and deferred) actually use all available GPUs, same as compress.
  Corrected the wording and the struct comment.
- **`--cold`.** Documented in the short help but only mentioned in passing in the
  long help; gave it its own entry in the long-help I/O section.

## v0.13.52 — fix GPU-compress hybrid rescue dropping mmap/view tasks (silent data loss)

Two correctness fixes in the failure/edge paths, found by code review.

**1. Hybrid GPU-compress rescue lost zero-copy (mmap) frames.** When a GPU failed
mid-batch in hybrid mode, the worker's `catch` re-routed the in-flight chunks to the
CPU rescue queue by reconstructing each task as `Task{ seq, data }`. But the default
compress reader for a regular file is the zero-copy **mmap** reader, whose tasks carry
their bytes in `view_ptr`/`view_len` with an **empty `data` vector**. Rebuilding from
`.data` alone dropped the view, so the rescue worker compressed **0 bytes** and emitted
an empty zstd frame for that sequence number — the output decompressed cleanly but was
**silently missing those chunks' bytes**. This corrupted output on the exact path the
rescue mechanism exists to handle gracefully (VRAM exhaustion / driver error), whenever
the input used the mmap reader (the default). Fix: `std::move` the whole `Task` into the
rescue queue instead of reconstructing it — the mmap region outlives the rescue join so
the view stays valid, and the move also preserves `direct_buf` ownership and avoids
copying owning data. The sibling paths were already correct (GPU-only failure and the
decompress failure path both `re_enqueue` the intact tasks); only this hybrid
`rescue->push` reconstructed the task.

**2. Throttle permit over-release in single-frame streaming decompress.** The CPU
decompress worker's streaming branch (single giant frame, e.g. `--ultra` / `zstd`
output) acquired exactly one `FrameThrottle` permit before the pop but pushed N result
chunks, and the writer releases one permit per frame written — so the writer
over-released by `(actual_chunks - 1)`, drifting `permits_` above its cap (and making
`in_flight()` read negative). Harmless in practice (only fires on a single-frame file at
end of work, and in-flight memory is independently bounded by the per-thread decomp
pool), but a real acquire/release asymmetry. Fix: acquire one additional permit per
streamed chunk beyond the first, so acquires match releases. Deadlock-free — chunks
ascend from the lowest seq, so the writer always drains the oldest first and frees a
permit.

## v0.13.51 — `--direct-read` for decompress (was compress-only)

`--direct-read` only honored the compress reader; decompress silently fell back to
buffered `fread`, so a benchmark or cold run would read the compressed input warm.
Wired O_DIRECT into `stream_frames_to_queue` (the shared reader behind both
`decompress_cpu_mt` and `decompress_nvcomp`): when `--direct-read` is set on a
regular-file input it opens its own O_DIRECT fd on `opt.input` and reads 4 KiB-aligned
`READ_CHUNK` (4 MiB) blocks into an aligned bounce buffer, copying into the existing
frame-parse buffer (frame boundaries don't align to reads, so a bounce copy is
required — but it's the same copy `fread` did internally). The FILE* `in` is at offset
0 here (`peek_first_frame_decomp_size` rewinds) and is simply unused for reading while
O_DIRECT is active; the streaming-fallback path (unknown content size) reads through
the same helper. Falls back to `fread` if O_DIRECT can't be set up. Byte-identical
decompressed output vs the buffered reader on cpu-only and gpu-only across multi-frame
+ unaligned inputs; round-trip clean; 290/290 extensive. (Known minor gap: the
single-giant-frame streaming path, `decompress_stream_from_file` for inputs above
`SINGLE_FRAME_STREAM_MIN` — i.e. `--sliding-window` / `zstd` outputs — still reads
buffered; gzstd's own chunked output is multi-frame and uses the wired path.)

## v0.13.50 — `--direct-read`: one contiguous pool region (fix the ~340 KiB request split)

After v0.13.49, `--direct-read` was still only 1.55 GiB/s vs the page-cache path's
4.46. `-vvv` showed the reader spends 99% of wall *inside* `pread` (279 s of a 283 s
run) yet moves only 1.55 GiB/s — not starvation (cores were 97% idle waiting on it),
the `pread`s themselves are slow. `iostat` found why and `dd` confirmed it:

| | rareq-sz | aqu-sz | throughput |
|---|---|---|---|
| dd (16 MiB O_DIRECT) | **638 KiB** | 8.5 | 3.8 GB/s |
| our reader | **340 KiB** | 15 | 1.8 GB/s |

Same requests/sec; dd's DMA requests are ~2× larger — the whole gap. Our 150 pool
buffers were separate `posix_memalign(16 MiB)` calls, and because 16 MiB is below
v0.13.48's `M_MMAP_THRESHOLD` they came from the **fragmented heap**: physically
scattered 4 KiB pages, so O_DIRECT's scatter-gather list hits the driver's
`max_segments=127` and each 16 MiB read shatters into ~340 KiB requests.

Fix: allocate the **whole pool as one large region** (> the mmap threshold ⇒ a fresh
dedicated `mmap`) and slice it. On an unfragmented box its pages fault in as long
physically-contiguous runs that merge into a few big DMA segments, so a `pread`
reaches the device's `max_sectors_kb`. Measured on the server: **rareq-sz 340 →
~1230 KiB** (≈ the 1280 KiB max), throughput **1.55 → 1.96 GiB/s**, run 4m43 → 3m41.
The region is 2 MiB-aligned + `MADV_HUGEPAGE` + lightly pre-faulted as
belt-and-suspenders for THP where it's healthy, but **THP did not engage on the 5.15
server** (`AnonHugePages=0`) — the win is the contiguous allocation, not huge pages.
Byte-identical cpu+gpu, round-trip clean, 290/290 extensive.

Remaining limiter (not a code issue): with `--direct --direct-read` on one drive,
O_DIRECT reads (~1.9 GB/s) and the O_DIRECT output writes (~0.8 GB/s) **contend for
the same device queue** — iostat shows ~2.7 GB/s mixed at 80% util. The page-cache
path avoids this only because its reads are free (served from RAM), leaving the whole
drive for writes. Reading and writing on separate drives removes the contention; on a
single big-RAM box the buffered path remains the throughput king and `--direct-read`'s
value stays honest-cold benchmarking + not evicting other users' cache.

## v0.13.49 — `--direct-read`: single-stream + zero-copy reader

Two `dd` facts settled the design. (1) On this NVMe a *single* O_DIRECT stream does
**4.5 GB/s** (4.9 at 128 MiB), while **4 independent streams collapse to ~3.0 GB/s
aggregate** (0.77 each) — concurrent O_DIRECT *contends*, it does not scale. So the
v0.13.46/47 multi-threaded reader was wrong for this hardware. (2) A single `dd`
stream already saturates the drive, yet our pipeline extracted only ~1.5 GB/s of
that 4.5 — because every chunk did a 16 MiB `memcpy` from the O_DIRECT buffer into
the Task, and on the 256-core box that copy competes for memory bandwidth with the
compressors and stalls the read stream between requests.

Rewritten as **one stream, zero copies**:
- **Single reader.** Dropped the work-stealing/multi-thread machinery; one
  uninterrupted O_DIRECT stream is fastest here.
- **Zero-copy (CPU path).** `pread` lands straight in a pooled 4 KiB-aligned buffer
  (`DirectReadPool`); the Task aliases it as a `view_ptr` (like the mmap path) and
  the worker recycles the slot on `release_input()`. No per-chunk copy — the stream
  reads continuously. `pool->acquire()` blocks when all buffers are in flight, so
  the pool *is* the producer backpressure (the queue byte-cap is a no-op for
  zero-byte view tasks). Pool sized to keep every worker fed plus a read-ahead
  backlog (`threads + 128`, capped by file size and 1024); a buffer is held only
  from pread until compression finishes (not during write), so peak RSS stays
  bounded. Read-byte metering moved fully to the reader for these views (workers
  skip `direct_buf >= 0` tasks) — no double-count.
- **GPU path unchanged.** It splits each host chunk into gpu subchunks, so one
  owning buffer per read doesn't map to one Task; it keeps the copy (pool == null,
  single scratch buffer), where PCIe dominates anyway.

Expected to close most of the 1.5 → ~4.5 GB/s gap on the 256-core box (capped by
the compressor at ~3.5); the win is memory-bandwidth-bound so it doesn't show on a
low-core workstation (where the copy was never the bottleneck) — local runs confirm
byte-identical output (cpu + gpu, multi-chunk + unaligned tail), clean round-trip,
correct read metering, and bounded RSS. 290/290 extensive. (v0.13.48's mallopt stays
— it still helps the GPU/fread/decompress and output-buffer paths.)

## v0.13.48 — Recycle frame buffers (pin mmap threshold) — kill the munmap TLB-shootdown storm

After v0.13.47 fixed the reader's access pattern, `--direct-read` still ran at only
~1.35 GiB/s on the 432 GiB compress while `dd` showed the same NVMe doing **4.5
GB/s single-stream O_DIRECT** (4.9 at 128 MiB blocks) — so ~70% of the drive was
left on the table, and the limiter was ours, not the disk. Root cause: our
per-frame buffers are 16 MiB (32 MiB ultra), above glibc's 32 MiB dynamic
`mmap` ceiling, so under the 4-producer/N-consumer hand-off the allocator `mmap`s
each chunk and **`munmap`s it on free**. The munmap — not the page faults — is the
killer: tearing down a 16 MiB mapping triggers a TLB shootdown (an IPI to every
other core), whose cost scales with core count, so on the 256-core server it
dominated (the run's enormous `sys` time).

Fix: at startup, `mallopt(M_MMAP_THRESHOLD, 128 MiB)` + `mallopt(M_TRIM_THRESHOLD,
256 MiB)` so frame buffers come from the heap and freed chunks are reused from the
arena bins — no munmap, no shootdown, no re-grow. Local A/B on a 4 GiB direct-read
(no 256-core shootdown tax to begin with): wall **2.97 s → 1.58 s** (~1.9×, 1.35 →
2.53 GB/s); minor faults essentially unchanged (255k → 243k), confirming the win is
the syscall/shootdown churn, not the faults. Expected to help *more* on the
256-core box where the shootdown cost is highest. Benefits every path that churns
large per-frame buffers (compress, decompress, GPU host staging), not just
`--direct-read`. Peak RSS stays bounded by the in-flight cap. 290/290 extensive.

## v0.13.47 — Fix the v0.13.46 parallel O_DIRECT reader (strided → work-stealing)

v0.13.46's parallel reader was catastrophically slow in practice — on a 432 GiB
real-data run it crawled at ~95 MiB/s (vs 1.47 GiB/s for the v0.13.44 single
thread it replaced, and 3.51 GiB/s for the page-cache path), starving the whole
pipeline so it looked like the writer had stalled. Two flaws, now fixed:

- **Strided assignment was the killer.** Each thread took `idx += ODIRECT_READERS`
  (a 64 MiB stride). That is only sequential while the threads stay in lockstep;
  the first copy/push stall desynchronises them and the in-flight reads scatter up
  to `N*cap` apart into a random-looking pattern. With O_DIRECT (no kernel
  readahead) that destroys NVMe locality. Replaced with **work-stealing**: a shared
  `atomic<size_t> next_idx` hands the next chunk to whichever reader is free, so the
  N outstanding reads are always on *consecutive* chunks — a contiguous window that
  slides forward at queue depth N. Near-sequential access, still deep-queued.
- **Shared fd.** All readers `pread` one fd; each now opens its own O_DIRECT fd to
  avoid serialising on the shared file struct.

seq is still the chunk index (file position), so output stays ordered and the
ordered writer keeps RAM bounded regardless of completion order. Output
byte-identical to the normal reader on cpu-only and gpu-only across multi-chunk +
unaligned-tail files; round-trip clean; 290/290 extensive.

## v0.13.46 — Parallel O_DIRECT reader (--direct-read was QD1-bound)

The v0.13.44 --direct-read used a single synchronous pread loop, which runs the
NVMe at queue-depth 1: O_DIRECT has no kernel readahead, so the drive sits idle
between reads and the reader starves the workers (measured ~1.47 GiB/s on a 432
GiB cold read vs ~4.5 the drive can do). Now ODIRECT_READERS=4 reader threads
each pread strided, 4 KiB-aligned host chunks, keeping multiple requests in
flight (deep queue) to saturate the NVMe. seq is assigned deterministically from
the chunk index (file position), not a shared counter, so frames stay correctly
ordered/contiguous despite out-of-order completion across threads (completion is
tracked by push count). The per-chunk copy lives in a noinline enqueue helper so
vector::assign is analysed in a clean context (avoids a -Wnonnull false positive
on the alloc when inlined into the threaded reader). Output byte-identical to the
normal reader on cpu-only and gpu-only across multi-chunk + unaligned-size files;
290/290.

## v0.13.45 — Document the mmap kernel-gate + `--direct-read` in `--help`

Help text only.  The extended (`--help`) and short (`-h`) entries for
`--mmap`/`--no-mmap` now describe the v0.13.43 auto-gate (kernel <6.4 + input
>4 GiB → fread; mmap on for 6.4+), and a new extended `--direct-read` entry
explains the O_DIRECT page-cache bypass, its one-pass-speedup vs honest-cold-
benchmark uses, that it implies fread and is independent of `--direct` (output).

## v0.13.44 — `--direct-read`: O_DIRECT input reader (page-cache bypass)

A first-class O_DIRECT input reader for the compress path (`compress_cpu_mt` and
`compress_nvcomp`).  O_DIRECT transfers straight disk→buffer, **bypassing the page
cache entirely** — it neither reads from nor populates it.  Two payoffs:

1. **One-pass speedup (real feature):** compressing a backup touches every input
   byte exactly once and never re-reads it, so the page cache provides zero reuse
   benefit — it's pure populate + writeback-pressure overhead.  Reading around it
   skips that.  (Caveat: O_DIRECT loses kernel readahead, so it needs large reads
   to keep the disk saturated — gzstd's 16 MiB chunks + pipelining cover that.)
2. **Honest cold benchmarking with zero system impact:** because nothing is cached
   or evicted, every run reads cold from disk deterministically — no warm-cache
   skew, and critically **no `kcompactd` storm**.  (The old `--cold` =
   `fadvise(DONTNEED)` *populates then drops* the cache; dropping a huge file
   fragments free memory and wakes kernel compaction, stalling the whole box.
   O_DIRECT sidesteps the drop entirely — see project_mmap_kernel_storm.)

Shared helper `odirect_read_chunks` (4 KiB-aligned `pread` into a `posix_memalign`
bounce buffer; EOF handled via O_DIRECT's short read; falls back to fread if
O_DIRECT can't be set up, e.g. tmpfs).  Takes precedence over mmap (mmap *is* the
page cache, so it can't bypass it).  Output is byte-identical to the normal reader
on both cpu-only and gpu-only (verified); round-trips incl. unaligned-size files;
290/290.  `--direct` (O_DIRECT *output*) and `--direct-read` (O_DIRECT *input*)
are independent — combine for a fully cache-bypassing run.

## v0.13.43 — Auto-fall-back to fread for large inputs on pre-6.4 kernels (mmap_lock storm)

On a 256-core box at kernel **5.15** (pre-6.4, no per-VMA locks), compressing a
432 GiB file with the default mmap reader cost **13–41%** vs `--no-mmap`: the
single `mmap_lock` rwsem serialises ~108M page faults across 256 cores (the file
fits in 1.5 TiB RAM, so it's pure lock contention, not eviction).  The v0.13.22
"mmap on everywhere" decision was calibrated on 20 GiB test files where the storm
was a tolerable regression; it scales with fault count = file size, so real
backup-scale files hit it ~20× harder.

Fix: gate mmap on **kernel version + input size** — on kernels `< 6.4`
(`kernel_has_per_vma_locks()` via `uname`), fall back to fread for inputs
`> 4 GiB`.  6.4+ kernels and small files are unchanged (mmap stays on, where its
zero-copy wins and few faults don't storm), so this is *not* the kernel-only gate
that regressed cold small files in v0.13.20–22, and *not* a core-count gate.
`--mmap`/`--no-mmap` hard-override it; the gate auto-retires when the box reaches
≥6.4.  Verified gate-off on a 6.17 box; 290/290.

---

## v0.13.42 — Fix `-T` with no numeric value crashing (uncaught `std::stoi`)

`-T`/`--threads` parsed their separate value with a raw `std::stoi(argv[++i])`,
so a bare `-T` followed by a non-numeric token — `gzstd -T --cpu-only file`, or
`-T file.zst`, or a trailing `-T` — threw an uncaught `std::invalid_argument`
and `abort()`ed (core dump).  The attached `-T4` form had the same unguarded
`stoi` (`-Tx` would crash too).

Fix: the separate form now only consumes the next token as the thread count when
it actually looks like an integer (new `looks_like_int` helper); otherwise the
token is left for normal parsing and `-T` falls back to the default (auto) thread
count — no crash, no error.  Both the attached (`-T4`, `--threads=N`) and
consumed-value paths now go through `parse_int_value`, which reports a clean
usage error (exit 2) on a genuinely malformed attached value (`-Tx`) instead of
aborting.  `-T0` (= all cores), `-T N`, `-T4`, `--threads=N`, `--threads N` all
unchanged.  Verified across the extensive suite's `-T` cases.

---

## v0.13.41 — Extend the byte cap to the compress producer (pipe/stdin RAM safety)

(Committed together with v0.13.40.)  The 7.8/v0.13.40 queue cap was decompress-only,
but **compress had the same exposure**: for a regular file the producer mmaps the
input (zero-copy views, no heap — a 1 TB file streams in bounded RAM regardless of
size), but for a **pipe/stdin** input it falls back to `fread`, reading frames onto
the heap, and the compress queue had *no* cap.  A producer that outruns the workers
(or a writer/disk bottleneck that blocks them on throttle permits, so they stop
popping) could then buffer the entire input in RAM → OOM.

Fix: call `queue.set_max_bytes(floor * host_chunk/2)` on both compress queues
(`compress_cpu_mt`, `compress_nvcomp`).  Bytes only, no frame cap — mmap views are
`data.size()==0` so it's a **no-op for the common regular-file path** and bounds only
fread.  Reuses the same `TaskQueue` machinery (`max_bytes_`/`queued_bytes_`/
`take_front_locked`, `!q_.empty()` deadlock guard) added in v0.13.40.

Demonstrated (Gen3, 2 GiB incompressible piped via `cat | gzstd --cpu-only -T2 -19`,
slow workers so the warm-cache pipe outruns them; max-RSS):

| build                  | maxRSS   | time    |
|------------------------|----------|---------|
| before (no compress cap) | 2232 MiB | 193.2 s |
| after  (byte cap)        |  568 MiB | 191.5 s |

−75% peak RSS, throughput unchanged.  At default level the workers keep up so the
queue never grows and the cap doesn't engage (both builds ~153 MiB) — it's a safety
net for the slow-worker / slow-output pathological case, where without it a large
pipe input OOMs.  Pipe + mmap round-trips verified across cpu/gpu/hybrid; 213/213.

## v0.13.40 — Byte-aware decompress reader queue cap (ROADMAP 7.8 follow-up): bound queue RAM

The 7.8 reader queue cap (`set_max_depth`) bounds the queue by **frame count**
(`parallelism * slack`), so the RAM it holds scales with compressibility — an
incompressible file (near-full-size compressed frames) buffers ~4× the RAM a
compressible one does for the same frame count.  Added a parallel **byte** cap to
`TaskQueue`: the reader now blocks when `frames >= max_depth_` **OR**
`queued_bytes_ >= max_bytes_`, whichever binds first.  `queued_bytes_` tracks
owned heap (`Task::data.size()`) so zero-copy mmap views (size 0) are correctly
ignored; a `!q_.empty()` guard on the byte cap guarantees progress even when a
single frame exceeds the whole budget (no deadlock).  Byte accounting is
centralized in one `take_front_locked()` helper so it can't drift across the pop
sites.  Budget = `floor * 8 MiB` (~half a standard 16 MiB frame per slot), set at
both decompress readers; tunable via `--throttle-factor`.

Measured (Gen3 2×2080Ti, `gpu-only` decompress, 4 GiB, max-RSS / best-of-3):

| profile         | RSS before | RSS after  | Δ      | time |
|-----------------|------------|------------|--------|------|
| low_compress    | 2127 MiB   | 1902 MiB   | −11%   | flat |
| medium_compress | 1748 MiB   | 1603 MiB   | −8%    | flat |

Throughput-neutral, RAM down 145–225 MiB — biggest on incompressible input, as
intended.  The reduced buffering for big frames *could* matter on a much faster
reader/consumer ratio (Gen4, 8×H100) — flagged in-code and tunable; validate
there.  Round-trips verified cpu/gpu × incompressible + sparse `zeros` (deadlock-
checked with timeouts); 213/213 tests pass.  CPU decompress RSS is unaffected (its
RAM is the output-buffer throttle budget, not the input queue).

---

## v0.13.39 — Default-init allocator on `FrameBuf`: eliminate the resize() zero-fill on direct-write paths

The `assign()` fixes (v0.13.37/38) only cover handoffs where bytes pass through
host memory first.  The **direct-write** paths — CPU decompress (`ZSTD_decompressDCtx`
writes straight into the buffer), GPU non-pinned D2H — still did
`out_buf->resize(decomp_size)` before the write, value-initializing (zeroing) the
grown region that the decompressor then fully overwrites.  Profiling CPU decompress
(callgrind) pinned this at **`_M_default_append`→memset = ~16% of instructions**:
the decomp buffer *pool* is large (`throttle_budget/n_workers`), so for any file up
to ~pool-size frames almost every frame grows a never-yet-used buffer and pays the
full ~16 MiB zero (it amortizes only on very large files).

Fix: `FrameBuf` now uses a `default_init_allocator<char>` (a `std::allocator`
subclass whose no-arg `construct()` default-initializes instead of value-initializes),
so `resize()`-grow leaves the new region uninitialized rather than zeroed.  Safe:
every producer fully fills `[0,size())` before the buffer is read (ZSTD / cudaMemcpy /
assign / memcpy), and no writer reads past `size()` — `DirectWriter` copies exactly
`size()` bytes into its own aligned buffer, so a buffer's `[size(),capacity())` tail
never reaches disk.  Ripple was contained: the typedef, 5 `make_shared` sites, and
the per-thread compress `scratch` (now `FrameVec` so it still `swap`s with the pooled
output).

Confirmed: `__memset` 16.2% → 1.9%, `_M_default_append` zeroing gone.  **Throughput-
neutral** in wall-clock, though — those cycles were parallel across workers and
overlapped with the memory-bandwidth/writer bottleneck, so this removes provably-
wasted CPU cycles + ~0.9× output-size of memory-write traffic per decompress rather
than adding speed.  Kept deliberately as resource-waste elimination.  All 30
compress×decompress combinations (5 profiles × cpu/gpu/hybrid × cpu/gpu decompress)
bit-identical incl. the sparse `zeros` path; 213/213 tests pass.

## v0.13.38 — `assign()` for the GPU decompress pinned readback

Extends v0.13.37's `assign()` change to the GPU decompress D2H readback pinned path
(`gpu_decomp_worker`, both completion paths): `h_out->resize(actual)+memcpy(pin_slot)`
→ `h_out->assign(pin_slot, pin_slot+actual)`.  Here `actual` is a full decompressed
frame (~16 MiB), so the avoided zero-fill is far from tiny when the pooled buffer
grows.  Non-pinned direct-D2H keeps `resize()` (dst must be pre-sized; superseded by
the v0.13.39 allocator anyway).  Round-trips verified; 213/213 tests pass.

---

## v0.13.37 — Use `assign()` for buffer handoffs to drop the residual resize() zero-fill

Follow-up to v0.13.36.  Three handoff sites took a recycled pooled `FrameBuf`,
`resize(csz)`'d it, then `memcpy`'d `csz` bytes over it — value-initializing
(zeroing) the grown region only to immediately overwrite it.  Replacing
`resize(csz)+memcpy` with `vector::assign(src, src+csz)` does the identical copy
but copy-constructs straight from the source, so it never zeroes — same bytes,
no wasted memset.  Sites:
- CPU compress worker, well-compressible (memcpy) branch.
- GPU compress D2H readback, **pinned** path (async-poll + sync-drain) — the bytes
  already pass through the pinned host slot, so `assign` from that slot applies.

This is the clean, local alternative to the `FrameBuf` default-init allocator
considered (and rejected as too invasive) for the GPU readback — no type change,
no extra copy, strictly less work than what it replaces.  The GPU **non-pinned**
direct-D2H fallback still needs `resize()` (the dst must be pre-sized before
`cudaMemcpy` writes into it; `assign` can't source from device memory) — that's
the slow fallback path, left as-is.

Throughput-neutral by design — the eliminated zeroing was the 0.59%-class
residual measured on a non-bottleneck host thread (see the closed GPU-readback
ROADMAP check) — but it stops doing provably useless work.  Round-trips verified
on cpu-only + gpu-only across all four profiles; 213/213 tests pass.

---

## v0.13.36 — CPU compress: stop zero-filling the output buffer every frame

Profiled the cpu-only compress path (callgrind, no root: `valgrind --tool=callgrind`
on the real `-T>1` per-frame path — note `-T1` takes the separate
`compress_cpu_stream` streaming path).  After zstd's own entropy kernels (~67%,
all BMI2-SIMD and not ours to touch), the single largest *gzstd-attributable*
cost was `std::vector<char>::_M_default_append → memset`, ~7% of total
instructions: `compress_one_cpu_frame` did `out.resize(bound)` before every
`ZSTD_compress2`, value-initializing (zeroing) up to ~16 MiB per frame — pure
waste, since ZSTD only writes `[0,csz)` and never reads the dst buffer, and the
`[csz,bound)` tail is undefined padding anyway.  Because the buffer was then
shrunk to `csz`, the next frame re-grew and re-zeroed `(bound − csz)` bytes —
worst on compressible data, where `csz` is tiny so nearly the whole buffer was
re-zeroed each frame.

Fix: `compress_one_cpu_frame` now grows the reusable per-thread buffer to
`compressBound` **once** (grow-only, never shrunk) and returns `csz` explicitly
instead of resizing the buffer down.  `resize()` then zeroes once on a thread's
first frame and no-ops thereafter.  The poorly-compressible "zero-copy swap"
branch (ROADMAP 7.4) resizes the swapped-in buffer down to `csz` after the swap
(a shrink — no zeroing).  Output bytes are identical (no ratio change).

Measured on the 24-core workstation, cpu-only `-T8`, 4 GiB inputs, compute-bound
(output to /dev/null), best of 5:

| profile          | before     | after      | gain  |
|------------------|------------|------------|-------|
| high_compress    | 17.1 GiB/s | 19.5 GiB/s | +14%  |
| medium_compress  |  8.9 GiB/s |  9.8 GiB/s | +9.5% |
| low_compress     | 11.1 GiB/s | 11.4 GiB/s | +2.8% |
| mixed            | 12.0 GiB/s | 12.4 GiB/s | +3.0% |

Round-trips verified on all four profiles; 213/213 tests pass.  (The win amortizes
over frames-per-thread, so it needs frames ≫ threads to show — a 4-frame/4-thread
microbenchmark gives each thread only its one-time first-frame zeroing and hides
it.)

---

## v0.13.35 — HybridSched: don't cap CPU when no GPU is active (ROADMAP 7.10)

Completes the 7.10 deep-dive by auditing the third target, `HybridSched`.  No
deadlock, missed-wakeup, or `gpus_waiting_` accounting bugs found: in fixed-share
mode `should_cpu_take`/`should_gpu_take` can never both be false (so one engine
always takes), the queue floor is enforced atomically in `may_take` and correctly
skipped in fixed mode, `push()` wakes one CPU per task while the exit paths
`notify_all`, and `gpu_got_data()` always fires before any throwing CUDA op.

One robustness gap fixed: in fixed `--cpu-share` mode, if the GPU(s) never
register (still initializing) or all fail/exit mid-run, nothing advances
`gpu_taken_`, so the share cap (`cpu/total < share+0.02`) stalls the main CPU
workers until the producer finishes — wasting the whole production phase before
the drain fast-path recovers.  `should_cpu_take` now short-circuits to `true`
when `active_gpu_streams_ == 0`, letting CPU run unrestricted whenever no GPU is
present to consume its share.  Adaptive mode already handled this via the
`gpus_waiting_`/floor path; healthy fixed-share runs keep `active_gpu_streams_ >
0` throughout, so this is a no-op for them (verified: `--cpu-share 0.5` hybrid
round-trip unchanged, GPU active throughout).

213/213 tests pass.  This closes the 7.10 audit — all three targets (auto-tuner,
failure rescue, HybridSched) are now done.

---

## v0.13.34 — GPU-failure rescue: fix permit leak + stranded-batch hang (ROADMAP 7.10)

Deep-dive audit of the VRAM-exhaustion / GPU-failure rescue path (one of the
7.10 targets) found two correctness bugs in the worker catch blocks, both of
which only fire when a device fails mid-run — exactly when graceful rescue
matters most.

1. **FrameThrottle permit leak (compress + decompress).**  The throttle invariant
   is "the popper acquires one permit per frame; the writer releases it after the
   frame is written."  The success paths honour this (the GPU bulk-acquires
   `pop_n` permits and the writer releases them one-by-one).  But both catch
   blocks handed in-flight frames to the rescue queue / re-enqueued them
   **without releasing the GPU's permits** — and the receiver (rescue worker, or
   the next popper after re_enqueue) re-acquires a fresh permit per frame.  That
   leaked one permit per rescued frame, up to `streams × per_stream_batch` per
   device failure.  Since the auto budget floor is `devices × streams ×
   per_stream_batch`, losing a device's permits could starve the surviving
   rescue/CPU workers into deadlock.  Both catches now release the held permits
   on hand-off (matching the in-loop decompress re_enqueue, which already did).

2. **Stranded batch on submit-time failure (compress only).**  The compress
   catch guarded rescue on `C.busy && !C.batch.empty()`, but `C.busy` isn't set
   until *after* the H2D copies and the `nvcompBatchedZstdCompressAsync` launch —
   any of which can throw.  A launch failure therefore left the just-popped batch
   un-rescued: those frames never reached the sequence-ordered `ResultStore`, so
   in hybrid mode (no abort) the writer blocked forever on the missing seq.  The
   guard is now `!C.batch.empty()` (matching the decompress side); both success
   paths clear `C.batch`, so a non-empty batch is always popped-but-undelivered
   and safe to rescue without duplicate-output risk.

Also removed dead code surfaced by the same audit: the per-stream
EXPLORE/REFINE/SETTLE batch-size tuner (the `TuneState` enum + `tune_*` /
`refine_*` fields in the compress `StreamCtx`, its ~26-line save/restore across
buffer reallocation, and the identical unreferenced block in the decompress
worker's per-stream struct).  It was fully superseded by the cross-GPU
`SharedTuneState` hill-climb that all streams/devices share — same class as the
7.7 `SequentialDispatcher` removal.  No behavioural change; batch size already
came solely from `shared_tune`.

213/213 tests pass.

---

## v0.13.33 — Recycle GPU compress output buffers; CLAUDE.md line count

Finishes the compress side of ROADMAP 7.2 (the decompress side landed in
v0.13.24).  Both GPU-compress completion paths (`gpu_worker` async-poll and sync
drain) did a fresh `make_shared<std::vector<char>>(csz)` per frame for the D2H
readback buffer; `StreamCtx` now owns the same recycled `out_pool` as the
decompress path (`acquire_out_buf`, `use_count()==1` reclaim, lazy growth to two
batches, drain-wait past the cap).  Lower-value than the decompress pool —
compress output is the *compressed* bytes (small for compressible data) — but it
removes the per-frame allocation churn on the GPU/hybrid compress paths, most
relevant for incompressible input where csz approaches the chunk size.

Round-trips verified on gpu-only and hybrid compress (incompressible + mixed);
213/213 tests pass.  This closes 7.2 — GPU result buffers are now pooled on both
compress and decompress.

Also: corrected the stale `~6,400 lines` figure in CLAUDE.md (the file is ~9,900).

---

## v0.13.32 — CPU compress: zero-copy swap for poorly-compressible frames (ROADMAP 7.4)

`cpu_worker` compressed into a per-thread `scratch` buffer, then `memcpy`'d the
result into a pooled `FrameBuf` for the writer.  For poorly-compressible data
(`mixed` ~50%, `low` ~90%) `csz` is a large fraction of the chunk, so that was a
near-full-chunk memcpy per frame on exactly the profiles where compress is
slowest.

Now, when `csz >= in_size / 2`, the worker `std::swap`s the scratch buffer
straight into the pooled `FrameBuf` (zero-copy) and takes the pool slot's old
buffer as the next scratch.  Well-compressible output (`csz` small) keeps the
memcpy path — the copy is cheap there, and swapping would leave every pool slot
carrying scratch's full `compressBound` capacity (a memory regression for tiny
output).  The threshold confines the capacity overhang to slots that already
hold large frames.

Correctness verified (round-trips on incompressible/swap path, zeros/memcpy path,
and mixed/both, multi-threaded); 213/213 tests pass.

**Benchmark verdict (Gen5, 256-core):** the throughput change is **within run
noise** — cpu-only `low` compress (the only profile that crosses the threshold)
moved +4%, identical to the memcpy-path controls (`high` +4%, `medium` +3%) and
no bigger than the swings on paths 7.4 can't touch (gpu-only compress, all
decompress, ±6%).  That's expected: the eliminated memcpy (~14 MiB at memory
bandwidth, ~1 ms) is only ~1–2% of per-frame compress time at level 3, below the
measurement floor.  **Kept anyway** — the old memcpy was pure data-shuffle
overhead (scratch → pool), so the swap path does strictly less work, is correct,
and carries a negligible RSS overhang (only `low` swaps, and its `csz` ≈ 14.4 MiB
is close to the 16 MiB `compressBound`, so ~1.6 MiB/slot; `mixed` at 49.9% stays
on memcpy).  No throughput regression, leaner code path.  ROADMAP 7.4 closed.

---

## v0.13.31 — Fix two Gen4+ regressions from the --direct default (sparse + log tag)

Two test failures surfaced on a Gen5 box, both fallout from making `--direct` the
default on Gen4+ (v0.13.25/26) — they only manifest where `--direct` auto-engages,
so Gen<4 never saw them.

- **Sparse output defeated by preallocate.**  Default `--direct` decompress uses
  the DirectWriter, which preallocates the output (`fallocate`); the sparse path
  then `lseek`s over zero regions whose blocks are *already allocated*, so the
  file came out fully allocated instead of sparse (a real disk-bloat regression
  for zero-heavy data, not just a test artifact).  Fix — **punch-hole hybrid**:
  `seek_forward` now `fallocate(FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE)`s the
  skipped region when the file was preallocated, deallocating those blocks back
  to a hole.  Keeps preallocate's extent-stall-free dense writes AND restores
  sparseness.  `write_sparse` now coalesces consecutive zero blocks into one
  skip, so it's one punch per zero run, not one per 4 KiB.  Best-effort:
  filesystems without punch support degrade to non-sparse (never incorrect).
  Verified (forced `--direct` on a Gen3 box): 64 MiB zeros decompress → 16 blocks
  sparse vs 131080 dense, both byte-correct; random-data `--direct` round-trip
  unaffected.

- **`[ASYMMETRIC]` log tag collision.**  The v0.13.25 `--direct` auto-default
  reused the `[ASYMMETRIC]` tag and (correctly) runs before the
  `backend_user_set` return, so on Gen4+ it logged `[ASYMMETRIC] … --direct` even
  with an explicit `--cpu-only`/`--hybrid` — tripping the asymmetric tests that
  assert explicit backends silence `[ASYMMETRIC]`.  Retagged that line
  `[O_DIRECT]` (it's an I/O decision, not backend selection).  No behavior change.

259/259 tests pass.  Validate the punch-hole + `[O_DIRECT]` retag on the Gen4+
box where the failures originally appeared.

---

## v0.13.30 — Review cleanups: --sync-output under --direct, unaligned load, dead code

Three small fixes from the Phase 7 review (ROADMAP 7.5–7.7):

- **7.5 — `--sync-output` was a no-op under `--direct`.** When the O_DIRECT writer
  owns the output, the `FILE* out` is closed and nulled, so the `fsync_file(out)`
  path in `main` never ran — `--direct --sync-output` returned without ever
  flushing.  Now `main` fsyncs the DirectWriter's own fd (device write cache +
  the size metadata from finalize's ftruncate) when `sync_output` is set.
  Confirmed via strace: `--direct --sync-output` issues one fsync, `--direct`
  alone issues none.

- **7.6 — `is_all_zero` did an unaligned `size_t` load.**
  `reinterpret_cast<const size_t*>(p)` on a `vector<char>::data()` pointer is UB
  on strict-alignment targets.  Replaced with a constant-size `memcpy` into a
  `size_t` (same wide load on x86, portable elsewhere).

- **7.7 — removed the dead `SequentialDispatcher` class.**  Superseded by the
  per-GPU result slots (v0.11.11); no callers remained (verified — the type and
  its methods appeared only in its own definition).  ~46 lines of concurrency
  surface gone.

259/259 tests pass; sparse and `--direct --sync-output` round-trips verified.

---

## v0.13.29 — Bound the decompress reader: queue-depth backpressure (slow-consumer RSS)

The FrameThrottle bounds *popped-but-unwritten* frames, but nothing bounded the
TaskQueue *ahead* of the workers.  On decompress, `stream_frames_to_queue` reads
each compressed frame into a fresh `Task.data` vector and pushes it; when the
consumer is slower than the reader — classically `--gpu-only` decompress, which
is D2H-bound — the reader races ahead and buffers the *entire* compressed input
in RAM.  The v0.13.24 Gen4 isolation caught this: gpu-only `-d` of a 9.75 GiB
input held 11.3 GiB RSS (vs ~1.9 GiB for cpu-only/hybrid), and it's a latent OOM
on inputs larger than RAM.  (ROADMAP 7.8; the original "hybrid excess faults"
hypothesis there was disproven — that was the buffered-write storm.)

`TaskQueue` gains an optional `max_depth_` (0 = unbounded, the default): `push()`
blocks on a new `space_cv_` once the queue is full, and the pop paths a bounded
queue uses (`try_pop_one`, `pop_batch_greedy`, `try_pop_one_cpu`) plus `set_done()`
wake it.  `re_enqueue` (push_front) bypasses the cap so it never blocks.  Both
decompress paths set the cap to a pipeline-depth multiple
(`max(THROTTLE_MIN_FRAMES, parallelism * slack)`), so queued RAM is O(pipeline),
not O(input) — skipped under `--no-throttle`.  The cap is deliberately ≥ the
auto-tuner's batch needs, so it bounds RAM without constraining GPU batch growth
(no throughput risk).  Compress queues are never bounded; with mmap input they're
zero-copy views anyway.

Verified: cpu-only/gpu-only/hybrid decompress round-trip with no deadlock (the
producer blocks and resumes correctly); a slow-consumer gpu-only `-d` of a 3 GB /
~2861-frame incompressible input holds 1.79 GiB RSS (queue capped, not the whole
input buffered); 259/259 tests pass.

---

## v0.13.28 — Size the compress throttle from the resolved chunk, not opt.chunk_mib

`compress_cpu_mt` built its `FrameThrottle` from `opt.chunk_mib`, but the frame
size actually used is `host_chunk` (= the resolved `chosen_mib`), which can be
auto-bumped for `--ultra` (e.g. level 22 forces a 128 MiB chunk) or shrunk by
`check_ram_budget`.  `compute_throttle_budget` divides `avail/2` by the frame
size for its RAM-cap term, so a stale 16 MiB there mis-sizes the in-flight cap:
on `--ultra` it under-counts in-flight RAM (thinks frames are 16 MiB when they're
128 MiB — 8× under), which on a RAM-constrained box could admit more frames than
memory holds.  Pass the resolved `host_chunk` instead.

Default (non-ultra) runs are unchanged — `opt.chunk_mib == chosen_mib == 16`
there.  Demonstrated at `-v`: a `--ultra -22 -T4` run now reports
`[THROTTLE] … 4.00 GiB in-flight max` (32 × 128 MiB) instead of the old
512 MiB (32 × 16 MiB).  The GPU compress path already used `chosen_mib`; the two
decompress paths keep `opt.chunk_mib` as a heuristic (the true decompressed frame
size isn't known until the stream is parsed, after the throttle is built).
259/259 tests pass.  ROADMAP Phase 7.3.

---

## v0.13.27 — Accept bundled short flags (`-dc`, `-dkf`, …) for zstd/gzip compat

`parse_args` exact-matched each argv token, so a bundled short-flag group like
`-dc` was rejected with `unknown option: -dc` — even though `zstd -dc` and
`gzip -dc` both accept it, and gzstd bills itself as a drop-in zstd replacement.
This bit common idioms carried over from zstd/gzip (`gzstd -dc archive | tar -xf -`,
`-dk`, `-df`, `-dcf`).

Fix: a pre-pass at the top of `parse_args` expands a bundled group into
individual flags *before* the match loop, so the loop and all its value-flag
(`argv[++i]`) handling are untouched.  A group is expanded only when every
character after a single leading `-` is a no-arg operation flag — `{d,t,k,f,c}`.
Anything else passes through unchanged: value flags (`-o`/`-T`/`-M`/`-B`/`-D`),
numeric levels (`-19`), attached-value short flags (`-T4`, `-M512`, `-b3`), the
repeat flags (`-vv`/`-vvv`/`-qq`), long options, and `--`/`-`.  `v`/`q` are
deliberately excluded so their repeat semantics survive — bundle verbosity flags
separately (`-d -vv`, not `-dvv`).  Unknown bundles (e.g. `-dz`) still error as
before.

Verified: `-dc`/`-dcf` decompress to stdout and round-trip; `-vv` still maps to
debug, `-19`/`-T4`/`-M512` parse, `-dz` rejected; 253/253 tests pass.  ROADMAP
Phase 7.9.

---

## v0.13.26 — Extend the Gen4+ `--direct` default to compress (was decompress-only)

v0.13.25 auto-enabled `--direct` for decompress on Gen4+; this extends the same
gate to **compress**.  The Gen4 server `--direct` data shows compress benefits
the same way decompress does — the win scales with output volume and is
backend-independent: cpu-only low_compress +103% / mixed +50% / medium +15%,
gpu-only +71% / +29% / +12%, hybrid +70% / +24% / +21%; tiny-output profiles
(high, zeros) are neutral.  No measured regression on Gen4 for any backend.

The decompress-only scoping in v0.13.25 existed solely to avoid the Gen<4
compress regression (a low-core box never saturates buffered writeback, so
O_DIRECT just adds alignment overhead) — but the `gen >= 4` gate already excludes
that case, so the extra `mode == DECOMPRESS` guard was redundant.

`apply_backend_defaults()` now enables `--direct` on Gen4+ for both compress and
decompress (test mode skipped — it writes nothing), unless the user passed
`--direct`/`--no-direct`.  The PCIe-gen probe moved above the compress branch so
it runs for all modes.  Verbose line generalized to
`[ASYMMETRIC] PCIe Gen4 detected; defaulting output to --direct`.

Caveats unchanged from v0.13.25: compress output size is unknown, so the O_DIRECT
path preallocates `input_size` as an upper bound and `ftruncate`s down at
finalize (already handled; `--no-preallocate` opts out).  O_DIRECT can raise
tail-latency variance (NVMe GC / journal commits); medians favor it on Gen4.
`--no-direct` forces the buffered baseline.

Verified: Gen3 compress stays buffered (no auto-enable), explicit
`--direct`/`--no-direct` honored, round-trip clean, 253/253 tests pass.  ROADMAP
Phase 5.3.

---

## v0.13.25 — Default `--direct` (O_DIRECT output) for decompress on PCIe Gen4+

Decompress on the Gen4 reference box is ~95% write-bound: with the write path
removed (`-c >/dev/null`) cpu-only decompress hits ~14 GiB/s, but buffered output
to disk runs at ~0.68 GiB/s — page-cache population + writeback throttling is the
whole cost.  O_DIRECT output bypasses that, and on fast-fabric / high-core boxes
where frame production outruns buffered writeback it is a large decompress win
(up to +130–230% on the Gen4 reference; mixed `-d` ~0.68 → ~2.0 GiB/s).  On
smaller Gen<4 boxes O_DIRECT regresses — the producer never saturates buffered
writeback, so it only adds alignment overhead — so they stay buffered.

`apply_backend_defaults()` now auto-enables `--direct` for **decompress** on PCIe
Gen4+ (reusing the `detect_min_pcie_gen()` probe that already drives the
cpu-only/hybrid decompress default), unless the user passed `--direct` or
`--no-direct`.  It is backend-independent — the win is in the output write path —
so it applies to cpu-only, hybrid, and gpu-only decompress alike.  Compress never
auto-enables it (O_DIRECT regresses compress on smaller boxes; strictly opt-in).
Test mode writes nothing, so it is skipped.  Visible at `-v` as
`[ASYMMETRIC] PCIe Gen4 detected; defaulting decompress output to --direct`.

Behavior notes:
- Override with `--no-direct` (e.g. to benchmark the buffered baseline on Gen4).
- The standard benchmark's plain decompress runs on a Gen4 box now use O_DIRECT
  by default; pass `--no-direct` for the buffered comparison.
- Gen<4 and detection-unavailable paths are unchanged (buffered).

Verified: Gen3 stays buffered (no auto-enable), explicit `--direct`/`--no-direct`
honored, round-trip clean, 253/253 tests pass.  ROADMAP Phase 5.3.

---

## v0.13.24 — Recycle GPU decompress output buffers (kill the per-frame D2H alloc churn)

The CPU workers recycle a bounded `FrameBuf` pool (the v0.13.7/v0.13.8 page-fault
storm fix), but the GPU decompress completion path never got the same treatment:
every readback frame did a fresh `make_shared<vector<char>>(actual)` — a full
decompressed-frame (~16 MiB) allocation that faults every page, on every frame
of every batch.  On the Gen4+ hybrid-decompress default this is the hot path and
the allocation cycles fast.

`DecompStreamCtx` now owns a recycled `out_pool`: `acquire_out_buf()` reuses a
slot whose `use_count()==1` (writer has drained it), grows the pool lazily up to
two batches' worth, and past that waits on the writer's drain signal rather than
allocating.  Recycled slots keep their resident pages, so after warm-up the path
stops faulting.  Deadlock-free by the same FIFO argument as the throttle: a
stream pushes frames in ascending seq, so the writer always has the oldest
in-flight frame to write and frees a slot when it drops the ref.

Measured on a consumer Gen3 box (2 GiB→2 GiB mixed, `--gpu-only -d`, isolating
the path): minor page-faults 636k → 538k (−15%), peak RSS 2.57 → 2.26 GiB
(−12%).  The Gen3 default is `--cpu-only` decompress so this only shows under a
forced GPU/hybrid run; the real target is the Gen4+ hybrid-decompress default,
where batches are larger and frames cycle faster — validate there.  Round-trip
verified on `--gpu-only` and `--hybrid`; 253/253 tests pass.

Compress-side GPU output buffers (`compress_nvcomp`) still allocate per frame but
hold only the *compressed* output (small), so the fault pressure is far lower;
left as a follow-up.  ROADMAP Phase 7.2.

---

## v0.13.23 — Fix: AsyncWritePool flush() waits for the physical write, not just the dequeue

Correctness fix found in a full-pipeline review.  `AsyncWritePool::flush()` waited
only on `pending_.empty()`, but the background write thread empties `pending_` by
*moving* the batch out before it writes it to disk.  So `flush()` could return
while the last batch was still in flight.  A write error on that final batch
(disk full, EIO, broken O_DIRECT tail) sets `error_` only *after* the single
`had_error()` check in `writer_thread`, so the run reported success (exit 0) and
the atomic `rename` proceeded over truncated/corrupt output.  Mid-stream errors
were already caught one batch late by the `had_error()` check inside `submit()`;
only the final batch escaped — i.e. precisely the disk-full-at-the-end case.

Fix: a `writing_` flag (guarded by the pool mutex) is set true when the worker
dequeues a batch and cleared once the batch is physically written — including on
the error-return path, which now also notifies so a blocked `flush()` wakes.
`flush()` now waits on `pending_.empty() && !writing_`, making the post-`flush()`
`had_error()` check reliable.  No hot-path change: the flag is touched twice per
batch under a mutex that was already taken on those transitions.

Roadmap Phase 7.1.  The remaining review items (GPU result-buffer pooling on the
Gen4 hybrid-decompress path, throttle-budget chunk size, CPU-compress memcpy,
and minor nits) are tracked in ROADMAP.md Phase 7.

---

## v0.13.22 — Revert v0.13.20's kernel gate; add `--cold` for honest benchmarking

Plain mmap is restored as the default for compress on every kernel.  v0.13.20
auto-switched to fread on pre-6.4 kernels (no per-VMA locks), but a follow-up
benchmark exposed the gate as a net regression: cpu-only `zeros` and
`high_compress` were **2.6-3× slower** than the prior mmap path, and other
configs were no better than the noise floor (hybrid `mixed` "won" by ~1%, well
inside benchmark jitter).

**Mechanism (why mmap beats fread even on a pre-6.4 high-core box).**  For
I/O-bound workloads the compressor finishes a chunk in microseconds and asks
for the next.  mmap lets all 256 worker threads fault their own pages directly
into their address space and read from the page cache *in parallel* — zero
copy, aggregate bandwidth scales with cores.  fread instead does a *single
producer thread* `fread` + `memcpy` into a `Task.data` buffer; that one thread
caps the whole pipeline at single-thread memcpy bandwidth (~4-5 GiB/s),
regardless of how many workers are downstream.  The `mmap_lock` cacheline
storm is real (28% `down_read_trylock` in the v0.13.20 profile) but doesn't
move wall time outside the run-to-run noise the kernel-gate ostensibly fixed.

**Removed:** `kernel_has_per_vma_locks()`, the gate in `apply_backend_defaults`,
the `mmap_user_set` flag, and `#include <sys/utsname.h>`.  `--mmap`/`--no-mmap`
still work as explicit overrides; default is mmap everywhere.

**Kept (added in v0.13.21, folded into this entry):** a `--cold` flag that
calls `posix_fadvise(POSIX_FADV_DONTNEED)` on the input fd right after open.
Without it, `gzstd-benchmark.sh`'s median-of-3 against a 20 GiB file on a
600+ GiB-RAM box was measuring memory-to-memory throughput — iteration 1
warmed the cache and iterations 2-3 served from RAM.  `--cold` makes every
iteration a real cold-disk read as an ordinary user (no root, no
`drop_caches`).  Documented in `--help` as benchmarking-only.
`gzstd-benchmark.sh` now passes `--cold` for both compress and decompress
invocations and no longer writes to `/proc/sys/vm/drop_caches` (system-wide
cache wipe under sudo, bad citizen on a shared host).

---

## v0.13.20 — Kernel-gated mmap/fread for compress (fixes the high-core mmap storm)

> **Reverted in v0.13.22 — net regression.**  Cold-cache benchmarks on the
> 256-core box (5.15 kernel) showed cpu-only `zeros`/`high_compress` 2.6-3×
> slower than plain mmap; the "fix" only helped warm-cache `mixed` by ~1%,
> within noise.  fread serializes the read on a single producer thread, which
> caps aggregate throughput at one core's memcpy bandwidth — a worse problem
> than the `mmap_lock` cacheline contention the gate was trying to avoid.
> See v0.13.22.

Resolve the high-core mmap compress slowdown for real, and **revert the failed
v0.13.17 prefault** (which made it worse). The root cause turned out to be the
kernel, not gzstd.

**Investigation.** On a 256-core box, mmap compress burned ~94s system time
(28% of perf samples in `down_read_trylock`) vs ~16s for `--no-mmap`. A toggle
matrix (v0.13.18, `GZSTD_MMAP_ADVISE` × `GZSTD_MMAP_PREFAULT`) showed the
v0.13.17 producer-side `MADV_POPULATE_READ` prefault made it *worse* in every
advise mode (~88-90s sys regardless), because the populated pages were reclaimed
before the workers reached them (minflt barely changed) — pure added faulting
work, not a fix. A binary-search sweep then showed `fread` beats mmap at *every*
thread count on that box (7.4 vs 7.8s at T=2, up to 7.8 vs 11s at T=256) — so
there's no thread-count crossover at all.

**Root cause.** Pre-6.4 kernels have no per-VMA locks, so every page fault takes
the single global `mm->mmap_lock` rwsem; many cores doing atomic ops on that one
counter cacheline dominates system time. Linux **6.4** (per-VMA locks, faults via
RCU + a fine-grained per-VMA lock) removes it. The slow box was on 5.15; the
workstation is already on a 6.x kernel, which is why mmap was always fine there.

**Fix.** `apply_backend_defaults` now checks the kernel (`uname`, parsed once):
on **< 6.4**, compress falls back to `fread`; on **6.4+**, it keeps mmap (the
faster path). Skipped for `--mmap`/`--no-mmap` (explicit override via a new
`mmap_user_set` flag), for `--gpu-only` (only a few H2D faulters, not the worker
storm), and for decompress (never used the mmap reader). It's a clean per-kernel
switch — no thread-count threshold, since fread wins at all thread counts on
pre-6.4. Distro backports (e.g. RHEL per-VMA locks on a `5.14.x-*.el9` string)
get a harmless false negative (fread ≈ mmap there); `--mmap` overrides.

Removed: the v0.13.17 `mmap_prefault` and the v0.13.18 `GZSTD_MMAP_*` diagnostic
toggles. `MmapRegion` is back to plain `MADV_SEQUENTIAL`. The slow box upgrades
to a 6.x kernel later this year, after which mmap will be the default there too.

---

## v0.13.17 — Pre-fault mmap input on the producer (kill the fault storm)

> **Reverted in v0.13.20 — this did NOT work.** The producer-side
> `MADV_POPULATE_READ` populated pages that were reclaimed before the workers
> used them, so it *added* system time instead of removing the storm. The real
> cause was a pre-6.4-kernel `mmap_lock` limitation; see v0.13.20.

Fix mmap compression being *slower* than `--no-mmap` on high-core machines,
despite being the "zero-copy" default.

**Cause.**  The compress producer hands workers `Task`s whose `view_ptr`
points into the mmap'd input; each worker faults its own pages in on first
touch.  With hundreds of workers hammering one mapping, concurrent faulting
storms the kernel's `mmap_lock` and per-page fault path.  On a 256-core
machine this showed as ~4× system time (~66s vs ~16s) and ~15% slower wall
than `--no-mmap` — the same `mmap_lock` storm already designed out of the
*decompress* path (which reads via `fread`).  Compress still defaulted to
mmap, so it ate the penalty.

**Fix.**  The producer now bulk-pre-faults each chunk with
`MADV_POPULATE_READ` *before* pushing its task (`mmap_prefault`, new helper
next to `MmapRegion`).  The faulting happens once, in bulk, on the single
producer thread — no concurrent storm — and because population precedes the
push, a worker can never touch an unpopulated page (no startup race, no need
for a separate prefetch thread).  Zero-copy reads and read/compress overlap
are preserved.  The producer paces itself to stay ≤ ~1 GiB ahead of
consumption (`m->read_bytes`, which workers bump per chunk), so the whole
file is never read up front.  Applied to both `compress_cpu_mt` and
`compress_nvcomp`.

**Portability.**  `MADV_POPULATE_READ` (Linux 5.14+) is `#define`d to its
stable UAPI value (22) when missing from build headers (the ubuntu-20.04
portable-build container, glibc 2.31), so the shipped binary still uses it on
a 5.14+ runtime kernel; on older kernels `madvise` returns `EINVAL` and we
fall back to lazy per-access faulting.

**Verified (24-core workstation):** 4 GiB compress round-trip byte-identical;
mmap+prefault now ~1.7s vs `--no-mmap` ~2.3s with low sys time (no storm);
full suite passes.  The decisive win is expected on a 256-core machine where
the storm was severe — re-measure `--direct` vs `--direct --no-mmap` there:
the goal is mmap matching or beating `--no-mmap` with the ~4× sys gap gone.

---

## v0.13.16 — Stream large single-frame files directly from the file

Fix a long-standing slow path: decompressing a single-frame `.zst`
(stock `zstd`, `--sliding-window`) was far slower than `zstd -d`,
spiked memory, and showed a frozen progress bar — in every mode,
including `--cpu-only`.

**Cause.**  A single zstd frame can't be split across CPU threads (nor
GPU subchunks).  The fallback routed the lone frame through the normal
queue: `stream_frames_to_queue` had to read and buffer the *entire*
compressed frame (growing its read buffer with realloc churn), `memcpy`
it into one Task, and only then could a CPU worker decompress it — and
even then the worker's streaming branch waits on the `producer_done`
gate (a v0.13.1 seq-collision guard) which can't fire until the reader
has consumed the whole file.  Net: read and decompress ran *serially*,
peak memory was input + frame-copy + output (~30 GiB on a 20 GiB file),
and neither meter moved until decompression started.

**Fix.**  In `main`'s decompress dispatch, peek the first frame; if it
decompresses to more than `SINGLE_FRAME_STREAM_MIN` (256 MiB) the input
is effectively a single-frame file, so hand it to a new
`decompress_stream_from_file` regardless of mode: a plain
`ZSTD_decompressStream` loop reading 4 MiB at a time straight from the
`FILE*`, writing each output chunk through the existing DirectWriter /
fwrite path.  Read, decompress, and write now overlap; peak RSS drops to
a couple of I/O buffers; the progress bar moves (we set `total_out` /
`total_out_final` from the peeked size up front); and for GPU modes no
CUDA is touched (no bringup thread, no cuInit).  Single-threaded by
nature — one zstd frame can't be split — which is inherent, not a
regression.

**Why a 256 MiB threshold, not 16 MiB.**  The threshold sits well above
gzstd's largest practical chunk (`--ultra` auto-bumps to 128 MiB) and the
v0.13.1 regression test's 100 MiB chunks, so genuinely *multi-frame*
inputs — even with large per-frame sizes — keep the parallel queue path.
`decompress_nvcomp` therefore retains its `gpu_disabled_by_peek` CPU
fallback for the 16–256 MiB multi-frame-oversize case (GPU can't subchunk
those frames, but the CPU pool still decompresses them in parallel).
Streaming a multi-frame file would needlessly serialise it.

Applies to all modes and both build configs (`decompress_stream_from_file`
is not GPU-gated).  Seekable input only — stdin (peek returns -1) keeps
the old path.  `ZSTD_decompressStream` also decodes the rare
trailing-frames-after-a-large-first-frame case correctly.

**Verified (24-core workstation):** 2 GiB single-frame `--sliding-window` round-trip
byte-identical via both `--cpu-only` and `--gpu-only` (peak RSS 24 MB, was
~2-3 GiB; no GPU bringup logged); a 4×100 MiB multi-frame file correctly
stays on the parallel queue path; full suite passes including the
`--sliding-window` round-trip and the v0.13.1 multi-frame-oversized guard.

---

## v0.13.15 — Overlap CUDA init with the reader (gpu-only decompress)

Extend the v0.13.13 bringup overlap to `--gpu-only` decompress, killing the
3-4s startup stall (the gap between `[O_DIRECT]` and `[INIT]` at `-v`) on
high-GPU-count boxes.

**Cause.**  v0.13.13 deferred the `cudaGetDeviceCount` cuInit (~2-3s on an
8-GPU box) to a background bringup thread, but *only* for adaptive hybrid —
the rationale was "no CPU pool to overlap with" in gpu-only.  That overlooked
the **reader**: `stream_frames_to_queue` reads and frame-parses the entire
compressed input, which on a multi-GiB file takes about as long as cuInit and
has to happen regardless.  In gpu-only the reader ran *after* the synchronous
cuInit + inline bringup, so the GPU sat idle through init and the reader sat
idle through nothing useful — pure serial cost.

**Fix.**  Generalize the deferral predicate from `hybrid_overlap` to
`defer_detect = hybrid_overlap || opt.gpu_only`.  In gpu-only the bringup
thread now does the deferred `cudaGetDeviceCount` + `select_best_gpus` +
worker spawn while the main thread goes straight to the reader, filling the
`TaskQueue`.  GPU workers consume a warm queue the instant their contexts are
ready instead of starting cold.  The `[INIT]` banner reports "GPUs detecting
in background" for gpu-only too.

**gpu-only edge cases** (handled synchronously before, now need explicit care
because detection is deferred):

- *No CUDA device.*  The synchronous path errored instantly at
  `cudaGetDeviceCount`.  Deferred, the bringup thread sets a
  `gpu_only_no_device` atomic; `stream_frames_to_queue` takes a new optional
  `abort` pointer and returns early when it's set, so main errors with the
  same `EXIT_USAGE` message instead of buffering a consumer-less queue to EOF.
- *Oversize first frame* (`--sliding-window` / `zstd` single frame).  The
  peek sets `gpu_disabled_by_peek`; the CPU-pool spawn condition now also
  fires on that flag (gpu-only has no `sched`-driven pool), so the file
  decompresses on CPU.  The deferred bringup still pays a (hidden, discarded)
  cuInit on the background thread for this case — acceptable for a rare path.

**Verified (2-GPU workstation):** 253/253 suite; gpu-only round-trip
byte-identical; masked-GPU run errors cleanly with exit 2 and removes the
partial output; sliding-window file falls back to CPU and round-trips.  The
win scales with GPU count, so the 8-GPU machine's gap should drop from ~3-4s
to ~0 — re-measure the `[O_DIRECT]`→`[INIT]` interval there.

---

## v0.13.14 — Fixed-share: wait for GPU registration before streaming

Fix a `--cpu-share` regression (introduced by the v0.13.11 device-probe
short-circuit) where the requested CPU/GPU split collapsed to all-CPU
on high-GPU-count machines.

**Cause.**  Before v0.13.11, `select_best_gpus` did a serial CUDA probe
that pre-created GPU contexts, so GPU workers registered almost
instantly.  v0.13.11 removed that probe; `warm_gpu_contexts` was meant
to compensate but only creates the CUDA *contexts* — the GPU worker
still does VRAM probe + cudaMalloc + `register_gpu_stream` afterward.
On an 8-GPU box with many fast CPU workers, the reader + CPU pool drain
a small input via the drain-phase fast path (`qs.done &&
!any_gpu_active()`) before any GPU registers, so `--cpu-share 0.0`
(all-GPU) produced 128c/0g — every frame went to CPU.  Surfaced by the
suite's `--cpu-share split responds to value` test on the 8-GPU
system; the 512 MiB test input gave enough lead time on 2-GPU hardware
but not on 8.

**Fix.**  In fixed-share mode only, wait for at least one GPU stream to
register (`any_gpu_active()`) — or for all GPUs to fail init
(`gpu_init_failures >= gpu_count`) — before starting the reader.  This
guarantees the drain-phase fast path can't fire before the GPU is in
the rotation, so the split is honored.  Applied to both compress and
decompress.  Adaptive mode (the default) skips the barrier: it promises
no exact split and wants the fastest possible start.

**Test suite changes:**
- New section "Hybrid GPU-bringup overlap (decompress)" guarding the
  v0.13.13 restructure: adaptive round-trip, fixed-share round-trip,
  stdout output, and a repeated-run teardown-stability check.
- Added a `--extensive` flag.  Lower-value / cosmetic sections are now
  gated behind it so the default run is leaner (253 tests vs 284 with
  `--extensive`): Stress tests, Help/version, Space-separated option
  values, and Completion summary format.  GPU correctness and
  regression-guard sections stay in the default run.  Gate further
  groups with `if $EXTENSIVE; then ... fi`.

---

## v0.13.13 — Overlap CUDA init with CPU decompression (hybrid)

Eliminate the startup stall before the progress meter moves in hybrid
decompress.  v0.13.12's timing showed the residual delay was almost
entirely `cudaSetDevice` context creation; this release also addresses
the *other* half — `cuInit`.

**Cause.**  `cudaGetDeviceCount` (decompress_nvcomp, the first CUDA call
in the process) triggers the one-time CUDA driver init `cuInit` —
~2s on an 8-GPU box.  It ran on the main thread *before* the CPU
decompression pool was spawned, so in hybrid mode the CPUs couldn't
start decompressing until cuInit finished.  The user's exact
observation: "the CPUs should be decompressing while the GPU detection
is going on" — and they were blocked from doing so.

**Fix.**  In hybrid mode, GPU detection + selection + worker spawn now
run on a background "bringup" thread:

- Main thread spawns the CPU pool and starts the frame reader
  immediately, using a *provisional* device count for throttle sizing
  (RAM-capped, so over-estimating is safe).
- CPU workers decompress from t≈0 — the scheduler already runs CPUs
  "wild" while `gpu_ready_` is false (no new scheduling logic needed).
- The bringup thread does `cudaGetDeviceCount` (the deferred cuInit),
  `select_best_gpus`, `init_slots`, and spawns GPU workers.  They
  register with the scheduler when ready and it rebalances.

**Concurrency safeguards:**
- `init_slots` (resizes `ResultStore::slots`) is called under
  `results.m`, which the writer's `drain_slots_locked` also holds — no
  resize/iterate race.
- Teardown joins the bringup thread before iterating `gpu_workers`
  (the bringup thread populates that vector).
- If the CPU pool drains the whole file during cuInit, the bringup
  thread skips GPU spawn entirely (no wasted context creation; process
  exits as soon as CPU+writer finish).  Late-spawned GPU workers that
  hit a done+empty queue exit cleanly via the existing
  `producer_done_seen` path.

gpu-only and cpu-only paths are unchanged (gpu-only has no CPU pool to
overlap with; detection stays inline).

**Result** (consumer Gen3, hybrid decompress): TTFB dropped from
0.256s (v0.13.12) to 0.035s.  The CPU pool starts before cuInit
instead of after it.  On high-GPU-count systems where cuInit is ~2s,
the win is correspondingly larger.  Tiny-file edge cases (CPU finishes
before GPU init) verified for correct output and clean teardown.
280/280 tests pass.

---

## v0.13.12 — Per-phase GPU init timing at -vv

Diagnostic only, no behavior change.  v0.13.11 removed the 5s serial
device-selection probe; a residual GPU-init delay remained (~4s on an
8-GPU box, ~2.5s hybrid).  To locate it, the GPU compress worker now
logs an init-phase breakdown at -vv:

  [GPU<d>] init phases: ctx=Nms probe=Nms malloc=Nms total=Nms

- **ctx**: `cudaSetDevice` forcing CUDA primary-context creation.
- **probe**: per-stream VRAM binary search (nvCOMP temp-size queries).
- **malloc**: `allocate_stream_buffers` (`cudaMalloc` of device buffers).

Measured on consumer Gen3 (2× 2080 Ti): ctx≈230ms, probe 1-11ms,
malloc≈1ms — context creation is ~99% of init.  The VRAM probe and
device allocation are negligible, so they're not worth optimizing.
The remaining startup cost is CUDA context creation, which on
multi-GPU systems appears to serialize on the driver's init lock
(8 × ~500ms ≈ the observed 4s).  This release just makes that
visible; reducing it (parallel context creation, or overlapping it
behind useful work) is follow-up.

---

## v0.13.11 — Skip serial GPU probe when using all devices

Fix a ~5s startup stall before the progress meter moves in `--gpu-only`
and `--hybrid` modes (absent in `--cpu-only`).

**Cause.**  `select_best_gpus()` ranks GPUs by free VRAM / utilization
so it can pick the best N when the user wants a subset.  The subset
path uses NVML (no CUDA context creation — fast).  But when using ALL
devices (the default), the NVML guard `if (want < total_devices)` is
false and the function fell through to an all-devices loop that calls
`cudaSetDevice(d)` + `cudaMemGetInfo()` on every device.  Those force
serial CUDA context creation on the main thread — ~0.6-1s per
datacenter GPU, ~5s for 8 — and this runs before the reader, progress
thread, and worker pool start.  The pipeline waits the whole time.

The probe was pointless in this case: when N == all devices there's
nothing to rank.  We paid 5s gathering ranking data we then ignored.

**Fix.**  Short-circuit when `want >= total_devices`: return the
trivial `[0..N)` device list without probing.  The GPU worker threads
create their CUDA contexts in parallel at startup (one per device on
its own thread) instead of serially on the main thread.  In hybrid
mode the reader and CPU pool also start immediately and overlap with
GPU context warm-up.  Expected: ~5s → ~1s (one parallel context init)
in gpu-only, near-zero perceived delay in hybrid.

**Fixed-share exception.**  Deferring context creation to the worker
threads broke `--cpu-share`: on a small input the CPU pool drains every
frame (via the `qs.done && !any_gpu_active()` path) before the GPU
finishes booting and registers, so the explicit split was silently
ignored — `--cpu-share 0.0` gave 100% CPU.  Fix: when `--cpu-share` is
set, `warm_gpu_contexts()` creates the primary contexts in parallel
(one thread per device, ~1s for 8 vs ~5s serial) before the pipeline
starts, so the GPU is ready to take its share.  Adaptive mode (the
default) still defers for fastest startup — there the GPU naturally
catches up on any non-trivial input, and a small input being
CPU-drained is the correct fast path.

**Telemetry.**  `[GPU] device selection: N ms` logged at `-v`, so this
startup cost is visible going forward.

**On the subset case** (`--gpu-devices N` with N < total): already
fast — it uses NVML to read utilization and free memory without
creating CUDA contexts (`cudaGetDeviceProperties` reads cached device
attributes, no context).  The only remaining slow path is NVML being
unavailable AND selecting a subset, where free-memory ranking requires
`cudaMemGetInfo` (hence a context).  That's unavoidable without NVML
and rare on NVIDIA systems.

---

## v0.13.10 — Condition-variable wait for the bounded pool

v0.13.9's bounded pool architecture is correct, but its acquire-when-
full path used `std::this_thread::yield()` to wait — which is a
`sched_yield` syscall on Linux.  With 96 workers each yielding hundreds
of thousands of times per run, sys time on cpu-only decompress mixed
jumped from 11.87s (v0.13.8) to 51.41s (v0.13.9): same throughput, 4×
more kernel cycles burned in `sched_yield`.

**Fix.**  Wait on a condition variable instead.  Added `drain_cv_` +
`drain_m_` to `FrameThrottle` with two methods:

- `notify_drain()` — called by `AsyncWritePool::worker_fn` after each
  `buf.reset()` (the moment a frame's `shared_ptr` ref drops from 2 to
  1, freeing a worker pool slot).  `notify_all()` because the writer
  doesn't know which worker owns the freed slot.
- `wait_for_drain(predicate)` — workers call this when their pool is
  full.  Standard CV `wait_for` with predicate, 10ms timeout as a
  safety net for any missed notify.

Both `cpu_worker` and `cpu_decomp_worker` now use `wait_for_drain`
instead of `yield`.  Predicate scans the per-worker pool for a slot
with `use_count() == 1`.

**Why a separate CV from the existing permit-acquire CV (`cv_`):**
sharing would force pool-waiters and permit-waiters onto the same
mutex, blocking `release()` while broadcasting.  `drain_m_` is
dedicated and never held by `release()`, so permit-acquire stays fast.

**Notify granularity.**  Per-frame `notify_all`, not per-batch.
Trades more wakeups for lower latency: workers wake immediately when
their slot frees rather than waiting for the writer's whole batch.
Wake cost is bounded — only workers currently in `wait_for_drain` are
woken, and the predicate check is a few atomic loads.

---

## v0.13.9 — Bounded per-worker buffer pool: route page-faults through backpressure

v0.13.8 introduced a per-worker output-buffer pool to eliminate the
per-iteration allocation storm.  Profiling on a 256-core / 8-GPU
system showed it didn't actually fix decompress at high thread counts
— the pool was UNBOUNDED and grew faster than the writer could drain.
This release makes the pool bounded so it participates in the existing
backpressure chain instead of bypassing it.

**The diagnosis (perf record, cpu-only decompress, zeros.bin):**

| Metric (T96 vs T16) | T96 | T16 |
|---|---|---|
| Wall time | 4.14s | 3.27s |
| Sys time | 29.2s | 3.11s |
| Sys/real ratio | **9.4×** | 1.0× |
| Page faults | 2.35M | 510k |
| IPC | 0.29 | 1.00 |

Hot path at T96: 82% of cycles in `std::vector::resize` → memset →
`asm_exc_page_fault` → `down_read_trylock` (the per-process mmap_lock
rwsem).  Same shape as the v0.13.7 compress diagnosis.

Hot path at T16: 68% in `AsyncWritePool::write_sparse` → `fseek` →
`lseek` syscall.  Writer was the bottleneck; 16 workers were enough.

**Root cause: the v0.13.8 pool bypassed the throttle's backpressure.**
The FrameThrottle bounds total in-flight frames (default 512).  With
96 workers, that's ~5 frames/worker on average.  But v0.13.8's
`acquire_decomp_buf()` grew the pool whenever `use_count() > 1` on all
existing slots — and since the writer was the bottleneck (~5 GiB/s
ceiling on sparse zeros), slots stayed in flight long enough that
workers grew their pools to 5+ entries.  Each new entry was a fresh
~64 MiB allocation → page-fault storm.

**Fix: pool is now bounded at startup and yields on full.**

```cpp
const int pool_size = std::max(2, throttle_budget / N_workers);
std::vector<FrameBuf> pool(pool_size);
for (auto & b : pool) b = std::make_shared<std::vector<char>>();

auto acquire = [&]() -> FrameBuf {
  while (true) {
    for (auto & b : pool) if (b.use_count() == 1) return b;
    std::this_thread::yield();  // backpressure: wait for writer
  }
};
```

The min-of-2 guarantees pipelining (one frame in flight + one being
worked on).  Above that, the throttle's global cap is divided across
workers.  This makes the chain work end-to-end:

```
writer slow → result store fills → pool slots stay in-flight
            → pool acquire yields → worker waits → no new alloc
            → frame production rate = writer drain rate

writer fast → result store drains → slots free fast
            → acquire returns immediately → worker proceeds full speed
            → throttle is the only cap (intended design)
```

**No thread-count cap.**  An arbitrary "decompress shouldn't exceed N
workers" rule was considered and rejected — it sidesteps the
architectural issue without fixing it, and gets the wrong answer on
hardware we haven't measured.  The bounded pool + existing throttle
lets workers scale to actual hardware while routing back-pressure
correctly.

**Applied to both `cpu_worker` (compress) and `cpu_decomp_worker`.**
GPU `h_out` allocations not changed — no evidence they're hitting the
same issue at current concurrency, but the pattern would transfer if
needed.

**Telemetry at -vv.**  Per-worker summary now includes `pool=N waits=K`
showing pool size and yield count.  Non-zero waits indicate the worker
was blocked waiting for the writer — useful for confirming the
backpressure is actually engaging.

---

## v0.13.8 — Result-store buffer pool: decompress page-fault fix

Apply the v0.13.7 page-fault diagnosis to the decompress path via a
proper buffer pool — the simple "per-thread scratch + copy" pattern
that worked for compress can't work for decompress because the output
is the same size as the buffer, so the copy would page-fault as many
bytes as the original allocation.  Instead, recycle whole buffers
through the writer.

**Mechanism.**  `ResultStore` and `AsyncWritePool` now carry
`std::shared_ptr<std::vector<char>>` (alias `FrameBuf`) end-to-end
instead of bare `std::vector<char>`.  Workers maintain a per-thread
pool of FrameBufs and reuse a slot once `use_count() == 1` (writer has
dropped its reference after writing to disk).  Buffers stay resident
across iterations — `resize()` only memsets resident memory, no kernel
page-fault path.

**cpu_decomp_worker now uses the pool.**  The single-frame path
(`acquire_decomp_buf()` → `resize(decomp_size)` → ZSTD writes → push)
and the streaming-frame path (16 MiB chunks in a loop) both pull from
the same per-thread pool.  Pool grows on demand and is bounded
implicitly by FrameThrottle's in-flight cap.

**Compress workers wrap their output in `make_shared` at the push
site.**  The v0.13.7 fix (per-thread scratch buffer, copy `csz` bytes
into a sized vector) is unchanged; that sized vector now becomes the
backing storage for a shared_ptr.  No reuse on the compress side
because `csz` is small for compressible data and the per-iteration
alloc is already cheap.

**GPU workers** (compress D2H, decompress D2H) also wrap `h_out` in
`make_shared` — uniform interface, negligible overhead.

**Expected impact** (needs re-benchmarking on high-core-count
multi-GPU systems):
- `cpu-only` decompress at the default high-thread cap: should
  approach the per-thread sweet-spot ceiling the same way v0.13.7
  lifted compress.  v0.13.7 cpu-only decompress on the worst-affected
  file was 3.71 GiB/s; the hand-tuned low-thread ceiling for the same
  file was 5.25 GiB/s.  Target: 5+ GiB/s.
- `hybrid` decompress should also benefit (same per-thread overhead).
- Compress behaviour unchanged from v0.13.7.

**Allocator overhead.**  shared_ptr's atomic refcount adds two atomic
ops per frame (one push, one writer drop) — measured at ~10-30ns each
on modern hardware.  At even 1000 frames/sec, that's 60µs/sec total.
Negligible relative to the page-fault savings.

---

## v0.13.7 — Hoist per-iteration output buffer in CPU workers

Fix the actual root cause of the hybrid-vs-gpu-only compress gap that
v0.13.6 only partially closed.  The 14-17% slowdown on high-core-count
multi-GPU systems wasn't scheduler overhead from idle threads — it was
**page-fault contention on the per-process mmap_lock from 96 worker
threads simultaneously allocating fresh 16+ MiB output buffers in their
hot loops**.

Discovered via `perf record` on a 256-core / 8-GPU server (zeros.bin
compress, 4-second run).  Counters told the story:

|                    | hybrid (T96) | gpu-only | ratio |
|--------------------|---|---|---|
| sys time           | 45.86s | 4.90s  | **9.35×** |
| context switches   | 107,784 | 1,728 | 62× |
| page faults        | 2.34M | 348k | 6.7× |
| IPC                | 0.43 | 1.55 | 0.28× |

The flame graph for hybrid showed **64% of all cycles in
`asm_exc_page_fault`**, with the call path:

```
compress_one_cpu_frame
  std::vector<char>::resize
    memset_avx512_unaligned_erms
      asm_exc_page_fault
        do_user_addr_fault
          down_read_trylock   ← 28% of fault time on mmap_lock
```

**Root cause.** Both `cpu_worker` and `cpu_worker_rescue` allocated a
fresh `std::vector<char> out_frame` inside the per-iteration block,
which `compress_one_cpu_frame` then grew to `ZSTD_compressBound(src)`
(~16 MiB for the default GPU subchunk size).  Vector growth value-
initializes new elements — for `char`, that's `memset(0)` across every
page, triggering one minor page fault per 4 KiB.  At 96 workers all
hitting this path during the "CPU runs wild while GPU initializes"
phase, the kernel serialized all 96 threads on the single per-process
`mmap_lock` rwsem.  Same pattern in `cpu_decomp_worker` (allocating
`out_buf(t.decomp_size)` per frame).

**Fix (compress only).**  Hoist the output buffer to per-thread
(lifetime = worker thread) in `cpu_worker` and `cpu_worker_rescue`.
On iteration 1, `resize()` pays the page-fault cost once.  On
iterations 2+, the pages are already resident — `resize` just memsets
resident memory (fast, no kernel involvement).  Then copy `csz` bytes
into a sized vector for the result store, preserving `scratch`'s
capacity for the next iteration.

For highly compressible data (csz ≈ 0) the copy is essentially free
and the fix recovers almost all lost throughput.  For poorly
compressible data (csz ≈ src_size) the copy still costs a memcpy of
the compressed output, but the worst-case page-fault storm is gone.

**Decompress is NOT patched** — and we tried, then reverted.  For
decompress, `actual ≈ decomp_size` (output is the decompressed
payload), so the scratch+copy pattern would page-fault as many bytes
on the copy as the original allocation, then ADD a memcpy.  Net cost:
original + memcpy = worse.  The decompress allocation pattern stays
as-is; the writer owns each `out_buf` after `std::move` and frees it
as it drains.  Fixing decompress requires a true buffer pool with
writer-side return — out of scope here.

**This also lifts `cpu-only` compress at the default high-thread cap**
by the same mechanism.  Measurements on a 256-core / 8-GPU system:
zeros.bin compress went from 7.91 to 12.62 GiB/s (+59%),
high_compress from 9.05 to 15.82 (+75%) — both now ABOVE the
hand-tuned low-thread sweet spot from the v0.13.2 baseline (10.5
GiB/s).  The empirical "lower thread count is best" finding is
retired for compress on this hardware class.

Validation on the same hardware:
- `--hybrid` and `--gpu-only` compress now converge to identical
  numbers (3.99 GiB/s on the highly-compressible files) — both
  bottlenecked downstream (writer/NVMe), CPU contribution can no
  longer go net-negative.
- `--cpu-only` (default high-thread cap) is now the clear winner on
  this hardware class: 12.6 GiB/s on zeros vs 3.99 for any GPU-using
  mode.
- Lower-core consumer hardware unchanged (single-thread page-fault
  storm doesn't exist on 24-ish cores).

---

## v0.13.6 — Hybrid mode: proactive batch reservation

Fix hybrid mode regressing below `--gpu-only` on high-core-count
multi-GPU systems.

A 930-measurement sweep at iterations=3 on a 256-core / 8-GPU server
showed hybrid compress at 2.45 GiB/s on zeros.bin vs gpu-only at 3.05
GiB/s — a 20% regression below the GPU-alone baseline.  Decompress was 25-60% slower on every file.
Hybrid mode is supposed to add CPU contribution on top of GPU; on this
hardware class it was instead displacing GPU work.

**Root cause: AUTO floor factor too small.**  `compute_auto_factor_()`
in HybridSched (added v0.12.12) computed the queue floor as
`(gpu_per_worker - cpu_per_worker) / gpu_per_worker`, clamped to [0, 1].
On systems where the per-worker GPU and CPU rates are similar (~0.13
GiB/s each), this produced factor ~0.15.  The effective floor was
`0.15 * streams * batch` — too shallow to actually reserve a GPU round.
96 CPU workers drained the queue during the millisecond-long GPU
processing window, so when a stream returned for its next batch via
`pop_batch_greedy(min_n=1)`, it got a tiny batch.  Small nvCOMP batches
don't amortize per-call kernel-launch overhead, and the throughput loss
from shrunken GPU batches exceeded CPU's contribution.

**Fix: "GPU first, CPU as surplus" policy.**  New AUTO formula keys off
CPU's measured share of aggregate throughput, not per-worker rates:

  cpu_share < 5%   : factor = 4.0  (CPU not contributing → heavy lockout,
                                     hybrid converges to gpu-only)
  cpu_share > 20%  : factor = 1.5  (CPU helping → reserve 1.5 batches,
                                     but let CPU work the surplus)
  in between       : linear interpolation
  warm-up          : factor = 2.0  (proactive default)

Floor is now always >= 1 full GPU round; a CPU pop can never leave the
next GPU batch short.  Cap on `--hybrid-floor-factor` raised from 1.0
to 4.0 so users can lock CPUs out further if needed.

**Second bug: drain-phase short-circuit ignored the floor.**  The
may_take predicate in cpu_worker and cpu_decomp_worker had an early
return when `qs.done && !is_fixed_mode()` — for AUTO mode, once the
producer finished, CPU took regardless of floor.  On a 20 GiB file with
mmap and warm page cache, the reader finishes in ~1s but the GPU drain
takes 10+s, during which CPU floods the queue and shrinks GPU batches.
Symptom: hybrid still ~10% slower than gpu-only after the AUTO formula
fix, on the same hardware class where the formula fix should have
sufficed.  Fix: drain-phase short-circuit only fires when no GPU is
active.  While any GPU stream is registered (working through its share
or about to), the floor applies in both fill and drain phases.

**Third bug: fixed-share mode deadlocked under the new floor.**  The
old code bypassed the floor entirely in fixed-share mode via the
drain-phase short-circuit; with that short-circuit removed,
`--cpu-share 0.5` could deadlock: `should_cpu_take` returns true when
CPU's share is below target, `should_gpu_take` returns true when CPU's
share is at-or-above target minus 2%.  If the floor blocks CPU, CPU's
share stays at 0, GPU's predicate also fails, both sides wait forever.
Fix: skip the floor check entirely in fixed-share mode — the user's
explicit share is the constraint, not GPU batch preservation.  Floor
only applies in adaptive AUTO/NOMINAL mode.

The diagnosis path: gpu-only's per-config sweep showed a flat 3.05
GiB/s ceiling across all batch×stream combinations (the signature of a
downstream bottleneck — writer/NVMe/ResultStore).  On mixed.bin where
nothing was saturated, gpu-only showed a clean "bigger batch wins"
gradient (0.85 → 0.93 GiB/s); hybrid's gradient was flat at ~0.88,
evidence that CPU was destroying the GPU's batch-size tuning by
draining the queue.

Consumer Gen3 hardware (24-core, 2× RTX 2080 Ti) validates the fix
preserves hybrid winning on its target tier: medium_compress.bin
compress shows hybrid 3.30-3.71 GiB/s vs gpu-only 1.87-2.12 GiB/s
(~70% faster).

No behavior change for explicit `--cpu-only`, `--gpu-only`, or users
who set `--hybrid-floor=nominal` or `--hybrid-floor-factor=X` manually.

---

## v0.13.4 — CLI arg-parser hardening + auto-tune log fix

Polish pass on the CLI surface plus one cosmetic bug in the GPU
decompress auto-tuner.  No behavior change on valid inputs; bad inputs
that previously crashed or silently truncated now produce a usage hint.

- **Argument-parser error handling.**  parse_num_arg / parse_int_arg /
  parse_double_arg called std::stoull / std::stoi / std::stod directly,
  with two failure modes:
    - `--gpu-streams=12abc` silently parsed as 12 — stoull stops at the
      first non-digit and never tells the caller.
    - `--gpu-streams=foo` let std::invalid_argument escape to main,
      printing a terminate-style backtrace instead of a usage hint.

  Added parse_u64_value / parse_int_value / parse_double_value helpers
  that catch invalid_argument and out_of_range, verify the full string
  was consumed, and call die_usage on failure.  All three parsing
  wrappers route through them.

- **--gpu-mem-frac validation.**  Hard-rejects values outside (0.0, 1.0)
  and warn-clamps anything outside [0.10, 0.95] so existing scripts that
  pass slightly aggressive values still run but the user learns why
  they did not get what they asked for.

- **--pinned auto|on|off.**  The old code combined parse_str_arg with an
  rfind prefix check and a manual `=` split; the space-separated form
  bypassed validation.  Replaced with a single parse_str_arg call into
  a scratch buffer.

- **Asymmetric default visibility.**  Promoted the PCIe Gen3 →
  --cpu-only notice from V_VERBOSE to V_DEFAULT.  Users on
  workstation-class hardware otherwise saw zero GPU activity during
  decompress and had no signal the runtime had switched backends on
  them.  Prefix changed from [ASYMMETRIC] to gzstd: to match the other
  default-verbosity notices.

- **GPU decomp auto-tune log fired twice per settle.**
  gpu_decomp_worker printed `[AUTO-TUNE] settled at batch=N` on every
  tune-step completion regardless of whether the next phase was REFINE
  or SETTLED, because the verbose-log sat outside the if/else.  Split
  into `refining [lo..hi] trying mid` vs `settled at N`.

---

## v0.13.2 — Build-system fixes for portable-build workflow

Two issues surfaced when first running scripts/build-portable.sh under
the new GitHub Actions release workflow:

- **NVCOMP_ROOT cache variable was ignored.**  CMakeLists.txt's
  find_path and find_library calls only checked `$ENV{NVCOMP_ROOT}`,
  so passing `-DNVCOMP_ROOT=/nvcomp` at the cmake command line had no
  effect — the build silently fell back to CPU-only.  Now the HINTS
  list includes both forms (`${NVCOMP_ROOT} $ENV{NVCOMP_ROOT}`).

- **CPU-only build broken.**  When HAVE_NVCOMP is undefined,
  try_reserve_pinned and release_pinned referenced PinMode and
  opt.pin_mode, which both live inside #ifdef HAVE_NVCOMP in the
  Options struct.  The whole pinned-budget infrastructure is now
  wrapped in #ifdef HAVE_NVCOMP since it's only ever called from GPU
  paths anyway.  CPU-only builds compile cleanly again.

No runtime behavior change for users on the GPU path.

---

## v0.13.1 — Multi-frame oversized decompress no longer corrupts output

The CPU streaming-decompress path added in v0.12.24 (for frames whose
decompressed size exceeds 64 MiB) was only safe for **single-frame**
inputs.  When fed multi-frame inputs with per-frame decomp_size > 64 MiB,
streaming chunks reused sequence numbers that collided with adjacent
frames' natural seqs in the ResultStore — chunks overwrote each other
and the writer either produced truncated output or got stuck waiting
for a frame that had been clobbered.

Surfaced on server during a benchmark sweep at `--ultra -22`: ultra
auto-bumps chunk size to 128 MiB (the windowLog 27 minimum), and the
RAM budget on server's 256 GiB allowed the full 128 MiB to survive,
so every frame qualified for the streaming path.  Two failure modes
observed:
- `cpu-ultra22 / mixed.bin` decompress: produced 2.7 GiB of output for a
  19.5 GiB input (clean exit, truncated data).
- `cpu-ultra22 / zeros.bin` decompress: writer-deadlock detector fired
  with `frame 163 of 577 missing (have 161 buffered)`.

Fix: in `cpu_decomp_worker`, before entering the streaming branch, wait
for the reader to set `producer_done` and only stream when
`results.total_tasks == 1`.  Multi-frame oversized inputs fall through
to the normal `ZSTD_decompressDCtx` path with a per-frame `decomp_size`
allocation — uses more peak RAM but is correct and parallelizable.
The original v0.12.24 motivation (single-frame `zstd` /
`--sliding-window` outputs) is preserved.

Regression test added in `gzstd-test.sh` (`--chunk-size 100` on a
200 MiB input forces 2 frames of 100 MiB each, which trips the bug
without needing the multi-minute `--ultra -22` workload).

---

## v0.13.0 — Asymmetric mode + Apache 2.0 relicense

**License: GPL v3 → Apache 2.0.**  Required for distributable binaries
that link nvCOMP: NVIDIA's nvCOMP license (§2.6) prohibits using the
SDK in a way that would subject it to a copyleft open-source license.
Apache 2.0 keeps gzstd's source permanently free and public, preserves
copyright, and adds an explicit patent grant + retaliation clause that
GPL doesn't have.  See LICENSE (root) and the SPDX header in gzstd.cpp.
Same license used by TensorFlow, PyTorch, RAPIDS, and every other
CUDA-using project that ships binaries.

### Asymmetric mode: smart, hardware-aware backend defaults

GPU compress wins consistently across hardware tiers, but on PCIe Gen3
(consumer cards: RTX 20-series, 30-series, etc.) the D2H transfer cost
makes hybrid *decompress* slower than CPU MT for every data type
measured on Workstation (2× RTX 2080 Ti):

| Data type | CPU-only | Hybrid | Asymmetric default wins by |
|-----------|----------|--------|----------------------------|
| zeros     | 4.88     | 3.50   | +39%                       |
| trivial   | 4.65     | 3.42   | +36%                       |
| medium    | 2.80     | 2.45   | +14%                       |
| mixed     | 1.40     | 1.31   | +7%                        |
| random    | 1.40     | 1.32   | +6%                        |

(GiB/s decompress on Workstation; raw v0.11.20 benchmark numbers.)

gzstd now picks the backend based on hardware *and* operation:
- **Compress (any GPU):** hybrid — GPU compress consistently wins.
- **Decompress / test, PCIe Gen<4:** cpu-only — D2H eats GPU benefit.
- **Decompress / test, PCIe Gen4+:** hybrid — D2H is cheap (Server's H100s).
- **Detection unavailable / no GPU:** hybrid (degrades gracefully).

PCIe gen detection uses `nvmlDeviceGetMaxPcieLinkGeneration()` (the
hardware ceiling, not `Curr` — idle GPUs drop their link to Gen1 for
power management and would otherwise mislead the heuristic).  Fallback
parses `/sys/bus/pci/devices/*/max_link_speed` when NVML isn't built in.

Visible at `-v` as `[ASYMMETRIC] PCIe Gen3 detected; defaulting
decompress to --cpu-only`.  Override with `--hybrid` or `--gpu-only`
when you specifically want to measure or use GPU decompress.

Implementation: `Options::backend_user_set` tracks whether the user
explicitly chose a backend (parsing `--cpu-only`/`--gpu-only`/`--hybrid`
or being implied by `--sliding-window`); the new `apply_backend_defaults()`
runs after `parse_args` and only fills in defaults when no explicit
choice was made.

**Tuning-flag promotion.** Asymmetric mode would silently route around
GPU-tuning flags on Gen3 — `gzstd -d --gpu-batch=64 file.zst` would auto-
flip to cpu-only and the user's tuning hint would do nothing.
`Options::gpu_hybrid_tuning_seen` now tracks any flag that only makes
sense in hybrid/GPU mode (`--gpu-batch`, `--gpu-streams`, `--gpu-devices`,
`--gpu-mem-frac`, `--pinned`/`--no-pinned`, `--cpu-share`, `--cpu-batch`,
`--cpu-backlog`, `--hybrid-floor`, `--hybrid-floor-factor`).
`apply_backend_defaults` promotes these to an implicit `--hybrid` when
no explicit backend flag was given — same precedent as `--sliding-window`
implying `--cpu-only`.  Explicit `--cpu-only` always wins over the
promotion (unchanged precedence).

---

## v0.12.51 — `--cpu-share` actually enforces the requested split

`--cpu-share X` was effectively a no-op: every value from 0.0 to 1.0
landed at ~85% CPU work on Workstation, because the `may_take` predicate
short-circuited on `qs.done` (`if (qs.done) return true;`) — so the
moment the reader called `set_done()`, CPUs drained everything
regardless of the user-set share.  The GPU side never consulted the
share at all, so even after fixing the drain bypass, high shares
(0.9, 1.0) capped at ~86% CPU because GPU kept stealing work.

Three coordinated fixes in `HybridSched` and the worker loops:
- The `qs.done` bypass now only triggers in fixed-share mode if every
  GPU stream has unregistered (real GPU exit, not just stuck in CUDA).
  Otherwise the share is honored through drain.
- New `should_gpu_take()` is the symmetric counterpart of
  `should_cpu_take()`: in fixed-share mode GPU yields when the
  cumulative CPU ratio is below `target − 0.02`, same hysteresis band
  the CPU side uses, so the ratio oscillates around the target instead
  of one side starving.  Adaptive mode is unchanged (always returns
  true; the EMA path drives sharing through `gpus_waiting_` and the
  queue floor).
- GPU workers propagate `producer_done_seen` when the share-yield path
  observes `queue->drained()`, otherwise a perpetually-yielding GPU
  would never exit its loop and the run would hang at high shares.

Measured on Workstation, 19.5 GiB medium-compressibility input, 22 CPU
threads + 2× RTX 2080 Ti.  Before: 0.0 → 0.82, 0.5 → 0.86, 0.9 → 0.84
(all within noise of each other).  After: 0.0 → 0.02, 0.1 → 0.12,
0.25 → 0.27, 0.5 → 0.51, 0.75 → 0.76, 0.9 → 0.87, 1.0 → 0.98.  The
slight undershoot at 0.9/1.0 is end-of-run drainage where GPU sweeps
the tail after CPU threads exit.

Adaptive mode (no `--cpu-share`) is untouched and still hits 5.6 GiB/s
compress on the same input.

---

## v0.12.50 — `--preallocate` / `--no-preallocate` toggle for fallocate

Adds the same on/off control over the `fallocate` upfront-preallocate
that `--mmap`, `--pinned`, and `--direct` already have.  Default stays
ON (matches the prior unconditional behaviour); `--no-preallocate`
skips fallocate so users can A/B test whether it actually helps on
their filesystem.

Touches all four call sites where `g_direct_writer->preallocate(...)`
fires today (compress/decompress, CPU-only and nvCOMP paths).
Preallocation only runs when:
- `--direct` (O_DIRECT) is in effect (fallocate is on `DirectWriter`)
- The expected size is known (input file size for compress, sum of
  frame_decomp sizes for decompress)
- `--preallocate` is on (new — was previously unconditional)

`--no-preallocate` is documented as useful on filesystems that
handle inline extent allocation efficiently (XFS, ZFS), or for
benchmarking the allocation cost.

Tests: 263/263.

---

## v0.12.49 — `[STARTUP]` banner + uniform `[TAG]` verbose-output style

Two related changes addressing the "no visual feedback for several
seconds after pressing enter" complaint on loaded servers:

**1. `[STARTUP]` banner before any heavy init.**  At -v+, the very
first line printed is `[STARTUP] gzstd vX.Y.Z MODE (backend)`.
Printed in main() right after `parse_args` returns, BEFORE `cudaGet
DeviceCount`, file open, output preallocate, or any other potentially
slow step.  Output is `fflush`'d immediately to bypass stderr line
buffering.  Examples:

```
[STARTUP] gzstd 0.12.49 COMPRESS (cpu-only)
[STARTUP] gzstd 0.12.49 COMPRESS (hybrid, CPU share adaptive)
[STARTUP] gzstd 0.12.49 DECOMPRESS (gpu-only)
[STARTUP] gzstd 0.12.49 TEST (auto-select backend)
```

**2. Uniform `[TAG]` style across all verbose output.**  The codebase
had drifted to a mix of `lowercase: prefix:`, `[lowercase]`, and
`[UPPERCASE]` formats.  Standardised every -v / -vv / -vvv message to
`[UPPERCASE_TAG] sentence-case body` with a single space between tag
and body.  Tags now used:

- `[STARTUP]`, `[INIT]`
- `[CPU]`, `[CPU/T#]`
- `[GPU]`, `[GPU#]`, `[GPU#/S#]`
- `[HYBRID]`, `[RESCUE]`, `[WRITER]`, `[READER]`, `[SPLIT]`
- `[THROTTLE]`, `[PINNED]`, `[AUTO-TUNE]`
- `[MMAP]`, `[O_DIRECT]`, `[FALLOCATE]`, `[FSYNC]`, `[RENAME]`
- `[ULTRA]`, `[SLIDING-WINDOW]`

Lines previously emitted as `throttle: ...`, `hybrid: ...`,
`writer: ...`, `using mmap...`, `using O_DIRECT...`, `preallocated
...`, `streamed N frames`, `[pinned] ...`, `[auto-tune] ...`,
`atomic rename: ...`, `fsync: ...`, `GPUs: ...` etc. are now all
prefixed with the appropriate uppercase tag.

The duplicate "Using hybrid mode: CPU share X%" line that used to
fire late inside `compress_nvcomp` is replaced by the early
`[STARTUP]` banner; a `[HYBRID]` confirmation line at -vv announces
when the scheduler actually starts.

Test grep patterns updated for the new format (5 tests).

---

## v0.12.48 — `--throttle-frames=0` / `--no-throttle` to fully disable throttling

For benchmarking the no-throttle baseline.  `FrameThrottle` now
recognizes a non-positive `max_in_flight` as "disabled":
`acquire`/`release`/`set_done` become no-op early returns (no lock
taken, no permit accounting, no peak/block stats).

`--throttle-frames` parsing extended:
- `N >= 1`  : explicit cap (existing behaviour, `source=user`).
- `N == 0`  : DISABLE throttle entirely.  -v shows `throttle: DISABLED`.
- `N == -1` : auto / formula (NEW DEFAULT — was 0 before; semantic
              shift on the sentinel only, default behaviour unchanged
              for users who never passed the flag).
- `N <= -2` : rejected with exit 2 (usage error).

`--no-throttle` is a convenience alias for `--throttle-frames=0`.

Both `compress_*` and `decompress_*` paths construct `FrameThrottle(0)`
when disabled; the existing acquire/release calls scattered through
the workers and the writer just no-op.  No control-flow changes
required outside the throttle class.

`log_throttle_stats` skips the stats line on disabled throttles and
just prints `DISABLED` instead.

**Quick A/B on Workstation** (24-core, 2 GiB mixed input):

| mode | throttle=auto | throttle=0 |
|---|---|---|
| compress | ~0.53 s | ~0.55 s |
| decompress | ~0.62 s | ~0.68 s |

Throttle off is *slightly slower* on this workload — without backpressure
the result store can grow large enough that L3 / RAM cache effects
hurt.  As suspected, the throttle is a guardrail not an optimization,
but you can now measure that directly.

Tests: 263/263 (+2 new — disabled-mode verification, `--no-throttle`
alias verification, replaced the old "must reject 0" rejection test).

---

## v0.12.47 — `--sweep-matrix` benchmark option for backend × mmap × pinned

Adds a small structured sweep that produces 10 configs (cpu-only × 2
mmap states + hybrid × 4 + gpu-only × 4, with pinned skipped on cpu-only
since it's a GPU-side knob).  Captures the same "which tricks actually
help on this system" analysis we did manually in v0.12.45/46 but as a
re-runnable harness, so the result interpretation can be reproduced on
any new machine without hand-rolling shell loops.

Smoke-tested against the 20 GiB `medium_compress.bin` profile:

| config | GiB/s |
|---|---|
| mtx-cpu-mmap | **7.05** (best) |
| mtx-hyb-mmap-pin0 | 5.23 |
| mtx-hyb-mmap-pin1 | 4.47 |
| mtx-cpu-nommap | 2.42 |
| mtx-gpu-mmap-pin0 | 2.33 |
| mtx-gpu-nommap-pin1 | 2.33 |
| ... | |

Confirms the v0.12.45/46 conclusions at 20 GiB scale on Workstation
(2× 2080 Ti): CPU-only crushes, mmap wins, pinned hurts.

`--sweep-all` now also enables `--sweep-matrix`.  Add `--sweep-matrix`
on its own for a focused 10-config run.

---

## v0.12.46 — `--mmap=on/off` toggle (default: on)

The mmap zero-copy reader has been the default for regular-file inputs
since early in the project but had no escape hatch — useful for
benchmarking against a stack of "tricks that don't help on this
system" (O_DIRECT, pinned RAM, atomic rename).  This pulls mmap up to
the same level: on by default, but `--no-mmap` lets you A/B against
fread to verify it's actually winning on your hardware.

**Local validation** (Workstation, 4 GiB mixed input, page cache warm):
| mode | mmap (default) | --no-mmap |
|---|---|---|
| `--cpu-only -T18` compress | ~1.4 s | ~2.2 s (~50% slower) |
| `--gpu-only` compress | ~3.9 s | ~3.9 s (wash) |

mmap wins clearly on CPU-only paths because workers read directly from
the page cache — no producer-side fread + memcpy through a userspace
buffer to serialise the input read.  GPU paths are a wash: the input
gets H2D-copied either way, and the page cache makes pageable cudaMemcpy
near-DMA-speed regardless of source.

`--mmap=on` / `--mmap=off` / `--no-mmap` accepted.  Pipes and stdin
always fall back to fread (mmap requires a regular file).

---

## v0.12.45 — `--pinned` default flipped to `off` (pinned was measured slower)

The plumbed-in pinned-host-memory infrastructure (v0.12.43, v0.12.44)
turned out to be slower than pageable on every workload tested.
On Workstation (2× 2080 Ti), 4 GiB mixed-compressibility input:

| mode       | --pinned=off | --pinned=on    |
|------------|--------------|----------------|
| compress   | ~3.6 s       | ~4.2 s (-15%)  |
| decompress | ~1.9 s       | ~4.4 s (-2.4×) |

Decompression was particularly bad — pinned cudaMemcpy + extra copy
into the result vector was 2-3× slower than direct device→pageable.

Likely causes:
- Input pages are usually already in the OS page cache (mmap'd file
  on a fast NVMe), so cudaMemcpy from pageable is near-DMA-speed
  anyway — the locked-page DMA path doesn't win.
- Locking pages out of the page cache hurts other parts of the
  pipeline (reader fread-ahead, writer fwrite cache).
- The mandatory pinned -> pageable memcpy for the result vector adds
  pure overhead with no offsetting gain.

**Fix.** `Options::pin_mode` default changed from `AUTO` to `OFF`.
The infrastructure is plumbed and exposed; users can opt in with
`--pinned=on` or `--pinned=auto` if their hardware/workload differs.
Help text updated to explain the trade-off.

Existing pinned tests (which verify flag acceptance) still pass.
`-v` no longer prints `[pinned]` lines unless explicitly enabled.

---

## v0.12.44 — Compress reuses one pinned buffer for both H2D and D2H

v0.12.43 only pinned H2D for compress (output went direct device →
pageable vector).  But the H2D pinned slot is unused after the upload
finishes — the GPU has the data on-device, the host slot sits idle for
the rest of the batch.  v0.12.44 reuses that same slot for the D2H
output readback:

1. H2D phase: input chunk is memcpy'd into `pinned[i]`, then
   `cudaMemcpyAsync` host → device.
2. Compute phase: GPU has the data; pinned slot is idle.
3. D2H phase: compressed output `cudaMemcpy`'d device → `pinned[i]`,
   then `memcpy` from pinned slot into the output `std::vector<char>`.

Each slot is sized to `max(gpu_chunk, max_out_chunk)` (≈ 16 MiB +
~3 KiB) so either direction's data fits.  Per-stream batches are
already serialised (`C.busy` gates re-pop until D2H + result delivery
complete), so there's no buffer-conflict race.

**Net effect:** compress now gets pinned D2H **for free** — same RAM
allocation as before, just slightly larger slot stride.  Pinned
cudaMemcpy uses a faster DMA path than pageable, so D2H finishes
sooner and the tot_ms / GPU-throughput numbers at -vv reflect that.

The `[pinned]` log line at -v changed:
```
[pinned] H2D 1.62 GiB reserved              # before v0.12.44
[pinned] H2D+D2H 1.63 GiB reserved (shared per slot)  # v0.12.44+
```

Decompress is unchanged (separate H2D-pageable / D2H-pinned scheme
from v0.12.43).  Adding pinned H2D there would mean an extra
mmap → pinned memcpy, which usually doesn't pay off because the
input pages are already cached.

---

## v0.12.43 — `--pinned auto` rations to ≤50% RAM + adds D2H pinning on decompress

### `--pinned auto` is now actually a heuristic

Before: `auto` and `on` were treated identically — both unconditionally
called `cudaHostAlloc`.  Misleading naming.

Now: `auto` rations pinned host memory to ≤50% of available system RAM,
summed across ALL gpu-worker threads (compress H2D + decompress D2H).
Streams that fit get pinned.  Streams that don't ("unlucky" ones) fall
back to pageable memory silently.  Same fallback if `cudaHostAlloc`
fails for any reason.

`--pinned on` keeps the prior behaviour (unconditional reserve, ignores
the budget).  `--pinned off` (and `--no-pinned`) skip pinning entirely.

Implementation: a global `g_pinned_bytes_reserved` atomic + `try_reserve_pinned` /
`release_pinned` helpers.  AUTO uses CAS to reserve from the global
budget; ON / OFF short-circuit.  The `[pinned]` log line at -v shows
each reservation and any skipped streams with the reason.

### Pinned D2H buffer added to decompress

Before: `DecompStreamCtx` had no pinned host memory at all — every D2H
copied straight from device to a freshly-allocated `std::vector<char>`
in pageable memory.

Now: each decompress stream allocates a pinned host staging buffer of
`alloc_batch * alloc_decomp` bytes (typically a few GiB per stream).
The D2H loop copies device → pinned slot, then `memcpy` from pinned
slot into the output `std::vector`.  Pinned cudaMemcpy uses a faster
DMA path; the pinned-to-pageable memcpy is a plain `memcpy` (which the
kernel optimises well).

Allocation honours the same `--pinned auto` budget; on `--pinned off`
the decompress path falls back to the previous direct-to-pageable
behaviour.  The pinned buffer is reused across batches and grown only
when `alloc_batch` or `alloc_decomp` increase, so per-batch overhead
is zero.

(Compress D2H still uses direct exact-size copies — output sizes are
variable, so a fixed-size pinned slot would either waste 2× memory or
need a per-chunk pinned allocator.  Could be added later if measured
to matter.)

---

## v0.12.42 — `--help`: throttle flags moved to CPU/GPU TUNING (apply to all modes)

`--throttle-factor` and `--throttle-frames` were listed only under
HYBRID SCHEDULER, which made them look hybrid-only.  They actually
affect every multi-threaded path (CPU-only compress with `-T ≥ 2`,
CPU-only decompress, GPU-only, hybrid).

Moved the canonical description into CPU TUNING with `[all modes]`
markers and concrete tuning guidance ("bump to 8 or 16 if you see
`source=pipeline` and the writer is bursty").  Added a cross-reference
in GPU TUNING with a GPU-specific note about permit starvation when
N_GPUs * streams * batch exceeds the default budget.  Removed the
duplicate entries from HYBRID SCHEDULER.

No behaviour change — purely documentation.

---

## v0.12.41 — `--overwrite`: unlink-then-create instead of truncate-on-fopen

**Symptom (Server, 432 GiB output).**
```
time ./build/gzstd -d --cpu-only -T18 --direct --overwrite -v ...
using O_DIRECT for output (--direct)
```
The `using O_DIRECT` line appeared 10–30 seconds after the command was
launched.  No output at all during that window.

**Cause.**  `--overwrite` opened the existing target with `fopen(path,
"wb")`, which truncates the file in place.  On ext4, `truncate(0)` on
a 432 GiB file has to free every extent the inode references — that's
O(file_size), and ext4's journal makes the freeing synchronous before
`fopen` returns.  All subsequent setup (the `using O_DIRECT` log,
throttle config, worker spawn) sat behind that truncate.

**Fix.**  `--overwrite` now `unlink()`s the existing target first, then
`fopen("wb")` creates a fresh empty inode in O(1).  The original
inode is unreferenced immediately; ext4 frees its extents in the
background.  No user-visible blocking.

Verified locally on a 4 GiB stand-in: time-to-first-output went from
visible delay to ~0.2 s end-to-end including round-trip.

`--sync-output` semantics are unchanged.  `-f` (atomic, with rename)
already wrote to a fresh `.gzstd.tmp` file and didn't have this
problem.

---

## v0.12.40 — Parameter-honor verification tests + `--overwrite` no progressive sync

### Tests

The existing test suite did round-trip checks for `--gpu-batch=N` etc.
("compress with the flag, decompress with the flag, output matches"),
but never verified the flag was actually applied at runtime.  The
v0.12.39 `--gpu-batch` regression slipped through because the tests
couldn't see that batches were popped at size 4 instead of N.

New `Parameter honor verification` section (+20 tests, 261 total) parses
verbose output to check runtime behaviour matches CLI input:

- **`--gpu-batch=N` honored at -vv**: parses every `[GPU/S] take batch=N`
  line, asserts all non-final batches equal N exactly.
- **`--gpu-streams=N` honored**: counts unique `[GPU#/S0..N-1]` indices
  in `pre-alloc` lines.
- **`--chunk-size=N` produces `ceil(file_size / N MiB)` frames**: counts
  `[CPU/T#] take seq=` lines emitted at -vv.
- **`-T N` spawns N workers**: greps for the worker-online line at -v
  (or single-thread streaming path for `-T 1`).
- **Verbosity escalates correctly**: `-v` has no `[CPU/T#] take seq=`
  (V_DEBUG content); `-vv` does, but no `[SPLIT]` (V_TRACE content);
  `-vvv` includes `[SPLIT]`.  Unique-line count strictly increases with
  verbosity level.
- **`-M N` round-trip**: re-verifies the v0.12.30 fix end-to-end.
- **`--throttle-frames=N` visible at -v**: greps for `source=user` or
  the explicit count.
- **`--no-sparse` vs default sparse**: compares `stat -c '%b'` block
  count on an all-zeros decompressed file.
- **`--ultra` is required for level 20+** and `--ultra -20` produces
  valid output.

### `--overwrite` skip progressive writeback

In v0.12.25 we enabled `sync_file_range(SYNC_FILE_RANGE_WRITE)` for all
decompress runs to fix the multi-second rename stall on ext4
`data=ordered`.  But `--overwrite` skips the tmp+rename dance entirely —
the rename stall doesn't apply, and the writeback hint just steals
bandwidth from `fwrite`.  Now disabled when `unsafe_overwrite` is set.

`--sync-output` is still opt-in (default off): the only thing that
forces an explicit `fsync()` on the output is `--sync-output`.  Plain
`fclose()` flushes user buffers to the kernel; the OS handles writeback
on its own schedule.

---

## v0.12.39 — Honour `--gpu-batch=N` exactly (full batches, not soft-min)

**Symptom (Workstation).**  `gzstd -d --gpu-only --gpu-batch=64 -vv` showed
`pre-alloc batch=64` (buffers correct) but actual pops were small:
```
[GPU0/S0] take batch=4 seq=[0..3] in=22.05 KiB
[GPU0/S0] take batch=8 seq=[8..15] in=128.00 MiB
[GPU1/S0] take batch=7 seq=[16..22] in=112.00 MiB
```

**Cause.**  Both `compress_nvcomp` and `decompress_nvcomp` pop with a
hardcoded soft minimum:
- decompress: `pop_batch_greedy(pop_n, ..., min_n=min(pop_n, 4))`
- compress:   `pop_batch_greedy(pop_n, ..., min_n=1)`

So when the queue had only 4 frames available, the GPU returned with 4
even though the user pinned `--gpu-batch=64`.  The soft minimum is
sensible during auto-tuning (multi-GPU shouldn't serialize behind a
single producer) but contradicts the user's explicit pin.

**Fix.**  When `shared_tune->locked` is set (user pinned
`--gpu-batch`), `min_n = pop_n` — wait for the full batch.  When
unlocked (auto-tuner active), the previous soft minimums apply.
`pop_batch_greedy` still returns early at end-of-queue regardless, so
no deadlock — but during steady-state operation the GPU now sees the
batch size the user asked for.

Applied to both compress (gzstd.cpp:5462) and decompress
(gzstd.cpp:6927).

---

## v0.12.38 — Restore concurrent worker spawn / parser (v0.12.21 architecture)

**Regression introduced in v0.12.22.**  When `--sliding-window` shipped
in v0.12.22, `decompress_nvcomp` was restructured to call
`stream_frames_to_queue` BEFORE spawning workers — so the producer's
`max_frame_decomp` could be checked against `GPU_SUBCHUNK_MAX` to
short-circuit GPU init for oversized single-frame files.

**Side effect.**  On large inputs the parse phase blocks for tens of
seconds.  All worker init (`throttle: …`, `[GPU] N device(s) online`,
`[GPU#/S#] pre-alloc batch=`, `hybrid decompress: N CPU threads`,
`hybrid: tick …`) was silent during that window — users saw nothing
but the producer's `[SPLIT] frame N` lines until parsing finished.

v0.12.21 had it the right way around: workers spawn first, parser runs
afterwards while workers are already consuming.  Init lines appeared
immediately at -v/-vv/-vvv.

**Fix.** Restored the v0.12.21 ordering in `decompress_nvcomp`:
1. Detect GPU device count (existing).
2. **NEW:** `peek_first_frame_decomp_size(in)` — read just the frame
   header bytes, get frame 0's decomp size, then `fseek` back to 0.
   If size > 16 MiB (single-frame oversize), set `device_count=0` and
   fall back to CPU.  Cheap because it touches only ~64 bytes.
3. Set up throttle, writer thread, hybrid scheduler.
4. Spawn CPU pool and GPU workers (init lines fire here).
5. Run `stream_frames_to_queue` (workers consume concurrently as the
   parser pushes).

The peek-only check covers the typical "oversize" case (zstd /
--sliding-window single-frame files where frame 0 IS the whole file).
For pathological multi-frame files where only a non-first frame is
oversize, the GPU runtime fallback path handles it as before.

User-visible result: at `-vvv` the throttle config, GPU device list,
`[GPU] N device(s) online`, and per-stream `[GPU#/S#] pre-alloc`
output all appear immediately when the command is run, instead of
after a 30-second delay on a 432 GiB input.

---

## v0.12.37 — CPU decompress worker verbose output parity with compress

**Symptom.** Compression's `cpu_worker` emits per-task and per-thread
verbose output:
- `[CPU/T#] take seq=N in=X` before each frame (`-vv`)
- `[CPU/T#] seq=N in=X out=Y ms=… thr=…` after each frame (`-vv`)
- `[CPU/T#] total tasks=… in=… out=… time=…ms thr=…` per-thread summary (`-vv`)
- `[CPU/T#] idle (0 tasks)` for unused workers (`-vvv`)

Decompression's `cpu_decomp_worker` only emitted the post-frame
`seq=…` line.  No "take" line, no per-thread summary, no idle reporting
— so `--cpu-only -d -vv` and `--hybrid -d -vv` looked drastically more
sparse than the equivalent compress runs.

**Fix.** Added the missing logs to `cpu_decomp_worker` so output now
matches the compress pattern:
- Per-task `[CPU/T#] take seq=N comp=X decomp=Y` before processing (V_DEBUG)
- Per-thread `[CPU/T#] total tasks=… comp=… decomp=… time=…ms thr=…`
  summary at exit (V_DEBUG)
- `[CPU/T#] idle (0 tasks)` for unused workers (V_TRACE)

Trace-mode users now see the same level of detail on the decompress
side that they've always had on compress.

---

## v0.12.36 — Visible init output during decompress pre-scan

**Symptom (Server, large `-d` runs).** With a 432 GiB `.zst` file the user
saw a long stretch of nothing but `[SPLIT] frame N` lines and asked
"where's the init output?".  No `[GPU]` device-online lines, no
`[GPU/S] pre-alloc batch=`, no throttle line — until the parse phase
finished tens of seconds later.

**Cause.** `decompress_nvcomp` does a full pre-scan of the input
(`stream_frames_to_queue`) *before* spawning GPU workers.  The pre-scan
is needed to detect oversized frames (sliding-window / `zstd`) and
fall back to CPU before allocating GPU buffers it can't fill.  But on
large inputs the pre-scan is the bulk of wall time, and during it only
the producer's `[SPLIT]` lines emit — the user-visible init lines
(throttle config, `[GPU] N device(s) online`, `[GPU#/S#] pre-alloc`,
etc.) all queue up behind the pre-scan.

**Fix.** Added three `[INIT]` log lines that fire BEFORE the pre-scan:
- `[INIT] decompress: N GPU(s) detected, mode=gpu-only|cpu-only|hybrid|auto`
- `[INIT] pre-scanning input frames (workers spawn after pre-scan)`
- (after pre-scan) `[INIT] pre-scan complete: N frames, max_decomp=X (Ts)`

Visible at `-v`/`-vv`/`-vvv`.  This doesn't change the architectural
ordering — workers still spawn after pre-scan — but the user now sees
that gzstd is alive and what phase it's in.  A future change can move
parsing into a thread that runs concurrently with worker spawn.

---

## v0.12.35 — Per-chunk `-vvv` output for GPU compress/decompress

**Symptom (Server, `--gpu-only -d -vvv`).** The trace output looked
sparse — mostly just the producer's `[SPLIT] frame N` lines every 1000
frames, with little visible GPU activity.

**Cause.** Side effect of v0.12.32: per-stream batches are now allowed
to grow up to 256 chunks (vs the previous 8-cap).  The existing
`[GPU#/S#] take batch=` and `[GPU#/S#] done batch=` lines fire once per
batch — at V_DEBUG (`-vv`).  After v0.12.32 a 16k-frame run produces
~63 batches instead of ~2000, so those lines show up ~30× less often.
At `-vvv` the user expects flood-of-detail, not "less than `-vv` used
to give."

**Fix.** Added per-chunk emission at V_TRACE (`-vvv`) in three places:
- GPU compress async-poll completion path
- GPU compress sync-drain completion path
- GPU decompress completion path

Each chunk in a completed batch now prints
`[GPU#/S#] chunk seq=N in=X out=Y` at -vvv.  V_DEBUG output is
unchanged.

---

## v0.12.34 — Test-count display in `gzstd-test.sh`

The runner's progress bar showed `N of 192` while the actual run
ended at 241 tests, finishing past 100% completion.  The `count_tests`
function had a hand-maintained per-section breakdown that drifted as
new sections were added.

Replaced with a single `EXPECTED_TESTS=241` constant at the top of the
file — bump it when you add/remove tests.  Two safety nets prevent
display breakage if the constant is forgotten:

- `progress_bar` clamps `pct` to 100 and auto-expands `TOTAL_TESTS` if
  the running count exceeds the planned count, so the bar never shows
  more than 100%.
- A drift-check line at the end of the run prints
  `note: EXPECTED_TESTS=N at top of script but M ran — please update.`
  whenever the actual ran count diverges from the constant.

Simpler than chasing a perfect static count or maintaining a cache file.

---

## v0.12.33 — Throttle starvation in hybrid mode (GPUs blocked on permits)

**Symptom (Server, hybrid compress).** Per-batch GPU subchunk count grew
fine after v0.12.32, but `nvtop` showed the H100s mostly idle.  CPUs
were doing the bulk of the work while GPUs sat blocked.  `-vvv` reported
`gpus_waiting=0`, which is technically correct (the wants/got window is
microseconds long) but obscured the real cause.

**Cause.** The frame-throttle budget in both `compress_nvcomp` and
`decompress_nvcomp` was sized off `opt.gpu_batch_cap` (default 8):

```cpp
const int comp_gpu_batch_floor = gpu_count * gpu_streams * opt.gpu_batch_cap;
const int comp_parallelism     = cpu_threads + comp_gpu_batch_floor;
FrameThrottle throttle(compute_throttle_budget(..., comp_parallelism, ...));
```

After v0.12.32 the auto-tuner can grow per-stream batches up to
`AUTO_TUNE_BATCH_CEILING` (256), but the throttle was sized for 8.  On
Server (8 GPUs × 1 stream + 96 CPU workers), every CPU that had popped a
frame was holding one permit (held until the writer drains it), so when
a GPU stream tried to `bp->acquire(pop_n)` for, say, 64 permits, it
blocked waiting for CPUs to drain.  Effectively the GPU pipeline was
serialised through CPU writeout speed.

**Fix.** Both `compress_nvcomp` (line 6125) and `decompress_nvcomp`
(line 7333) now compute the throttle floor using the *effective* per-
stream max:

```cpp
per_stream_budget = opt.gpu_batch_user_set
    ? opt.gpu_batch_cap
    : std::max(opt.gpu_batch_cap, AUTO_TUNE_BATCH_CEILING);
gpu_batch_floor   = gpu_count * gpu_streams * per_stream_budget;
```

When `--gpu-batch=N` is set, the budget honours that value exactly (no
auto-grow either, so no headroom needed).  Otherwise it provisions
enough permits for the auto-tuner's full growth path.

**Server example.** Before: floor = 8×1×8 = 64; throttle ≈ 640 frames
total → 8 streams × 64 = 512 GPU permits + 96 CPUs ≈ over budget.
After: floor = 8×1×256 = 2048; throttle ≈ 8192 frames (RAM-capped) →
2048 GPU + 96 CPU = 2144, well under budget.

---

## v0.12.32 — Fix GPU batch frozen by allocation (auto-tuner had no headroom)

**Symptom (Server, `--gpu-only` compress).** The per-batch GPU subchunk
count was stuck at 8 across the entire run regardless of throughput.  The
shared auto-tuner appeared to do nothing.  Hybrid compression had the same
problem (same code path).  Decompression was partially affected for files
under ~10 GiB.

**Cause.** Two interacting pieces, present in both compress and decompress:
1. The GPU init path allocates per-stream buffers based on
   `per_stream_cap = std::min(opt.gpu_batch_cap, HARD_BATCH_CAP)` — for
   compress that defaults to `min(8, 1024) = 8`; for decompress on small
   files it's `min(16, 1024) = 16`.  A VRAM-fit search lowers this further
   if needed but never raises it.
2. The pop site clamps the per-batch size: `pop_n = std::min(pop_n,
   C.per_stream_batch)`.  So even when `SharedTuneState::batch_size` grew
   to 16, 32, etc., the actual pop was still 8 (compress) or 16
   (small-file decompress) because the buffers were only big enough for
   that many subchunks.

The auto-tuner's growth path was therefore silently dead on those paths.
Long-standing: the clamp was introduced in v0.10.34 alongside
`SharedTuneState`.

**Fix.** Both `compress_nvcomp` and `decompress_nvcomp` now size per-stream
buffers up to `AUTO_TUNE_BATCH_CEILING` (256) when `--gpu-batch` is not
user-pinned, giving the shared tuner real room to grow.  The VRAM-fit
halve loop still shrinks this if the GPU can't hold it.

**Compress.**  Pure win — buffer was previously capped at 8.

**Decompress.**  Already-large files (>75 GiB → cap=256, >10 GiB →
cap=64) are unchanged because `max(cap, 256) == cap` (or close to it).
Small files (<10 GiB) now allocate up to 256 per stream instead of 16.

**VRAM impact.**  Each subchunk needs `gpu_chunk + max_out_chunk +
temp/N` in device memory.  With 16 MiB chunks that's ~33 MiB per slot
plus nvCOMP scratch.  256 slots per stream is ~8.5 GiB plus scratch —
fits comfortably in H100 VRAM under the default `--gpu-mem-frac=0.60`.
On smaller GPUs the binary-search VRAM-fit loop already halves the
allocation when needed.

**User override.** Pass `--gpu-batch=N` to pin a specific size and skip
auto-tuning; that path is unchanged on both compress and decompress.

---

## v0.12.31 — Fix `out:%` jumping to ~90% immediately on `--cpu-only` compress

**Symptom (Server, 432 GiB tar via `--overwrite --cpu-only --direct`):**
```
in:12.8% 55.34 GiB 4.56 GiB/s | out:91.5% 14.95 GiB 1.23 GiB/s
```
The `out:%` jumped to ~90% almost immediately and stayed there for the
duration of the run, while `in:%` ticked up normally.

**Cause.** `compress_cpu_mt` set `meter.total_out_final = true` inside the
producer-done block.  That flag was designed for decompression — where
`total_out` is summed from `frame_header.decomp_size` during the pre-scan
and IS a known final total.  For compress, `total_out` is a *running
accumulator* incremented by the writer-collector at line 3016 as compressed
batches arrive.  Setting `total_out_final = true` makes the progress code
take the "decompress, reader done" branch
(`wrote_bytes / total_out_so_far`), which is just the writer's catch-up
ratio — typically 80–95%.

The GPU compress path (`compress_nvcomp`) correctly leaves
`total_out_final` unset.  Now `compress_cpu_mt` matches that.

**Result.** With the flag unset, the percentage logic falls through to the
frame-level branch (`tasks_done / total_frames`), giving a percentage that
tracks `in:%` instead of jumping to 90% right away.

---

## v0.12.30 — `-M` / `--memlimit` / `--memory` now real flags

Promoted from v0.12.29's warn-no-op set to actual implementations.

**Accepted forms:** `-M N`, `-M N` (joined or separated), `--memlimit N`,
`--memlimit=N`, `--memory N`, `--memory=N` — value is in MiB to match zstd.

**Decompression.** The value is pushed to every `ZSTD_DCtx` via
`ZSTD_d_windowLogMax` with `wlog = floor(log2(N * 1 MiB))`, clamped to the
`[10, 31]` range zstd accepts.  Streams whose frames require a larger window
are rejected with zstd's `Frame requires too much memory for decoding`
error (exit 4 = data error) rather than being allowed to allocate unbounded
memory.  This matches zstd's own semantics for `-M`.

**Compression.** zstd itself ignores `-M` for compress; gzstd uses the
value to tighten the in-flight frame-throttle budget in
`compute_throttle_budget`.  Without `-M`, the RAM cap is
`min(pipeline_parallelism * slack, RAM/2 / frame_bytes)`.  With `-M N`
the cap is lowered to `max(1, N * 1 MiB / frame_bytes)` if that's
smaller, and the throttle source in `-vv` output shows `source=ram`
whenever the user's limit is the binding constraint.

**Not applied to nvCOMP decompression.**  The GPU path allocates its
VRAM buffers through nvCOMP, which has its own memory accounting via
`--gpu-mem-frac` — the host-side `-M` cap doesn't directly apply there.
Frames that fall back to CPU rescue respect the limit through the
worker thread's `tl_dctx`.

---

## v0.12.29 — zstd-compat flag layer

gzstd now accepts the full zstd CLI flag set so it can truly serve as a
drop-in replacement.  Flags fall into four buckets:

**Real aliases** (map to existing gzstd semantics):
`--decompress`, `--uncompress`, `--force`, `--keep`, `--test`, `--verbose`,
`--stdout`, `--to-stdout`, `-H` (long help), `--single-thread` (≡ `-T 1`),
`--fast=#`.

**Silent no-ops** (zstd defaults that gzstd already matches — accepted without
comment): `--asyncio`, `--no-asyncio`, `--check`, `--no-check`,
`--format=zstd`, `--no-dictID`, `--compress-literals`,
`--no-compress-literals`, `--row-match-finder`, `--no-row-match-finder`,
`--mmap-dict`, `--no-mmap-dict`, `--stream-size=…`, `--size-hint=…`,
`--target-compressed-block-size=…`, `--auto-threads=…`.

**Warn no-ops** (zstd features gzstd does not implement — accepted with a
`gzstd: warning: <flag> accepted for zstd compatibility but ignored` line):
`--adapt`, `--long[=#]`, `--patch-from[=REF]`, `--rsyncable`,
`--exclude-compressed`, `--format=gzip|xz|lzma|lz4`, `--pass-through`,
`--no-pass-through`, `-r`/`--recursive`, `-l`/`--list`, `--filelist`,
`--output-dir-flat`, `--output-dir-mirror`, `--trace`, `-D`/`--dict`/
`--dictionary`, `--train`/`--train-*`, `--maxdict`, `--dictID=#`, `-B#`,
zstd benchmark flags (`-b#`/`-e#`/`-i#`/`-S`/`--priority=rt`).

(`-M#` / `--memlimit` / `--memory` started here as warn-no-ops and were
promoted to real flags in v0.12.30 — see that entry.)

The warn stream respects verbosity: `-q` / `-qq` / `--quiet` / `--silent`
suppress the compat warnings.  A pre-scan of `argv` sets the suppression
threshold so the quieting flag can appear in any position.

---

## v0.12.28 — Help split: concise `-h` / `-?`, detailed `--help` with examples

`-h` and the new `-?` alias print a short, grouped option list (Operation /
Output / Compression / Backend / Tuning / I/O / Logging / Misc) intended to
fit on a single terminal screen.  `--help` now prints a long reference with
per-flag descriptions, flag interactions, exit codes, and a block of runnable
examples covering the common workflows (compress, decompress with
`--overwrite`, piped tar, CPU-only baseline, GPU-only tuning,
`--sliding-window`, integrity check, forced progress, stats JSON).

---

## v0.12.27 — `--gpu-batch` is now per-stream on compress (BEHAVIOR CHANGE)

**Symptom.** `--gpu-only --gpu-batch=512 --gpu-streams=4` on the compression path was allocating only **128** subchunks per stream, not 512. The user expected "each stream gets batches of 512."

**Cause.** The compression producer-side batch cap was computed as `ceil(gpu_batch_cap / stream_count)` — treating `--gpu-batch` as a *per-device* total and dividing across streams. The decompression path treated the same flag as *per-stream* (see comment at `decompress_nvcomp`: "kernel launch overhead dominates, so each stream needs large batches"). The help text ("Max GPU subchunks per device") agreed with compress, disagreed with decompress.

**Fix.** Compress now uses `--gpu-batch` as a per-stream cap, matching decompress. With `--gpu-batch=512 --gpu-streams=4`, each of the 4 streams now aims for 512 subchunks. VRAM safety is preserved: `gpu_mem_fraction` is still divided across streams, and the per-stream binary search clamps down when the requested batch doesn't fit.

**Compatibility.** Runs that relied on the old semantics (compress dividing the flag) will see more subchunks in flight and higher VRAM usage. If VRAM is tight the binary search will report "VRAM-fit: batch=N (requested M)" at `-v`. To restore the previous effective per-stream batch, divide the old value by `--gpu-streams` (e.g., old `--gpu-batch=512 --gpu-streams=4` → new `--gpu-batch=128 --gpu-streams=4`).

**Help text updated:** `Max GPU subchunks per CUDA stream (default: 16)`.

---

## v0.12.26 — `--overwrite` (non-atomic) + perf-breakdown reader stats fix

### 1. New `--overwrite` flag

**Symptom (Workstation).** Running `gzstd -d -f big.zst` against a pre-existing output file stalled for tens of seconds at the final rename, while deleting the target first and letting gzstd create a fresh file was fast. v0.12.23 already reduced this stall with `sync_file_range`, but on ext4 with large outputs a substantial rename cost remained.

**Cause.** `-f` always used the `.gzstd.tmp` + `rename()` atomic-overwrite dance, which on ext4 `data=ordered` ties rename commit to flushing dirty pages. For workloads where atomicity isn't worth that cost, users want to opt out.

**Fix.** New `--overwrite` flag (implies `-f`) bypasses the atomic dance: gzstd calls `fopen(target, "wb")` directly, truncating the target in place. No tmp file, no rename. Trade-off: if gzstd is interrupted, the target is partial/corrupt.

- Default `-f` behaviour (atomic) is unchanged.
- Regular-file check still applies (FIFOs, devices, stdout are unaffected).
- The target is still registered with the cleanup handler so `Ctrl-C` removes the half-written file.

### 2. Reader stats showing all zeros under `-vvv`

**Symptom.** The `PERFORMANCE BREAKDOWN` table printed `Reader: 0.000 s (0.00 GiB, 0.00 GiB/s)` for any run that used the default mmap zero-copy reader.

**Cause.** `compress_cpu_mt` and `compress_nvcomp` had two producer paths: a `fread` path (which recorded `read_ns` / `read_bytes_total`) and an `mmap` path (which didn't). For regular files on Linux, gzstd always takes the mmap path, so `PerfCounters` never saw any bytes.

**Fix.** Both mmap producer loops now record `read_bytes_total` (= mapped file size) and `read_ns` (time spent enqueuing view tasks). The timing is small — pointer arithmetic, not I/O — but the bytes column now reflects reality.

---

## v0.12.25 — Compression I/O fixes + GPU D2H timing correction

**Three independent fixes surfaced while investigating why gzstd was 6.5× slower than `zstd -T0` on barely-compressible data and inconsistent across runs.**

### 1. Output `setvbuf` — multi-MiB buffer instead of glibc default

**Symptom.** Compression of large barely-compressible data was dominated by `write()` syscall overhead. A 14.4 MiB `fwrite` was being split into 1800–3600 individual `write()` syscalls by the glibc default ~4–8 KiB FILE buffer.

**Fix.** Set a 1 MiB `_IOFBF` buffer on every output `FILE *` opened by gzstd (both `open_output_atomic` and the two `fopen` call sites in `main`). This collapses the syscall count by ~128–256×.

**Long-standing issue.** This has been present for the entire history of the tool — not a v0.12.x regression. Affected both compression and decompression output paths.

### 2. `sync_file_range` gated behind `progressive_sync_` flag

**Symptom.** v0.12.23 added unconditional `sync_file_range(..., SYNC_FILE_RANGE_WRITE)` in `AsyncWritePool::worker_fn` to fix a 46s `rename()` stall on decompression. This caused a measurable regression on compression: the non-blocking writeback hint created I/O contention with subsequent `fwrite` calls, triggering `balance_dirty_pages` throttling inside the writer thread.

**Root cause.** Compression produces much smaller output than decompression (e.g., 19 GiB in → 2 MiB out for trivially-compressible data), so the writeback hint buys nothing but steals bandwidth from the next `fwrite`.

**Fix.** `AsyncWritePool` now takes a `progressive_sync` bool (default `false`). Only enabled for decompression, where the ordered-journal rename stall is the real concern.

### 3. GPU D2H timing always reporting `0.00 ms` at `-vvv`

**Symptom.** `--gpu-only` compression with `-vvv` consistently reported `d2h=0.00ms` per batch, even though real D2H copies were happening.

**Root cause.** `cudaEventRecord(C.ev_comp_end)` and `cudaEventRecord(C.ev_d2h_end)` were recorded back-to-back in the CUDA stream with no D2H operation between them. The actual D2H copies are synchronous host-side `cudaMemcpy` per chunk, which happen *after* stream completion — the CUDA events couldn't see them.

**Fix.** Replaced CUDA event timing with wall-clock `now_ns()` timing bracketed around the host-side D2H memcpy loop in both the async poll path and the sync drain path. Total time is now correctly computed as `h2d_ms + comp_ms + d2h_ms`. The `ev_d2h_end` event is still created/destroyed for ABI simplicity but is unused.

---

## v0.12.24 — Streaming decompression for oversized frames

**Symptom.** Decompressing single-frame .zst files (e.g., from `zstd` or `gzstd --sliding-window`) showed the `out:` progress bar stuck at 0% for the entire decompression, then jumping to 99.9% at the end. Memory usage spiked to the full decompressed size (e.g., 125 GiB) because the worker allocated one giant buffer for the entire frame.

**Root cause.** `cpu_decomp_worker` called `ZSTD_decompressDCtx` with a single output buffer sized to the full decompressed frame. Nothing was written to disk until the entire frame was decompressed, so the writer (and its progress tracking) had no work to do until the very end.

**Fix.** For frames larger than 64 MiB (`STREAM_THRESHOLD`), the worker now uses `ZSTD_decompressStream` with 16 MiB output chunks. Each chunk is pushed to `ResultStore` with its own sequence number, and `total_tasks` is adjusted upward to account for the sub-chunks. This lets the writer start writing (and updating `out:` progress) as soon as the first 16 MiB is decompressed, rather than waiting for the entire frame.

**Key details:**
- `n_chunks_est = ceil(decomp_size / 16 MiB)` — pre-calculated from the frame header
- `total_tasks` adjusted atomically before streaming begins; corrected after if actual chunk count differs
- Only triggers for frames > 64 MiB; normal multi-frame files (16 MiB frames) take the existing fast path
- FrameThrottle naturally releases permits per sub-chunk, providing backpressure
- Memory usage drops from full-frame to ~16 MiB working set per worker

---

## v0.12.23 — Progressive writeback fix + --sliding-window compression

**Symptom (decompression overwrite stall).** Decompressing low_compress.bin.zst (19.53 GiB output) with `-f` (overwrite) showed a 46-second stall after all decompression work completed, reported as "atomic rename: 46155 ms" in verbose output.

**Root cause.** Buffered `fwrite` of 19.53 GiB accumulated dirty pages in the page cache faster than the kernel's background writeback could drain them. When `rename()` was called on the `.gzstd.tmp` file, ext4's `data=ordered` journaling required flushing ALL dirty pages to disk before committing the metadata transaction — a synchronous 46s wait. Compression didn't suffer because: (1) it's CPU-intensive, giving writeback time to drain; (2) compressed output is smaller.

**Fix.** Added `sync_file_range(fd, offset, len, SYNC_FILE_RANGE_WRITE)` in `AsyncWritePool::worker_fn` after each batch of writes. This non-blocking call tells the kernel to start writing dirty pages to disk immediately rather than letting them accumulate. By the time `rename()` executes, most pages are already on disk.

**Result on Workstation** (low_compress.bin.zst → existing file with `-f`):
- Atomic rename: 46,155 ms → ~700 ms (**66× faster**)
- No regression to compression or fresh-file decompression paths

---

## v0.12.22 — `--sliding-window` single-frame compression mode

**Motivation.** For highly repetitive data (e.g., random word lists repeated across 125 GiB), gzstd's multi-frame architecture (8000 independent 16 MiB frames) achieved 0.29% ratio while `zstd` achieved 0.01% (31× better). The difference: zstd produces a single frame with a 2 MiB sliding window that maintains context across the entire file, while gzstd's frames each start with a cold window.

**Feature.** New `--sliding-window` flag delegates compression to zstd's built-in multi-threaded mode (`ZSTD_c_nbWorkers`), producing a single standard zstd frame. Trade-offs:
- Ratio matches `zstd` exactly (shared sliding window context)
- Output is a standard .zst file — `zstd -d` can decompress it
- Decompression is single-threaded (one frame = one unit of work for any decompressor)
- Implies `--cpu-only` (GPU/nvCOMP has no sliding window API)

**Validation:**
- `--sliding-window --gpu-only` and `--sliding-window --hybrid` rejected with clear error
- `--sliding-window -d` rejected (compression-only)
- `--sliding-window` without `--cpu-only` auto-enables it with a warning
- Round-trip verified; `zstd --list` confirms single frame; `zstd -d` interop confirmed

**GPU fallback for oversized frames.** When decompressing a single-frame file (from `zstd` or `--sliding-window`), the frame's decompressed size can be hundreds of GiB — far exceeding nvCOMP's 16 MiB per-slot VRAM allocation. gzstd now detects oversized frames during the pre-scan and automatically falls back to CPU-only decompression with a clear warning. `--gpu-only` is gracefully overridden rather than crashing.

**Progress bar fix (mmap compression).** The mmap zero-copy reader enqueued all tasks instantly (pointer arithmetic, no I/O), causing the `in:` progress to jump to 100% immediately. Fixed by deferring `read_bytes` updates to when workers actually pick up each task, so the progress bar reflects real processing throughput.

**Test coverage.** Full `./gzstd-test.sh` suite passes (200/200).

---

## v0.12.21 — mmap zero-copy compression input + benchmark accuracy fix

**Symptom (compression).** CPU-only compression of mixed.bin (19.5 GiB) took 9.9s vs zstd's 6.1s. Profiling showed the single-threaded `fread` producer was the bottleneck — 22 worker threads were starved, achieving only ~1.5 effective cores of utilization.

**Fix.** Memory-map input files for both CPU (`compress_cpu_mt`) and GPU (`compress_nvcomp`) compression paths. Workers read directly from the mapped pages via `view_ptr`/`view_len` on `Task`, eliminating the `fread` + `memcpy` bottleneck. Pipes and stdin fall back to the existing `fread` path. Key changes:

- Added `MmapRegion` RAII class (read-only mmap with `MADV_SEQUENTIAL`)
- Extended `Task` struct with `view_ptr`/`view_len` for borrowed (mmap) data vs owned `std::vector<char> data`, plus `ptr()`, `len()`, `release_input()` helpers
- Updated all consumer touch points: `t.data.data()` → `t.ptr()`, `t.data.size()` → `t.len()`, `std::vector<char>().swap(t.data)` → `t.release_input()`

**Result on Workstation** (24-core, mixed.bin CPU-only compress):
- Before: 9.9s (1.97 GiB/s)
- After: 3.1s (6.3 GiB/s) — **3.2× faster**, now 1.9× faster than zstd

**Symptom (benchmark).** `gzstd-benchmark.sh` reported compression at ~3.4s / 5.7 GiB/s, roughly 2× faster than manual `time` measurements (~7.2s). Results appeared suspiciously fast.

**Root cause.** `rm -f "$comp_out"` before each iteration deleted the output file. This forced the kernel to create a fresh file (fast new-block allocation) instead of exercising the atomic overwrite path (`write .gzstd.tmp` → `rename`), which contends with the old file's dirty-page writeback. The `sync` in `run_timed` was defeated because `rm` discards dirty pages before sync runs.

**Fix.** Removed `rm -f` calls before both compress and decompress iterations so iterations 2+ exercise the realistic overwrite path. Also changed hardcoded version string to generic "gzstd benchmark suite".

**Failed experiments (reverted, documented here for future reference):**
- **mmap output for decompression**: MAP_SHARED writes from multiple threads caused 9+ min of kernel sys time for 20 GiB due to page-fault contention. Buffered fwrite through the sequential writer thread is faster.
- **MADV_HUGEPAGE on write mmap**: THP defragmentation overhead made everything 2× worse.
- **32K write chunking**: Increased syscall count without improving I/O scheduling.
- **Removing atomic overwrite**: The overwrite penalty comes from kernel dirty-page throttling on gzstd's 16 MiB write chunks (one per decompressed frame), not from the .tmp + rename mechanism. zstd avoids this via streaming decompression with 32 KB writes. This is an architectural difference, not a quick fix.

**Decompression status.** On fresh files, gzstd matches zstd (6.5s vs 5.1s on mixed.bin). The overwrite penalty (2–3×) remains an open issue tied to frame-at-a-time vs streaming decompression architecture.

**Test coverage.** Full `./gzstd-test.sh` suite passes (193/193).

---

## v0.12.20 — Fix re_enqueue FIFO violation causing throttle deadlock on disk

**Symptom.** Hybrid decompression stalled at ~39.7% on medium_compress.bin when writing to a real file, but completed reliably to `/dev/null`. The v0.12.19 `may_take` fix resolved the GPU-waits-on-CPU deadlock, but a second deadlock remained on disk-backed output.

**Root cause.** `TaskQueue::re_enqueue()` used `push_back`, sending GPU-skipped trivial frames (low sequence numbers) to the *back* of the queue. CPU workers then processed higher-sequence frames first, consuming all `FrameThrottle` permits. The writer needed the low-sequence frames to release permits (sequential ordering), but those frames were stuck behind the high-sequence work. Classic circular wait: workers need permits to produce frames, writer needs low-seq frames to release permits, low-seq frames are queued behind work that needs permits.

This only manifested with real disk I/O because `/dev/null` releases permits instantly (no write latency), so the writer could always drain fast enough to recycle permits before exhaustion.

**Fix.** Changed `re_enqueue` from `push_back` to `push_front` with reverse iteration:
```cpp
for (auto it = batch.rbegin(); it != batch.rend(); ++it)
    q_.push_front(std::move(*it));
```
Reverse iteration preserves original sequence order at the front of the queue. This restores the FIFO invariant that `FrameThrottle` depends on for deadlock freedom: "the frame the writer needs next is always among the oldest in-flight frames."

**Result on Workstation** (24-core, 2× RTX 2080 Ti, medium_compress.bin.zst → real file):
- Before: stalled at ~39.7% (4/5 runs to disk, 0/5 to `/dev/null` after v0.12.19)
- After: 5/5 real-file completions at 2.96–3.15 GiB/s, 15/15 `/dev/null` at 6.68–7.54 GiB/s

**Test coverage.** Full `./gzstd-test.sh` suite passes (193/193).

---

## v0.12.19 — Fix hybrid compress/decompress deadlock

**Symptom.** Hybrid decompression hung on medium_compress.bin (~4 out of 5 runs). Diagnostic showed `cpu_taken=0, gpu_taken=0, queue_floor=29, gpus_waiting=0` repeating indefinitely — no worker was making progress.

**Root cause (primary).** The `may_take` predicate in both CPU compress and decompress workers called `sched->should_cpu_take()` unconditionally. `should_cpu_take()` returns false when `gpus_waiting_ > 0`. But there was no `done` bypass: once the producer finished (`queue.set_done()`), CPU workers still deferred to the GPU scheduler. If the GPU was stuck in `bp->acquire` (throttle exhaustion), a CUDA operation, or `pop_batch_greedy` while holding `gpus_waiting_=1`, CPU workers would never take work — permanent deadlock.

**Root cause (secondary).** GPU workers never deregistered their streams from `HybridSched` on exit, so `gpu_queue_floor_` persisted at 29 after GPU exit, blocking CPU workers when `depth <= floor` and `done=false`.

**Fix.**
1. **`may_take` predicate**: Added `if (qs.done) return true;` before the `should_cpu_take()` check in both compress and decompress CPU workers. Once the producer is done, CPU workers drain the queue regardless of GPU state. The redundant `&& !qs.done` guards on floor/cpu_queue_min checks were removed (now handled by the early return).
2. **GPU stream deregistration**: Added `HybridSched::unregister_gpu_stream()` — decrements `active_gpu_streams_`, recalculates `gpu_queue_floor_` to 0, and calls `notify_cpu_waiters()`. Both GPU compress and decompress workers call it on all exit paths.
3. **Defensive wake**: After GPU worker threads are joined, `notify_cpu_waiters()` is called as a safety net.

---

## v0.12.18 — Default to buffered I/O; O_DIRECT now opt-in via --direct

**Motivation.** Despite fixing CPU-side contention in v0.12.17 (thundering herd via CV mismatch), wall-time variance on disk-backed runs remained severe: 2–10× on the same file between consecutive runs. Reference tool zstd showed 1.2× variance on the same workload. The difference: zstd uses buffered `fwrite` (OS page cache absorbs write latency); gzstd used `O_DIRECT` by default, bypassing the cache and exposing every write to NVMe-internal GC, ext4 journal commits, and writeback contention from prior runs' dirty pages.

**Root cause.** `O_DIRECT` writes are synchronous to the device: if the NVMe controller is busy (garbage collection, NAND erase, journal commit from a prior buffered write), each 4 KiB–4 MiB `write()` stalls until the device is ready. With buffered I/O, the kernel coalesces writes in the page cache and flushes to the device at its own pace — the application sees consistent ~1.5–1.9 GiB/s regardless of device state.

**Fix.** O_DIRECT is now off by default. `--direct` opts in; `--no-direct` is accepted for explicitness (already the default). Both the explicit-output-file path and the stdout-redirect-to-file path are gated on `opt.direct_io`.

**Result on Workstation** (24-core, ext4/NVMe, `mixed.bin` 19.5 GiB, 5-run median):

| Mode | Before (O_DIRECT) | After (buffered) |
|------|-------------------|-------------------|
| CPU compress | 5.7–48.2 s | 5.81–6.67 s (1.15×) |
| CPU decompress | 5.4–31.1 s | 5.37–5.83 s (1.09×) |
| Hybrid compress | 6.6–105.8 s | 6.56–6.93 s (1.06×) |
| Hybrid decompress | stalled at 55% | 6.44–6.75 s (1.05×) |

The "stalled" hybrid decompress was caused by O_DIRECT writes contending with NVMe GC, stalling the aio thread, exhausting throttle permits, and blocking all workers.

**v0.12.16 push_to_slot change reverted.** The earlier "only notify writer when seq == next_to_write" optimization was reverted — not safe in hybrid mode where GPU batch-completion and CPU per-frame notifications interact. Per-CPU-push `notify_one` is cheap (single writer waiter, not a herd).

**Test coverage.** Full `./gzstd-test.sh` suite passes (193/193).

---

## v0.12.17 — Kill the CPU-side thundering herd (wait_for_work / notify fixes)

**Motivation.** v0.12.10–0.12.15 fixed several pipeline-depth and throttle issues but left a ~1.7× run-to-run variance on CPU-only decompress at high thread counts (22 workers on Workstation): fast runs ~4.0 s to `/dev/null`, slow runs ~7.0 s on the same cached input. Reducing `-T` from 22 to 4 collapsed both the variance and the absolute time (3.1–3.6 s). Variance scaled with worker count — a contention signature, not a hardware one.

**Root cause 1 — `TaskQueue::wait_for_work()` was waiting on the wrong CV.** `TaskQueue` exposes two condition variables by design: `cv_` for GPU batch waiters (woken by `notify_all` because batch predicates need every waiter to re-check) and `cpu_cv_` for CPU workers (woken by `notify_one` in the push path). Non-hybrid CPU workers called `wait_for_work()`, which — incorrectly — waited on `cv_`. So `push()`'s targeted `cpu_cv_.notify_one()` hit nothing, and the `cv_.notify_all()` it fires for GPU waiters woke **every** CPU worker in the pool on every frame push. 22 threads × ~8000 frames = ~176k spurious wakeups per run, all contending on the same queue mutex as they raced to pop, 21 of them losing each race and going back to sleep.

Fix: `wait_for_work()` and `pop_one()` now wait on `cpu_cv_`. One CPU worker wakes per pushed frame, matching the actual work. `set_done()` still notifies both CVs so shutdown wakes everyone.

Result on `-T 22 --cpu-only -c mixed.bin.zst > /dev/null` (5-run sample): 4.1–7.0 s → 3.25–3.84 s. Variance collapsed from 1.7× to 1.18×, and the *floor* improved — i.e. the "good" runs got faster too, confirming the herd was costing work even in the best case.

**Root cause 2 — `FrameThrottle::release()` used `notify_all`.** With 22+ workers saturated against the throttle, `notify_all` on every single-permit release fan-out-woke the whole pool per aio write. One woke-up worker won the mutex, took the single new permit, 21 went back to sleep after redundantly contending. Replaced with a `notify_one` loop over `n`: wake exactly the number of waiters that the capacity change can satisfy. (v0.12.16 change kept here for clarity.)

**Root cause 3 — `ResultStore::push_to_slot()` CPU path notified on every push.** CPU workers finish out-of-order and push to the shared results map; the writer only cares when its `next_to_write` seq arrives. Waking the writer for every out-of-order push churned it through wake/recheck/sleep cycles. Now only notifies when `seq == next_to_write`; out-of-order frames get drained on the next natural writer cycle. (v0.12.16 change kept here for clarity.)

**Residual variance.** CPU-side variance is resolved (≤1.18×). Writes to `/tmp` (ext4) still show a bimodal pattern (11.5 s fast / ~28 s slow) that does *not* track worker count — this is OS page-cache flush / NVMe writeback behavior between consecutive large writes, not algorithmic. Left for a future pass; out of scope here.

**Test coverage.** Full `./gzstd-test.sh` suite passes (193/193).

---

## v0.12.15 — Throttle diagnostics, tunables, and coverage

**Deadlock guardrail (late add).** Initial smoke-testing of the new knobs caught a real bug: `--throttle-frames=1` with a GPU path active would hang indefinitely. GPU workers greedy-acquire up to `gpu_batch_cap` permits per batch in one call — with a 1-permit budget, the first stream takes the single permit and blocks waiting for the rest, which only the writer can release, which it can never do because no frame has been pushed yet. Classic producer-consumer circular wait.

`compute_throttle_budget` now takes a `gpu_batch_floor` argument (= `devices × streams × gpu_batch_cap`) and clamps user-provided `--throttle-frames` up to it when GPU is active, with a warning at `-q` or above. The default-formula path sets source to `user+gpu-floor` in the startup log when clamping fires. CPU-only paths pass `gpu_batch_floor=0` and are unaffected.

Regression test added to section 36: hybrid `--throttle-frames=1` round-trip, gated on `has_gpu`, must warn-and-complete in under 30 s.



Follow-up to v0.12.14: instrument the throttle so its behaviour is observable, expose the knobs needed to sweep it, and lock the new surface in with tests.

**FrameThrottle instrumentation.** New atomic counters on every throttle: `block_count` (how many `acquire()` calls actually had to wait), `block_nanos` (cumulative blocked time), `peak_in_flight` (high-water mark vs budget). `acquire()` times the wait from the first `cv_.wait` until the permit is taken; exits without a wait skip the timing hot path entirely. A `Stats` snapshot struct + `stats()` method expose them to the end-of-run logger.

**New CLI knobs.**
- `--throttle-factor=N` — override the default `SLACK_FACTOR=4` slack multiplier. Lets you sweep pipeline tightness without changing source.
- `--throttle-frames=N` — explicit in-flight frame cap; bypasses the parallelism formula entirely. Useful for repro cases and deadlock stress tests.

Both validated (must be >= 1) and reflected in help output. The startup summary labels where the budget came from: `source=user` when `--throttle-frames` is set, `source=pipeline` when the parallelism formula wins, `source=ram` when the RAM cap binds, `source=floor` when the 32-frame minimum binds.

**Verbose output.**
- `-v` (V_VERBOSE): one-line startup summary — `throttle: N frames (X GiB in-flight max, source=..., parallelism=..., slack=...)`.
- `-vv` (V_DEBUG): adds `throttle detail: pipeline_cap=..., ram_cap=..., floor=..., avail_ram=...` at startup and an end-of-run `throttle stats [phase]: peak=P/M (S%), block_count=N, block_time=Xms` line tagged by path (`compress-cpu`, `decompress-hybrid`, `compress-gpu`, etc.).

Saturation (`peak/max`) plus `block_count` tells you at a glance whether the throttle is the bottleneck: low saturation means something upstream (reader, GPU, CPU) is the limiter; high saturation with low `block_time` means workers fill the pipeline but never stall; high saturation with significant `block_time` means the writer is the limiter and backpressure is engaged as designed.

**Test coverage** (section 36 of `gzstd-test.sh`, 8 new tests → 192 total):
- Round-trips at `--throttle-frames=1`, `=32`, and `--throttle-factor=1`, `=16` (deadlock guards at both extremes).
- `-v` output contains the throttle startup line.
- `-vv` output contains the end-of-run stats line.
- `--throttle-frames=0` and `--throttle-factor=0` are rejected with `EXIT_USAGE` (2).

**Benchmark sweep.** `gzstd-benchmark.sh` gains `--sweep-throttle`, which sweeps `--throttle-factor` over `{1,2,4,8,16}` (or `{1,4,16}` with `--quick`) across all enabled paths (cpu-only, hybrid, gpu-only). Rolled into `--sweep-all`.

---

## v0.12.14 — Pipeline-depth throttle budget (principled scaling)

**Motivation:** v0.12.13 fixed the lost-backpressure bug by capping the throttle budget at a hard 8 GiB. That number worked on the two test systems but was arbitrary: too restrictive for a 256-core / 8×H100 server whose pipeline can legitimately hold hundreds of GiB in flight, too generous for a 16 GiB VM where 8 GiB is half of physical RAM. The budget needs to track the machine, not a magic constant.

**Fix:** Replace the fixed byte cap with a formula rooted in observable hardware parallelism:

```cpp
pipeline_frames = (cpu_threads + gpu_count * streams * batch_cap) * SLACK_FACTOR
ram_cap_frames  = (avail_ram / 2) / frame_bytes
frames = max(min(pipeline_frames, ram_cap_frames), 32)
```

`SLACK_FACTOR = 4` gives each active producer ~4 frames of headroom — enough to ride out writer jitter, not so much that we queue hundreds of frames per producer with no throughput payoff. The RAM cap stays as a safety net; the 32-frame floor guards against pathological low-parallelism or huge-chunk configs.

Expected budgets:

| System                         | Parallelism             | Frames   | In-flight |
|--------------------------------|-------------------------|----------|-----------|
| Laptop (8 CPU, no GPU)         | 8                       | 32 (floor) | 512 MiB  |
| 16 GiB VM (4 CPU, no GPU)      | 4                       | 32 (floor) | 512 MiB  |
| Workstation (24 CPU, 2×1×16 GPU)  | 24 + 32 = 56            | 224      | ~3.5 GiB  |
| Server (256 CPU, 8×2×64 GPU)    | 256 + 1024 = 1280       | 5120     | ~80 GiB   |

On Workstation this is ~2× tighter than the old 8 GiB cap but well above the ~320 MiB the writer actually drains before the next producer wakeup, so no throughput regression is expected. On Server it unlocks the pipeline the hardware can actually sustain.

The `-vvv` throttle debug line now shows all inputs: `parallelism=`, `pipeline=` (pre-clamp), `ram_cap=`, plus the chosen frame count and in-flight byte equivalent.

---

## v0.12.13 — Throttle budget byte cap (restore writer backpressure)

**Bug:** On Workstation (256 GiB RAM) decompression appeared to lose all writer backpressure. Reader, GPUs, and CPUs finished in seconds; the writer then ground through a massive in-RAM backlog at ~88 MiB/s with no throttling of producers. The throttle budget formula `avail_ram / (2 × frame_bytes)` gave ~7,800 frames on a 246 GiB-available box — 123 GiB of permitted in-flight data. Files under that size (mixed.bin.zst at ~20 GiB decompressed = 1,220 frames) fit entirely within the budget, so workers never blocked, decompressed everything immediately, and the writer drained alone.

**Fix:** Cap the budget at an absolute byte ceiling in addition to the RAM-relative calculation:

```cpp
budget_bytes = min(avail_ram / 2, THROTTLE_MAX_BYTES);   // 8 GiB cap
frames = budget_bytes / frame_bytes;
frames = max(frames, 32);                                // min pipeline depth
```

On Workstation: budget = 512 frames (8 GiB in-flight) instead of ~7,800. Workers fill the pipeline, then block on `acquire(1)` — writer releases permits as frames are written, producers resume. Lockstep backpressure restored.

Fast-I/O systems are unaffected: when the writer can release permits faster than workers acquire them, nobody blocks. Only slow-I/O systems (relative to producer throughput) feel the cap — which is exactly where it's needed.

Prior floor of 1024 was also wrong on low-RAM systems (forced a minimum 16 GiB in-flight on 24 GiB boxes, would have caused swap once the reader's buffer was resident). Floor is now 32 frames.

**Debug log change:** `-vvv` throttle message now reports both the frame count and the byte in-flight max, and the available-RAM figure:
```
throttle: 512 frame budget (8.00 GiB in-flight max, 246.03 GiB avail RAM)
```

---

## v0.12.12 — EMA-scaled hybrid queue floor + tuning knobs

**Bug:** The fixed queue floor introduced in v0.12.9 — `active_gpu_streams × gpu_batch_size` — assumed GPU was strictly faster than CPU per frame. For compression that holds (GPU batches pay off), but for decompression nvCOMP throughput ≈ CPU zstd throughput, so reserving frames for the GPU just idles CPUs. Workstation benchmarks (v0.12.11) showed hybrid decompression 2–8% slower than the best pure path on every file, and 18% slower than either pure config on `zeros.bin` (3.675 vs 4.482 GiB/s).

**Fix:** `HybridSched` now scales the nominal floor by observed GPU advantage. Each tick (every ≥0.5s) feeds per-side EMA throughput (`cpu_rate_ema_`, `gpu_rate_ema_`, α=0.3) from the already-tracked `cpu_bytes_` / `gpu_bytes_` counters. The floor factor is `clamp((gpu_per_stream − cpu_per_thread) / gpu_per_stream, 0, 1)`:
- Compression: GPU ≫ CPU → factor ≈ 1.0 (nominal reservation, preserves v0.12.9 gains).
- Decompression: GPU ≈ CPU → factor → 0 (CPUs compete freely).

During warm-up (<2 EMA samples on either side), factor defaults to 1.0, matching v0.12.9. Convergence is typical after ~2s (≈6 ticks at α=0.3). The constructor now consumes the `cpu_threads` argument it previously ignored.

**New CLI flags** (gated; defaults preserve v0.12.12 AUTO behaviour):
- `--hybrid-floor=auto|nominal|off`
  - `auto` (default): EMA-scaled as above.
  - `nominal`: v0.12.9 behaviour (`streams × batch`).
  - `off`: no reservation — CPUs compete freely, relying on the `gpus_waiting_` semaphore for GPU priority.
- `--hybrid-floor-factor=X` — manual override in `[0.0, 1.0]`; bypasses mode selection.

`-vvv` tick output adds `floor_factor=` alongside `queue_floor=` for diagnosis.

---

## v0.12.11 — Progress bar: freeze input rate after read completes

**Bug:** After the reader finished (in:100.0%), the displayed input rate (GiB/s) kept declining because it was computed as `read_bytes / total_elapsed_time` — a cumulative average where the denominator grows but the numerator is fixed.

**Fix:** The progress loop snapshots the elapsed time the first time it sees `read_bytes >= total_in` and reuses that frozen duration for subsequent input rate calculations. Added `read_elapsed_ms` (mutable atomic) to `Meter`. The output rate continues using wall-clock time since writes are still in progress.

---

## v0.12.10 — Progress bar: ratio-estimated output percentage

**Bug:** The `out:` progress percentage was misleading during the read phase for both compression and decompression. During decompression, `out:` showed ~56% at only 10% input read because `wrote_bytes / total_out` used a partial (still-growing) denominator — workers complete frames faster than the reader parses new ones, so the ratio is inflated and then drops as the denominator catches up. Compression had the same class of issue (showed `---` until the reader finished).

**Fix:** While the reader is still running (`total_frames` not yet set), the progress bar estimates total output from the current input/output ratio: `estimated_total = total_in × (total_out_so_far / read_bytes_so_far)`. This works for both compression (ratio < 1) and decompression (ratio > 1), converges as more data is read, and only increases monotonically for files with uniform compressibility. Once the reader finishes: decompression switches to exact `wrote_bytes / total_out` (byte-level); compression uses `tasks_done / total_frames` (frame-level) then `wrote_bytes / total_out` during AIO drain. Added `total_out_final` flag to `Meter` to distinguish finalized vs partial `total_out`.

---

## v0.12.9 — GPU queue depth reservation (hybrid scheduler)

**Bug:** On Server, hybrid mode was ~10% slower than `--gpu-only`. Scheduler stats showed CPUs took 18,344 tasks vs GPUs 9,341 — despite CPUs being ~6× slower per task (0.21 vs 1.19 GiB/s). The `should_cpu_take()` gate only blocked CPUs when `gpus_waiting > 0`, but GPUs cycle through wants→got in microseconds. During the much longer GPU processing phase (milliseconds), `gpus_waiting == 0` and all 96 CPUs flooded the queue, leaving it empty when GPUs came back for their next batch.

**Fix:** `HybridSched` now tracks total active GPU streams (`register_gpu_stream()`) and current batch size (`set_gpu_batch_size()`). A dynamic queue floor = `active_streams × batch_size` reserves enough tasks for every GPU stream to fill one full batch. The `may_take` lambda in both compress and decompress CPU workers checks `qs.depth <= floor` and yields if so. The floor updates automatically as the auto-tuner adjusts batch size. When the queue is draining (`qs.done`), the floor is bypassed so CPUs can process remaining tasks. The floor is logged in `-vvv` tick output for diagnosis.

---

## v0.12.8 — FrameThrottle ordering deadlock + memory-based throttle + thundering herd

**Three fixes targeting hybrid mode reliability and performance:**

### 1. FrameThrottle ordering deadlock (acquire-before-pop)

**Bug:** Hybrid compress stalled with `cpu_rate=0 gpu_rate=0` — everything frozen. The v0.12.7 fix (acquire after pop) prevented hoarding but introduced a new circular deadlock: a worker pops the frame the writer needs, then blocks on `acquire()` because all permits are consumed by ResultStore. Writer waits for the frame, nobody releases permits.

**Fix:** CPU workers now use a three-step pattern: (1) wait for predicate with no permit held (no hoarding), (2) acquire a permit (may block, but no task is held, so writer can progress), (3) non-blocking `try_pop_one_cpu` (if pop fails because state changed, release permit and retry). Workers only hold both a task AND a permit simultaneously — never one without the other. Added `try_pop_one_cpu()` to `TaskQueue` for the non-blocking pop step. Rescue workers use simple acquire-before-pop (hoarding is acceptable — rescue queue only fills on GPU failure).

### 2. Memory-based throttle sizing

**Bug:** The throttle was sized from a worker-count formula (`max(512, 2 × (cpu + gpu×batch + rescue))`). On a 256-core + 8×GPU system the ceiling of 512 was routinely exhausted by ResultStore write lag, triggering the ordering deadlock above.

**Fix:** Throttle budget is now `(available_ram / 2) / chunk_size` with a floor of 1024. A 256 GiB system gets ~8192 permits; a 16 GiB system gets ~512; a 2 TiB cluster gets ~65536. Scales naturally from embedded to thousand-GPU without any per-configuration tuning. Uses `get_available_ram_bytes()` (/proc/meminfo on Linux, 8 GiB fallback elsewhere). Applied to all four throttle sites (GPU compress, GPU decompress, CPU compress, CPU decompress).

### 3. Thundering herd reduction

**Bug:** `gpu_got_data()` called `notify_cpu_waiters()` (notify_all) every time the last waiting GPU got data. With 8 GPUs × 24 streams, this woke all 96 CPU threads — each grabbed the queue mutex, found the predicate false, and went back to sleep. The resulting `futex` churn increased `sys` time by ~24%, making hybrid 10% slower than gpu-only.

**Fix:** `gpu_got_data()` now calls `notify_cpu_one()` (new method). GPU workers call `notify_cpu_waiters()` (all) on exit so CPU stragglers always get woken for the drain path — this was the specific edge case that killed the previous `notify_one` attempt in v0.11.41.

---

## v0.12.7 — FrameThrottle deadlock + VRAM-starved stream auto-decrement

**Two separate deadlocks fixed:**

### 1. Stream count auto-decrement on VRAM starvation

**Bug:** `./build/gzstd -f -k --gpu-only --gpu-batch=1 --gpu-streams=64 …` hung with:
```
[GPU1] insufficient VRAM for even batch=1  skipping device
[GPU0] insufficient VRAM for even batch=1  skipping device
in:100.0% 19.53 GiB ... out:0.0% 0.00 B
```

**Root cause:** When per_stream_batch reached 1 and the allocator still failed, the GPU worker skipped the *entire device* — even if earlier streams in the loop had already initialized successfully. With `--gpu-only` and every GPU skipped, the producer kept reading into a queue no one would ever consume.

**Fix (compress + decompress):**
- If stream `s` can't fit at batch=1 but streams `[0..s)` initialized fine, cleanly destroy the failed stream, `ctxs.resize(s)`, and continue running with `s` streams instead of the requested count. Emits `WARNING: [GPU#] VRAM insufficient for N streams at batch=1; auto-reducing to M stream(s)` at `V_DEFAULT` (suppressed under `-q`).
- Only skip the GPU entirely when `s == 0` (zero usable streams).
- Added `std::atomic<int> gpu_init_failures` shared with the producer. In `--gpu-only`, the compress producer now bails out of the read loop once `gpu_init_failures == gpu_count`, instead of buffering the entire input into RAM before the post-join `die(EXIT_GPU_FAIL)` fires.

### 2. FrameThrottle permit starvation on multi-GPU with large batches

**Bug:** On 8× H100 with `--gpu-batch=64 --gpu-streams=4`, hybrid compress hung indefinitely. `hybrid: tick` showed `cpu_rate=0 gpu_rate=0 gpus_waiting=0 cpu_taken=0 gpu_taken=0` — everything frozen before any task moved.

**Root cause:** `FrameThrottle` had a hard default of 512 permits, and idle worker threads were hoarding permits:
- `cpu_worker`, `cpu_worker_rescue`, and `cpu_decomp_worker` called `bp->acquire(1)` at the top of their loop — *before* popping a task. An idle rescue thread blocked on an empty rescue queue held 1 permit forever.
- Rescue pool = `hw_concurrency/2` (128 on a 256-core box) → 128 permits locked idle.
- Plus CPU pool (96) + GPU workers needing `stream_count × per_stream_cap × gpu_count` = 4 × 16 × 8 = 512 → total demand 736 against 512 supply. Classic permit-starvation deadlock, made worse by `acquire()` hoarding partial grabs while waiting for the rest.

**Fix:**
1. Moved `bp->acquire(1)` to *after* a successful pop in all three CPU-path workers. Idle workers no longer hold permits. Matching `release(1)` calls on exit paths removed (no permit to release).
2. `compress_nvcomp` and `decompress_nvcomp` now size the throttle to `max(512, 2 × (cpu_threads + gpu_batch_cap × gpu_count + rescue_threads))`. Small runs unchanged; large fleets get proportional headroom with a 2× pipeline factor.

---

## v0.12.6 — Multi-GPU parallelism fix + verbose output improvements

- **Multi-GPU compress starvation fix**: `pop_batch_greedy` was called with `min_n = max_n`, causing all GPU workers to block until a full batch was available. Only one GPU could ever win the batch race; the other remained idle. Fixed by using `min_n=1` (same fix already applied to decompress), allowing GPUs to interleave and take partial batches
- **Decompress GPU verbose output**: `take` and `done` log lines now match compress format — `seq=[lo..hi]` added to take, and done now shows `in=`, `h2d=`, `comp=`, `d2h=`, `tot=`, `thr=` breakdown with CPU-side timing
- **CPU worker take log**: added `[CPU/T#] take seq=N in=X` log at `-vv` so early frames grabbed during GPU init are visible (previously only completion was logged)
- **Number colorizer**: changed `!isalpha` to `!isalnum` predecessor check — digits embedded in identifiers like `h2d` and `d2h` are no longer colorized

---

## v0.12.5 — Progress bar UI polish + decompression % fix

- **Only `XX.X%` values are bold/bright**; labels, sizes, and rates use dim cyan/green
- **Dark grey `|` separator** (`\033[90m`) between in and out sections
- **Completion summary colorized**: `OK` bold green, input size cyan, output size/rate green, ratio bold
- **Decompression `out%` fix**: was stuck while size display changed because frame-completion counter (`tasks_done/total_frames`) updated before AIO finished writing. Now uses `wrote_bytes/total_out` (byte-level, matches the size display) throughout decompression
- **Compression early `out%` fix**: spurious 99.9% at startup caused by `wrote_bytes/total_out` where `total_out` was only a partial running sum. Now shows `---` until frame tracking (`total_frames`) is established
- **Benchmark script**: falls back to `in:XX.X%` when `out%` is not yet available; status suffix shortened (`gzstd ` dropped, `ETA ` → `~`) to stay under 80 columns

---

## v0.12.4 — Colorized progress bar

Progress bar now uses ANSI colors for readability (already required ANSI for cursor control):

- **`in:` label** — cyan; **`in%` value** — bold bright cyan
- **`out:` label** — green; **`out%` value** — bold bright green (bold yellow when unknown)
- **rates and separator** — dim

Test mode (`--test`) colorizes `in%` and `verified:` bytes consistently.

---

## v0.12.3 — Dual-percentage progress bar (in% and out% shown independently)

The v0.12.2 frame-based progress caused a visible jump: read-based % reached 100% quickly, then switched to frame-based % near 0% and climbed again. Confusing.

**Fix:** Show two independent percentages side by side — no single number ever jumps backwards.

```
in:34.2% 2.10 GiB  out:12.7% 780 MiB | in:1.20 GiB/s out:450 MiB/s
```

- **in%** — `read_bytes / total_input`: how much input has been consumed; climbs fast on fast NVMe
- **out%** — `tasks_done / total_frames` while compressing/decompressing, then switches to `wrote_bytes / total_out` during the AIO flush phase; reflects actual CPU/GPU work
- Shows `---` when a metric is unknown (pipe input, single-thread stream path)
- Test mode updated to match new format

---

## v0.12.2 — Progress bar tracks frame completion instead of reader

Previously the progress percentage was based on `read_bytes / total_input_size`. The reader finishes quickly (it's just I/O), so at ultra compression levels with large chunks the bar jumps to 100% while workers are still compressing — useless as a progress indicator.

**Fix:** Added `total_frames` to `Meter` (set by the producer after enqueuing all work) and moved `tasks_done` tracking to `writer_thread` (incremented per frame batch handed off for writing). Progress is now `tasks_done / total_frames`, which reflects actual compression work.

- Falls back to read-based % before `total_frames` is known (single-thread stream path, or brief start-of-run window)
- Write drain phase (all frames done, AIO still flushing): shows `[X.X%] writing: A / B @ C/s`
- Also fixed: removed premature `[done]` flash from inside the progress loop (race where `wrote_bytes` transiently caught up to the incrementally-accumulated `total_out`)

---

## v0.12.1 — Ultra compression window fix

### Ultra levels (--ultra -20/-21/-22) now set ZSTD_c_windowLog — POSITIVE (correctness fix)

Previously, `--ultra` enabled level 20–22 but never set `ZSTD_c_windowLog`, causing zstd to silently clamp the window to its default (~8 MiB). The result: ultra levels incurred the CPU cost of the extended search strategy with none of the compression benefit.

**Root cause:** `compress_one_cpu_frame()` and `compress_cpu_stream()` only called `ZSTD_CCtx_setParameter(ZSTD_c_compressionLevel)`. Without an explicit `ZSTD_c_windowLog`, the library ignores the intended 32–128 MiB window.

**Fix:**
- Added `ultra_window_log()`, `ultra_min_chunk_mib()`, and `apply_ultra_cctx()` helpers
- `compress_one_cpu_frame()` now takes a `bool ultra` parameter and calls `apply_ultra_cctx()` after setting the level
- `compress_cpu_stream()` sets `ZSTD_c_windowLog` on its direct `CCtx` and logs it at `-v`
- All three compress paths (stream, MT, nvCOMP) auto-increase chunk size to match window size (32/64/128 MiB for levels 20/21/22) when `--chunk-size` was not explicitly set; warns if user-specified chunk is too small
- `check_ram_budget()` now accounts for ~8× window size per thread for CCtx hash/chain tables (~256 MiB/thread at -20, ~512 MiB at -21, ~1 GiB at -22); auto-reduces thread count rather than OOMing

**Window sizes:**
- `-20 --ultra`: windowLog=25 (32 MiB window, min chunk 32 MiB)
- `-21 --ultra`: windowLog=26 (64 MiB window, min chunk 64 MiB)
- `-22 --ultra`: windowLog=27 (128 MiB window, min chunk 128 MiB)

**Result:** `gzstd --ultra -22` now produces compression ratios comparable to `zstd --ultra -22 -T0` and runs at similar speed (both doing the intended work). Output is fully interoperable with `zstd -d`.

---

## v0.12.0 — FrameThrottle (counting semaphore replaces byte-based backpressure)

### FrameThrottle Refactor — POSITIVE (simplification)
Replaced `WriterBackpressure` (byte-based high/low water marks + `writer_stalled_` escape hatch) with `FrameThrottle`, a counting semaphore that bounds the number of in-flight frames (popped from queue but not yet written to disk).

**How it works:**
- Workers call `acquire(N)` before popping (1 for CPU, `pop_n` for GPU batches)
- Writer calls `release(1)` per frame after physical disk write (via AsyncWritePool)
- GPU batches release excess permits if fewer frames are returned than requested
- Default: 512 permits (max in-flight frames)

**What was removed (-57 net lines):**
- `mark_produced()` — 6 call sites across CPU/GPU compress/decompress workers
- `mark_written()` — byte-level tracking in AsyncWritePool
- `writer_stalled_` flag + `set_writer_stalled()` — the deadlock escape hatch
- High/low water mark hysteresis (4 GiB / 2 GiB byte thresholds)
- `produced_` / `written_` atomic counters

**Why it's deadlock-free by construction:** The task queue is FIFO. If all 512 permits are consumed, the frame the writer needs (the oldest) was the first one popped and is guaranteed to be in-flight. The writer never waits for a frame that hasn't been popped yet while all permits are consumed. No `writer_stalled_` escape hatch needed.

**Why the old design was fragile:** `WriterBackpressure` counted total `produced - written` bytes, which included out-of-order frames sitting in ResultStore that the writer couldn't drain yet. This inflated the apparent backlog, triggering backpressure even when the writer had capacity. The resulting deadlock (all workers blocked on backpressure while the writer waited for frame N still in the queue) required progressively more complex fixes: first a 100ms timeout (v0.11.43), then the `writer_stalled_` signal (v0.11.44). The counting semaphore eliminates the root cause.

### v0.11.38–v0.11.44 (intermediate fixes, subsumed by v0.12.0)
- **v0.11.38:** Fixed backpressure disabled prematurely — `set_done()` moved after `pool.join()` in all 4 teardown paths
- **v0.11.39:** Added `fallocate()` preallocation for all write paths (compress + decompress, all modes). Avoids per-write extent allocation on NVMe.
- **v0.11.40:** Fixed hybrid decompress deadlock at 55.5% — `gpu_wants_data()` called before backpressure check blocked GPUs and CPUs simultaneously. Fixed by swapping order.
- **v0.11.41:** Fixed thundering herd — `gpu_got_data()` only notifies CPUs when last GPU is satisfied (`gpus_waiting_` drops from 1→0), not on every GPU completion.
- **v0.11.42:** Fixed CPU hang at end — reverted `notify_one` back to `notify_all` for `set_done()` path. Removed `-D` suffix from CPU decompress labels.
- **v0.11.43:** Timeout-based fix for out-of-order ResultStore deadlock (`cv_.wait_for(100ms)`). Replaced by proper solution in v0.11.44.
- **v0.11.44:** Replaced timeout with `writer_stalled_` signal approach. Subsumed by FrameThrottle in v0.12.0.

---

## Write Path Optimizations

### O_DIRECT Writer (v0.9.71)  POSITIVE
Bypasses page cache for sequential writes. Uses 16 MiB aligned buffer, flushes in aligned chunks.
- **Server:** Writer I/O improved 1.1 → 2.72 GiB/s on 432 GiB file
- **Why it works:** Avoids double-buffering through page cache for large sequential writes
- **Caveat:** Unaligned tail requires dropping O_DIRECT via fcntl for final write

### pwrite for Out-of-Order Decompress (v0.9.72)  NEGATIVE (reverted)
Tried using pwrite() to write decompressed frames directly to their final offset without waiting for in-order delivery.
- **Server:** 0.93 GiB/s (worse than sequential 2.72 GiB/s)
- **Why it failed:** 27k individual O_DIRECT pwrite calls = massive kernel DMA setup overhead. sys time: 12m45s.
- **Lesson:** O_DIRECT pwrite per-frame is catastrophically expensive. Sequential batch drain is better.

### Async Double-Buffered Write Pool (v0.9.73)  POSITIVE
Background write thread with one pending slot. Writer collects batch → submits to pool (non-blocking) → collects next batch while pool writes previous.
- **Server:** Improved overlap between GPU D2H and disk writes
- **Why it works:** Writer thread doesn't block on disk I/O; can collect next batch while previous is being written

### Sparse File Support (v0.9.73)  POSITIVE (for zero-heavy data)
Scans 4K blocks for zeros, lseek past them instead of writing. Integrated with both O_DIRECT (DirectWriter::seek_forward) and fwrite paths.
- **Server:** zeros.bin decompress: sparse=5.2s vs no-sparse=6.9s (~25% faster)
- **Why it works:** Avoids physical writes for zero-filled regions
- **Caveat:** O_DIRECT seek_forward must flush internal buffer before seeking. Added --[no-]sparse flag matching zstd syntax.

### io_uring Writer  NOT YET TRIED
Proposed: Replace O_DIRECT write() with io_uring for less syscall overhead per write.
- **Expected:** 10-20% improvement on NVMe drives where per-syscall overhead is significant
- **Rationale:** NVMe drives have deep internal queues; io_uring can submit multiple writes without syscalls

### mmap + memcpy Writer  NOT YET TRIED
Proposed: mmap output file at target size, memcpy frames directly. Kernel handles writeback.
- **Expected:** Good for sparse data (unmapped pages stay as holes), possibly worse for dense data
- **Risk:** mmap as INPUT was already tried and was negative (v0.9.53-54)

### Multiple pwrite Threads  NOT YET TRIED
Proposed: Open output file multiple times, pwrite from multiple threads at known offsets.
- **Expected:** Could double NVMe throughput by increasing queue depth
- **Risk:** O_DIRECT pwrite per-frame was catastrophic (v0.9.72); would need large contiguous writes

### Page-Cache Path for Trivial Data  NOT YET TRIED
Proposed: When >90% of blocks are zero, drop O_DIRECT and use fwrite + ftruncate. This is what zstd does  the page cache handles sparse much more efficiently.
- **Expected:** Match zstd's 2-3s on zeros.bin (currently 4-5s)
- **Rationale:** zstd achieves 0.3s sys time on zeros vs our 5-8s with O_DIRECT sparse

---

## Read Path Optimizations

### mmap Input (v0.9.53-54)  NEGATIVE (reverted)
Replaced fread with mmap for zero-copy reading.
- **Why it failed:** mmap with t.data.assign() still copies from mapped pages (not zero-copy). Worse than fread for sequential I/O due to page fault overhead and TLB pressure.
- **Lesson:** mmap only wins with true zero-copy (string_view/span) or random access patterns. Sequential fread is hard to beat.

### Offset-Based Buffer (v0.9.50)  NEGLIGIBLE (kept)
Replaced buf.erase(0,N) O(n) memmove with offset cursor.
- Correct optimization, prevents pathological quadratic behavior, but invisible at 128-frame scale.

---

## GPU Memory & Transfer Optimizations

### Pinned (Page-Locked) Memory for GPU Decompress (v0.9.53)  NEGATIVE (catastrophic, reverted)
cudaHostAlloc for H2D/D2H staging buffers to enable true async DMA.
- **Server:** GPU decompress nearly doubled: 13.4s → 25.6s
- **Why it failed:** Massive pinned allocations (512 MiB per stream) starved system memory, caused page faults in other threads, and the copy-to-pinned + DMA was slower than direct pageable transfer for our access pattern.
- **Lesson:** Pinned memory requires small rotating pools, not batch-sized allocations. The extra memcpy to/from pinned staging negated any DMA benefit.

### Frame-Level Pinned Buffer Pool  NOT YET TRIED (proposed v0.9.55)
Small rotating pool of frame-sized pinned buffers (2-4 × 16 MiB) shared across streams. True async overlap.
- **Key difference from failed #7:** Small pool vs massive per-stream allocation. Would enable cudaMemcpyAsync to actually overlap with kernel.

### Pre-Allocated GPU Decompress Buffers (v0.9.51)  NEGLIGIBLE (kept)
ensure_buffers() allocates once, reuses across batches. Saves ~150-300ms of cudaMalloc/cudaFree per file.
- Invisible at 8 GiB scale but correct for repeated small files.

### VRAM-Aware Batch Sizing (v0.9.96-98)  POSITIVE
Binary search for largest compress batch that fits in VRAM. Includes nvCOMP temp workspace in estimate.
- **Workstation (10 GiB VRAM):** Finds batch=104 instead of hanging on batch=256
- **Why it matters:** cudaMalloc can hang on some drivers if request exceeds VRAM. Pre-check avoids this.
- Fixed partial allocation leak on retry (free_stream_buffers_only before halving).

---

## CUDA Context & Init Optimizations

### CUDA Context Warm-Up (v0.9.58-59)  NEGATIVE (reverted)
Pre-initialize CUDA contexts on all devices before GPU workers start.
- **Both sync and async versions added ~3s overhead**
- **Why it failed:** CUDA contexts are per-thread. Warming up from a temporary thread creates a throwaway context; the actual GPU worker creates its own anyway.
- **Lesson:** CUDA per-thread context model makes warm-up ineffective. Would need cuDevicePrimaryCtxRetain (driver API) to share contexts.

---

## Scheduling & Routing Optimizations

### Hybrid Scheduler: 256 Threads (v0.9.52)  NEGATIVE (reverted)
Full hardware_concurrency() threads at 80% CPU start share.
- **Regression:** 0.75x across all configs
- **Why it failed:** 256 worker threads starved the reader/writer I/O threads. Even GPU-only mode degraded because I/O pipeline couldn't keep up.
- **Lesson:** I/O pipeline (reader + writer) is the critical path. Cap CPU threads below full hardware count.

### Adaptive CPU/GPU Share via EMA (v0.9.52-78)  MIXED
Various attempts at throughput-based adaptive scheduling.
- 50/50 start: CPU ate everything before GPU initialized
- 10/90 GPU-favored start: CPU still drained queue during GPU init
- Throughput measurement: CPU always appeared faster because GPU was starved for data
- **Final solution:** Semaphore-based scheduling (v0.9.83)

### GPU-Priority Semaphore Scheduler (v0.9.83)  POSITIVE
`gpus_waiting` atomic counter. GPU increments before pop, decrements after. CPU yields when counter > 0.
- **Why it works:** Direct, instant priority signaling. No measurement delay. GPU always gets fed first.
- CPU runs wild during GPU init, then yields once GPU signals ready
- CPU helps when all GPUs are busy processing (counter = 0)

### Trivially-Compressed Frame Detection (v0.9.93)  POSITIVE
Decompress: peek at front frame's ratio. If < 2%, CPU takes it regardless of GPU priority.
- **Why it works:** Frames decompressing to mostly zeros are faster on CPU (no PCIe D2H overhead). CPU + sparse writes = near-instant.
- **Server:** zeros.bin: CPU path 1.4s vs GPU path 4.4s

### Auto CPU Thread Cap at 96 (v0.9.80)  POSITIVE
Default auto: min(hw-1, 96). -T0 = all threads (matches zstd).
- **Why:** Diminishing returns beyond 96 threads on large-core machines. Leaves headroom for I/O threads.

### --cpu-batch as Queue Depth Threshold (v0.9.92-94)  POSITIVE
Minimum queue depth before CPU workers activate. Each CPU takes 1 frame (no CPU batching benefit).
- **Why:** Keeps queue stocked for GPUs. CPU only helps when there's overflow.

---

## Batch Size Auto-Tuning

### Decompress Greedy Batch Pop (v0.9.69)  POSITIVE (massive)
pop_batch_greedy waits for full batch before GPU processes. DEFAULT_GPU_DECOMP_BATCH_CAP = 256.
- **Server:** medium_compress kernel dropped 24.7s → 1.27s (55× speedup!)
- **Why:** Default batch=8 caused 64 kernel launches × 385ms each. Batch=256 = 3 launches × 424ms.

### Continuous Binary-Search Auto-Tuner (v0.10.0-0.10.6)  POSITIVE
Runtime throughput-aware batch sizing for compress. Explores both directions from default.
1. Record baseline throughput at starting batch size
2. Try halving  if better, continue halving
3. Try doubling  if better, continue doubling
4. Settle at best when throughput drops
5. Periodically probe to detect data character changes
- **Workstation:** Correctly finds batch=8 optimal for compress, settles in 2 steps
- **Fixed bugs:** free_stream_buffers_only wiped tune state (v0.10.4), tune ceiling was default not VRAM limit (v0.10.2), baseline never recorded (v0.10.3)

---

## GPU Selection & Topology

### NVML/NUMA-Aware GPU Selection (v0.9.63-68)  POSITIVE
Queries GPU utilization and NUMA topology. Penalizes GPUs on busy NUMA nodes.
- **Why:** Prevents selecting GPU 6 when GPU 4 (same NUMA node) is busy at 29%.

### --gpu-devices N (v0.9.62)  POSITIVE
Decompress default: 2 GPUs (PCIe bandwidth optimal for 1-2 GPUs).
Compress default: all GPUs.

---

## Performance Instrumentation

### -vvv Breakdown (v0.9.61)  ESSENTIAL
PerfCounters struct with atomic accumulators for every pipeline phase.
- **Bug found (v0.9.89):** Compress GPU worker had TWO completion paths (async poll + sync drain). Only async poll recorded to g_perf. Sync drain path handled majority of completions for small batches.
- **Lesson:** Comment both paths with "MUST record to g_perf  see also other path"

---

## Key Architectural Lessons

1. **PCIe is the wall for GPU decompress.** GPU kernel is fast; moving 8 GiB D2H at 1.5-3.5 GiB/s dominates. 1 GPU often beats 8 GPUs due to PCIe contention.

2. **Writer I/O is the wall for CPU decompress.** CPU decompresses at 5-20 GiB/s aggregate but NVMe writes at 1.8-3.0 GiB/s.

3. **Never starve the I/O pipeline.** Reader and writer are serial bottlenecks. Too many CPU threads, too-high I/O priority, or GPU-induced memory pressure all cause regression.

4. **CUDA contexts are per-thread.** Warm-up on temporary threads is useless. Pinned memory from wrong context causes slowdown. Always design around the thread that will actually use the GPU.

5. **Measure before optimizing.** The -vvv breakdown has been the single most valuable tool. Every successful optimization was guided by perf data. Every failed one was based on hypothesis alone.

6. **Small GPUs need different tuning than large GPUs.** H100 (95 GiB): batch=256 decompress, 8 GPUs. RTX 2080 Ti (10 GiB): batch=8 compress, batch=16 decompress, 2 GPUs. The auto-tuner handles this automatically.

---

## Benchmark Snapshots

### Server (H100 × 8)  v0.9.74 vs zstd -T0, 8 GiB files, decompress
| File | zstd -T0 | gzstd | Speedup |
|------|----------|-------|---------|
| zeros | 4.85s | 4.40s | 1.10× |
| high_compress | 9.52s | 7.07s | **1.35×** |
| medium_compress | 15.39s | 9.69s | **1.59×** |
| mixed | 9.12s | 6.55s | **1.39×** |
| low_compress | 9.25s | 7.14s | **1.30×** |
| **Total** | **48.13s** | **34.85s** | **1.38×** |

### Workstation (RTX 2080 Ti × 2)  v0.10.6 vs zstd -T0, 8 GiB files
**Decompress:** gzstd wins 2/5 (medium_compress 1.22×, low_compress 1.06×). Loses on trivial data where zstd's page-cache sparse dominates.
**Compress:** gzstd wins 4/5 (high 1.83×, low 1.54×, medium 1.11×, mixed 1.26×). Only loses zeros.

---

### io_uring Writer (v0.10.22-0.10.28)  NEGATIVE (reverted)
Replaced DirectWriter + AsyncWritePool with Linux io_uring for async writes.
- **v0.10.22-26:** O_DIRECT + io_uring. Writes submitted but never completed  `io_uring_wait_cqe` hung forever. Likely kernel/NVMe driver incompatibility with O_DIRECT + io_uring on Server.
- **v0.10.27:** Tried `io_uring_submit_and_wait()`  still hung.
- **v0.10.28:** Dropped O_DIRECT, tried buffered io_uring  still hung.
- **Root cause:** Unknown kernel-level issue. io_uring write completions never arrived despite successful submission. Possibly a kernel config, seccomp policy, or filesystem limitation.
- **Decision:** Reverted to DirectWriter + AsyncWritePool.

### Multi-threaded pwrite Pool (v0.10.29)  NEGATIVE (reverted)
4 threads doing pwrite() at known offsets through the page cache.
- **Server:** 10m30s (vs 4m with DirectWriter). `sys: 38m40s` (vs 12m).
- **Why it failed:** Without O_DIRECT, 432 GiB went through the page cache. The pwrite() calls returned fast (page cache absorb), but kernel writeback stalled massively. The page cache backlog created 9.5 minutes of post-completion flush.
- **Key lesson:** You cannot beat the NVMe's physical write speed (~2-3 GiB/s on Server). O_DIRECT + single-threaded sequential write is already optimal for this workload. The 220s writer drain IS the hardware limit  not a software bottleneck.
- **Decision:** Reverted to DirectWriter + AsyncWritePool (v0.10.30).

---

### Removed fsync on output (v0.10.31-33)  POSITIVE
Removed fsync() call before closing output file. Like zstd, the OS handles writeback in the background after close(). With O_DIRECT, data is already on physical media  only the tiny unaligned tail goes through the page cache.
- Added `--sync-output` flag for users who need guaranteed persistence before exit.
- Renamed misleading "flushing to disk" messages to "draining write queue" / "writing..."
- **Decision:** Default off. Matches zstd behavior.

### File-size-based decompress batch start (v0.10.34)  POSITIVE
Starting batch size for decompress auto-tuner now scales with input file size:
- >75 GiB: start at 256 (was 16, wasted minutes exploring upward on large files)
- >10 GiB: start at 64
- ≤10 GiB: start at 16
Auto-tuner still refines from the starting point. On 217 GiB file, converges to 512 in 3 steps.

---

## Benchmark Snapshots (Updated)

### Server (H100 × 2 GPUs)  v0.10.34, 432 GiB file (rpfrancis.tar)

**Decompress test mode (-t, no disk I/O):**
- 432.58 GiB decompressed in **53.5 seconds** = **8.13 GiB/s**
- Auto-tuned to batch=512
- 96 CPU threads + 2 GPU devices

**Decompress to disk (O_DIRECT):**
- 432.58 GiB in ~3m37s-5m22s = **1.3-2.0 GiB/s** (varies with NVMe contention)
- Writer drain: ~220s (NVMe write bandwidth ceiling)
- Compute pipeline runs at 4+ GiB/s, storage is the bottleneck

**Compress (4 GPUs, v0.10.11):**
- 432.58 GiB → 217 GiB in **3m21s** = **2.16 GiB/s**
- Auto-tuned to batch=48→816 over the run

### Workstation (RTX 2080 Ti × 2)  v0.10.6, 8 GiB files

**Compress: gzstd wins 4/5 vs zstd -T0**
| File | zstd -T0 | gzstd | Speedup |
|------|----------|-------|---------|
| high_compress | 7.23s | 3.96s | **1.83×** |
| low_compress | 5.85s | 3.80s | **1.54×** |
| medium_compress | 4.33s | 3.91s | **1.11×** |
| mixed | 4.28s | 3.40s | **1.26×** |
| zeros | 2.47s | 3.88s | 0.64× |

**Decompress: gzstd wins 2/5 (storage-limited on consumer NVMe)**

### Workstation (RTX 2080 Ti × 2)  v0.11.20, 8 GiB files, 3 iterations

**Compress (GiB/s):**
| File | CPU | GPU-only | Hybrid | Best config |
|------|-----|----------|--------|-------------|
| high_compress | 1.46 | **2.04** | 2.02 | GPU ≈ Hybrid |
| medium_compress | 1.36 | **2.10** | 2.06 | GPU ≈ Hybrid |
| mixed | 1.50 | 1.73 | **2.14** | Hybrid |
| low_compress | **1.50** | 1.22 | 1.51 | CPU ≈ Hybrid |
| zeros | 1.49 | **2.06** | 2.02 | GPU ≈ Hybrid |

Hybrid compress matches or beats the best single backend on every data type.
Mixed data (2.14 GiB/s) beats both CPU (1.50) and GPU-only (1.73)  the scheduler
correctly splits work between CPU and GPU based on observed throughput.

**Decompress (GiB/s):**
| File | CPU | GPU-only | Hybrid | Best config |
|------|-----|----------|--------|-------------|
| zeros | **4.88** | 3.49 | 3.50 | CPU |
| medium_compress | **2.80** | 1.93 | 2.11 | CPU |
| mixed | **2.05** | 1.95 | 1.85 | CPU |
| high_compress | **1.52** | 1.44 | 1.40 | CPU |
| low_compress | 1.44 | 1.45 | 1.43 | ~Tied |

CPU wins decompress across the board on Workstation. PCIe Gen3 bandwidth makes D2H
the bottleneck  the GPU can't transfer decompressed data back fast enough to
justify the round-trip. Trivial frame detection helps (zeros at 4.88 GiB/s).
Confirms that asymmetric mode (GPU compress + CPU decompress) would be the
ideal default for consumer GPUs with PCIe Gen3.

### Workstation (RTX 2080 Ti × 2)  v0.11.22, 8 GiB files, 3 iterations

Machine was under student load (~12% lower baseline than v0.11.20 run).
Back-to-back v0.11.21 → v0.11.22 comparison is valid (same load).

**Hybrid Compress (GiB/s):**
| File | v0.11.21 | v0.11.22 | Delta |
|------|----------|----------|-------|
| mixed | 1.850 | **1.989** | **+7.5%** |
| high_compress | 1.844 | 1.845 |  |
| medium_compress | 1.845 | 1.845 |  |
| low_compress | 1.438 | 1.439 |  |
| zeros | 1.844 | 1.844 |  |

**Hybrid Decompress (GiB/s):**
| File | v0.11.21 | v0.11.22 | Delta |
|------|----------|----------|-------|
| mixed | 1.719 | **1.841** | **+7.1%** |
| medium_compress | 1.981 | 1.985 |  |
| zeros | 3.218 | 3.205 |  |
| high_compress | 1.288 | 1.289 |  |
| low_compress | 1.357 | 1.357 |  |

Early memory release (v0.11.22) improved mixed.bin by ~7% in both directions.
Mixed data has high frame churn (alternating compressible/random blocks) where
freeing input buffers sooner reduces page allocation contention. Other data
types are flat  bottlenecked by PCIe or NVMe, not memory lifecycle.

## Key Lessons Learned (Updated)

7. **io_uring may not work on all kernels.** Server's kernel accepted io_uring submissions but never completed writes. Possibly a seccomp policy, kernel config, or NVMe driver limitation. Always have a fallback.

8. **Page cache is not free.** Multi-threaded pwrite through page cache caused 38 minutes of sys time (vs 12 min with O_DIRECT). The page cache absorbed writes instantly but kernel writeback created a massive backlog. O_DIRECT + sequential single-thread is optimal for large sequential output.

9. **Don't fsync unless asked.** zstd doesn't fsync, cp doesn't fsync. O_DIRECT data is already on disk. Removing fsync saves seconds and matches user expectations. Provide `--sync-output` for paranoid users.

10. **The disk is the ceiling.** At 8.13 GiB/s compute vs 1.5-2.0 GiB/s NVMe write, the decompression pipeline is 4-5× faster than storage. No software optimization can fix this. Faster NVMe (Gen5, RAID) is the only path forward.

---

### Batched H2D Transfer (v0.11.6-0.11.8)  NEGATIVE (reverted)
Packed all compressed frames into contiguous host buffer, one cudaMemcpyAsync.
- **Why it failed:** `alloc_comp` per frame is max size (16 MiB) but actual compressed data is smaller. Packing copies 4 GiB of mostly padding. Per-frame async only copies actual bytes. CUDA driver already coalesces async transfers internally.
- H2D went from ~2 GiB/s to 0.22 GiB/s.

### Batched D2H Transfer (v0.11.6)  NEGATIVE (reverted)
Single cudaMemcpy for entire decompressed batch, deliver all frames at once.
- **Why it failed:** Blocked writer thread for entire 4 GiB transfer. Writer could no longer pipeline disk writes with GPU D2H. Per-frame D2H feeds writer continuously.
- D2H: 0.14 GiB/s. Result lock contention: 451 seconds (8 GPUs fighting one mutex).

### Thread Pinning (v0.11.5)  NEGATIVE (disabled)
Pinned reader to core 0, writer to core 1.
- **Why it failed on Server:** Students had ALL cores at 97-99%. Pinning forced I/O threads onto busy cores instead of letting the OS scheduler find idle moments on any core.
- **When it would help:** Dedicated machine with no competing workloads.

### GPU Utilization Backoff (v0.11.3)  REPLACED by proportional scaling
Paused GPU workers when utilization >50%, resumed at ≤30%.
- **Why it was wrong:** Blocking wastes a GPU that could still contribute at reduced capacity.
- **Replaced by:** `util_scale` factor (v0.11.4)  GPU at 50% gets half the batch size, still contributes.

### Proportional GPU Utilization Scaling (v0.11.4)  POSITIVE
`util_scale = max(0.05, (100 - gpu_util%) / 100)` applied to batch size.
- Updated via NVML after each batch completion.
- GPU at 0% → full batch, 50% → half, 90% → 10%.
- No wasted GPU cycles, no blocking.

### Sequential Frame Dispatcher (v0.11.1)  NEGATIVE (reverted)
Round-robin ticket system forcing GPUs to pop in order.
- **Why it failed:** Serialized the pop operation  GPU 1 couldn't pop until GPU 0 finished popping. With `pop_batch_greedy` blocking for enough frames, 7 GPUs sat idle while 1 waited.

## Key Lessons Learned (Updated)

11. **Don't batch what CUDA already batches.** `cudaMemcpyAsync` in a stream is already coalesced by the driver. Manual packing adds host-side memcpy overhead and padding waste.

12. **Writer parallelism > transfer efficiency.** Per-frame D2H is "inefficient" per-transfer but keeps the writer pipeline full. Batched D2H is "efficient" but starves the writer. Pipeline throughput wins.

13. **Thread pinning hurts on shared machines.** The OS scheduler is better at finding idle moments across all cores than a fixed pin on a busy core.

14. **Proportional > binary.** Don't block a resource (GPU, core)  scale its allocation proportionally. A 50%-loaded GPU with half the batch is better than an idle GPU.

---

### Per-GPU Result Slots (v0.11.11)  POSITIVE (major)
Each GPU pushes decompressed frames to its own slot (own mutex). Writer drains all slots periodically. Eliminates cross-GPU mutex contention.
- **Result lock: 451s → 0.06s** (7,500× improvement)
- Why: 8 GPUs doing per-frame lock/unlock on one shared mutex = massive contention. Per-GPU slots = zero contention (one producer per slot).

### Batch-Completion Writer Notification (v0.11.14-15)  POSITIVE
Only notify writer after full D2H batch completes (not per-frame). CPU fallback path still notifies per-frame (low volume).
- **Writer wakeups: 23,185 → 254** (91× reduction)
- Each wakeup now drains 200+ frames instead of checking and sleeping.

### Pinned D2H Buffer (v0.11.17)  NEGATIVE (reverted, 3rd attempt)
Pinned host buffer per stream for D2H, then memcpy to frame vector.
- 9% slower than pageable. Two copies (DMA→pinned→vector) worse than CUDA's internal staging (DMA→internal_pinned→vector, optimized by driver).
- **Three failed pinned attempts documented.** CUDA's pageable transfer is highly optimized internally. Don't try to outsmart it unless you can eliminate ALL copies.

### Rate-Match CPU Throttle (v0.11.0, disabled v0.11.9)  MIXED
`cpu_may_take()` throttled CPU workers to match GPU batch timing.
- Correctly reduced CPU usage on loaded machines (user time dropped from 8m to 2m)
- Disabled for debugging; needs re-evaluation on quiet machine.

### Thread Pinning (v0.11.5, disabled v0.11.9)  NEGATIVE on shared machines
Reader pinned to core 0, writer to core 1. Hurts when cores are loaded by other users.
- Would help on a dedicated machine. Keep disabled by default, consider `--pin-io` flag.

### Remove dead liburing references (v0.11.20)  CLEANUP
Removed `#include <liburing.h>` and stale io_uring comment left over from the reverted io_uring writer (v0.10.22-0.10.28).
- **Why:** The include created an unnecessary build dependency on liburing despite io_uring code being fully reverted in v0.10.28.
- No functional change.

### CV-Based CPU Worker Scheduling (v0.11.21)  POSITIVE (correctness + scalability)
Replaced 9 × `sleep_for(1ms)` poll loops in CPU compress and decompress workers with proper condition variable waits. CPU workers now block on a dedicated `cpu_cv_` and wake in microseconds when conditions change.
- **TaskQueue:** Added `cpu_cv_` (dedicated CV for CPU workers), `wait_for_cpu(predicate)`, and `pop_one_cpu(task, predicate)`. Predicates receive a `QueueState` snapshot to avoid recursive lock deadlocks. `push()` uses `cv_.notify_all()` to ensure all waiting GPUs see new frames.
- **HybridSched:** `gpu_got_data()` and `set_gpu_ready()` now call `notify_cpu_waiters()` so CPU threads wake instantly when scheduling state changes instead of sleeping up to 1ms.
- **Workstation (8 GiB files):** No measurable throughput change on small workloads (127 frames  sleep overhead was ~2% of runtime). The win is on large files with thousands of frames where 22 threads × 1ms × thousands of iterations compounds to minutes of waste.
- **Bug fixed during development:** Initial implementation deadlocked because predicate lambdas called `tq->peek_front_ratio()` / `tq->size()` / `tq->drained()` while `pop_one_cpu` held `m_` (non-recursive mutex). Fixed by passing a `QueueState` snapshot to predicates instead.
- **Bug fixed during development:** `push()` with `cv_.notify_one()` could deliver notifications to a GPU that was busy processing (not waiting), starving the other GPU. Changed to `cv_.notify_all()`.

### Early Memory Release (v0.11.22)  POSITIVE
Release input data buffers immediately after they're consumed instead of holding them until end of processing cycle. Reduces peak memory by freeing frames as soon as they're no longer needed.
- **CPU compress worker:** `t.data` (up to 32 MiB) released via swap immediately after `compress_one_cpu_frame`. Previously held through logging, stats, and result delivery.
- **GPU compress worker:** Batch input data (up to 16 × 16 MiB = 256 MiB) released after H2D upload. Guarded by `!rescue`  in hybrid mode data stays alive for potential CPU rescue on GPU failure; in gpu-only mode released immediately.
- **GPU decompress worker:** Batch compressed data released before kernel launch (after re-upload path). Saved `batch_seqs[]` and `batch_comp_sizes[]` for completion paths.
- **CPU decompress worker:** Already had early release (swap at line 2280)  no change needed.
- **Workstation (8 GiB files):** +7.1% decompress and +7.5% compress on mixed.bin. Other data types flat (bottlenecked elsewhere). Mixed data benefits most because alternating compressible/random blocks cause high frame churn  freeing memory sooner reduces page allocation contention.

### Write Drain Progress Bar (v0.11.23)  IMPROVEMENT
Progress bar now shows write drain percentage when the compute pipeline finishes but disk I/O is still in progress. Previously showed a static "writing..." message or sat at 100% while the NVMe caught up.
- **Meter:** Added `total_out` atomic tracking expected total output bytes. Set by `stream_frames_to_queue` (decompress, from frame headers) or accumulated by writer thread (compress, as frames complete).
- **AsyncWritePool:** `wrote_bytes` now updated by the AIO worker after physical write completes (not on submit), so progress reflects actual disk I/O.
- **Progress format:** `[85.3%] writing: 6.75 GiB / 7.91 GiB @ 2.01 GiB/s` during drain phase.
- **Verbose output cleanup:** `vlog()` now checks `g_progress_active` flag and clears the progress line (`\r\033[K`) before printing, so `-v`/`-vv`/`-vvv` messages don't overlap the progress bar.
- **GPU ready message:** Now shows device ID: `GPU 7 ready  semaphore scheduling active`.
- **Decompress progress lifetime:** Progress bar stays alive through the entire DirectWriter finalize + file close, replacing the old one-shot "writing..." message.

### Writer Backpressure (v0.11.24)  POSITIVE (major)
Prevents CPU decompression workers from producing data faster than the NVMe can write, which was causing massive kernel writeback pressure in hybrid mode.
- **WriterBackpressure class:** Tracks produced vs physically-written bytes with hysteresis (4 GiB high-water / 2 GiB low-water). CPU workers block on a CV when backlog exceeds high-water, wake instantly when it drops below low-water. GPUs are never throttled (batches in-flight).
- **AIO worker** calls `mark_written()` after each physical write, waking blocked CPU workers.
- **CPU decomp workers** call `wait_if_backlogged()` before popping a new task, `mark_produced()` after delivering a decompressed frame.
- **GPU decomp workers** call `mark_produced()` for accurate backlog accounting but are never blocked.
- No artificial sleeps  all coordination via condition variables with instant wakeup.

**Server (H100 × 8, 432 GiB file) hybrid decompress:**

| Metric | v0.11.22 (before) | v0.11.24 (after) | Change |
|--------|-------------------|------------------|--------|
| Wall clock | 6m07s | **3m56s** | **-36%** |
| user time | 4m36s | 2m55s | -37% |
| sys time | **19m11s** | **6m27s** | **-66%** |
| Throughput | 1.18 GiB/s | **1.84 GiB/s** | **+56%** |

Hybrid went from worst of all three modes (6m07s) to best of all three (3m56s), beating both CPU-only (4m42s, 1.53 GiB/s) and GPU-only tuned (4m13s, 1.72 GiB/s). The sys time drop from 19m to 6m confirms the root cause: 96 CPU threads were flooding the kernel with write syscalls. Backpressure keeps CPU workers productive during GPU batch gaps while preventing writer saturation.

### Test Mode & Progress Fixes (v0.11.25)  BUGFIX
- **`wrote_bytes` double-counting:** Worker-side updates removed; writer thread is now the sole source of truth for output bytes.
- **Test mode backpressure stall:** Backpressure pointer set to `nullptr` in test mode (`-t`) since there's no AIO, so `mark_written()` is never called. Without this, CPU workers would block forever waiting for writes that never happen.
- **Progress bar in test mode:** Shows `verified:` label instead of `out:`.

### Graceful GPU VRAM Handling (v0.11.26)  POSITIVE (robustness)
GPU workers now survive VRAM exhaustion instead of crashing the process. Critical for shared GPU environments where other users consume VRAM mid-run.
- **Graceful skip:** GPU workers return early when batch=1 allocation fails. `TaskQueue::re_enqueue()` returns in-flight frames to the queue (without incrementing `total_tasks_`) so other GPUs or CPU workers process them.
- **VRAM reserve:** Each GPU holds a half-batch-sized reserve. On allocation failure, the reserve is freed for retry before giving up.
- **VRAM retry limit:** 10 attempts max in both compress and decompress allocation loops, preventing infinite retry when VRAM fluctuates near the threshold.
- **Reader no longer aborts early:** Removed `abort_on_failure && any_gpu_failed` check from compress reader loop. A single GPU VRAM failure no longer truncates the output  surviving GPUs handle the work.
- **Post-join GPU failure:** Only calls `die()` if ALL GPUs failed (count-based), not on any single failure.

### Structured Exit Codes & Argument Hardening (v0.11.26)  IMPROVEMENT
- **Exit codes:** 0=OK, 1=runtime, 2=usage, 3=I/O, 4=data, 5=GPU_FAIL. All `die()` calls categorized via `die_io()`, `die_data()`, `die_usage()`. Help text documents exit codes.
- **Unknown option rejection:** Flags starting with `-` that aren't recognized exit with code 2 (EXIT_USAGE) instead of being treated as filenames.
- **`--` end-of-options:** Everything after `--` is treated as a filename, matching POSIX convention.
- **`--threads=N` form:** Now recognized alongside `-T N`, `-T2`, `--threads N`.
- **Argument order independence:** `-22 --ultra` works (deferred ultra check to post-parse validation).
- **Truncated stream detection:** Decompress checks `ret > 0` after loop and dies on >8 trailing bytes at EOF.
- **`.zst` double-compression warning:** Warns when compressing a file that already has `.zst` extension.

### Writer Deadlock Detection & Cleanup Safety (v0.11.27)  IMPROVEMENT
- **Writer deadlock detection:** 5-second timed wait when `workers_done` is set. If the next expected frame never arrives, calls `die()` with diagnostic instead of hanging forever or silently producing truncated output.
- **`die()` reports cleanup:** Shows `gzstd: removing incomplete output: path` on fatal error.
- **Atomic temp file cleanup:** When using `-f` to overwrite, the `.tmp` file is registered for cleanup on failure/signal. Original file is preserved untouched.
- **Consistent log format:** `[GPU-D...]` → `[GPU...]` everywhere. `[GPU2] ready, semaphore scheduling active` format.

### Compress Backpressure (v0.11.29)  POSITIVE (major)
Same backpressure mechanism from decompress (v0.11.24) now applied to all compress paths. Prevents CPU workers from producing compressed data faster than the NVMe can write.
- **`compress_cpu_mt`:** `WriterBackpressure` created and passed to writer + all CPU workers.
- **`compress_nvcomp`:** Backpressure passed to writer, GPU workers, CPU hybrid workers, and rescue workers.
- **CPU workers** call `wait_if_backlogged()` before popping, `mark_produced(csz)` after delivering compressed frame.
- **GPU workers** call `mark_produced(out_sum)` on both async and sync D2H paths. Never throttled.
- **`--cpu-batch` in `--cpu-only` mode:** Now ignored with a note, since the stop-and-go pattern caused massive sys overhead (10m26s sys on 432 GiB file).

### Default Chunk Size = 16 MiB (v0.11.30)  POSITIVE
All paths (CPU-only, hybrid, GPU-only) now default to 16 MiB chunk size. Removed auto-chunk scaling that previously used 32512 MiB based on file size and device count.
- **Why 16 MiB wins:** More tasks for better load balancing (27,685 vs 3,461 tasks for 432 GiB), lower per-thread memory (3 GiB vs 24 GiB for 96 threads), and matches `GPU_SUBCHUNK_MAX` so no splitting is needed in hybrid mode.
- **Removed:** `auto_chunk_mib_cpu`, `auto_chunk_mib_gpu`, `is_regular_file_stream`, `AUTO_HOST_CHUNK_*` constants.
- **`--chunk-size=N`** still available for manual override.

### RAM Budget Check (v0.11.29)  IMPROVEMENT
Pre-flight check reads `/proc/meminfo` `MemAvailable` and estimates memory needed for N threads × chunk_mib. If estimated usage exceeds 75% of available RAM, auto-reduces chunk size (halving until it fits) with a warning instead of OOMing.

### Progress Bar Improvements (v0.11.30)  IMPROVEMENT
- **Dual rate display:** Progress bar now shows both `in:` and `out:` rates: `[45.2%] in:195.6 GiB out:97.3 GiB | in:1.84 GiB/s out:918 MiB/s`
- **Write drain states:** Three states  AIO writing (capped 99.9%), flushing to disk (elapsed timer), finalizing message.
- **Finalize message:** Shows `[done] finalizing 217.3 GiB ...` for large files during file close/rename.

### Comprehensive Test Suite (v0.11.26v0.11.30)  NEW
`gzstd-test.sh`: ~170+ tests across 35 sections with live progress bar, per-test timing, CTRL-C handling, and auto GPU detection. Key coverage: round-trip, compression levels with ratios, integrity, pipes, tar, file management, threading forms, chunk sizes, verbosity validation, stats JSON, exit codes, zstd interop, VRAM pressure, wildcards, `--` end-of-options, output redirection, space-separated options, GPU options, error handling, cross-level decompress, argument order, completion summary format.

### Stdout O_DIRECT Detection (v0.11.31)  POSITIVE (major)
When stdout is redirected to a regular file (`gzstd -d < file.zst > output.tar`), gzstd now detects this via `fstat(fileno(stdout))` + `/proc/self/fd/N` and reopens with O_DIRECT, bypassing the page cache.
- **Safety checks:** Skips O_APPEND (undefined with O_DIRECT), `/dev/*`, deleted files, non-regular files. Falls back silently to buffered fwrite on any failure.
- **Result:** `tar | gzstd > file.zst` gets full NVMe speed without the user needing to know about `-o`.

**Server (432 GiB decompress via stdin redirect):**

| Method | Wall | sys | GiB/s |
|--------|------|-----|-------|
| gzstd stdout → page cache | 8m59s | 27m27s | 0.83 |
| zstd stdout → page cache | 10m46s | 11m01s | 0.67 |
| **gzstd stdout → O_DIRECT** | **3m33s** | **19m53s** | **2.05** |

### GPU Backpressure on Pop (v0.11.31)  BUGFIX (major)
GPUs now call `wait_if_backlogged()` before `pop_batch_greedy` in both compress and decompress workers. Previously GPUs were exempt from backpressure ("never throttled"), but 8 H100s decompressing at full speed overwhelmed the NVMe writer  decompression finished with only 28% of data written to disk.
- GPU workers block before grabbing new work, not mid-kernel
- One batch per GPU remains in-flight beyond the high-water mark at most
- Writer drain after decompress: was 72% remaining, now <1 second

### Test Mode Defaults to 2 GPU Streams (v0.11.31)  POSITIVE
`-t` verify mode now defaults to `--gpu-streams=2` instead of 1. No write bottleneck in verify mode, so stream overlap helps.
- **Server (432 GiB verify):** 1 stream: 4.09 GiB/s (1m47s) → 2 streams: 6.39 GiB/s (1m09s)  **56% faster**
- Compress/decompress stays at 1 stream (NVMe is the bottleneck, larger batches win)
- Fixed help text: was incorrectly showing default as 3

## Key Lessons Learned (Updated)

19. **Stdout O_DIRECT is a free 2.5× win.** Users don't think about O_DIRECT  they just write `> file`. Detecting stdout-to-file and auto-enabling O_DIRECT gives them NVMe speed without any knowledge of I/O internals. The page cache adds 17 GiB of dirty pages and halves throughput on 432 GiB files.

20. **GPUs need backpressure too.** The original design exempted GPUs ("batches in-flight, can't throttle"). But on H100 × 8, GPU decompression throughput vastly exceeds NVMe write speed. The result: 300+ GiB buffered in RAM, massive kernel writeback, and a frozen "28% writing" progress bar after decompression finished. Throttling GPUs before their next `pop_batch_greedy`  not mid-kernel  keeps the pipeline balanced with <1s drain.

21. **Bound frames, not bytes.** Byte-based backpressure conflated two concerns: memory pressure (bytes in RAM) and frame ordering (which frames the writer can drain). Out-of-order frames in ResultStore inflated the byte count, triggering false backpressure. A counting semaphore on frames separates these concerns: frame ordering lives in ResultStore, flow control lives in the semaphore. The FIFO queue guarantees the writer's next-needed frame is always in-flight, making the design deadlock-free without escape hatches.

18. **GPU VRAM is a shared resource  design for it.** On a multi-user machine (Server, 8× H100), any GPU can lose VRAM at any moment. Infinite retry loops, early reader aborts on single GPU failure, and missing frame deadlocks all surfaced under real student workloads. The fix: retry limits, graceful skip with re-enqueue, deadlock detection with hard error, and never abort the reader on partial failure.

---

## Early Benchmark History (Server, v0.9.50v0.9.59)

Baseline: v0.9.51 CPU-default avg compress 7.7s (1.06 GiB/s), decompress 11.2s (0.72 GiB/s).
All times are averages across 5 data types (8 GiB each).

| Version | Key Changes | CPU Compress | CPU Decompress | GPU Decompress | Hybrid Compress | Hybrid Decompress |
|---------|------------|-------------|----------------|----------------|-----------------|-------------------|
| v0.9.50 | Pre-optimization baseline | 10.65s | 11.02s | -- | -- | -- |
| v0.9.50-opt | DCtx, offset buf, early release | 10.74s | 11.29s | -- | -- | -- |
| v0.9.51 | Batch 8, streams 1, pre-alloc | 7.69s | 11.19s | 13.37s | 7.90s | 11.66s |
| v0.9.52 | 256 threads, 80% start | 8.26s | 11.10s | 16.54s | 10.41s | 15.53s |
| v0.9.53 | +mmap, +pinned decomp | 9.25s | 12.55s | 25.55s | 9.90s | 17.24s |
| v0.9.54 | -pinned (mmap only) | 8.14s | 11.32s | 15.04s | 9.81s | 13.80s |
| v0.9.55 | -mmap (scheduler+IO prio only) | 7.65s | 10.84s | 19.70s | 7.43s | 13.39s |
| v0.9.56 | nice(-5) skip in GPU-only | 8.14s* | -- | 17.69s | -- | -- |
| v0.9.57 | Decomp pinned fully removed | 8.01s* | -- | 12.75s | -- | -- |
| v0.9.58 | CUDA sync warm-up | 11.27s* | -- | 14.72s* | -- | -- |
| v0.9.59 | CUDA async warm-up | 11.35s* | -- | 14.94s* | -- | -- |

*GPU-only compress only (targeted benchmark). -- = not tested in isolation.
v0.9.56 and v0.9.57 were targeted GPU-only benchmarks (1 iteration). v0.9.55 was a full 3-iteration sweep.

**Key early observations:**

- **Scale matters.** DCtx reuse, offset buffer, and early release showed negligible impact at 8 GiB / 128 frames. They would matter at 500+ GiB / 10,000+ frames where per-frame overhead accumulates. This was confirmed by the CV scheduling work (v0.11.21) which targets the same scaling issue.

- **Hybrid mode works when conditions are right.** v0.9.51 hybrid beat both CPU and GPU for high-compressibility compression (5.86s vs 7.23s CPU vs 6.63s GPU). The adaptive scheduler can find the optimal balance  but only if the I/O pipeline is not starved and the thread count is sensible.

- **mmap needs true zero-copy to win.** mmap with `t.data.assign()` (which copies from mapped pages) is worse than fread for sequential I/O. The benefit only materializes with zero-copy access (`string_view`/`span` into mapped region) or random-access patterns.

- **Pinned memory requires architectural fit.** Pinned memory is a powerful optimization when used correctly (async overlap, modest sizes). Naively bolting it onto a synchronous pipeline with massive allocations makes things worse. Three separate attempts (v0.9.53, v0.11.17, and a third) all failed. CUDA's pageable transfer path is internally optimized with its own pinned staging.

---

*Note: This file supersedes the former PERFORMANCE_LOG.md, which covered v0.9.50v0.9.59 in detail. All content from that file has been integrated here.*
