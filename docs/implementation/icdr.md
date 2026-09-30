# ICDR: reproducibility investigation

This records what the real ArielCyber/ICDR tool is, whether it runs in this
environment, and the exact reason for the verdict. The short version is at the
bottom (`STATUS`). This is the tool the D4 defense adapter in
`src/defenses/icdr.py` is meant to call.

## Source

- Repository: https://github.com/ArielCyber/ICDR
- Commit cloned: `64492694c220b7e549e1b88e025453ab0cb3e4ba`
  (only commit on the default branch, "Update icdr.java", 2022-10-30)
- Clone location: `external/ICDR/` (git-ignored)
- License: Apache 2.0 (the repository's own code; see below for the runtime
  dependency's separate commercial license)

## What the repository contains

The whole repository is four tracked source files plus documentation images:

- `icdr.java` — the disarm/reconstruct operations (single file, ~230 lines).
- `image_quality.py` — post-hoc quality measurement (uses `sewar`, PIL, numpy).
- `vt_ider.py`, `vt_analyse.py` — VirusTotal upload/query helpers (use
  `vtapi3`, `pandas`, `requests`).
- `assets/images/*.PNG` — figures used in the README.

It is a research artifact, not a packaged tool. There is no `pom.xml`, no
Gradle build, no Ant `build.xml`, no bundled jar, and no manifest. The image
work is Java; the measurement and VirusTotal scripts are Python.

## The processing pipeline as implemented

The operations live in `icdr.java`. The documented "optimal" pipeline is
`final_fun()` (line 179), which does, in order:

1. Load the image, save to a temporary PNG, reload it. This drops metadata
   (the transcode step).
2. Resize to 0.97 of each dimension, then resize back to the original size
   (compress/restore).
3. Apply a Gaussian blur filter, then a sharpen filter (`AdvFilter`).
4. Save the result as JPEG.

Other operations exist as separate methods: `transCode()` (line 53),
`clean()` (LSB randomization, line 62), `resize()` (line 82), `alpha()`
(the ImageDetox gamma map, line 93), `AdvFilter()` (line 147), and a
filter-sweep test `filter_types()` (line 155). Input and output are JPEG files.

This matches the pipeline the adapter docstring describes
(`transCode -> resize (~97%) -> AdvFilter`), so the adapter's intent is
faithful to the real tool. The problem is not the adapter; it is that the tool
cannot be run here.

## Dependency situation (the crux)

`icdr.java` is built entirely on **Aspose.Imaging for Java**. Every image
primitive comes from `com.aspose.imaging.*`:

- `com.aspose.imaging.Image`, `RasterImage`, `Color`
- `com.aspose.imaging.imageoptions.*` (`JpegOptions`, `PngOptions`)
- `com.aspose.imaging.ImageResizeSettings`, `ResizeType`
- `com.aspose.imaging.imagefilters.filteroptions.*` (Gaussian blur, sharpen,
  median, bilateral, Gauss-Wiener)

`main()` (line 17) explicitly installs a commercial license before doing
anything:

```java
com.aspose.imaging.License lici = new com.aspose.imaging.License();
lici.setLicense(icdr.class.getClassLoader()
        .getResourceAsStream("Aspose.Total.Product.Family.lic"));
```

The README states the requirement directly: "To run the files and scripts you
must have access to the Aspose.imaging library and a valid VirusTotal key,
along with numpy, swear [sewar], pandas, vtapi3 python libraries."

**Aspose.Imaging is a paid commercial library.** The repository bundles neither
the Aspose jar nor the `Aspose.Total.Product.Family.lic` file (confirmed by
`git ls-files`: only the four sources, the LICENSE, and the PNGs are tracked).
So running `icdr.java` requires two things this project does not have and cannot
obtain for free: the Aspose.Imaging jar and a valid Aspose license.

What happens without a license is the decisive point. Per Aspose's own
documentation, when Aspose.Imaging for Java runs unlicensed (evaluation mode):

- it stamps an "Evaluation Only. Created with Aspose.Imaging ..." watermark on
  every image it saves or modifies, or draws two diagonal lines across images
  too small to hold the text, and
- core drawing functionality is unsupported: image pixels cannot be loaded or
  saved in evaluation mode.

Both consequences make unlicensed output useless for this study. A defense that
stamps a watermark across every image, or that cannot execute the pixel load
and save that `clean()`, `alpha()`, and the filter/resize steps depend on, is
not measuring the CDR transform. It is measuring an Aspose evaluation artifact.
There is no threshold at which watermarked, pixel-disabled output supports a
scientific claim about CDR. A temporary 30-day Aspose license would remove the
limits but is time-boxed and personal, which defeats reproducibility for a
graded, re-runnable experiment.

## Two more blockers, independent of Aspose

Even if an Aspose license appeared, the cloned code would not run as-is:

1. **The committed `icdr.java` does not compile.** After `alpha()` ends at line
   110, lines 111 to 146 are an orphaned method body: statements and a closing
   brace with no method signature (the combined "all" method whose declaration
   was removed in the "Update icdr.java" commit). `javac icdr.java` fails with
   "illegal start of type" at line 116, before any dependency is even resolved.

2. **It is not a single-image runner.** `main()` scans a hardcoded `src/`
   directory, writes to hardcoded `res/` and `temp/` directories, and every
   operation call inside the loop is commented out. There is no input to output
   argument mapping, which is what the adapter's `{input}`/`{output}` contract
   needs. Wiring that up would mean editing the tool, and editing an
   uncompilable, license-gated tool does not produce a reproducible result.

## Is there a legitimate open-source fallback inside the project?

No. The entire image path in ICDR is Aspose. The repository provides no
non-Aspose code path, no pure-Java or pure-Python reimplementation of the
operations, and nothing that reproduces `final_fun` without the commercial
dependency. Reimplementing the pipeline with an open library (PIL, OpenCV) would
be a look-alike, not the real ICDR, so it is deliberately not done here. The
adapter in `src/defenses/icdr.py` stays as an honest external-command shim that
raises if `ICDR_CMD` is unset, and `ICDR_CMD` is left unset.

## Whether it runs in this environment

No. Attempted and confirmed:

- `git clone` — succeeded.
- `javac icdr.java` — failed (exit 1, syntax error at line 116), independent of
  Aspose.
- Build/run — not reached: no build system, no Aspose jar, no Aspose license.

Environment for the record: OpenJDK 21.0.11, Python via uv, Docker available,
network reachable. None of these remove the commercial-license blocker.

## Commands (for the record; they do not yield a working tool)

```bash
git clone https://github.com/ArielCyber/ICDR external/ICDR
git -C external/ICDR rev-parse HEAD   # 64492694c220b7e549e1b88e025453ab0cb3e4ba
javac external/ICDR/icdr.java          # fails: illegal start of type (line 116)
```

There is no valid `ICDR_CMD` to record. It is left unset in
`configs/experiment.yaml`, and the D4 defense is not run.

## Failure modes summary

- Commercial dependency: Aspose.Imaging jar not bundled and not free to obtain.
- License gate: no `.lic` in the repo; unlicensed runs are watermarked and have
  pixel load/save disabled, so output is not scientifically usable.
- Compilation: committed `icdr.java` has an orphaned method body and does not
  build.
- Interface: no single-image input/output entrypoint; `main()` uses hardcoded
  directories with all operations commented out.

## Containerization does not help

A reasonable question is whether a Docker/Podman image would make ICDR reproducible.
It would not. The blocker is not environment setup; it is that every image operation
calls the commercial Aspose.Imaging library, which needs a paid license file at
runtime, and unlicensed use is watermarked with pixel load/save disabled. A container
would still need the licensed Aspose jar and `.lic` baked in, which cannot be
distributed, and the committed `icdr.java` does not compile regardless. So a container
reproduces the blocker, not a working tool. ICDR stays BLOCKED unless a licensed Aspose
build is supplied, in which case `ICDR_CMD` can be set and the honest adapter runs it.

## STATUS: BLOCKED

Reason: the real ArielCyber/ICDR implements its entire pipeline on the
commercial Aspose.Imaging for Java library. The repository ships no jar and no
license, unlicensed Aspose watermarks output and disables pixel load/save, and
the committed `icdr.java` does not even compile. There is no open-source-only
path inside the project to fall back to, and substituting a homemade
reimplementation would not be ICDR.

## Impact on scientific claims

With ICDR (D4) unavailable, the study compares JPEG-based defenses and
Dangerzone only. Any CDR-specific claim must be scoped to Dangerzone's
document-oriented CDR, and the paper cannot report results for the ICDR
image-CDR pipeline. Statements about "image CDR as an adversarial defense" that
rely on ICDR should be dropped or clearly marked as not evaluated, and the
proposal/report should note that ICDR was excluded because its only
implementation requires a paid Aspose license and does not run reproducibly.
