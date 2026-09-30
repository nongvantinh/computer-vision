# Dangerzone: reproducibility investigation

This records whether the real Freedom of the Press "Dangerzone" tool runs in this
environment and can sanitize an image through the adapter in
`src/defenses/dangerzone.py` (defense condition D5). The short verdict is at the
bottom (`STATUS`).

## Source and version

- Project: https://github.com/freedomofpress/dangerzone
- Version installed: **0.11.0** (release tag `v0.11.0`, git commit
  `6cc19db50166658feff4d72df605f30e97308f16`). `version.txt` in the package
  reports `0.11.0`.
- Container image referenced by this version:
  `ghcr.io/freedomofpress/dangerzone/v1`, pulled digest
  `sha256:6193957fd3c831d5c4287fadf6129fce6eeee9abd477897d290775f556269ec2`
  (about 1.58 GB on disk).
- cosign used for image signature verification: sigstore cosign `v3.1.3`.

## What Dangerzone is

Dangerzone takes a document that might be malicious and produces a clean PDF. It
does this by rendering every input page to raw RGB pixels inside a locked-down
container, then rebuilding a fresh PDF from those pixels (optionally adding an OCR
text layer). All original file structure is discarded, so any embedded scripts,
fonts, or exploit payloads do not survive. The sandbox is a gVisor application
kernel running inside the container, and on Linux the container is started with
Podman.

## Environment checked

- OS: Ubuntu 25.10 (questing).
- Container engine: Docker 29.2.1, working and usable without root. The daemon
  uses the containerd image store (`overlayfs` /
  `io.containerd.snapshotter.v1`).
- Podman: not installed.
- `pdftoppm` (Poppler) 25.03.0: present. The adapter's rasterize-back step works.
- `uv` 0.10.2: present. No `pipx`.
- `sudo`: no passwordless sudo. Interactive authentication is required, which an
  automated run cannot satisfy.
- Disk: the root filesystem was about 94% full (57 GB free) at the start. The
  1.58 GB image pull was safe; it was removed again after the investigation to
  return the space.

## Install: what worked

The official Debian/Ubuntu path installs the `dangerzone` (or `dangerzone-full`)
`.deb` from `packages.freedom.press` and pulls in Podman plus the Qt GUI. That
path needs root for `apt`, and there is no passwordless sudo here, so it was not
usable. Installing Podman separately has the same problem: rootless Podman needs
`/etc/subuid` and `/etc/subgid` entries and the setuid `newuidmap`/`newgidmap`
helpers, all of which require root.

The tool was instead installed from source into a throwaway virtualenv, which
needs no root:

```sh
uv venv --python 3.12 dz312
source dz312/bin/activate
uv pip install "git+https://github.com/freedomofpress/dangerzone.git@v0.11.0"
```

Python 3.12 was chosen deliberately. Dangerzone declares `python >=3.10,<3.15`
and depends on PySide6 (Qt); PySide6 wheels were available for 3.12 but not for
the 3.14 interpreter that `uv` picks by default here.

A source install does not ship the `share/dangerzone` data directory that the
`.deb` provides, so `dangerzone-cli` first failed with a missing
`ocr-languages.json`. Copying `share/` from a checkout of the same tag fixes it:

```sh
git clone --depth 1 --branch v0.11.0 \
  https://github.com/freedomofpress/dangerzone.git dz-src
mkdir -p dz312/share/dangerzone
cp -r dz-src/share/* dz312/share/dangerzone/
```

After this, `dangerzone-cli --help` and `--version` run headless with no GUI and
no X server. The CLI is a real console entry point (`dangerzone-cli =
dangerzone.cli:run`), separate from the GUI (`dangerzone = dangerzone.gui.run`),
so the desktop app is not needed for command-line conversion.

The CLI options match what the adapter already invokes: `--output-filename` sets
the output path and the input file is a positional argument. There is also
`--set-container-runtime`, `--ocr-lang`, and `--debug`.

## Container engine: the blocker

Dangerzone 0.11.0 is built for Podman on Linux, and the code only knows how to
drive Podman. `dangerzone/container_utils.py` builds a `PodmanCommand` and, on
Linux, expects a system `podman` binary. The `--set-container-runtime` option
does exist, and its own help text says it takes "the name or full path of the
container runtime", but in the code the chosen path is still handed to the Podman
command wrapper. There is no separate Docker code path.

With no runtime configured, `dangerzone-cli` stops immediately:

```
dangerzone.podman.errors.exceptions.PodmanNotInstalled:
    The Podman command is not installed in the system
```

Pointing it at Docker (`--set-container-runtime /usr/bin/docker`) gets
surprisingly far, because Docker happens to accept the basic subcommands
Dangerzone issues first (`docker version -f ...`, `docker ps -a`). The image
download and signature check also succeed once cosign is placed where the
packaged build would vendor it (`share/dangerzone/vendor/cosign`):

```
dz312/bin/dangerzone-image upgrade
# pulls ghcr.io/freedomofpress/dangerzone/v1 by digest, then:
# ✅ Signatures have been verified and stored locally
```

The conversion still fails, for two separate reasons, both traceable to the fact
that this is a Podman tool running on Docker:

1. **Image lookup.** Dangerzone locates its image with
   `docker images <name> --format '{{.Digest}}'`. Under Docker 29's containerd
   image store, an image pulled by digest has a `<none>` tag, and that command
   returns an empty string unless `--digests` is also passed. So Dangerzone
   reports "No container image found" even though the image is present and
   verified. (Tagging the image locally works around this one step, but is not
   something the experiment should have to do.)

2. **Container flags.** With the image found, Dangerzone launches the sandbox
   with, among others, `--userns nomap`. That is a Podman-only value. Docker
   rejects it outright:

   ```
   docker: --userns: invalid USER mode
   ```

   Docker returns exit code 125 and the conversion aborts with
   "Unknown error code '125'". Removing that flag by hand and running the image
   directly under Docker's `runc` then fails inside the container instead, at the
   gVisor re-exec step:

   ```
   Error executing inside namespace: re-executing self:
   fork/exec /proc/self/exe: operation not permitted
   ```

   The image is designed to set up its gVisor sandbox under Podman's rootless
   security model, and Docker's default runtime does not provide the same
   conditions.

So the tool installs and reaches the point of starting the sandbox container, but
the actual page-rendering conversion never runs here. Podman is required, and
Podman cannot be installed on this host without root.

## Supported input types (verified) and adapter legitimacy

Dangerzone's declared input types include images, read directly from
`get_supported_extensions()` in the source: `.pdf`, the office formats
(`.docx`, `.xlsx`, `.pptx`, ODF, `.epub`), and images `.jpg`, `.jpeg`, `.gif`,
`.png`, `.tif`, `.tiff`, `.bmp`, `.pnm`, `.pbm`, `.ppm`, `.svg`. The CLI does not
gate on the extension; `Document.validate_input_filename` only checks that the
file opens. Every accepted input is rendered to pixels in the sandbox, and the
output is always a PDF (`-safe.pdf`).

Two consequences for the adapter:

- The image is genuinely legitimate. Feeding a single-page PDF built from the
  image sends it through Dangerzone's real doc-to-pixels path: the page is
  rendered to raw RGB inside the sandbox and a clean PDF is rebuilt from those
  pixels. This is the same pipeline any document goes through. Rasterizing that
  PDF back with `pdftoppm` and resizing to the original dimensions recovers an
  image. No part of this fakes or approximates Dangerzone's behavior.
- The PDF wrapping is not strictly required. Because Dangerzone accepts `.png`
  and other raster formats natively, the adapter could pass the image file
  directly and get the same kind of sanitized PDF, then rasterize that. The
  current PDF-wrap approach is still valid; the native-image path is just one
  fewer step. Either way the output is a PDF that must be rasterized back, so the
  `pdftoppm` stage stays.

## Exact invocation and the config value

The adapter runs, per image:

```
dangerzone-cli --output-filename <safe.pdf> <in.pdf>
pdftoppm -png -singlefile -r 150 <safe.pdf> <page>
```

The experiment reads the command from `configs/experiment.yaml` under
`tools.dangerzone_cmd`. `src/evaluation/runner.py` passes that value straight to
`DangerzoneDefense(cli=...)`, so the config key is what actually takes effect;
the `DANGERZONE_CMD` environment variable in the adapter is only a fallback for
when no `cli` is passed. On a host where Dangerzone is installed from the `.deb`
alongside Podman, the value is simply:

```yaml
tools:
  dangerzone_cmd: "dangerzone-cli"
```

For a source install like the one above, point it at the venv entry point, for
example `dz312/bin/dangerzone-cli`, and make sure `share/dangerzone` and a
vendored `cosign` are in place. In both cases Podman must be present, since that
is the only runtime the tool drives on Linux.

## Runtime and resource use

- Install from source: about two minutes, dominated by downloading PySide6
  (roughly 250 MB of wheels).
- Container image: `ghcr.io/freedomofpress/dangerzone/v1`, about 1.58 GB pulled,
  cosign-verified against the embedded `freedomofpress-dangerzone.pub` key.
- Per-image conversion runtime could not be measured here, because the sandbox
  container does not start under Docker. It is not the sub-second numbers seen in
  the failed attempts above; those stop at the flag-rejection step before any
  rendering. Dangerzone runs two container invocations per document
  (doc-to-pixels, then pixels-to-PDF), so on a Podman host the realistic cost is
  on the order of seconds to tens of seconds per image, plus a one-time image
  load. This matters for the experiment: at up to 200 images this is minutes of
  wall time, and it should be measured on the actual Podman host before the full
  run is scheduled.

## Failure modes observed

- No runtime configured: `PodmanNotInstalled`, then a `ContainerException` saying
  the Podman binary is missing.
- Source install without `share/`: `FileNotFoundError` on
  `share/dangerzone/ocr-languages.json`.
- `dangerzone-image upgrade` without a vendored cosign: aborts with
  `No such file or directory: .../share/dangerzone/vendor/cosign`.
- Docker as runtime, image present but untagged: "No container image found",
  because `docker images --format '{{.Digest}}'` is empty under the containerd
  image store.
- Docker as runtime, image tagged: `docker: --userns: invalid USER mode`, exit
  125, conversion aborts.

## Note on the adapter

`src/defenses/dangerzone.py` is functionally correct. Its invocation
(`--output-filename` plus a positional input, then `pdftoppm`) matches the real
`dangerzone-cli` interface verified here, and the image-to-PDF-to-image approach
is a legitimate use of the tool. No code bug was found, so the file was not
changed.

One inaccuracy is worth fixing in the comments when someone next touches the
file: the module docstring says the tool "needs a container engine (Docker
present in this project's cloud env)". Dangerzone 0.11.0 does not use Docker on
Linux; it requires Podman. The wording should say Podman so nobody plans around
Docker. This is a comment correction, not a behavior change, so it is left for a
deliberate edit rather than changed as part of this investigation.

## What it would take to unblock

Run the experiment's Dangerzone condition on a host where Podman is available:
install the `dangerzone` `.deb` (which pulls in Podman and, on first use, the signed
image) with root. The Colab and cloud-GPU paths provide root, so this is a
host-provisioning step, not a code problem. Verified working on Colab (Ubuntu 24.04
"noble", Podman 4.9.3); the FPF apt repo has both `noble` and `jammy`. The signing
key comes from a keyserver, not a hosted file (a hosted-file URL 404s):

```sh
apt-get update && apt-get install -y ca-certificates curl gnupg podman
install -dm755 /etc/apt/keyrings
# Use keyserver.ubuntu.com, not keys.openpgp.org: the repo Release is signed by the
# signing subkey (04CABEB5DD76BACF2BD43D2FF3ACC60F62EA51CB), and keys.openpgp.org
# strips subkeys, causing "NO_PUBKEY F3ACC60F62EA51CB ... repository is not signed".
gpg --keyserver hkps://keyserver.ubuntu.com --no-default-keyring --no-permission-warning \
    --homedir "$(mktemp -d)" \
    --keyring gnupg-ring:/etc/apt/keyrings/fpf-apt-tools-archive-keyring.gpg \
    --recv-keys DE28AB241FA48260FAC9B8BAA7C9B38522604281
. /etc/os-release
echo "deb [signed-by=/etc/apt/keyrings/fpf-apt-tools-archive-keyring.gpg] \
https://packages.freedom.press/apt-tools-prod ${VERSION_CODENAME} main" \
    > /etc/apt/sources.list.d/fpf-apt-tools.list
apt-get update && apt-get install -y dangerzone
dangerzone-cli --version   # 0.11.0
```

This is wired into `scripts/colab_bootstrap.py` and the notebook (best-effort, so a
failure skips D5 rather than crashing the run). With `tools.dangerzone_cmd:
"dangerzone-cli"`, re-run and record the measured per-image runtime. The container
image (~1.6 GB) pulls and cosign-verifies on the first sanitize.

## STATUS: BLOCKED

The real Dangerzone 0.11.0 was installed and reached the point of launching its
sandbox container, and the signed container image was pulled and cosign-verified.
It could not sanitize the test image in this environment. Dangerzone requires
Podman on Linux; this host has only Docker and no way to install Podman without
root. Forcing Docker fails because Docker rejects the Podman-only `--userns
nomap` flag (exit 125) and the gVisor image will not run under Docker's default
runtime. The blocker is the container engine, not the adapter.
