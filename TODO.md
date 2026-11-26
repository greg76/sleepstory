# Security TODO

## Pin the MLX-Audio dependency

`requirements.txt` installs `mlx-audio` from the upstream `main` branch. That
branch can change between installations, so the same requirements file can
install different code over time.

- [ ] Identify and test a specific upstream commit that supports the Ming 3
  model, then pin the dependency to its full commit hash. Update the setup
  notes to match.

Until this is fixed, a new installation can pick up an incompatible change or
compromised upstream code. A full commit hash makes the selected source
revision reproducible, although its dependencies still need their own version
controls.

## Protect audio output from path replacement

`generate_audio.py` reserves an output name by creating a file exclusively,
then closes it before FFmpeg opens that path with overwrite enabled. In a
directory writable by another user, that user can replace the path during
generation and redirect FFmpeg's write. Cleanup can also remove a replacement
at that path after a failure.

- [ ] Encode to a file in a private temporary directory on the destination
  filesystem, then publish it with an atomic operation that fails if the
  destination already exists. Preserve automatic naming and cleanup behavior,
  including when concurrent runs choose the same name.

Until this is fixed, running the generator in a shared writable directory can
overwrite a file accessible to the process or remove a replacement path. The
normal case of a directory writable only by the current user has a smaller
exposure.
