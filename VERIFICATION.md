# Local verification

Verified on this Apple Silicon Mac, October 22, 2026, using native Python 3.12 and PyTorch 2.5.1.

- Downloaded and extracted the authors' apple2orange archive; counts and SHA-256 are in `data/dataset.json`.
- Three automated tests passed: cycle gradients through both generators, image preprocessing/subset selection, and detached bounded replay buffers.
- Trained the default 128px, 32-channel, six-block model on **MPS** for two epochs with two updates each, using four images per domain. Losses were finite, and checkpoints and previews were written to `runs/smoke/`.
- Loaded the checkpoint and generated four held-out translations in each direction on **MPS**, in `outputs/smoke_ab/` and `outputs/smoke_ba/`.
- Loaded model/optimizer/replay/RNG checkpoint state through the resume command. Since both configured smoke epochs were already complete, this was a checkpoint-loading check, not an additional resumed training epoch.
- Python compilation and shell syntax checks passed.

The app sandbox hides MPS access; the GPU checks succeeded when run outside it. Run the documented commands from your regular Terminal.

At the initial verification, only the smoke experiment had been run. The default training budget was subsequently reduced to 10 epochs. Smoke outputs are pipeline checks, not useful trained models. `bash run.sh` starts the actual experiment in `runs/apple2orange/`, or resumes it if that checkpoint already exists, then performs inference in both directions.
