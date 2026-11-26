# Meta MHR-to-SMPL conversion files

The files in this directory are copied without modification from
[`facebookresearch/MHR`](https://github.com/facebookresearch/MHR), commit
`1b358558f3644a472ed9d79fddb0db19dce1a0e9`, under the included Apache 2.0
license.

Vendored files and SHA-256 digests:

- `pytorch_fitting.py`:
  `2eb4fab49a16205b5eefe781b1d9300fc0afb8ec4b3a4ec02a4884097b6ca26c`
- `assets/mhr2smpl_mapping.npz`:
  `4bfe2fd6abb598f38ba3a3aa5506ac45b4efb3cdaed7fe4757cfbbf20bbe6ac7`
- `LICENSE`:
  `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`

`workers/sam3d_body_worker.py` verifies the first two digests, transfers the SAM 3D
Body MHR mesh with the official 6,890-vertex SMPL mapping (SMPL and SMPL-H share
this topology), and fits a neutral SMPL-H model with the unmodified
`PyTorchSMPLFitting` class, using its 63-D body + articulated-hand
parameterization.
