# RoCEnante proxy provenance

`proxy.c` is copied from the supplied Sparknet snapshot:

- Snapshot: `/home/staticduo/tmp/deepseek-work/nvfp4-migration-20261023/transport-diff`
- Upstream: `local-inference-lab/b12x`, revision
  `f8069b2c0be1311df3b112591c6b8876a843f8be`
- Local B12X patch head: `abe4b2afd7b60e2699bf4a62ac8543ed63d1d937`
- Snapshot tree: `cd615bd69ed1a99d6b59217329a3ff429b5277fe`
- Wire ABI: 10
- License: Apache-2.0; see `LICENSE` and applicable attribution in `NOTICE`.
- The copied source is adapted for this backend: only `world == 3` and direct
  topology (`ring4 == 0`) are accepted. This deliberately excludes the
  snapshot's four-rank relay/mesh branches; record the post-adaptation digest
  in the build provenance at integration.

The source does not claim compatibility with the Python/Torch runtime.
`roce_abi_version()` remains 10 so the ABI shared with the host adapter is
explicit. Any future wire/layout ABI change must bump it and update the
independent tests.

The source inherits the upstream B12X RoCEnante copyright and attribution
notices. Sparknet's `NOTICE` identifies upstream authors and contributors
separately from its local routing changes. This adaptation must retain those
notices and the Apache license in source distributions.

## Native adaptation

The Strata adaptation uses a named C context, atomically publishes the completed
sequence only after draining the local CQ, and makes host-read diagnostic counters
atomic. The wire layout is unchanged; this does not claim binary equivalence with
the source snapshot. CMake records the adapted proxy source SHA-256 and rebuilds
that identity when the source changes.

Adapted proxy SHA-256: `df3b2e3225877cf4c70e18b72c9233b470d60d2e5ace2105ac343069cf0ae4c8`.
