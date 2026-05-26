# Smart Contract Vulnerability Dataset

This directory contains Solidity smart contract datasets for three vulnerability types:
reentrancy, timestamp dependency, and transaction state dependency. The contracts are
organized by vulnerability type and label.

## Source

The dataset is from the SCVHUNTER study:

```bibtex
@inproceedings{luo2024scvhunter,
  title={Scvhunter: Smart contract vulnerability detection based on heterogeneous graph attention network},
  author={Luo, Feng and Luo, Ruijie and Chen, Ting and Qiao, Ao and He, Zheyuan and Song, Shuwei and Jiang, Yu and Li, Sixing},
  booktitle={Proceedings of the IEEE/ACM 46th international conference on software engineering},
  pages={1--13},
  year={2024}
}
```

## Directory Structure

```text
datasets/
├── reentrancy/
│   ├── dependency/      # vulnerable contracts
│   └── undependency/    # non-vulnerable contracts
├── timestamp/
│   ├── dependency/      # vulnerable contracts
│   └── undependency/    # non-vulnerable contracts
└── origin/
    ├── dependency/      # vulnerable contracts
    └── undependency/    # non-vulnerable contracts
```

`origin` stores the transaction state dependency samples. Each contract file uses the
`.sol` suffix.

## Statistics

| Vulnerability | Total | Vulnerable | Non-vulnerable |
| --- | ---: | ---: | ---: |
| Reentrancy | 300 | 203 | 97 |
| Timestamp | 104 | 60 | 44 |
| Transaction State | 300 | 180 | 120 |

The `dependency` directories contain vulnerable contracts, and the `undependency`
directories contain non-vulnerable contracts.

## Patched Samples

Some dataset contracts were minimally patched so they can be compiled by Slither
for graph generation. For each patched Solidity file, the original source is kept
next to it with the `.sol.orig` suffix.

- `origin/undependency/0x1a3f7583c0af24ef78cdb1a1eb48d957df793824.sol`
  - Original backup: `origin/undependency/0x1a3f7583c0af24ef78cdb1a1eb48d957df793824.sol.orig`
  - Reason: extracted one Kyber call into a helper to avoid a Solidity 0.5.17
    `Stack too deep` compiler error.
- `origin/undependency/0x6c8f2a135f6ed072de4503bd7c4999a1a17f824b.sol`
  - Original backup: `origin/undependency/0x6c8f2a135f6ed072de4503bd7c4999a1a17f824b.sol.orig`
  - Reason: fixed minimal Solidity 0.4.24 compatibility issues: missing
    modifier semicolons, trailing struct-literal commas, fallback return
    values, enum conversion, and payable constructor creation.
- `timestamp/dependency/0x1eee197a40ea98185535f0e7d93d09be6bfcd5cb.sol`
  - Original backup: `timestamp/dependency/0x1eee197a40ea98185535f0e7d93d09be6bfcd5cb.sol.orig`
  - Reason: added `pragma solidity 0.4.11;` so this old-style contract compiles
    with the matching compiler.
