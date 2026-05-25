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
