// SPDX-License-Identifier: MIT
pragma solidity 0.8.35;

library MathLib {
    function inc(uint256 value) internal pure returns (uint256) {
        return value + 1;
    }
}

interface IExternal {
    function ping() external;
}

contract MixedKinds {
    uint256 public value;

    function set(uint256 newValue) public {
        value = MathLib.inc(newValue);
        afterSet();
    }

    function afterSet() internal {}
}
