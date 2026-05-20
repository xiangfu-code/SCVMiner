// SPDX-License-Identifier: MIT
pragma solidity 0.4.24;

contract Legacy0424 {
    uint256 private total;

    function Legacy0424(uint256 initialTotal) public {
        total = initialTotal;
    }

    function add(uint256 value) public {
        total = total + value;
        afterAdd();
    }

    function afterAdd() internal {}

    function getTotal() public view returns (uint256) {
        return total;
    }
}
