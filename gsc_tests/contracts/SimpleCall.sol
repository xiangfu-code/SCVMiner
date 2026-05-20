// SPDX-License-Identifier: MIT
pragma solidity 0.8.35;

contract SimpleCall {
    address private owner;

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }

    function entry() public onlyOwner {
        stepOne();
        stepTwo();
    }

    function stepOne() internal {
        leaf();
    }

    function stepTwo() internal {}

    function leaf() private {}
}
