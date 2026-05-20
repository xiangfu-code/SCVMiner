// SPDX-License-Identifier: MIT

contract NoPragma {
    uint256 private counter;

    function start() public {
        counter = counter + 1;
        finish();
    }

    function finish() internal {}
}
