<?php

namespace Rubix\ML\Tests\Other\Functions;

use Rubix\ML\Other\Functions\Sign;
use PHPUnit\Framework\TestCase;

class SignTest extends TestCase
{
    public function setUp()
    {
        //
    }

    public function test_compute()
    {
        $this->assertEquals(1, Sign::compute(2000));
        $this->assertEquals(0, Sign::compute(0));
        $this->assertEquals(-1, Sign::compute(-0.5));
    }
}
