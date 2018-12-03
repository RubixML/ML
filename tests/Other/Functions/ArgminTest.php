<?php

namespace Rubix\ML\Tests\Other\Functions;

use Rubix\ML\Other\Functions\Argmin;
use PHPUnit\Framework\TestCase;

class ArgminTest extends TestCase
{
    public function setUp()
    {
        //
    }

    public function test_compute()
    {
        $value = Argmin::compute(['yes' => 0.8, 'no' => 0.2, 'maybe' => 0.0]);

        $this->assertEquals('maybe', $value);
    }
}
