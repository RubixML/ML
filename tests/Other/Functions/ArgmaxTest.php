<?php

namespace Rubix\ML\Tests\Other\Functions;

use Rubix\ML\Other\Functions\Argmax;
use PHPUnit\Framework\TestCase;

class ArgmaxTest extends TestCase
{
    public function setUp()
    {
        //
    }

    public function test_compute()
    {
        $value = Argmax::compute(['yes' => 0.8, 'no' => 0.2, 'maybe' => 0.0]);

        $this->assertEquals('yes', $value);
    }
}
