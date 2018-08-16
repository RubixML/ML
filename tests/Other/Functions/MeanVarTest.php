<?php

namespace Rubix\Tests\Other\Functions;

use Rubix\ML\Other\Functions\MeanVar;
use PHPUnit\Framework\TestCase;

class MeanVarTest extends TestCase
{
    protected $values;

    public function setUp()
    {
        $this->values = [
            0.5, 0.4, 0.9, 1.0, 0.2, 0.9, 0.1, 0.5, 0.7,
        ];
    }

    public function test_compute()
    {
        list($mean, $variance) = MeanVar::compute($this->values);

        $this->assertEquals(0.5777777777777778, $mean, '', 1e-3);
        $this->assertEquals(0.09061728395061729, $variance, '', 1e-3);
    }
}
