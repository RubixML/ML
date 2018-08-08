<?php

namespace Rubix\Tests\Other\Functions;

use Rubix\ML\Other\Functions\MeanVar;
use PHPUnit\Framework\TestCase;

class MeanVarTest extends TestCase
{
    protected $values;

    protected $outcome;

    public function setUp()
    {
        $this->values = [
            0.5, 0.4, 0.9, 1.0, 0.2, 0.9, 0.1, 0.5, 0.7,
        ];

        $this->outcome = [0.5777777777777778, 0.09061728395061729];
    }

    public function test_compute()
    {
        $value = MeanVar::compute($this->values);

        $this->assertEquals($this->outcome[0], $value[0], '', 1e-3);
        $this->assertEquals($this->outcome[1], $value[1], '', 1e-3);
    }
}
