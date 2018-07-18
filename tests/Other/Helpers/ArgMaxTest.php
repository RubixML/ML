<?php

namespace Rubix\Tests\Other\Helpers;

use Rubix\ML\Other\Helpers\ArgMax;
use PHPUnit\Framework\TestCase;

class ArgMaxTest extends TestCase
{
    protected $values;

    protected $outcome;

    public function setUp()
    {
        $this->values = [
            'yes' => 0.8, 'no' => 0.2, 'maybe' => 0.0,
        ];

        $this->outcome = 'yes';
    }

    public function test_compute()
    {
        $value = ArgMax::compute($this->values);

        $this->assertEquals($this->outcome, $value);
    }
}
