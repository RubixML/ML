<?php

namespace Rubix\Tests\Other\Helpers;

use Rubix\ML\Other\Helpers\ArgMin;
use PHPUnit\Framework\TestCase;

class ArgMinTest extends TestCase
{
    protected $values;

    protected $outcome;

    public function setUp()
    {
        $this->values = [
            'yes' => 0.8, 'no' => 0.2, 'maybe' => 0.0,
        ];

        $this->outcome = 'maybe';
    }

    public function test_compute()
    {
        $value = ArgMin::compute($this->values);

        $this->assertEquals($this->outcome, $value);
    }
}
