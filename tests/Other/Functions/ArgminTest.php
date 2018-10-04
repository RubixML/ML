<?php

namespace Rubix\ML\Tests\Other\Functions;

use Rubix\ML\Other\Functions\Argmin;
use PHPUnit\Framework\TestCase;

class ArgminTest extends TestCase
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
        $value = Argmin::compute($this->values);

        $this->assertEquals($this->outcome, $value);
    }
}
