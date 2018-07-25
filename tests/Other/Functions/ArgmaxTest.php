<?php

namespace Rubix\Tests\Other\Functions;

use Rubix\ML\Other\Functions\Argmax;
use PHPUnit\Framework\TestCase;

class ArgmaxTest extends TestCase
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
        $value = Argmax::compute($this->values);

        $this->assertEquals($this->outcome, $value);
    }
}
