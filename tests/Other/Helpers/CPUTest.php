<?php

namespace Rubix\ML\Tests\Other\Helpers;

use Rubix\ML\Other\Helpers\CPU;
use PHPUnit\Framework\TestCase;

class CPUTest extends TestCase
{
    public function test_cores()
    {
        $cores = CPU::cores();

        $this->assertInternalType('integer', $cores);
        $this->assertGreaterThanOrEqual(0, $cores);
    }
}
