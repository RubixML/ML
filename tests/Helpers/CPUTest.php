<?php

namespace Rubix\ML\Tests\Helpers;

use Rubix\ML\Helpers\CPU;
use PHPUnit\Framework\TestCase;

/**
 * @group Helpers
 * @covers \Rubix\ML\Helpers\CPU
 */
class CPUTest extends TestCase
{
    /**
     * @before
     */
    protected function setUp() : void
    {
        ini_set('precision', '28');
    }

    /**
     * @test
     */
    public function epsilon() : void
    {
        $epsilon = CPU::epsilon();

        $this->assertLessThan(1.0, $epsilon);
        $this->assertGreaterThan(0.0, $epsilon);
    }
}
