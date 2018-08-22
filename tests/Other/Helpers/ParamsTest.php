<?php

namespace Rubix\Tests\Other\Helpers;

use Rubix\ML\Other\Helpers\Params;
use PHPUnit\Framework\TestCase;

class ParamsTest extends TestCase
{
    public function setUp()
    {
        //
    }

    public function test_generate_ints()
    {
        $values = Params::ints(0, 100, 5);

        $this->assertThat($values[0], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[1], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[2], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[3], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[4], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
    }

    public function test_generate_floats()
    {
        $values = Params::floats(0, 100, 5);

        $this->assertThat($values[0], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[1], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[2], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[3], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[4], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
    }

    public function test_generate_grid()
    {
        $values = Params::grid(0, 100, 5);

        $this->assertEquals(0, $values[0]);
        $this->assertEquals(25, $values[1]);
        $this->assertEquals(50, $values[2]);
        $this->assertEquals(75, $values[3]);
        $this->assertEquals(100, $values[4]);
    }
}
