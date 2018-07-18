<?php

namespace Rubix\Tests\Other\Helpers;

use Rubix\ML\Other\Helpers\RandomParams;
use PHPUnit\Framework\TestCase;

class RandomParamsTest extends TestCase
{
    public function setUp()
    {
        //
    }

    public function test_generate_ints()
    {
        $values = RandomParams::ints(1, 100, 3);

        $this->assertThat($values[0], $this->logicalAnd($this->greaterThanOrEqual(1), $this->lessThanOrEqual(100)));
        $this->assertThat($values[1], $this->logicalAnd($this->greaterThanOrEqual(1), $this->lessThanOrEqual(100)));
        $this->assertThat($values[2], $this->logicalAnd($this->greaterThanOrEqual(1), $this->lessThanOrEqual(100)));
    }

    public function test_generate_floats()
    {
        $values = RandomParams::floats(1, 100, 3);

        $this->assertThat($values[0], $this->logicalAnd($this->greaterThanOrEqual(1), $this->lessThanOrEqual(100)));
        $this->assertThat($values[1], $this->logicalAnd($this->greaterThanOrEqual(1), $this->lessThanOrEqual(100)));
        $this->assertThat($values[2], $this->logicalAnd($this->greaterThanOrEqual(1), $this->lessThanOrEqual(100)));
    }
}
