<?php

namespace Rubix\Tests;

use Rubix\ML\RandomParams;
use PHPUnit\Framework\TestCase;

class RandomParamsTest extends TestCase
{
    public function setUp()
    {
        //
    }

    public function test_generate_ints()
    {
        $results = RandomParams::ints(1, 100, 3);

        $this->assertThat($results[0], $this->logicalAnd($this->greaterThanOrEqual(1), $this->lessThanOrEqual(100)));
        $this->assertThat($results[1], $this->logicalAnd($this->greaterThanOrEqual(1), $this->lessThanOrEqual(100)));
        $this->assertThat($results[2], $this->logicalAnd($this->greaterThanOrEqual(1), $this->lessThanOrEqual(100)));
    }

    public function test_generate_unique_ints()
    {
        $results = RandomParams::ints(1, 3, 3);

        sort($results);

        $this->assertEquals($results, [1, 2, 3]);
    }

    public function test_generate_floats()
    {
        $results = RandomParams::floats(1, 100, 3);

        $this->assertThat($results[0], $this->logicalAnd($this->greaterThanOrEqual(1), $this->lessThanOrEqual(100)));
        $this->assertThat($results[1], $this->logicalAnd($this->greaterThanOrEqual(1), $this->lessThanOrEqual(100)));
        $this->assertThat($results[2], $this->logicalAnd($this->greaterThanOrEqual(1), $this->lessThanOrEqual(100)));
    }
}
