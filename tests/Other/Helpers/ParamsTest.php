<?php

namespace Rubix\ML\Tests\Other\Helpers;

use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Classifiers\KNearestNeighbors;
use PHPUnit\Framework\TestCase;

class ParamsTest extends TestCase
{
    public function test_generate_ints() : void
    {
        $values = Params::ints(0, 100, 5);

        $this->assertThat($values[0], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[1], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[2], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[3], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[4], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
    }

    public function test_generate_floats() : void
    {
        $values = Params::floats(0, 100, 5);

        $this->assertThat($values[0], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[1], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[2], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[3], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[4], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
    }

    public function test_generate_grid() : void
    {
        $values = Params::grid(0, 100, 5);

        $this->assertEquals(0, $values[0]);
        $this->assertEquals(25, $values[1]);
        $this->assertEquals(50, $values[2]);
        $this->assertEquals(75, $values[3]);
        $this->assertEquals(100, $values[4]);
    }

    public function test_extract_args() : void
    {
        $expected = ['k', 'weighted', 'kernel'];

        $this->assertEquals($expected, Params::args(new KNearestNeighbors()));
    }

    public function test_stringify() : void
    {
        $expected = 'learning_rate=0.1 alpha=0.0001';

        $this->assertEquals($expected, Params::stringify([
            'learning_rate' => 0.1,
            'alpha' => 1e-4,
        ]));
    }

    public function test_get_short_name() : void
    {
        $this->assertEquals('KNearestNeighbors', Params::shortName(new KNearestNeighbors()));
        $this->assertEquals('KNearestNeighbors', Params::shortName(KNearestNeighbors::class));
    }
}
