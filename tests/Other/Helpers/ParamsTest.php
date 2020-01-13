<?php

namespace Rubix\ML\Tests\Other\Helpers;

use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Classifiers\KNearestNeighbors;
use PHPUnit\Framework\TestCase;

/**
 * @group Helpers
 * @covers \Rubix\ML\Other\Helpers\Params
 */
class ParamsTest extends TestCase
{
    /**
     * @test
     */
    public function ints() : void
    {
        $values = Params::ints(0, 100, 5);

        $this->assertThat($values[0], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[1], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[2], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[3], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[4], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
    }
    
    /**
     * @test
     */
    public function floats() : void
    {
        $values = Params::floats(0, 100, 5);

        $this->assertThat($values[0], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[1], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[2], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[3], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
        $this->assertThat($values[4], $this->logicalAnd($this->greaterThanOrEqual(0), $this->lessThanOrEqual(100)));
    }
    
    /**
     * @test
     */
    public function grid() : void
    {
        $values = Params::grid(0, 100, 5);

        $this->assertEquals(0, $values[0]);
        $this->assertEquals(25, $values[1]);
        $this->assertEquals(50, $values[2]);
        $this->assertEquals(75, $values[3]);
        $this->assertEquals(100, $values[4]);
    }
    
    /**
     * @test
     */
    public function args() : void
    {
        $expected = ['k', 'weighted', 'kernel'];

        $this->assertEquals($expected, Params::args(new KNearestNeighbors()));
    }
    
    /**
     * @test
     */
    public function stringify() : void
    {
        $expected = 'learning_rate=0.1 alpha=0.0001';

        $this->assertEquals($expected, Params::stringify([
            'learning_rate' => 0.1,
            'alpha' => 1e-4,
        ]));
    }
    
    /**
     * @test
     */
    public function shortName() : void
    {
        $this->assertEquals('KNearestNeighbors', Params::shortName(new KNearestNeighbors()));
        $this->assertEquals('KNearestNeighbors', Params::shortName(KNearestNeighbors::class));
    }
}
