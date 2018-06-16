<?php

use Rubix\ML\Metrics\Distance\Diagonal;
use Rubix\ML\Metrics\Distance\Distance;
use PHPUnit\Framework\TestCase;

class DiagonalTest extends TestCase
{
    protected $kernel;

    public function setUp()
    {
        $this->kernel = new Diagonal();
    }

    public function test_build_distance_function()
    {
        $this->assertTrue($this->kernel instanceof Diagonal);
        $this->assertTrue($this->kernel instanceof Distance);
    }

    public function test_compute_distance()
    {
        $this->assertEquals(8.0, round($this->kernel->compute(['x' => 2, 'y' => 1], ['x' => 7, 'y' => 9]), 2));
    }
}
