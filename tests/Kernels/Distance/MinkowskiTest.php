<?php

use Rubix\ML\Kernels\Distance\Minkowski;
use Rubix\ML\Kernels\Distance\Distance;
use PHPUnit\Framework\TestCase;

class MinkowskiTest extends TestCase
{
    protected $kernel;

    public function setUp()
    {
        $this->kernel = new Minkowski();
    }

    public function test_build_distance_function()
    {
        $this->assertTrue($this->kernel instanceof Minkowski);
        $this->assertTrue($this->kernel instanceof Distance);
    }

    public function test_compute_distance()
    {
        $this->assertEquals(8.6, round($this->kernel->compute(['x' => 2, 'y' => 1], ['x' => 7, 'y' => 9]), 2));
    }
}
