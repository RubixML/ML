<?php

use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\Kernels\Distance\Distance;
use PHPUnit\Framework\TestCase;

class ManhattanTest extends TestCase
{
    protected $kernel;

    public function setUp()
    {
        $this->kernel = new Manhattan();
    }

    public function test_build_distance_function()
    {
        $this->assertTrue($this->kernel instanceof Manhattan);
        $this->assertTrue($this->kernel instanceof Distance);
    }

    public function test_compute_distance()
    {
        $this->assertEquals(13.0, $this->kernel->compute(['x' => 2, 'y' => 1], ['x' => 7, 'y' => 9]));
    }
}
