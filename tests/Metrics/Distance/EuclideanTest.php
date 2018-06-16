<?php

use Rubix\ML\Metrics\Distance\Euclidean;
use Rubix\ML\Metrics\Distance\Distance;
use PHPUnit\Framework\TestCase;

class EuclideanTest extends TestCase
{
    protected $kernel;

    public function setUp()
    {
        $this->kernel = new Euclidean();
    }

    public function test_build_distance_function()
    {
        $this->assertTrue($this->kernel instanceof Euclidean);
        $this->assertTrue($this->kernel instanceof Distance);
    }

    public function test_compute_distance()
    {
        $this->assertEquals(9.43, round($this->kernel->compute(['x' => 2, 'y' => 1], ['x' => 7, 'y' => 9]), 2));
    }
}
