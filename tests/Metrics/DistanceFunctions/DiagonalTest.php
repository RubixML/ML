<?php

use Rubix\Engine\Metrics\DistanceFunctions\Diagonal;
use Rubix\Engine\Metrics\DistanceFunctions\DistanceFunction;
use PHPUnit\Framework\TestCase;

class DiagonalTest extends TestCase
{
    protected $distanceFunction;

    public function setUp()
    {
        $this->distanceFunction = new Diagonal();
    }

    public function test_build_distance_function()
    {
        $this->assertTrue($this->distanceFunction instanceof Diagonal);
        $this->assertTrue($this->distanceFunction instanceof DistanceFunction);
    }

    public function test_compute_distance()
    {
        $this->assertEquals(8.0, round($this->distanceFunction->compute(['x' => 2, 'y' => 1], ['x' => 7, 'y' => 9]), 2));
    }
}
