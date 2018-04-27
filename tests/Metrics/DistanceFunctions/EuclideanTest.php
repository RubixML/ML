<?php

use Rubix\Engine\Metrics\DistanceFunctions\Euclidean;
use Rubix\Engine\Metrics\DistanceFunctions\DistanceFunction;
use PHPUnit\Framework\TestCase;

class EuclideanTest extends TestCase
{
    protected $distanceFunction;

    public function setUp()
    {
        $this->distanceFunction = new Euclidean();
    }

    public function test_build_distance_function()
    {
        $this->assertTrue($this->distanceFunction instanceof Euclidean);
        $this->assertTrue($this->distanceFunction instanceof DistanceFunction);
    }

    public function test_compute_distance()
    {
        $this->assertEquals(9.43, round($this->distanceFunction->compute(['x' => 2, 'y' => 1], ['x' => 7, 'y' => 9]), 2));
    }
}
