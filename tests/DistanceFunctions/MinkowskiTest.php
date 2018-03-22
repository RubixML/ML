<?php

use Rubix\Engine\DistanceFunctions\Minkowski;
use Rubix\Engine\DistanceFunctions\DistanceFunction;
use Rubix\Engine\GraphNode;
use PHPUnit\Framework\TestCase;

class MinkowskiTest extends TestCase
{
    protected $distanceFunction;

    public function setUp()
    {
        $this->distanceFunction = new Minkowski();
    }

    public function test_build_distance_function()
    {
        $this->assertTrue($this->distanceFunction instanceof Minkowski);
        $this->assertTrue($this->distanceFunction instanceof DistanceFunction);
    }

    public function test_compute_distance()
    {
        $start = new GraphNode(1, ['x' => 2, 'y' => 1]);

        $end = new GraphNode(2, ['x' => 7, 'y' => 9]);

        $this->assertEquals(8.6, round($this->distanceFunction->compute($start, $end, ['x', 'y']), 2));
    }
}
