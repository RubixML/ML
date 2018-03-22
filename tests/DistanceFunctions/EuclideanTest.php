<?php

use Rubix\Engine\DistanceFunctions\Euclidean;
use Rubix\Engine\DistanceFunctions\DistanceFunction;
use Rubix\Engine\Graph\GraphNode;
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
        $start = new GraphNode(1, ['x' => 2, 'y' => 1]);

        $end = new GraphNode(2, ['x' => 7, 'y' => 9]);

        $this->assertEquals(9.43, round($this->distanceFunction->compute($start, $end, ['x', 'y']), 2));
    }
}
