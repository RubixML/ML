<?php

use Rubix\Engine\Graph\DistanceFunctions\Manhattan;
use Rubix\Engine\Graph\DistanceFunctions\DistanceFunction;
use PHPUnit\Framework\TestCase;

class ManhattanTest extends TestCase
{
    protected $distanceFunction;

    public function setUp()
    {
        $this->distanceFunction = new Manhattan();
    }

    public function test_build_distance_function()
    {
        $this->assertTrue($this->distanceFunction instanceof Manhattan);
        $this->assertTrue($this->distanceFunction instanceof DistanceFunction);
    }

    public function test_compute_distance()
    {
        $this->assertEquals(13.0, $this->distanceFunction->compute(['x' => 2, 'y' => 1], ['x' => 7, 'y' => 9]));
    }
}
