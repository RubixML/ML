<?php

use Rubix\Engine\Metrics\DistanceFunctions\Ellipsoidal;
use Rubix\Engine\Metrics\DistanceFunctions\DistanceFunction;
use PHPUnit\Framework\TestCase;

class ElliposoidalTest extends TestCase
{
    protected $distanceFunction;

    public function setUp()
    {
        $this->distanceFunction = new Ellipsoidal();
    }

    public function test_build_distance_function()
    {
        $this->assertTrue($this->distanceFunction instanceof Ellipsoidal);
        $this->assertTrue($this->distanceFunction instanceof DistanceFunction);
    }

    public function test_compute_distance()
    {
        $this->assertEquals(0.61, round($this->distanceFunction->compute(['x' => 2, 'y' => 3, 'z' => 5], ['x' => 7, 'y' => 9, 'z' => 4]), 2));
    }
}
