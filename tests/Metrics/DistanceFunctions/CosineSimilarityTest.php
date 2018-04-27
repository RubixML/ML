<?php

use Rubix\Engine\Metrics\DistanceFunctions\CosineSimilarity;
use Rubix\Engine\Metrics\DistanceFunctions\DistanceFunction;
use PHPUnit\Framework\TestCase;

class CosineSimilarityTest extends TestCase
{
    protected $distanceFunction;

    public function setUp()
    {
        $this->distanceFunction = new CosineSimilarity();
    }

    public function test_build_distance_function()
    {
        $this->assertTrue($this->distanceFunction instanceof CosineSimilarity);
        $this->assertTrue($this->distanceFunction instanceof DistanceFunction);
    }

    public function test_compute_distance()
    {
        $this->assertEquals(0.1, round($this->distanceFunction->compute(['x' => 2, 'y' => 1], ['x' => 7, 'y' => 9]), 2));
    }
}
