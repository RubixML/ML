<?php

namespace Rubix\Tests\Kernels\Distance;

use Rubix\ML\Kernels\Distance\CosineSimilarity;
use Rubix\ML\Kernels\Distance\Distance;
use PHPUnit\Framework\TestCase;

class CosineSimilarityTest extends TestCase
{
    protected $kernel;

    public function setUp()
    {
        $this->kernel = new CosineSimilarity();
    }

    public function test_build_distance_function()
    {
        $this->assertTrue($this->kernel instanceof CosineSimilarity);
        $this->assertTrue($this->kernel instanceof Distance);
    }

    public function test_compute_distance()
    {
        $this->assertEquals(0.1, round($this->kernel->compute(['x' => 2, 'y' => 1], ['x' => 7, 'y' => 9]), 2));
    }
}
