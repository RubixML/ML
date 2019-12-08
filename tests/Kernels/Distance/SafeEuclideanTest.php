<?php

namespace Rubix\ML\Tests\Kernels\Distance;

use Rubix\ML\Kernels\Distance\NaNSafe;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\SafeEuclidean;
use PHPUnit\Framework\TestCase;
use Generator;

class SafeEuclideanTest extends TestCase
{
    /**
     * @var \Rubix\ML\Kernels\Distance\SafeEuclidean
     */
    protected $kernel;

    public function setUp() : void
    {
        $this->kernel = new SafeEuclidean();
    }

    public function test_build_distance_kernel() : void
    {
        $this->assertInstanceOf(SafeEuclidean::class, $this->kernel);
        $this->assertInstanceOf(NaNSafe::class, $this->kernel);
        $this->assertInstanceOf(Distance::class, $this->kernel);
    }

    /**
     * @dataProvider compute_provider
     */
    public function test_compute(array $a, array $b, float $expected) : void
    {
        $distance = $this->kernel->compute($a, $b);

        $this->assertGreaterThanOrEqual(0., $distance);
        $this->assertEquals($expected, $distance);
    }

    public function compute_provider() : Generator
    {
        yield [[2, 1, 4, NAN], [-2, 1, 8, -2],  6.531972647421808];

        yield [[7.4, -2.5, 0.001], [NAN, -1, 0.075], 1.8393515161599752];
        
        yield [[1000, NAN, 3000], [1000, NAN, 3000], 0.0];
    }
}
