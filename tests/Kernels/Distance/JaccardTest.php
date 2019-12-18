<?php

namespace Rubix\ML\Tests\Kernels\Distance;

use Rubix\ML\Kernels\Distance\Jaccard;
use Rubix\ML\Kernels\Distance\Distance;
use PHPUnit\Framework\TestCase;
use Generator;

class JaccardTest extends TestCase
{
    /**
     * @var \Rubix\ML\Kernels\Distance\Jaccard
     */
    protected $kernel;

    public function setUp() : void
    {
        $this->kernel = new Jaccard();
    }

    public function test_build_distance_kernel() : void
    {
        $this->assertInstanceOf(Jaccard::class, $this->kernel);
        $this->assertInstanceOf(Distance::class, $this->kernel);
    }

    /**
     * @param (int|float)[] $a
     * @param (int|float)[] $b
     * @param float $expected
     *
     * @dataProvider compute_provider
     */
    public function test_compute(array $a, array $b, float $expected) : void
    {
        $distance = $this->kernel->compute($a, $b);

        $this->assertGreaterThanOrEqual(0., $distance);
        $this->assertEquals($expected, $distance);
    }

    /**
     * @return \Generator<array>
     */
    public function compute_provider() : Generator
    {
        yield [[2, 1, 4, 0], [-2, 1, 8, -2],  0.9090909090909091];

        yield [[7.4, -2.5], [0.01, -1], 1.3890625];
        
        yield [[1000, -2000, 3000], [1000, -2000, 3000], 0.0];
    }
}
