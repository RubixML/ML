<?php

namespace Rubix\ML\Tests\Kernels\Distance;

use Rubix\ML\Kernels\Distance\Gower;
use Rubix\ML\Kernels\Distance\NaNSafe;
use Rubix\ML\Kernels\Distance\Distance;
use PHPUnit\Framework\TestCase;
use Generator;

class GowerTest extends TestCase
{
    /**
     * @var \Rubix\ML\Kernels\Distance\Gower
     */
    protected $kernel;

    public function setUp() : void
    {
        $this->kernel = new Gower(1.);
    }

    public function test_build_distance_kernel() : void
    {
        $this->assertInstanceOf(Gower::class, $this->kernel);
        $this->assertInstanceOf(NaNSafe::class, $this->kernel);
        $this->assertInstanceOf(Distance::class, $this->kernel);
    }

    /**
     * @dataProvider compute_provider
     */
    public function test_compute(array $a, array $b, $expected) : void
    {
        $distance = $this->kernel->compute($a, $b);

        $this->assertGreaterThanOrEqual(0., $distance);
        $this->assertEquals($expected, $distance);
    }

    public function compute_provider() : Generator
    {
        yield [['toast', 1., 0.5, NAN], ['pretzels', 1., 0.2, 0.1], 0.43333333333333335];

        yield [[0., 1., 0.5, 'ham'], [0.1, 0.9, 0.4, 'ham'], 0.07499999999999998];
        
        yield [[1, NAN, 1], [1, NAN, 1], 0.0];
    }
}
