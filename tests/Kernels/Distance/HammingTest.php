<?php

namespace Rubix\ML\Tests\Kernels\Distance;

use Rubix\ML\Kernels\Distance\Hamming;
use Rubix\ML\Kernels\Distance\Distance;
use PHPUnit\Framework\TestCase;
use Generator;

class HammingTest extends TestCase
{
    protected $kernel;

    public function setUp()
    {
        $this->kernel = new Hamming();
    }

    public function test_build_distance_kernel()
    {
        $this->assertInstanceOf(Hamming::class, $this->kernel);
        $this->assertInstanceOf(Distance::class, $this->kernel);
    }

    /**
     * @dataProvider compute_provider
     */
    public function test_compute(array $a, array $b, $expected)
    {
        $distance = $this->kernel->compute($a, $b);

        $this->assertGreaterThanOrEqual(0., $distance);
        $this->assertEquals($expected, $distance);
    }

    public function compute_provider() : Generator
    {
        yield [['soup', 'turkey', 'broccoli', 'cake'], ['salad', 'turkey', 'broccoli', 'pie'], 2.0];
        yield [['salad', 'ham', 'peas', 'jello'], ['bread', 'steak', 'potato', 'cake'], 4.0];
        yield [['toast', 'eggs', 'bacon'], ['toast', 'eggs', 'bacon'], 0.0];
    }
}
