<?php

namespace Rubix\ML\Tests\Kernels\Distance;

use Rubix\ML\Kernels\Distance\SparseCosine;
use Rubix\ML\Kernels\Distance\Distance;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Distances
 * @covers \Rubix\ML\Kernels\Distance\Cosine
 */
class SparseCosineTest extends TestCase
{
    /**
     * @var SparseCosine
     */
    protected $kernel;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->kernel = new SparseCosine();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(SparseCosine::class, $this->kernel);
        $this->assertInstanceOf(Distance::class, $this->kernel);
    }

    /**
     * @test
     * @dataProvider computeProvider
     *
     * @param (int|float)[] $a
     * @param (int|float)[] $b
     * @param float $expected
     */
    public function compute(array $a, array $b, float $expected) : void
    {
        $distance = $this->kernel->compute($a, $b);

        $this->assertGreaterThanOrEqual(0.0, $distance);
        $this->assertEqualsWithDelta($expected, $distance, 1e-8);
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function computeProvider() : Generator
    {
        yield [
            [2, 1, 4, 0], [-2, 1, 8, -2],
            0.2593263058537443,
        ];

        yield [
            [7.4, -2.5], [0.01, -1],
            0.6704765571747832,
        ];

        yield [
            [1000, -2000, 3000], [1000, -2000, 3000],
            0.0,
        ];

        yield [
            [1000, -2000, 3000], [-1000, 2000, -3000],
            2.0,
        ];

        yield [
            [1.0, 2.0, 3.0], [0.0, 0.0, 0.0],
            2.0,
        ];

        yield [
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            0.0,
        ];
    }
}
