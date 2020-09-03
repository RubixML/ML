<?php

namespace Rubix\ML\Tests\Kernels\Distance;

use Rubix\ML\Kernels\Distance\Cosine;
use Rubix\ML\Kernels\Distance\Distance;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Distances
 * @covers \Rubix\ML\Kernels\Distance\Cosine
 */
class CosineTest extends TestCase
{
    /**
     * @var \Rubix\ML\Kernels\Distance\Cosine
     */
    protected $kernel;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->kernel = new Cosine();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Cosine::class, $this->kernel);
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
        $this->assertEquals($expected, $distance);
    }

    /**
     * @return \Generator<array>
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
