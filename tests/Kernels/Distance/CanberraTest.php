<?php

namespace Rubix\ML\Tests\Kernels\Distance;

use Rubix\ML\Kernels\Distance\Canberra;
use Rubix\ML\Kernels\Distance\Distance;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Distances
 * @covers \Rubix\ML\Kernels\Distance\Canberra
 */
class CanberraTest extends TestCase
{
    /**
     * @var \Rubix\ML\Kernels\Distance\Canberra
     */
    protected $kernel;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->kernel = new Canberra();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Canberra::class, $this->kernel);
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
     * @return \Generator<mixed[]>
     */
    public function computeProvider() : Generator
    {
        yield [
            [2, 1, 4, 0], [-2, 1, 8, -2],
            2.333333333333333,
        ];

        yield [
            [7.4, -2.5], [0.01, -1],
            1.4258723732407943,
        ];

        yield [
            [1000, -2000, 3000], [1000, -2000, 3000],
            0.0,
        ];
    }
}
