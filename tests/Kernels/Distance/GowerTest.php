<?php

namespace Rubix\ML\Tests\Kernels\Distance;

use Rubix\ML\Kernels\Distance\Gower;
use Rubix\ML\Kernels\Distance\NaNSafe;
use Rubix\ML\Kernels\Distance\Distance;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Distances
 * @covers \Rubix\ML\Kernels\Distance\Gower
 */
class GowerTest extends TestCase
{
    /**
     * @var Gower
     */
    protected $kernel;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->kernel = new Gower(1.0);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Gower::class, $this->kernel);
        $this->assertInstanceOf(NaNSafe::class, $this->kernel);
        $this->assertInstanceOf(Distance::class, $this->kernel);
    }

    /**
     * @test
     * @dataProvider computeProvider
     *
     * @param list<string|int|float> $a
     * @param list<string|int|float> $b
     * @param float $expected
     */
    public function compute(array $a, array $b, $expected) : void
    {
        $distance = $this->kernel->compute($a, $b);

        $this->assertGreaterThanOrEqual(0.0, $distance);
        $this->assertEquals($expected, $distance);
    }

    /**
     * @return \Generator<array<mixed>>
     */
    public function computeProvider() : Generator
    {
        yield [['toast', 1.0, 0.5, NAN], ['pretzels', 1.0, 0.2, 0.1], 0.43333333333333335];

        yield [[0.0, 1.0, 0.5, 'ham'], [0.1, 0.9, 0.4, 'ham'], 0.07499999999999998];

        yield [[1, NAN, 1], [1, NAN, 1], 0.0];
    }
}
