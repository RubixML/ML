<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Kernels\Distance;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Kernels\Distance\Euclidean;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Distances')]
#[CoversClass(Euclidean::class)]
class EuclideanTest extends TestCase
{
    protected Euclidean $kernel;

    /**
     * @return Generator<array<int, int|float|list<int>|list<float>>>
     */
    public static function computeProvider() : Generator
    {
        yield [
            [2, 1, 4, 0], [-2, 1, 8, -2],
            6.0,
        ];

        yield [
            [7.4, -2.5], [0.01, -1],
            7.540696254325591,
        ];

        yield [
            [1000, -2000, 3000], [1000, -2000, 3000],
            0.0,
        ];
    }

    protected function setUp() : void
    {
        $this->kernel = new Euclidean();
    }

    /**
     * @param list<int|float> $a
     * @param list<int|float> $b
     * @param float $expected
     */
    #[DataProvider('computeProvider')]
    public function testCompute(array $a, array $b, float $expected) : void
    {
        $distance = $this->kernel->compute(a: $a, b: $b);

        $this->assertGreaterThanOrEqual(0., $distance);
        $this->assertEquals($expected, $distance);
    }
}
