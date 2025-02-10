<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Kernels\Distance;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Kernels\Distance\SafeEuclidean;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Distances')]
#[CoversClass(SafeEuclidean::class)]
class SafeEuclideanTest extends TestCase
{
    protected SafeEuclidean $kernel;

    public static function computeProvider() : Generator
    {
        yield [
            [2, 1, 4, NAN], [-2, 1, 8, -2],
            6.531972647421808,
        ];

        yield [
            [7.4, -2.5, 0.001], [NAN, -1, 0.075],
            1.8393515161599752,
        ];

        yield [
            [1000, NAN, 3000], [1000, NAN, 3000],
            0.0,
        ];
    }

    protected function setUp() : void
    {
        $this->kernel = new SafeEuclidean();
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
