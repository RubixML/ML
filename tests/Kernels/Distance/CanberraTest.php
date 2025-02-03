<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Kernels\Distance;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Kernels\Distance\Canberra;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Distances')]
#[CoversClass(Canberra::class)]
class CanberraTest extends TestCase
{
    protected Canberra $kernel;

    /**
     * @return Generator<array<int, int|float|list<int>|list<float>>>
     */
    public static function computeProvider() : Generator
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

    protected function setUp() : void
    {
        $this->kernel = new Canberra();
    }

    /**
     * @param list<float|int> $a
     * @param list<float|int> $b
     * @param float $expected
     */
    #[DataProvider('computeProvider')]
    public function testCompute(array $a, array $b, float $expected) : void
    {
        $distance = $this->kernel->compute(a: $a, b: $b);

        $this->assertGreaterThanOrEqual(0.0, $distance);
        $this->assertEquals($expected, $distance);
    }
}
