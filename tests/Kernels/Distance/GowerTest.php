<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Kernels\Distance;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Kernels\Distance\Gower;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Distances')]
#[CoversClass(Gower::class)]
class GowerTest extends TestCase
{
    protected Gower $kernel;

    public static function computeProvider() : Generator
    {
        yield [['toast', 1.0, 0.5, NAN], ['pretzels', 1.0, 0.2, 0.1], 0.43333333333333335];

        yield [[0.0, 1.0, 0.5, 'ham'], [0.1, 0.9, 0.4, 'ham'], 0.07499999999999998];

        yield [[1, NAN, 1], [1, NAN, 1], 0.0];
    }

    protected function setUp() : void
    {
        $this->kernel = new Gower(1.0);
    }

    /**
     * @param list<string|int|float> $a
     * @param list<string|int|float> $b
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
