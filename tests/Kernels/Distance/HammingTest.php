<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Kernels\Distance;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Kernels\Distance\Hamming;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Distances')]
#[CoversClass(Hamming::class)]
class HammingTest extends TestCase
{
    protected Hamming $kernel;

    public static function computeProvider() : Generator
    {
        yield [
            ['soup', 'turkey', 'broccoli', 'cake'], ['salad', 'turkey', 'broccoli', 'pie'],
            2.0,
        ];

        yield [
            ['salad', 'ham', 'peas', 'jello'], ['bread', 'steak', 'potato', 'cake'],
            4.0,
        ];

        yield [
            ['toast', 'eggs', 'bacon'], ['toast', 'eggs', 'bacon'],
            0.0,
        ];
    }

    protected function setUp() : void
    {
        $this->kernel = new Hamming();
    }

    /**
     * @param list<string> $a
     * @param list<string> $b
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
