<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Helpers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Helpers\Stats;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Helpers')]
#[CoversClass(Stats::class)]
class StatsTest extends TestCase
{
    protected const array TEST_VALUES = [15, 12.5, 13, 2, 1.5, 6, 9.5, 10, 13, 5];

    public static function meanProvider() : Generator
    {
        yield [self::TEST_VALUES, 8.75];

        yield [[5.0], 5.0];
    }

    public static function weightedMeanProvider() : Generator
    {
        yield [self::TEST_VALUES, [3, 2, 5, 1, 2, 4, 4, 2, 3, 5], 9.225806451612904];

        yield [self::TEST_VALUES, array_fill(0, count(self::TEST_VALUES), 1), 8.75];
    }

    public static function quantileProvider() : Generator
    {
        yield [self::TEST_VALUES, 0.5, 9.75];

        yield [self::TEST_VALUES, 0.99, 14.82];

        yield [[5.0], 0.5, 5.0];
    }

    public static function centralMomentProvider() : Generator
    {
        yield [self::TEST_VALUES, 1, 0.0];

        yield [self::TEST_VALUES, 2, 21.1125];

        yield [self::TEST_VALUES, 3, -30.9375];

        yield [self::TEST_VALUES, 4, 747.26015625];
    }

    /**
     * @param (int|float)[] $values
     * @param float $expected
     */
    #[DataProvider('meanProvider')]
    public function testMean(array $values, float $expected) : void
    {
        $this->assertSame($expected, Stats::mean(values: $values));
    }

    /**
     * @param (int|float)[] $values
     * @param (int|float)[] $weights
     * @param float $expected
     */
    #[DataProvider('weightedMeanProvider')]
    public function testWeightedMean(array $values, array $weights, float $expected) : void
    {
        $this->assertSame($expected, Stats::weightedMean(values: $values, weights: $weights));
    }

    public function testVariance() : void
    {
        $this->assertSame(21.1125, Stats::variance(values: self::TEST_VALUES));
    }

    public function testMedian() : void
    {
        $this->assertSame(9.75, Stats::median(values: self::TEST_VALUES));
    }

    /**
     * @param (int|float)[] $values
     * @param float $q
     * @param float $expected
     */
    #[DataProvider('quantileProvider')]
    public function testQuantile(array $values, float $q, float $expected) : void
    {
        $this->assertSame($expected, Stats::quantile(values: $values, q: $q));
    }

    public function testMad() : void
    {
        $this->assertEquals(3.5, Stats::mad(values: self::TEST_VALUES));
    }

    public function testSkewness() : void
    {
        $this->assertEquals(-0.31891556974589724, Stats::skewness(values: self::TEST_VALUES));
    }

    /**
     * @param (int|float)[] $values
     * @param int $moment
     * @param float $expected
     */
    #[DataProvider('centralMomentProvider')]
    public function testCentralMoment(array $values, int $moment, float $expected) : void
    {
        $this->assertEquals($expected, Stats::centralMoment(values: $values, moment: $moment));
    }

    public function testKurtosis() : void
    {
        $this->assertEquals(-1.3235426808299866, Stats::kurtosis(values: self::TEST_VALUES));
    }

    public function testMeanVar() : void
    {
        [$mean, $variance] = Stats::meanVar(values: self::TEST_VALUES);

        $this->assertEquals(8.75, $mean);
        $this->assertEquals(21.1125, $variance);
    }

    public function testMedMad() : void
    {
        [$median, $mad] = Stats::medianMad(values: self::TEST_VALUES);

        $this->assertEquals(9.75, $median);
        $this->assertEquals(3.5, $mad);
    }
}
