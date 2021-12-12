<?php

namespace Rubix\ML\Tests\Helpers;

use Rubix\ML\Helpers\Stats;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Helpers
 * @covers \Rubix\ML\Helpers\Stats
 */
class StatsTest extends TestCase
{
    protected const TEST_VALUES = [15, 12.5, 13, 2, 1.5, 6, 9.5, 10, 13, 5];

    /**
     * @test
     * @dataProvider meanProvider
     *
     * @param (int|float)[] $values
     * @param float $expected
     */
    public function mean(array $values, float $expected) : void
    {
        $this->assertSame($expected, Stats::mean($values));
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function meanProvider() : Generator
    {
        yield [self::TEST_VALUES, 8.75];

        yield [[5.0], 5.0];
    }

    /**
     * @test
     * @dataProvider weightedMeanProvider
     *
     * @param (int|float)[] $values
     * @param (int|float)[] $weights
     * @param float $expected
     */
    public function weightedMean(array $values, array $weights, float $expected) : void
    {
        $this->assertSame($expected, Stats::weightedMean($values, $weights));
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function weightedMeanProvider() : Generator
    {
        yield [self::TEST_VALUES, [3, 2, 5, 1, 2, 4, 4, 2, 3, 5], 9.225806451612904];

        yield [self::TEST_VALUES, array_fill(0, count(self::TEST_VALUES), 1), 8.75];
    }

    /**
     * @test
     */
    public function variance() : void
    {
        $this->assertSame(21.1125, Stats::variance(self::TEST_VALUES));
    }

    /**
     * @test
     */
    public function median() : void
    {
        $this->assertSame(9.75, Stats::median(self::TEST_VALUES));
    }

    /**
     * @test
     * @dataProvider quantileProvider
     *
     * @param (int|float)[] $values
     * @param float $q
     * @param float $expected
     */
    public function quantile(array $values, float $q, float $expected) : void
    {
        $this->assertSame($expected, Stats::quantile($values, $q));
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function quantileProvider() : Generator
    {
        yield [self::TEST_VALUES, 0.5, 9.75];

        yield [self::TEST_VALUES, 0.99, 14.82];

        yield [[5.0], 0.5, 5.0];
    }

    /**
     * @test
     */
    public function mad() : void
    {
        $this->assertEquals(3.5, Stats::mad(self::TEST_VALUES));
    }

    /**
     * @test
     */
    public function skewness() : void
    {
        $this->assertEquals(-0.31891556974589724, Stats::skewness(self::TEST_VALUES));
    }

    /**
     * @test
     * @dataProvider centralMomentProvider
     *
     * @param (int|float)[] $values
     * @param int $moment
     * @param float $expected
     */
    public function centralMoment(array $values, int $moment, float $expected) : void
    {
        $this->assertEquals($expected, Stats::centralMoment($values, $moment));
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function centralMomentProvider() : Generator
    {
        yield [self::TEST_VALUES, 1, 0.0];

        yield [self::TEST_VALUES, 2, 21.1125];

        yield [self::TEST_VALUES, 3, -30.9375];

        yield [self::TEST_VALUES, 4, 747.26015625];
    }

    /**
     * @test
     */
    public function kurtosis() : void
    {
        $this->assertEquals(-1.3235426808299866, Stats::kurtosis(self::TEST_VALUES));
    }

    /**
     * @test
     */
    public function meanVar() : void
    {
        [$mean, $variance] = Stats::meanVar(self::TEST_VALUES);

        $this->assertEquals(8.75, $mean);
        $this->assertEquals(21.1125, $variance);
    }

    /**
     * @test
     */
    public function medMad() : void
    {
        [$median, $mad] = Stats::medianMad(self::TEST_VALUES);

        $this->assertEquals(9.75, $median);
        $this->assertEquals(3.5, $mad);
    }
}
