<?php

namespace Rubix\ML\Tests\Other\Helpers;

use Rubix\ML\Other\Helpers\Stats;
use PHPUnit\Framework\TestCase;

class StatsTest extends TestCase
{
    /**
     * @var (int|float)[]
     */
    protected $values;

    public function setUp() : void
    {
        $this->values = [15, 12.5, 13, 2, 1.5, 6, 9.5, 10, 13, 5];
    }

    public function test_mean() : void
    {
        $this->assertEquals(8.75, Stats::mean($this->values));
    }

    public function test_weighted_mean() : void
    {
        $weights = [3, 2, 5, 1, 2, 4, 4, 2, 3, 5];

        $this->assertEquals(9.225806451612904, Stats::weightedMean($this->values, $weights));
    }

    public function test_variance() : void
    {
        $this->assertEquals(21.1125, Stats::variance($this->values));
    }

    public function test_median() : void
    {
        $this->assertEquals(9.75, Stats::median($this->values));
    }

    public function test_range() : void
    {
        $this->assertEquals(13.5, Stats::range($this->values));
    }

    public function test_percentile() : void
    {
        $this->assertEquals(9.75, Stats::percentile($this->values, 50.));
    }

    public function test_mode() : void
    {
        $this->assertEquals(13, Stats::mode($this->values));
    }

    public function test_mad() : void
    {
        $this->assertEquals(3.5, Stats::mad($this->values));
    }

    public function test_iqr() : void
    {
        $this->assertEquals(8., Stats::iqr($this->values));
    }

    public function test_skewness() : void
    {
        $this->assertEquals(-0.31891556974589724, Stats::skewness($this->values));
    }

    public function test_central_moment() : void
    {
        $this->assertEquals(747.26015625, Stats::centralMoment($this->values, 4));
    }

    public function test_kurtosis() : void
    {
        $this->assertEquals(-1.3235426808299866, Stats::kurtosis($this->values));
    }

    public function test_mean_var() : void
    {
        [$mean, $variance] = Stats::meanVar($this->values);

        $this->assertEquals(8.75, $mean);
        $this->assertEquals(21.1125, $variance);
    }

    public function test_med_mad() : void
    {
        [$median, $mad] = Stats::medianMad($this->values);

        $this->assertEquals(9.75, $median);
        $this->assertEquals(3.5, $mad);
    }
}
