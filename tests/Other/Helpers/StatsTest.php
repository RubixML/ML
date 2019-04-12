<?php

namespace Rubix\ML\Tests\Other\Helpers;

use Rubix\ML\Other\Helpers\Stats;
use PHPUnit\Framework\TestCase;

class StatsTest extends TestCase
{
    protected $values;

    public function setUp()
    {
        $this->values = [15, 12.5, 13, 2, 1.5, 6, 9.5, 10, 13, 5];
    }

    public function test_mean()
    {
        $this->assertEquals(8.75, Stats::mean($this->values));
    }

    public function test_weighted_mean()
    {
        $weights = [3, 2, 5, 1, 2, 4, 4, 2, 3, 5];

        $this->assertEquals(9.225806451612904, Stats::weightedMean($this->values, $weights));
    }

    public function test_variance()
    {
        $this->assertEquals(21.1125, Stats::variance($this->values));
    }

    public function test_median()
    {
        $this->assertEquals(9.75, Stats::median($this->values));
    }

    public function test_range()
    {
        $this->assertEquals(13.5, Stats::range($this->values));
    }

    public function test_percentile()
    {
        $this->assertEquals(9.75, Stats::percentile($this->values, 50.));
    }

    public function test_mode()
    {
        $this->assertEquals(13, Stats::mode($this->values));
    }

    public function test_mad()
    {
        $this->assertEquals(3.5, Stats::mad($this->values));
    }

    public function test_iqr()
    {
        $this->assertEquals(8., Stats::iqr($this->values));
    }

    public function test_skewness()
    {
        $this->assertEquals(-0.31891556974589724, Stats::skewness($this->values));
    }

    public function test_central_moment()
    {
        $this->assertEquals(747.26015625, Stats::centralMoment($this->values, 4));
    }

    public function test_kurtosis()
    {
        $this->assertEquals(-1.3235426808299866, Stats::kurtosis($this->values));
    }

    public function test_mean_var()
    {
        [$mean, $variance] = Stats::meanVar($this->values);

        $this->assertEquals(8.75, $mean);
        $this->assertEquals(21.1125, $variance);
    }

    public function test_med_mad()
    {
        [$median, $mad] = Stats::medMad($this->values);

        $this->assertEquals(9.75, $median);
        $this->assertEquals(3.5, $mad);
    }
}
