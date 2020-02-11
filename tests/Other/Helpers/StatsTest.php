<?php

namespace Rubix\ML\Tests\Other\Helpers;

use Rubix\ML\Other\Helpers\Stats;
use PHPUnit\Framework\TestCase;

/**
 * @group Helpers
 * @covers \Rubix\ML\Other\Helpers\Stats
 */
class StatsTest extends TestCase
{
    protected const VALUES = [15, 12.5, 13, 2, 1.5, 6, 9.5, 10, 13, 5];
    
    /**
     * @test
     */
    public function mean() : void
    {
        $this->assertEquals(8.75, Stats::mean(self::VALUES));
    }
    
    /**
     * @test
     */
    public function weightedMean() : void
    {
        $weights = [3, 2, 5, 1, 2, 4, 4, 2, 3, 5];

        $this->assertEquals(9.225806451612904, Stats::weightedMean(self::VALUES, $weights));
    }
    
    /**
     * @test
     */
    public function variance() : void
    {
        $this->assertEquals(21.1125, Stats::variance(self::VALUES));
    }
    
    /**
     * @test
     */
    public function median() : void
    {
        $this->assertEquals(9.75, Stats::median(self::VALUES));
    }
    
    /**
     * @test
     */
    public function range() : void
    {
        $this->assertEquals(13.5, Stats::range(self::VALUES));
    }
    
    /**
     * @test
     */
    public function percentile() : void
    {
        $this->assertEquals(9.75, Stats::percentile(self::VALUES, 50.0));
    }
    
    /**
     * @test
     */
    public function mode() : void
    {
        $this->assertEquals(13, Stats::mode(self::VALUES));
    }
    
    /**
     * @test
     */
    public function mad() : void
    {
        $this->assertEquals(3.5, Stats::mad(self::VALUES));
    }
    
    /**
     * @test
     */
    public function iqr() : void
    {
        $this->assertEquals(8., Stats::iqr(self::VALUES));
    }
    
    /**
     * @test
     */
    public function skewness() : void
    {
        $this->assertEquals(-0.31891556974589724, Stats::skewness(self::VALUES));
    }
    
    /**
     * @test
     */
    public function centralMoment() : void
    {
        $this->assertEquals(747.26015625, Stats::centralMoment(self::VALUES, 4));
    }
    
    /**
     * @test
     */
    public function kurtosis() : void
    {
        $this->assertEquals(-1.3235426808299866, Stats::kurtosis(self::VALUES));
    }
    
    /**
     * @test
     */
    public function meanVar() : void
    {
        [$mean, $variance] = Stats::meanVar(self::VALUES);

        $this->assertEquals(8.75, $mean);
        $this->assertEquals(21.1125, $variance);
    }
    
    /**
     * @test
     */
    public function medMad() : void
    {
        [$median, $mad] = Stats::medianMad(self::VALUES);

        $this->assertEquals(9.75, $median);
        $this->assertEquals(3.5, $mad);
    }
}
