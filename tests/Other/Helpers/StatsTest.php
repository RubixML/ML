<?php

namespace Rubix\ML\Tests\Other\Helpers;

use MathPHP\Statistics\Average;
use Rubix\ML\Other\Helpers\Stats;
use PHPUnit\Framework\TestCase;

class StatsTest extends TestCase
{
    protected $values;

    public function setUp()
    {
        $this->values = [
            15, 12.5, 13, 2, 1.5, 6, 9.5, 10, 13, 5,
        ];
    }

    public function test_mean()
    {
        $this->assertEquals(8.75, Stats::mean($this->values));
    }

    public function test_variance()
    {
        $this->assertEquals(21.1125, Stats::variance($this->values, 8.75));
    }

    public function test_median()
    {
        $this->assertEquals(9.75, Stats::median($this->values));
    }

    public function test_range()
    {
        $this->assertEquals([1.5, 15], Stats::range($this->values));
    }

    public function test_mode()
    {
        $this->assertEquals(13, Stats::mode($this->values));
    }

    public function test_mad()
    {
        $this->assertEquals(3.5, Stats::mad($this->values, 9.75));
    }

    public function test_mean_var()
    {
        list($mean, $variance) = Stats::meanVar($this->values);

        $this->assertEquals(8.75, $mean);
        $this->assertEquals(21.1125, $variance);
    }

    public function test_med_mad()
    {
        list($median, $mad) = Stats::medMad($this->values);

        $this->assertEquals(9.75, $median);
        $this->assertEquals(3.5, $mad);
    }
}
