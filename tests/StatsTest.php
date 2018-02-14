<?php

use Rubix\Engine\Stats;
use PHPUnit\Framework\TestCase;

class StatsTest extends TestCase
{
    protected $data;

    public function setUp()
    {
        $this->data = [
            10, 50, 40, 20, 80, 100, 10, 35,
        ];
    }

    public function test_calculate_sum()
    {
        $this->assertEquals(345, Stats::sum($this->data));
    }

    public function test_calculate_mean()
    {
        $this->assertEquals(43.125, Stats::mean($this->data));
    }

    public function test_calculate_average()
    {
        $this->assertEquals(43.125, Stats::average($this->data));
    }

    public function test_calculate_median()
    {
        $this->assertEquals(37.5, Stats::median($this->data));
        $this->assertEquals(35, Stats::median($this->data, 'LOW'));
        $this->assertEquals(40, Stats::median($this->data, 'HIGH'));
        $this->assertEquals(37.5, Stats::median($this->data, 'AVERAGE'));

        $this->expectException(\RuntimeException::class);

        Stats::median(['bad', 'data']);
    }

    public function test_calculate_mode()
    {
        $this->assertEquals(10, Stats::mode($this->data));
    }

    public function test_calculate_variance()
    {
        $this->assertEquals(930.859375, Stats::variance($this->data));
    }

    public function test_calculate_standard_deviation()
    {
        $this->assertEquals(30.50998811864731, Stats::stddev($this->data));
    }

    public function test_calculate_min()
    {
        $this->assertEquals(10, Stats::min($this->data));
    }

    public function test_calculate_max()
    {
        $this->assertEquals(100, Stats::max($this->data));
    }
}
