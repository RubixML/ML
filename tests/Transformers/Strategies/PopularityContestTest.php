<?php

use Rubix\ML\Transformers\Strategies\Strategy;
use Rubix\ML\Transformers\Strategies\PopularityContest;
use PHPUnit\Framework\TestCase;

class PopularityContestTest extends TestCase
{
    protected $values;

    protected $strategy;

    public function setUp()
    {
        $this->values = ['a', 'a', 'b', 'a', 'c'];

        $this->strategy = new PopularityContest();
    }

    public function test_build_local_celebrity_strategy()
    {
        $this->assertInstanceOf(PopularityContest::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_guess_value()
    {
        $this->strategy->fit($this->values);

        $value = $this->strategy->guess();

        $this->assertContains($value, $this->values);
    }
}
