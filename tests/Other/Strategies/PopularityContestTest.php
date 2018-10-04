<?php

namespace Rubix\ML\Tests\Other\Strategies;

use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Strategies\Categorical;
use Rubix\ML\Other\Strategies\PopularityContest;
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
        $this->assertInstanceOf(Categorical::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_make_guess()
    {
        $this->strategy->fit($this->values);

        $value = $this->strategy->guess();

        $this->assertContains($value, $this->values);
    }
}
