<?php

use Rubix\Engine\Transformers\Strategies\PopularityContest;
use PHPUnit\Framework\TestCase;

class PopularityContestTest extends TestCase
{
    protected $strategy;

    public function setUp()
    {
        $this->strategy = new PopularityContest();
    }

    public function test_build_local_celebrity_strategy()
    {
        $this->assertInstanceOf(PopularityContest::class, $this->strategy);
    }

    public function test_guess_value()
    {
        $data = ['a', 'a', 'b'];

        $this->assertTrue(in_array($this->strategy->guess($data), ['a', 'b']));
    }
}
