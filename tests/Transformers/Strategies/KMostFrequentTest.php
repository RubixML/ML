<?php

use Rubix\Engine\Transformers\Strategies\Strategy;
use Rubix\Engine\Transformers\Strategies\KMostFrequent;
use PHPUnit\Framework\TestCase;

class KMostFrequentTest extends TestCase
{
    protected $strategy;

    public function setUp()
    {
        $this->strategy = new KMostFrequent();
    }

    public function test_build_k_most_frequent_strategy()
    {
        $this->assertInstanceOf(KMostFrequent::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_guess_value()
    {
        $data = ['a', 'a', 'b'];

        $value = $this->strategy->guess($data);

        $this->assertContains($value, $data);
    }
}
