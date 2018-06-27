<?php

namespace Rubix\Tests\Transformers\Strategies;

use Rubix\ML\Transformers\Strategies\Strategy;
use Rubix\ML\Transformers\Strategies\KMostFrequent;
use PHPUnit\Framework\TestCase;

class KMostFrequentTest extends TestCase
{
    protected $values;

    protected $strategy;

    public function setUp()
    {
        $this->values = ['a', 'a', 'b', 'b', 'c'];

        $this->strategy = new KMostFrequent(2);
    }

    public function test_build_k_most_frequent_strategy()
    {
        $this->assertInstanceOf(KMostFrequent::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_guess_value()
    {
        $this->strategy->fit($this->values);

        $value = $this->strategy->guess();

        $this->assertContains($value, $this->values);
    }
}
