<?php

namespace Rubix\ML\Tests\Other\Strategies;

use Rubix\ML\Other\Strategies\Prior;
use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Strategies\Categorical;
use PHPUnit\Framework\TestCase;

class PriorTest extends TestCase
{
    protected $values;

    protected $strategy;

    public function setUp()
    {
        $this->values = ['a', 'a', 'b', 'a', 'c'];

        $this->strategy = new Prior();
    }

    public function test_build_local_celebrity_strategy()
    {
        $this->assertInstanceOf(Prior::class, $this->strategy);
        $this->assertInstanceOf(Categorical::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_priors_guess()
    {
        $this->strategy->fit($this->values);

        $expected = [
            'a' => 0.6,
            'b' => 0.2,
            'c' => 0.2,
        ];

        $this->assertEquals($expected, $this->strategy->priors());

        $value = $this->strategy->guess();

        $this->assertContains($value, $this->values);
    }
}
