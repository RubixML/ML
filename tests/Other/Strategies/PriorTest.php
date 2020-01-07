<?php

namespace Rubix\ML\Tests\Other\Strategies;

use Rubix\ML\Other\Strategies\Prior;
use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Strategies\Categorical;
use PHPUnit\Framework\TestCase;

class PriorTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Strategies\Prior
     */
    protected $strategy;

    public function setUp() : void
    {
        $this->strategy = new Prior();
    }

    public function test_build_strategy() : void
    {
        $this->assertInstanceOf(Prior::class, $this->strategy);
        $this->assertInstanceOf(Categorical::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_priors_guess() : void
    {
        $values = ['a', 'a', 'b', 'a', 'c'];

        $this->strategy->fit($values);

        $value = $this->strategy->guess();

        $this->assertContains($value, $values);
    }
}
