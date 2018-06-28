<?php

namespace Rubix\Tests\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use PHPUnit\Framework\TestCase;

class StochasticTest extends TestCase
{
    protected $optimizer;

    public function setUp()
    {
        $this->optimizer = new Stochastic(0.001);
    }

    public function test_build_optimizer()
    {
        $this->assertInstanceOf(Stochastic::class, $this->optimizer);
    }
}
