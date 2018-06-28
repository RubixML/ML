<?php

namespace Rubix\Tests\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Optimizers\Momentum;
use PHPUnit\Framework\TestCase;

class MomentumTest extends TestCase
{
    protected $optimizer;

    public function setUp()
    {
        $this->optimizer = new Momentum(0.001);
    }

    public function test_build_optimizer()
    {
        $this->assertInstanceOf(Momentum::class, $this->optimizer);
    }
}
