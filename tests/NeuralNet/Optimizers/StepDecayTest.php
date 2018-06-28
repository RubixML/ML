<?php

namespace Rubix\Tests\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Optimizers\StepDecay;
use PHPUnit\Framework\TestCase;

class StepDecayTest extends TestCase
{
    protected $optimizer;

    public function setUp()
    {
        $this->optimizer = new StepDecay(0.001);
    }

    public function test_build_optimizer()
    {
        $this->assertInstanceOf(StepDecay::class, $this->optimizer);
    }
}
