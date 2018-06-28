<?php

namespace Rubix\Tests\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Optimizers\AdaGrad;
use PHPUnit\Framework\TestCase;

class AdaGradTest extends TestCase
{
    protected $optimizer;

    public function setUp()
    {
        $this->optimizer = new AdaGrad(0.001);
    }

    public function test_build_optimizer()
    {
        $this->assertInstanceOf(AdaGrad::class, $this->optimizer);
    }
}
