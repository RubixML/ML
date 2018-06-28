<?php

namespace Rubix\Tests\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Optimizers\RMSProp;
use PHPUnit\Framework\TestCase;

class RMSPropTest extends TestCase
{
    protected $optimizer;

    public function setUp()
    {
        $this->optimizer = new RMSProp(0.001);
    }

    public function test_build_optimizer()
    {
        $this->assertInstanceOf(RMSProp::class, $this->optimizer);
    }
}
