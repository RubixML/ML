<?php

namespace Rubix\Tests\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Optimizers\Adam;
use PHPUnit\Framework\TestCase;

class AdamTest extends TestCase
{
    protected $optimizer;

    public function setUp()
    {
        $this->optimizer = new Adam(0.001);
    }

    public function test_build_optimizer()
    {
        $this->assertInstanceOf(Adam::class, $this->optimizer);
    }
}
