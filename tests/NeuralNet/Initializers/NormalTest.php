<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Initializers\Normal;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\TestCase;

class NormalTest extends TestCase
{
    protected $initializer;

    public function setUp()
    {
        $this->initializer = new Normal(0.05);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(Normal::class, $this->initializer);
        $this->assertInstanceOf(Initializer::class, $this->initializer);
    }

    public function test_initialize()
    {
        $w = $this->initializer->init(4, 3);

        $this->assertInstanceOf(Matrix::class, $w);
        $this->assertEquals([3, 4], $w->shape());
    }
}
