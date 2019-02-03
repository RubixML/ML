<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Initializers\Uniform;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\TestCase;

class UniformTest extends TestCase
{
    protected $initializer;

    public function setUp()
    {
        $this->initializer = new Uniform(0.05);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(Uniform::class, $this->initializer);
        $this->assertInstanceOf(Initializer::class, $this->initializer);
    }

    public function test_initialize()
    {
        $w = $this->initializer->initialize(4, 3);

        $this->assertInstanceOf(Matrix::class, $w);
        $this->assertEquals([3, 4], $w->shape());
    }
}
