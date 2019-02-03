<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Initializers\Xavier2;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\TestCase;

class Xavier2Test extends TestCase
{
    protected $initializer;

    public function setUp()
    {
        $this->initializer = new Xavier2();
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(Xavier2::class, $this->initializer);
        $this->assertInstanceOf(Initializer::class, $this->initializer);
    }

    public function test_initialize()
    {
        $w = $this->initializer->initialize(4, 3);

        $this->assertInstanceOf(Matrix::class, $w);
        $this->assertEquals([3, 4], $w->shape());
    }
}
