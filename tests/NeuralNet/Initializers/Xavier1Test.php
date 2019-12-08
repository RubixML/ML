<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Initializers\Xavier1;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\TestCase;

class Xavier1Test extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\Initializers\Xavier1
     */
    protected $initializer;

    public function setUp() : void
    {
        $this->initializer = new Xavier1();
    }

    public function test_build_layer() : void
    {
        $this->assertInstanceOf(Xavier1::class, $this->initializer);
        $this->assertInstanceOf(Initializer::class, $this->initializer);
    }

    public function test_initialize() : void
    {
        $w = $this->initializer->initialize(4, 3);

        $this->assertInstanceOf(Matrix::class, $w);
        $this->assertEquals([3, 4], $w->shape());
    }
}
