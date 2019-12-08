<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Initializers\LeCun;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\TestCase;

class LeCunTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\Initializers\LeCun
     */
    protected $initializer;

    public function setUp() : void
    {
        $this->initializer = new LeCun();
    }

    public function test_build_layer() : void
    {
        $this->assertInstanceOf(LeCun::class, $this->initializer);
        $this->assertInstanceOf(Initializer::class, $this->initializer);
    }

    public function test_initialize() : void
    {
        $w = $this->initializer->initialize(4, 3);

        $this->assertInstanceOf(Matrix::class, $w);
        $this->assertEquals([3, 4], $w->shape());
    }
}
