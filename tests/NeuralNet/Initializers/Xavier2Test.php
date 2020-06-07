<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Initializers\Xavier2;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\TestCase;

/**
 * @group Initializers
 * @covers \Rubix\ML\NeuralNet\Initializers\Xavier2
 */
class Xavier2Test extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\Initializers\Xavier2
     */
    protected $initializer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->initializer = new Xavier2();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Xavier2::class, $this->initializer);
        $this->assertInstanceOf(Initializer::class, $this->initializer);
    }

    /**
     * @test
     */
    public function initialize() : void
    {
        $w = $this->initializer->initialize(4, 3);

        $this->assertInstanceOf(Matrix::class, $w);
        $this->assertEquals([3, 4], $w->shape());
    }
}
