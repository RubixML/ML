<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Initializers\Xavier1;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\TestCase;

/**
 * @group Initializers
 * @covers \Rubix\ML\NeuralNet\Initializers\Xavier1
 */
class Xavier1Test extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\Initializers\Xavier1
     */
    protected $initializer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->initializer = new Xavier1();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Xavier1::class, $this->initializer);
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
