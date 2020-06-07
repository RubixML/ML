<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Initializers\Normal;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\TestCase;

/**
 * @group Initializers
 * @covers \Rubix\ML\NeuralNet\Initializers\Normal
 */
class NormalTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\Initializers\Normal
     */
    protected $initializer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->initializer = new Normal(0.05);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Normal::class, $this->initializer);
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
