<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\TestCase;

/**
 * @group Initializers
 * @covers \Rubix\ML\NeuralNet\Initializers\He
 */
class HeTest extends TestCase
{
    /**
     * @var He
     */
    protected $initializer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->initializer = new He();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(He::class, $this->initializer);
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
