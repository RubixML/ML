<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use PHPUnit\Framework\TestCase;

/**
 * @group Layers
 * @covers \Rubix\ML\NeuralNet\Layers\Placeholder1D
 */
class Placeholder1DTest extends TestCase
{
    /**
     * @var \Tensor\Matrix
     */
    protected $input;

    /**
     * @var \Rubix\ML\NeuralNet\Layers\Placeholder1D
     */
    protected $layer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->input = Matrix::quick([
            [1.0, 2.5],
            [0.1, 0.0],
            [0.002, -6.0],
        ]);

        $this->layer = new Placeholder1D(3);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Placeholder1D::class, $this->layer);
        $this->assertInstanceOf(Input::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
    }

    /**
     * @test
     */
    public function forwardInfer() : void
    {
        $this->assertEquals(3, $this->layer->width());

        $expected = [
            [1.0, 2.5],
            [0.1, 0.0],
            [0.002, -6.0],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals($expected, $forward->asArray());

        $infer = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }
}
