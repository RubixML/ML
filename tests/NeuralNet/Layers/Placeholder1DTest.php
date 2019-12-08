<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use PHPUnit\Framework\TestCase;

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

    public function setUp() : void
    {
        $this->input = Matrix::quick([
            [1., 2.5,],
            [0.1, 0.],
            [0.002, -6.],
        ]);

        $this->layer = new Placeholder1D(3);
    }

    public function test_build_layer() : void
    {
        $this->assertInstanceOf(Placeholder1D::class, $this->layer);
        $this->assertInstanceOf(Input::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);

        $this->assertEquals(3, $this->layer->width());
    }

    public function test_forward_infer() : void
    {
        $expected = [
            [1., 2.5,],
            [0.1, 0.],
            [0.002, -6.],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals($expected, $forward->asArray());

        $infer = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }
}
