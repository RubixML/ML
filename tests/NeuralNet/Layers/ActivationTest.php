<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Nonparametric;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
use PHPUnit\Framework\TestCase;

class ActivationTest extends TestCase
{
    protected $fanIn;

    protected $input;

    protected $prevGrad;

    protected $optimizer;

    protected $layer;

    public function setUp()
    {
        $this->fanIn = 3;

        $this->input = Matrix::quick([
            [1., 2.5, -0.1],
            [0.1, 0., 3.],
            [0.002, -6., -0.5],
        ]);

        $this->prevGrad = function () {
            return Matrix::quick([
                [0.25, 0.7, 0.1],
                [0.50, 0.2, 0.01],
                [0.25, 0.1, 0.89],
            ]);
        };

        $this->optimizer = new Stochastic();

        $this->layer = new Activation(new ReLU());

        $this->layer->initialize($this->fanIn);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(Activation::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
        $this->assertInstanceOf(Nonparametric::class, $this->layer);
    }

    public function test_width()
    {
        $this->assertEquals($this->fanIn, $this->layer->width());
    }

    public function test_forward_back_infer()
    {
        $forward = $this->layer->forward($this->input);

        $output = [
            [1., 2.5, 0.],
            [0.1, 0., 3.],
            [0.002, 0., 0.],
        ];

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals([3, 3], $forward->shape());
        $this->assertEquals($output, $forward->asArray());

        $back = $this->layer->back($this->prevGrad, $this->optimizer);

        $this->assertInternalType('callable', $back);

        $back = $back();

        $dYdX = [
            [0.25, 0.7, 0.],
            [0.5, 0., 0.01],
            [0.25, 0, 0.],
        ];

        $this->assertInstanceOf(Matrix::class, $back);
        $this->assertEquals([3, 3], $back->shape());
        $this->assertEquals($dYdX, $back->asArray());

        $infer = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals([3, 3], $infer->shape());
        $this->assertEquals($output, $infer->asArray());
    }
}
