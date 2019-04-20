<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Initializers\Constant;
use PHPUnit\Framework\TestCase;

class PReLUTest extends TestCase
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

        $this->layer = new PReLU(new Constant(0.25));

        $this->layer->initialize($this->fanIn);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(PReLU::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);
    }

    public function test_width()
    {
        $this->assertEquals(3, $this->layer->width());
    }

    public function test_forward_back_infer()
    {
        $forward = $this->layer->forward($this->input);

        $output = [
            [1., 2.5, -0.025],
            [0.1, 0., 3.],
            [0.002, -1.5, -0.125],
        ];

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals([3, 3], $forward->shape());
        $this->assertEquals($output, $forward->asArray());

        $back = $this->layer->back($this->prevGrad, $this->optimizer);

        $this->assertInternalType('callable', $back);

        $back = $back();

        $dYdX = [
            [0.25, 0.7, 0.025001000000000002],
            [0.5, 0.05, 0.01],
            [0.25, 0.025104500000000002, 0.22343005000000002],
        ];

        $this->assertInstanceOf(Matrix::class, $back);
        $this->assertEquals([3, 3], $back->shape());
        $this->assertEquals($dYdX, $back->asArray());

        $infer = $this->layer->infer($this->input);

        $output = [
            [1., 2.5, -0.025001000000000002],
            [0.1, 0., 3.],
            [0.002, -1.5062700000000002, -0.1255225],
        ];

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals([3, 3], $infer->shape());
        $this->assertEquals($output, $infer->asArray());
    }
}
