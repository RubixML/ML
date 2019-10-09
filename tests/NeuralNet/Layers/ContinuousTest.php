<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Continuous;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use PHPUnit\Framework\TestCase;

class ContinuousTest extends TestCase
{
    protected const RANDOM_SEED = 0;

    protected $fanIn;

    protected $input;

    protected $labels;

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

        $this->labels = [90, 260, 180];

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Continuous(1e-4, new LeastSquares());

        srand(self::RANDOM_SEED);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(Continuous::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Output::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);

        $this->layer->initialize($this->fanIn);

        $this->assertEquals(1, $this->layer->width());
    }

    public function test_forward_back_infer()
    {
        $this->layer->initialize($this->fanIn);

        $expected = [
            [0.15612618258583866, -1.094611267759001, 1.304346510116554],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals($expected, $forward->asArray());

        [$back, $loss] = $this->layer->back($this->labels, $this->optimizer);

        $this->assertInstanceOf(Deferred::class, $back);
        $this->assertInternalType('float', $loss);

        $expected = [
            [-3.2356374595741064, -9.403063868827228, -6.435546672344759],
            [-14.263978940320465, -41.45245154167989, -28.370453535254192],
            [-6.8117425441129775, -19.795558371589877, -13.548269115541029],
        ];

        $this->assertInstanceOf(Matrix::class, $back->result());
        $this->assertEquals($expected, $back->result()->asArray());

        $expected = [
            [0.5913063305845302, 2.9973302670132633, 2.2777613635524014],
        ];

        $infer = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }
}
