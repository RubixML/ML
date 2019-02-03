<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use PHPUnit\Framework\TestCase;

class MulticlassTest extends TestCase
{
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

        $this->labels = ['hot', 'cold', 'ice cold'];

        $this->optimizer = new Stochastic();

        $this->layer = new Multiclass(['hot', 'cold', 'ice cold'], 1e-4, new CrossEntropy());

        $this->layer->initialize($this->fanIn);
    }

    public function test_width()
    {
        $this->assertEquals(3, $this->layer->width());
    }

    public function test_forward_back_infer()
    {
        $forward = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals([3, 3], $forward->shape());

        [$back, $loss] = $this->layer->back($this->labels, $this->optimizer);

        $this->assertInternalType('callable', $back);
        $this->assertInternalType('float', $loss);

        $back = $back();

        $this->assertInstanceOf(Matrix::class, $back);
        $this->assertEquals([3, 3], $back->shape());

        $infer = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals([3, 3], $infer->shape());
    }
}
