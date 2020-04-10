<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use PHPUnit\Framework\TestCase;

/**
 * @group Layers
 * @covers \Rubix\ML\NeuralNet\Layers\Multiclass
 */
class MulticlassTest extends TestCase
{
    protected const RANDOM_SEED = 0;

    /**
     * @var int
     */
    protected $fanIn;

    /**
     * @var \Tensor\Matrix
     */
    protected $input;

    /**
     * @var string[]
     */
    protected $labels;

    /**
     * @var \Rubix\ML\Deferred
     */
    protected $prevGrad;

    /**
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * @var \Rubix\ML\NeuralNet\Layers\Multiclass
     */
    protected $layer;
    
    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->fanIn = 3;

        $this->input = Matrix::quick([
            [1., 2.5, -0.1],
            [0.1, 0., 3.],
            [0.002, -6., -0.5],
        ]);

        $this->labels = ['hot', 'cold', 'ice cold'];

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Multiclass(['hot', 'cold', 'ice cold'], 1e-4, new CrossEntropy());

        srand(self::RANDOM_SEED);
    }
    
    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Multiclass::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Output::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);
    }
    
    /**
     * @test
     */
    public function initializeForwardBackInfer() : void
    {
        $this->layer->initialize($this->fanIn);

        $this->assertEquals($this->fanIn, $this->layer->width());
        
        $forward = $this->layer->forward($this->input);

        $expected = [
            [0.2595391959758536, 0.02923162203401248, 0.12450877926117593],
            [0.46982815260469185, 0.023090868121796462, 0.10799036932220654],
            [0.27063265141945453, 0.9476775098441911, 0.7675008514166175],
        ];

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals($expected, $forward->asArray());

        [$computation, $loss] = $this->layer->back($this->labels, $this->optimizer);

        $this->assertInstanceOf(Deferred::class, $computation);
        $this->assertIsFloat($loss);

        $gradient = $computation->compute();

        $expected = [
            [0.0918320711672522, -0.1949031158033154, 0.02187986261379058],
            [0.04900732237179448, 0.15427079036690672, -0.03871887274402862],
            [-0.007885243780857604, -0.27716042855055123, 0.055465226407507576],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEquals($expected, $gradient->asArray());

        $infer = $this->layer->infer($this->input);

        $expected = [
            [0.2596071187697675, 0.029611917045991417, 0.12452927259570575],
            [0.4701228858592578, 0.023700158558071664, 0.10809787871952142],
            [0.27026999537097474, 0.9466879243959369, 0.7673728486847728],
        ];

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }
}
