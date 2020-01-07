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

    public function setUp() : void
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

    public function test_build_layer() : void
    {
        $this->assertInstanceOf(Multiclass::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Output::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);

        $this->layer->initialize($this->fanIn);

        $this->assertEquals(3, $this->layer->width());
    }

    public function test_forward_back_infer() : void
    {
        $this->layer->initialize($this->fanIn);
        
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
            [0.09187323354697499, -0.19486195342359264, 0.021921024993513363],
            [0.049037390613305525, 0.15430085860841775, -0.03868880450251759],
            [-0.007839801389529954, -0.27711498615922364, 0.05551066879883521],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEquals($expected, $gradient->asArray());

        $infer = $this->layer->infer($this->input);

        $expected = [
            [0.25960713436540017, 0.029611904494947872, 0.1245292575454179],
            [0.470122847941923, 0.02370011220607803, 0.10809781499994708],
            [0.27027001769267694, 0.9466879832989741, 0.767372927454635],
        ];

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }
}
