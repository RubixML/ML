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
            [0.36144604660573537, 0.628174317130086, 0.3102818056614218],
            [0.33836247814464154, 0.3670863635665911, 0.05172435977226342],
            [0.3001914752496232, 0.004739319303323032, 0.6379938345663148],
        ];

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals($expected, $forward->asArray());

        [$back, $loss] = $this->layer->back($this->labels, $this->optimizer);

        $this->assertInstanceOf(Deferred::class, $back);
        $this->assertInternalType('float', $loss);

        $expected = [
            [-0.02314956691486992, 0.0013034187833920589, 0.026704012638151413],
            [-0.030372854684915476, 0.12361950968924176, -0.05261841965930736],
            [0.08201144203904855, -0.017002135254245234, -0.08555498623142435],
        ];

        $this->assertInstanceOf(Matrix::class, $back->result());
        $this->assertEquals($expected, $back->result()->asArray(), '', 1e-4);

        $infer = $this->layer->infer($this->input);

        $expected = [
            [0.3612945932454064, 0.6241341958483243, 0.30971379982708824],
            [0.3385303579193887, 0.37111565247260486, 0.05173667944399345],
            [0.30017504883520496, 0.0047501516790706905, 0.6385495207289182],
        ];

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }
}
