<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Rubix\ML\Deferred;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Binary;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use PHPUnit\Framework\TestCase;

class BinaryTest extends TestCase
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

        $this->labels = ['hot', 'cold', 'hot'];

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Binary(['hot', 'cold'], 1e-4, new CrossEntropy());

        srand(self::RANDOM_SEED);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(Binary::class, $this->layer);
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
            [0.5430883638324137, 0.2294558178526538, 0.8089955514755949],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals($expected, $forward->asArray());

        [$back, $loss] = $this->layer->back($this->labels, $this->optimizer);

        $this->assertInstanceOf(Deferred::class, $back);
        $this->assertInternalType('float', $loss);

        $expected = [
            [0.021648941873997945, -0.0307072728685828, 0.03224695412670032],
            [0.09543716032128477, -0.1353698920180607, 0.1421574203845685],
            [0.045575878089120406, -0.06464569644342781, 0.06788707081287679],
        ];

        $this->assertInstanceOf(Matrix::class, $back->result());
        $this->assertEquals($expected, $back->result()->asArray(), '', 1e-4);

        $expected = [
            [0.5431400980394667, 0.23113347851058044, 0.8086830533932138],
        ];

        $infer = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }
}
