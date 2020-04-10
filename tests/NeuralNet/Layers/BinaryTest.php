<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Binary;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use PHPUnit\Framework\TestCase;

/**
 * @group Layers
 * @covers \Rubix\ML\NeuralNet\Layers\Binary
 */
class BinaryTest extends TestCase
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
     * @var \Rubix\ML\NeuralNet\Layers\Binary
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

        $this->labels = ['hot', 'cold', 'hot'];

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Binary(['hot', 'cold'], 1e-4, new CrossEntropy());

        srand(self::RANDOM_SEED);
    }
    
    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Binary::class, $this->layer);
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

        $this->assertEquals(1, $this->layer->width());

        $expected = [
            [0.5357798817266803, 0.053977226104562924, 0.6003099946844843],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals($expected, $forward->asArray());

        [$computation, $loss] = $this->layer->back($this->labels, $this->optimizer);

        $this->assertInstanceOf(Deferred::class, $computation);
        $this->assertIsFloat($loss);

        $gradient = $computation->compute();

        $expected = [
            [0.021354074010898085, -0.03770473849196098, 0.0239259899320254],
            [0.040616032691930426, -0.07171544364080662, 0.045507887102479515],
            [0.09413726514384431, -0.16621750785286177, 0.10547529510811594],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEquals($expected, $gradient->asArray());

        $expected = [
            [0.5359044875329853, 0.05460698301591459, 0.6000497043635915],
        ];

        $infer = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }
}
