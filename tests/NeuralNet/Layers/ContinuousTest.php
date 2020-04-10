<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Continuous;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use PHPUnit\Framework\TestCase;

/**
 * @group Layers
 * @covers \Rubix\ML\NeuralNet\Layers\Continuous
 */
class ContinuousTest extends TestCase
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
     * @var int[]
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
     * @var \Rubix\ML\NeuralNet\Layers\Continuous
     */
    protected $layer;
    
    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->fanIn = 3;

        $this->input = Matrix::quick([
            [1.0, 2.5, -0.1],
            [0.1, 0.0, 3.0],
            [0.002, -6.0, -0.5],
        ]);

        $this->labels = [90, 260, 180];

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Continuous(1e-4, new LeastSquares());

        srand(self::RANDOM_SEED);
    }
    
    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Continuous::class, $this->layer);
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
            [0.1295445178808929, -2.587649055132072, 0.36754636758929626],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals($expected, $forward->asArray());

        [$computation, $loss] = $this->layer->back($this->labels, $this->optimizer);

        $this->assertInstanceOf(Deferred::class, $computation);
        $this->assertIsFloat($loss);

        $gradient = $computation->compute();

        $expected = [
            [-3.2365958251552804, -9.456835220316393, -6.4692856664639535],
            [-6.156093763559253, -17.9871591844509, -12.304758239033854],
            [-14.26820377229217, -41.689497006624315, -28.519188411746832],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEquals($expected, $gradient->asArray());

        $expected = [
            [0.566852840529922, 1.5270197607920448, 1.346042108950773],
        ];

        $infer = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }
}
