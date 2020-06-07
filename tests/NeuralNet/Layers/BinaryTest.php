<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Binary;
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
     * @var \Tensor\Matrix
     */
    protected $input;

    /**
     * @var string[]
     */
    protected $labels;

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
        $this->input = Matrix::quick([
            [1.0, 2.5, -0.1],
        ]);

        $this->labels = ['hot', 'cold', 'hot'];

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Binary(['hot', 'cold'], new CrossEntropy());

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Binary::class, $this->layer);
        $this->assertInstanceOf(Output::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
    }

    /**
     * @test
     */
    public function initializeForwardBackInfer() : void
    {
        $this->layer->initialize(1);

        $this->assertEquals(1, $this->layer->width());

        $expected = [
            [0.7310585786300049, 0.9241418199787566, 0.47502081252106],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals($expected, $forward->asArray());

        [$computation, $loss] = $this->layer->back($this->labels, $this->optimizer);

        $this->assertInstanceOf(Deferred::class, $computation);
        $this->assertIsFloat($loss);

        $gradient = $computation->compute();

        $expected = [
            [0.2436861928766683, -0.02528606000708115, 0.15834027084035332],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEquals($expected, $gradient->asArray());

        $expected = [
            [0.7310585786300049, 0.9241418199787566, 0.47502081252106],
        ];

        $infer = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }
}
