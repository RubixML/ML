<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use PHPUnit\Framework\TestCase;

/**
 * @group Layers
 * @covers \Rubix\ML\NeuralNet\Layers\Dropout
 */
class DropoutTest extends TestCase
{
    protected const RANDOM_SEED = 0;

    /**
     * @var positive-int
     */
    protected $fanIn;

    /**
     * @var Matrix
     */
    protected $input;

    /**
     * @var Deferred
     */
    protected $prevGrad;

    /**
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * @var Dropout
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

        $this->prevGrad = new Deferred(function () {
            return Matrix::quick([
                [0.25, 0.7, 0.1],
                [0.50, 0.2, 0.01],
                [0.25, 0.1, 0.89],
            ]);
        });

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Dropout(0.5);

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Dropout::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
    }

    /**
     * @test
     */
    public function initializeForwardBackInfer() : void
    {
        $this->layer->initialize($this->fanIn);

        $this->assertEquals($this->fanIn, $this->layer->width());

        $expected = [
            [2.0, 5.0, -0.2],
            [0.2, 0.0, 6.0],
            [0.004, -12.0, 0.0],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals($expected, $forward->asArray());

        $gradient = $this->layer->back($this->prevGrad, $this->optimizer)->compute();

        $expected = [
            [0.5, 1.4, 0.2],
            [1.0, 0.4, 0.02],
            [0.5, 0.2, 0.0],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEquals($expected, $gradient->asArray());

        $expected = [
            [1.0, 2.5, -0.1],
            [0.1, 0.0, 3.0],
            [0.002, -6.0, -0.5],
        ];

        $infer = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }
}
