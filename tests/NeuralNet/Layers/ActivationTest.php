<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(Activation::class)]
class ActivationTest extends TestCase
{
    /**
     * @var positive-int
     */
    protected int $fanIn;

    protected Matrix $input;

    protected Deferred $prevGrad;

    protected Optimizer $optimizer;

    protected Activation $layer;

    protected function setUp() : void
    {
        $this->fanIn = 3;

        $this->input = Matrix::quick([
            [1.0, 2.5, -0.1],
            [0.1, 0.0, 3.0],
            [0.002, -6.0, -0.5],
        ]);

        $this->prevGrad = new Deferred(fn: function () {
            return Matrix::quick([
                [0.25, 0.7, 0.1],
                [0.50, 0.2, 0.01],
                [0.25, 0.1, 0.89],
            ]);
        });

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Activation(new ReLU());
    }

    public function testInitializeForwardBackInfer() : void
    {
        $this->layer->initialize($this->fanIn);

        $this->assertEquals($this->fanIn, $this->layer->width());

        $expected = [
            [1.0, 2.5, 0.0],
            [0.1, 0.0, 3.0],
            [0.002, 0.0, 0.0],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertEquals($expected, $forward->asArray());

        $gradient = $this->layer
            ->back(prevGradient: $this->prevGrad, optimizer:  $this->optimizer)
            ->compute();

        $expected = [
            [0.25, 0.7, 0.0],
            [0.5, 0.0, 0.01],
            [0.25, 0, 0.0],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEquals($expected, $gradient->asArray());

        $expected = [
            [1.0, 2.5, 0.0],
            [0.1, 0.0, 3.0],
            [0.002, 0.0, 0.0],
        ];

        $infer = $this->layer->infer($this->input);
        $this->assertEquals($expected, $infer->asArray());
    }
}
