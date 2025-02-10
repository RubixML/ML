<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Initializers\Constant;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(PReLU::class)]
class PReLUTest extends TestCase
{
    protected const int RANDOM_SEED = 0;

    /**
     * @var positive-int
     */
    protected int $fanIn;

    protected Matrix $input;

    protected Deferred $prevGrad;

    protected Optimizer $optimizer;

    protected PReLU $layer;

    protected function setUp() : void
    {
        $this->fanIn = 3;

        $this->input = Matrix::quick([
            [1., 2.5, -0.1],
            [0.1, 0., 3.],
            [0.002, -6., -0.5],
        ]);

        $this->prevGrad = new Deferred(fn: function () {
            return Matrix::quick([
                [0.25, 0.7, 0.1],
                [0.50, 0.2, 0.01],
                [0.25, 0.1, 0.89],
            ]);
        });

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new PReLU(new Constant(0.25));

        srand(self::RANDOM_SEED);
    }

    public function testInitializeForwardBackInfer() : void
    {
        $this->layer->initialize($this->fanIn);

        $this->assertEquals($this->fanIn, $this->layer->width());

        $forward = $this->layer->forward($this->input);

        $expected = [
            [1.0, 2.5, -0.025],
            [0.1, 0.0, 3.0],
            [0.002, -1.5, -0.125],
        ];

        $this->assertEquals($expected, $forward->asArray());

        $gradient = $this->layer->back(
            prevGradient: $this->prevGrad,
            optimizer: $this->optimizer
        )->compute();

        $expected = [
            [0.25, 0.7, 0.025001000000000002],
            [0.5, 0.05, 0.01],
            [0.25, 0.025104500000000002, 0.22343005000000002],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEquals($expected, $gradient->asArray());

        $expected = [
            [1.0, 2.5, -0.025001000000000002],
            [0.1, 0.0, 3.0],
            [0.002, -1.5062700000000002, -0.1255225],
        ];

        $infer = $this->layer->infer($this->input);

        $this->assertEquals($expected, $infer->asArray());
    }
}
