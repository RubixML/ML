<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Noise;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(Noise::class)]
class NoiseTest extends TestCase
{
    protected const int RANDOM_SEED = 0;

    /**
     * @var positive-int
     */
    protected int $fanIn;

    /**
     * @var Matrix
     */
    protected Matrix $input;

    /**
     * @var Deferred
     */
    protected Deferred $prevGrad;

    protected Optimizer $optimizer;

    protected Noise $layer;

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

        $this->layer = new Noise(0.1);

        srand(self::RANDOM_SEED);
    }

    public function testInitializeForwardBackInfer() : void
    {
        $this->layer->initialize($this->fanIn);

        $this->assertEquals($this->fanIn, $this->layer->width());

        $expected = [
            [0.9396596259960941, 2.408572590287506, -0.16793207202614419],
            [0.1457098686524435, -0.0783513312152093, 3.063132246060683],
            [-0.08825748362793215, -5.936776081560676, -0.5918333225801408],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertEqualsWithDelta($expected, $forward->asArray(), 1e-8);

        $gradient = $this->layer->back(
            prevGradient: $this->prevGrad,
            optimizer: $this->optimizer
        )->compute();

        $expected = [
            [0.25, 0.7, 0.1],
            [0.5, 0.2, 0.01],
            [0.25, 0.1, 0.89],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEqualsWithDelta($expected, $gradient->asArray(), 1e-8);

        $expected = [
            [1.0, 2.5, -0.1],
            [0.1, 0.0, 3.0],
            [0.002, -6.0, -0.5],
        ];

        $infer = $this->layer->infer($this->input);

        $this->assertEqualsWithDelta($expected, $infer->asArray(), 1e-8);
    }
}
