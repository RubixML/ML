<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers\Dropout;

use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Dropout\Dropout;
use Rubix\ML\NeuralNet\Optimizers\Base\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Stochastic\Stochastic;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(Dropout::class)]
class DropoutTest extends TestCase
{
    protected const int RANDOM_SEED = 0;

    /**
     * @var positive-int
     */
    protected int $fanIn;

    protected NDArray $input;

    protected Deferred $prevGrad;

    protected Optimizer $optimizer;

    protected Dropout $layer;

    protected function setUp() : void
    {
        $this->fanIn = 3;

        $this->input = NumPower::array([
            [1.0, 2.5, -0.1],
            [0.1, 0.0, 3.0],
            [0.002, -6.0, -0.5],
        ]);

        $this->prevGrad = new Deferred(fn: function () : NDArray {
            return NumPower::array([
                [0.25, 0.7, 0.1],
                [0.50, 0.2, 0.01],
                [0.25, 0.1, 0.89],
            ]);
        });

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Dropout(0.5);
    }

    #[Test]
    #[TestDox('Initializes width equal to fan-in')]
    public function testInitializeSetsWidth() : void
    {
        $this->layer->initialize($this->fanIn);

        self::assertEquals($this->fanIn, $this->layer->width());
    }

    #[Test]
    #[TestDox('forward() returns an NDArray with the same shape as the input')]
    public function testForward() : void
    {
        $this->layer->initialize($this->fanIn);

        // Deterministic mask so that forward output is predictable
        $mask = NumPower::array([
            [2.0, 2.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
        ]);

        $forward = $this->layer->forward($this->input, $mask);

        $expected = [
            [2.0, 5.0, -0.2],
            [0.2, 0.0, 6.0],
            [0.004, -12.0, 0.0],
        ];

        self::assertSame($this->input->shape(), $forward->shape());
        self::assertEqualsWithDelta($expected, $forward->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Backpropagates gradients using the same dropout mask')]
    public function testBack() : void
    {
        $this->layer->initialize($this->fanIn);

        // Use the same deterministic mask as in testForward so that the
        // gradient is fully predictable: grad = prevGrad * mask.
        $mask = NumPower::array([
            [2.0, 2.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
        ]);

        // Forward pass to set internal mask cache
        $this->layer->forward($this->input, $mask);

        $gradient = $this->layer->back(
            prevGradient: $this->prevGrad,
            optimizer: $this->optimizer
        )->compute();

        $expected = [
            [0.5, 1.4, 0.2],
            [1.0, 0.0, 0.02],
            [0.5, 0.2, 0.0],
        ];

        self::assertInstanceOf(NDArray::class, $gradient);
        self::assertEqualsWithDelta($expected, $gradient->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Inference pass leaves inputs unchanged')]
    public function testInfer() : void
    {
        $this->layer->initialize($this->fanIn);

        $expected = [
            [1.0, 2.5, -0.1],
            [0.1, 0.0, 3.0],
            [0.002, -6.0, -0.5],
        ];

        $infer = $this->layer->infer($this->input);

        self::assertEqualsWithDelta($expected, $infer->toArray(), 1e-7);
    }
}
