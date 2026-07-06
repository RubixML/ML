<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\Attributes\DataProvider;
use NDArray;
use NumPower;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
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

    protected NDArray $input;

    protected Deferred $prevGrad;

    protected Optimizer $optimizer;

    protected Activation $layer;

    /**
     * @return array<int, array{NDArray,array}>
     */
    public static function forwardProvider() : array
    {
        return [
            [
                NumPower::array([
                    [1.0, 2.5, -0.1],
                    [0.1, 0.0, 3.0],
                    [0.002, -6.0, -0.5],
                ]),
                [
                    [1.0, 2.5, 0.0],
                    [0.1, 0.0, 3.0],
                    [0.002, 0.0, 0.0],
                ],
            ],
        ];
    }

    /**
     * @return array<int, array{NDArray,NDArray,array}>
     */
    public static function backProvider() : array
    {
        return [
            [
                NumPower::array([
                    [1.0, 2.5, -0.1],
                    [0.1, 0.0, 3.0],
                    [0.002, -6.0, -0.5],
                ]),
                NumPower::array([
                    [0.25, 0.7, 0.1],
                    [0.50, 0.2, 0.01],
                    [0.25, 0.1, 0.89],
                ]),
                [
                    [0.25, 0.7, 0.0],
                    [0.5, 0.0, 0.01],
                    [0.25, 0, 0.0],
                ],
            ],
        ];
    }

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

        $this->layer = new Activation(new ReLU());
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        self::assertEquals('Activation (activation fn: ReLU)', (string) $this->layer);
    }

    #[Test]
    #[TestDox('Initializes width equal to fan-in')]
    public function testInitializeSetsWidth() : void
    {
        $this->layer->initialize($this->fanIn);

        self::assertEquals($this->fanIn, $this->layer->width());
    }

    #[Test]
    #[TestDox('Computes forward activations')]
    #[DataProvider('forwardProvider')]
    public function testForward(NDArray $input, array $expected) : void
    {
        $this->layer->initialize($this->fanIn);

        $forward = $this->layer->forward($input);
        self::assertEqualsWithDelta($expected, $forward->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Computes backpropagated gradients after forward pass')]
    #[DataProvider('backProvider')]
    public function testBack(NDArray $input, NDArray $prevGrad, array $expected) : void
    {
        $this->layer->initialize($this->fanIn);

        // Forward pass to set internal input/output state
        $this->layer->forward($input);

        $gradient = $this->layer
            ->back(prevGradient: new Deferred(fn: fn () => $prevGrad), optimizer: $this->optimizer)
            ->compute();

        self::assertEqualsWithDelta($expected, $gradient->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Computes inference activations')]
    #[DataProvider('forwardProvider')]
    public function testInfer(NDArray $input, array $expected) : void
    {
        $this->layer->initialize($this->fanIn);

        $infer = $this->layer->infer($input);
        self::assertEqualsWithDelta($expected, $infer->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Computes gradient correctly given input, output, and previous gradient')]
    #[DataProvider('backProvider')]
    public function testGradient(NDArray $input, NDArray $prevGrad, array $expected) : void
    {
        $this->layer->initialize($this->fanIn);

        // Produce output to pass explicitly to gradient
        $output = $this->layer->forward($input);

        $gradient = $this->layer->gradient(
            $input,
            $output,
            new Deferred(fn: fn () => $prevGrad)
        );

        self::assertEqualsWithDelta($expected, $gradient->toArray(), 1e-7);
    }
}
