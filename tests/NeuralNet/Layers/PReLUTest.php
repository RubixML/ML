<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers\PReLU;

use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Parameter as TrainableParameter;
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

    protected NDArray $input;

    protected Deferred $prevGrad;

    protected Optimizer $optimizer;

    protected PReLU $layer;

    /**
     * @return array<string, array{0:int}>
     */
    public static function initializeProvider() : array
    {
        return [
            'fanIn=3' => [3],
        ];
    }

    /**
     * @return array<string, array{0:array}>
     */
    public static function forwardProvider() : array
    {
        return [
            'expectedForward' => [[
                [1.0, 2.5, -0.025],
                [0.1, 0.0, 3.0],
                [0.002, -1.5, -0.125],
            ]],
        ];
    }

    /**
     * @return array<string, array{0:array}>
     */
    public static function backProvider() : array
    {
        return [
            'expectedGradient' => [[
                [0.25, 0.6999999, 0.0250010],
                [0.5, 0.05, 0.01],
                [0.25, 0.0251045, 0.2234300],
            ]],
        ];
    }

    /**
     * @return array<string, array{0:array}>
     */
    public static function gradientProvider() : array
    {
        return [
            'expectedGradient' => [[
                [0.25, 0.7, 0.025],
                [0.5, 0.05, 0.01],
                [0.25, 0.025, 0.2225],
            ]],
        ];
    }

    /**
     * @return array<string, array{0:array}>
     */
    public static function inferProvider() : array
    {
        return [
            'expectedInfer' => [[
                [1.0, 2.5, -0.0250000],
                [0.1, 0.0, 3.0],
                [0.0020000, -1.5, -0.125],
            ]],
        ];
    }

    /**
     * @return array<string, array{0:array,1:array}>
     */
    public static function activateProvider() : array
    {
        return [
            'defaultInput' => [
                [
                    [1.0, 2.5, -0.1],
                    [0.1, 0.0, 3.0],
                    [0.002, -6.0, -0.5],
                ],
                [
                    [1.0, 2.5, -0.025],
                    [0.1, 0.0, 3.0],
                    [0.002, -1.5, -0.125],
                ],
            ],
        ];
    }

    /**
     * @return array<string, array{0:array,1:array}>
     */
    public static function differentiateProvider() : array
    {
        return [
            'defaultInput' => [
                [
                    [1.0, 2.5, -0.1],
                    [0.1, 0.0, 3.0],
                    [0.002, -6.0, -0.5],
                ],
                [
                    [1.0, 1.0, 0.25],
                    [1.0, 0.25, 1.0],
                    [1.0, 0.25, 0.25],
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

        $this->layer = new PReLU(new Constant(0.25));

        srand(self::RANDOM_SEED);
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        self::assertEquals('PReLU (initializer: Constant (value: 0.25))', (string) $this->layer);
    }

    #[Test]
    #[TestDox('Initializes width equal to fan-in')]
    public function testInitializeSetsWidth() : void
    {
        $this->layer->initialize($this->fanIn);

        self::assertEquals($this->fanIn, $this->layer->width());
    }

    #[Test]
    #[TestDox('Initializes and returns fan out equal to fan-in')]
    #[DataProvider('initializeProvider')]
    public function testInitializeReturnsFanOut(int $fanIn) : void
    {
        $fanOut = $this->layer->initialize($fanIn);

        self::assertEquals($fanIn, $fanOut);
        self::assertEquals($fanIn, $this->layer->width());
    }

    #[Test]
    #[TestDox('Computes forward activations')]
    #[DataProvider('forwardProvider')]
    public function testForward(array $expected) : void
    {
        $this->layer->initialize($this->fanIn);

        $forward = $this->layer->forward($this->input);

        self::assertEqualsWithDelta($expected, $forward->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Backpropagates and returns gradient for previous layer')]
    #[DataProvider('backProvider')]
    public function testBack(array $expected) : void
    {
        $this->layer->initialize($this->fanIn);

        // Forward pass to set internal input state
        $this->layer->forward($this->input);

        $gradient = $this->layer->back(
            prevGradient: $this->prevGrad,
            optimizer: $this->optimizer
        )->compute();

        self::assertInstanceOf(NDArray::class, $gradient);
        self::assertEqualsWithDelta($expected, $gradient->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Computes gradient for previous layer directly')]
    #[DataProvider('gradientProvider')]
    public function testGradient(array $expected) : void
    {
        $this->layer->initialize($this->fanIn);

        $gradient = $this->layer->gradient(
            $this->input,
            ($this->prevGrad)(),
        );

        self::assertEqualsWithDelta($expected, $gradient->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Computes inference activations')]
    #[DataProvider('inferProvider')]
    public function testInfer(array $expected) : void
    {
        $this->layer->initialize($this->fanIn);

        $infer = $this->layer->infer($this->input);

        self::assertEqualsWithDelta($expected, $infer->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Yields trainable alpha parameter')]
    public function testParameters() : void
    {
        $this->layer->initialize($this->fanIn);

        $params = iterator_to_array($this->layer->parameters());

        self::assertArrayHasKey('alpha', $params);
        self::assertInstanceOf(TrainableParameter::class, $params['alpha']);
    }

    #[Test]
    #[TestDox('Restores alpha parameter from array')]
    public function testRestore() : void
    {
        $this->layer->initialize($this->fanIn);

        $alphaNew = new TrainableParameter(NumPower::full([$this->fanIn], 0.5));

        $this->layer->restore([
            'alpha' => $alphaNew,
        ]);

        $restored = iterator_to_array($this->layer->parameters());

        self::assertSame($alphaNew, $restored['alpha']);
        self::assertEquals(
            array_fill(0, $this->fanIn, 0.5),
            $restored['alpha']->param()->toArray(),
        );
    }
}
