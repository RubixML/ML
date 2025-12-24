<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers\Swish;

use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Deferred;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\NeuralNet\Initializers\Constant\Constant;
use Rubix\ML\NeuralNet\Layers\Swish\Swish;
use Rubix\ML\NeuralNet\Optimizers\Base\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Stochastic\Stochastic;
use Rubix\ML\NeuralNet\Parameters\Parameter;

#[Group('Layers')]
#[CoversClass(Swish::class)]
class SwishTest extends TestCase
{
    /**
     * @var positive-int
     */
    protected int $fanIn;

    protected NDArray $input;

    protected Deferred $prevGrad;

    protected Optimizer $optimizer;

    protected Swish $layer;

    protected function setUp() : void
    {
        $this->fanIn = 3;

        $this->input = NumPower::array([
            [1.0, 2.5, -0.1],
            [0.1, 0.1, 3.0],
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

        $this->layer = new Swish(new Constant(1.0));
    }

    /**
     * @return array<int, array<string, array<int, array<int, float>>>>
     */
    public static function initializeForwardBackInferProvider() : array
    {
        return [
            [
                'forwardExpected' => [
                    [0.7310585, 2.3103545, -0.0475020],
                    [0.0524979, 0.0524979, 2.8577223],
                    [0.0010009, -0.0148357, -0.1887703],
                ],
                'backExpected' => [
                    [0.2319176, 0.7695808, 0.0450083],
                    [0.2749583, 0.1099833, 0.0108810],
                    [0.1252499, -0.0012326, 0.2314345],
                ],
                'inferExpected' => [
                    [0.7306671, 2.3094806, -0.0475070],
                    [0.0524976, 0.0524976, 2.8576817],
                    [0.0010010, -0.0147432, -0.1887089],
                ],
            ],
        ];
    }

    /**
     * @return array<string, array{0: float, 1: string}>
     */
    public static function toStringProvider() : array
    {
        return [
            'value one' => [1.0, 'Swish (initializer: Constant (value: 1))'],
            'value zero' => [0.0, 'Swish (initializer: Constant (value: 0))'],
        ];
    }

    #[DataProvider('initializeForwardBackInferProvider')]
    public function testInitializeForwardBackInfer(
        array $forwardExpected,
        array $backExpected,
        array $inferExpected,
    ) : void
    {
        $this->layer->initialize($this->fanIn);

        self::assertEquals($this->fanIn, $this->layer->width());

        $forward = $this->layer->forward($this->input);

        self::assertEqualsWithDelta($forwardExpected, $forward->toArray(), 1e-7);

        $gradient = $this->layer->back(
            prevGradient: $this->prevGrad,
            optimizer: $this->optimizer
        )->compute();

        self::assertInstanceOf(NDArray::class, $gradient);
        self::assertEqualsWithDelta($backExpected, $gradient->toArray(), 1e-7);

        $infer = $this->layer->infer($this->input);

        self::assertEqualsWithDelta($inferExpected, $infer->toArray(), 1e-7);
    }

    #[DataProvider('toStringProvider')]
    public function testToString(float $value, string $expected) : void
    {
        $layer = new Swish(new Constant($value));

        self::assertSame($expected, (string) $layer);
    }

    public function testWidthThrowsIfNotInitialized() : void
    {
        $layer = new Swish();

        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Layer has not been initialized.');

        $layer->width();
    }

    public function testInitializeReturnsFanOutAndSetsWidth() : void
    {
        $fanIn = 4;
        $layer = new Swish(new Constant(1.0));

        $fanOut = $layer->initialize($fanIn);

        self::assertSame($fanIn, $fanOut);
        self::assertSame($fanIn, $layer->width());
    }

    public function testParametersAndRestore() : void
    {
        $this->layer->initialize($this->fanIn);

        $parameters = iterator_to_array($this->layer->parameters());

        self::assertArrayHasKey('beta', $parameters);
        self::assertInstanceOf(Parameter::class, $parameters['beta']);

        $betaParam = $parameters['beta'];
        $originalBeta = $betaParam->param()->toArray();

        $newLayer = new Swish(new Constant(0.0));
        $newLayer->initialize($this->fanIn);

        $newLayer->restore($parameters);

        $restoredParams = iterator_to_array($newLayer->parameters());

        self::assertArrayHasKey('beta', $restoredParams);
        self::assertInstanceOf(Parameter::class, $restoredParams['beta']);

        $restoredBeta = $restoredParams['beta']->param()->toArray();

        self::assertEquals($originalBeta, $restoredBeta);
    }

    public function testGradientMatchesBackpropagatedGradient() : void
    {
        $this->layer->initialize($this->fanIn);

        $output = $this->layer->forward($this->input);

        $backGradient = $this->layer->back(
            prevGradient: $this->prevGrad,
            optimizer: $this->optimizer
        )->compute();

        $directGradient = $this->layer->gradient(
            $this->input,
            $output,
            ($this->prevGrad)()
        );

        self::assertInstanceOf(NDArray::class, $directGradient);
        self::assertEqualsWithDelta($backGradient->toArray(), $directGradient->toArray(), 1e-7);
    }
}
