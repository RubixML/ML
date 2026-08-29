<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers;

use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Deferred;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Layers\Swish;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Parameter;

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

    /**
     * @return array<int, array<string, array<int, array<int, float>>>>
     */
    public static function initializeForwardBackInferProvider() : array
    {
        return [
            [
                'forwardExpected' => [
                    [0.7310586, 2.3103545, -0.0475021],
                    [0.0524979, 0.0524979, 2.8577223],
                    [0.0010010, -0.0148357, -0.1887703],
                ],
                'backExpected' => [
                    [0.2319176, 0.7695808, 0.0450083],
                    [0.2749584, 0.1099834, 0.0108810],
                    [0.1252500, -0.0012326, 0.2314346],
                ],
                'inferExpected' => [
                    [0.7309885, 2.3101985, -0.0475030],
                    [0.0524979, 0.0524979, 2.8577199],
                    [0.0010010, -0.0148412, -0.1887739],
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

    #[DataProvider('initializeForwardBackInferProvider')]
    #[Test]
    public function initializeForwardBackInfer(
        array $forwardExpected,
        array $backExpected,
        array $inferExpected,
    ) : void {
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

    #[Test]
    public function widthThrowsIfNotInitialized() : void
    {
        $layer = new Swish();

        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Layer has not been initialized.');

        $layer->width();
    }

    #[Test]
    public function initializeReturnsFanOutAndSetsWidth() : void
    {
        $fanIn = 4;
        $layer = new Swish(new Constant(1.0));

        $fanOut = $layer->initialize($fanIn);

        self::assertSame($fanIn, $fanOut);
        self::assertSame($fanIn, $layer->width());
    }

    #[Test]
    public function parametersAndRestore() : void
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

    #[Test]
    public function gradientMatchesBackpropagatedGradient() : void
    {
        $this->layer->initialize($this->fanIn);

        $output = $this->layer->forward($this->input);

        $parameters = iterator_to_array($this->layer->parameters());
        $beta = clone $parameters['beta']->param();

        $backGradient = $this->layer->back(
            prevGradient: $this->prevGrad,
            optimizer: $this->optimizer
        )->compute();

        $directGradient = $this->layer->gradient(
            input: $this->input,
            dOut: ($this->prevGrad)(),
            beta: $beta,
        );

        self::assertInstanceOf(NDArray::class, $directGradient);
        self::assertEqualsWithDelta($backGradient->toArray(), $directGradient->toArray(), 1e-7);
    }

    #[Test]
    public function initializeForwardBackInferWithNonDefaultBeta() : void
    {
        $layer = new Swish(new Constant(0.75));

        $layer->initialize(3);

        $input = NumPower::array([
            [1.5, 0.0, -2.0],
            [0.75, -0.25, 4.0],
            [0.0, -7.5, 0.001],
        ]);

        $prevGrad = new Deferred(function () {
            return NumPower::array([
                [0.9, 0.33, 0.05],
                [0.61, 0.44, 0.02],
                [0.77, 0.08, 0.95],
            ]);
        });

        $forward = $layer->forward($input);

        $expected = [
            [1.1323725, 0.0, -0.3648511],
            [0.4777731, -0.1133155, 3.8102965],
            [0.0, -0.0269520, 0.0005002],
        ];

        $this->assertInstanceOf(NDArray::class, $forward);
        $this->assertEqualsWithDelta($expected, $forward->toArray(), 1e-5);

        $gradient = $layer->back($prevGrad, $this->optimizer)->compute();

        $expected = [
            [0.8667545, 0.1650000, -0.0020647],
            [0.4679271, 0.1789904, 0.0217621],
            [0.3850000, -0.0013238, 0.4753563],
        ];

        $this->assertInstanceOf(NDArray::class, $gradient);
        $this->assertEqualsWithDelta($expected, $gradient->toArray(), 1e-5);

        $expected = [
            [1.1322041, 0.0, -0.3650924],
            [0.4777600, -0.1133170, 3.8102238],
            [0.0, -0.0269553, 0.0005002],
        ];

        $infer = $layer->infer($input);

        $this->assertInstanceOf(NDArray::class, $infer);
        $this->assertEqualsWithDelta($expected, $infer->toArray(), 1e-5);
    }

    #[Test]
    public function parametersRestoreRoundTrip() : void
    {
        $this->layer->initialize($this->fanIn);

        $parameters = iterator_to_array($this->layer->parameters());

        $this->assertArrayHasKey('beta', $parameters);
        $this->assertCount(1, $parameters);
        $this->assertInstanceOf(Parameter::class, $parameters['beta']);

        $fresh = new Swish(new Constant(1.0));
        $fresh->initialize($this->fanIn);
        $fresh->restore(['beta' => $parameters['beta']]);

        $restored = iterator_to_array($fresh->parameters())['beta'];

        $this->assertSame(
            $parameters['beta']->param()->toArray(),
            $restored->param()->toArray()
        );
    }
}
