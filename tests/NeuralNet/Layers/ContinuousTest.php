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
use Rubix\ML\NeuralNet\Layers\Continuous;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(Continuous::class)]
class ContinuousTest extends TestCase
{
    protected NDArray $input;

    /**
     * @var (int|float)[]
     */
    protected array $labels;

    protected Optimizer $optimizer;

    protected Continuous $layer;

    /**
     * @return array<int, array{0: array}>
     */
    public static function forwardProvider() : array
    {
        return [
            [
                [
                    [2.5, 0.0, -6.0],
                ],
            ],
        ];
    }

    /**
     * @return array<int, array{0: array}>
     */
    public static function gradientProvider() : array
    {
        return [
            [
                [
                    [0.8333333, 0.8333333, -32.0],
                ],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->input = NumPower::array([
            [2.5, 0.0, -6.0],
        ]);

        $this->labels = [0.0, -2.5, 90.0];

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Continuous(new LeastSquares());
    }

    #[TestDox('Returns string representation')]
    public function testToString() : void
    {
        $this->layer->initialize(1);

        self::assertEquals('Continuous (cost function: Least Squares)', (string) $this->layer);
    }

    #[Test]
    #[TestDox('Initializes and reports width')]
    public function initializeWidth() : void
    {
        $this->layer->initialize(1);
        self::assertEquals(1, $this->layer->width());
    }

    #[Test]
    #[TestDox('Initialize rejects fan-in not equal to 1')]
    public function initializeRejectsInvalidFanIn() : void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->layer->initialize(2);
    }

    #[Test]
    #[TestDox('Computes forward pass')]
    #[DataProvider('forwardProvider')]
    public function forward(array $expected) : void
    {
        $this->layer->initialize(1);

        $forward = $this->layer->forward($this->input);
        self::assertEqualsWithDelta($expected, $forward->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Backpropagates and returns gradient for previous layer')]
    #[DataProvider('gradientProvider')]
    public function back(array $expectedGradient) : void
    {
        $this->layer->initialize(1);
        $this->layer->forward($this->input);

        [$computation, $loss] = $this->layer->back(labels: $this->labels, optimizer: $this->optimizer);

        self::assertInstanceOf(Deferred::class, $computation);
        self::assertIsFloat($loss);

        $gradient = $computation->compute();

        self::assertInstanceOf(NDArray::class, $gradient);
        self::assertEqualsWithDelta($expectedGradient, $gradient->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Computes gradient directly given input and expected')]
    #[DataProvider('gradientProvider')]
    public function gradient(array $expectedGradient) : void
    {
        $this->layer->initialize(1);

        $input = $this->input;
        $expected = NumPower::array([$this->labels]);

        $gradient = $this->layer->gradient($input, $expected);

        self::assertInstanceOf(NDArray::class, $gradient);
        self::assertEqualsWithDelta($expectedGradient, $gradient->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Computes inference activations')]
    #[DataProvider('forwardProvider')]
    public function infer(array $expected) : void
    {
        $this->layer->initialize(1);

        $infer = $this->layer->infer($this->input);
        self::assertEqualsWithDelta($expected, $infer->toArray(), 1e-7);
    }
}
