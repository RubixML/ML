<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers\Dense;

use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\Deferred;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\Initializers\Constant\Constant;
use Rubix\ML\NeuralNet\Layers\Dense\Dense;
use Rubix\ML\NeuralNet\Optimizers\Base\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Stochastic\Stochastic;
use Rubix\ML\NeuralNet\Initializers\He\HeUniform;
use Rubix\ML\NeuralNet\Parameters\Parameter as TrainableParameter;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(Dense::class)]
class DenseTest extends TestCase
{
    protected const int RANDOM_SEED = 0;

    /**
     * @var positive-int
     */
    protected int $fanIn;

    protected NDArray $input;

    protected Deferred $prevGrad;

    protected Optimizer $optimizer;

    protected Dense $layer;

    /**
     * @return array<int, array{array<array<float>>, array<float>, array<array<float>>}>
     */
    public static function forwardProvider() : array
    {
        return [
            [
                // weights 2x3
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                // biases length-2
                [0.0, 0.0],
                // expected forward output 2x3 for the fixed input in setUp()
                // input = [
                //   [1.0, 2.5, -0.1],
                //   [0.1, 0.0, 3.0],
                //   [0.002, -6.0, -0.5],
                // ];
                // so W * input = first two rows of input
                [
                    [1.0, 2.5, -0.1],
                    [0.1, 0.0, 3.0],
                ],
            ],
        ];
    }

    /**
     * @return array<int, array{array<array<float>>, array<float>, array<array<float>>, array<array<float>>}>
     */
    public static function backProvider() : array
    {
        return [
            [
                // weights 2x3
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                // biases length-2
                [0.0, 0.0],
                // prev gradient 2x3
                [
                    [0.50, 0.2, 0.01],
                    [0.25, 0.1, 0.89],
                ],
                // expected gradient for previous layer 3x3
                [
                    [0.50, 0.2, 0.01],
                    [0.25, 0.1, 0.89],
                    [0.0, 0.0, 0.0],
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
                [0.50, 0.2, 0.01],
                [0.25, 0.1, 0.89],
            ]);
        });

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Dense(
            neurons: 2,
            l2Penalty: 0.0,
            bias: true,
            weightInitializer: new HeUniform(),
            biasInitializer: new Constant(0.0)
        );

        srand(self::RANDOM_SEED);
    }

    #[Test]
    #[TestDox('Throws an exception for invalid constructor arguments')]
    public function testConstructorValidation() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Dense(
            neurons: 0,
            l2Penalty: -0.1,
            bias: true,
            weightInitializer: new HeUniform(),
            biasInitializer: new Constant(0.0)
        );
    }

    #[Test]
    #[TestDox('Computes forward activations for fixed weights and biases')]
    #[DataProvider('forwardProvider')]
    public function testForward(array $weights, array $biases, array $expected) : void
    {
        $this->layer->initialize($this->fanIn);
        self::assertEquals(2, $this->layer->width());

        $this->layer->restore([
            'weights' => new TrainableParameter(NumPower::array($weights)),
            'biases' => new TrainableParameter(NumPower::array($biases)),
        ]);

        $forward = $this->layer->forward($this->input);

        self::assertEqualsWithDelta($expected, $forward->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Method weights() returns the restored weight matrix')]
    public function testWeightsReturnsExpectedValues() : void
    {
        $this->layer->initialize($this->fanIn);

        $weightsArray = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];

        $this->layer->restore([
            'weights' => new TrainableParameter(NumPower::array($weightsArray)),
            'biases' => new TrainableParameter(NumPower::array([0.0, 0.0])),
        ]);

        $weights = $this->layer->weights();

        self::assertEqualsWithDelta($weightsArray, $weights->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('width() returns the number of neurons')]
    public function testWidthReturnsNeuronsCount() : void
    {
        // Layer is constructed in setUp() with neurons: 2
        self::assertSame(2, $this->layer->width());
    }

    #[Test]
    #[TestDox('Computes backpropagated gradients for previous layer')]
    #[DataProvider('backProvider')]
    public function testBack(array $weights, array $biases, array $prevGrad, array $expected) : void
    {
        $this->layer->initialize($this->fanIn);

        $this->layer->restore([
            'weights' => new TrainableParameter(NumPower::array($weights)),
            'biases' => new TrainableParameter(NumPower::array($biases)),
        ]);

        $prevGradNd = NumPower::array($prevGrad);

        // Forward pass to set internal input cache
        $this->layer->forward($this->input);

        $gradient = $this->layer->back(
            prevGradient: new Deferred(fn: fn () => $prevGradNd),
            optimizer: $this->optimizer
        )->compute();

        self::assertInstanceOf(NDArray::class, $gradient);
        self::assertEqualsWithDelta($expected, $gradient->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Computes inference activations equal to forward for fixed parameters')]
    #[DataProvider('forwardProvider')]
    public function testInfer(array $weights, array $biases, array $expected) : void
    {
        $this->layer->initialize($this->fanIn);

        $this->layer->restore([
            'weights' => new TrainableParameter(NumPower::array($weights)),
            'biases' => new TrainableParameter(NumPower::array($biases)),
        ]);

        $infer = $this->layer->infer($this->input);

        self::assertEqualsWithDelta($expected, $infer->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Method restore() correctly replaces layer parameters')]
    public function testRestoreReplacesParameters() : void
    {
        $this->layer->initialize($this->fanIn);

        // Use the same deterministic weights and biases as in forwardProvider
        $weights = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];

        $biases = [0.0, 0.0];

        $expected = [
            [1.0, 2.5, -0.1],
            [0.1, 0.0, 3.0],
        ];

        $this->layer->restore([
            'weights' => new TrainableParameter(NumPower::array($weights)),
            'biases' => new TrainableParameter(NumPower::array($biases)),
        ]);

        $forward = $this->layer->forward($this->input);

        self::assertEqualsWithDelta($expected, $forward->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Method parameters() yields restored weights and biases')]
    public function testParametersReturnsRestoredParameters() : void
    {
        $this->layer->initialize($this->fanIn);

        $weightsArray = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];

        $biasesArray = [0.0, 0.0];

        $weightsParam = new TrainableParameter(NumPower::array($weightsArray));
        $biasesParam = new TrainableParameter(NumPower::array($biasesArray));

        $this->layer->restore([
            'weights' => $weightsParam,
            'biases' => $biasesParam,
        ]);

        $params = iterator_to_array($this->layer->parameters());

        self::assertArrayHasKey('weights', $params);
        self::assertArrayHasKey('biases', $params);

        self::assertSame($weightsParam, $params['weights']);
        self::assertSame($biasesParam, $params['biases']);

        self::assertEqualsWithDelta($weightsArray, $params['weights']->param()->toArray(), 1e-7);
        self::assertEqualsWithDelta($biasesArray, $params['biases']->param()->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('It returns correct string representation')]
    public function testToStringReturnsCorrectValue() : void
    {
        $expected = 'Dense (neurons: 2, l2 penalty: 0, bias: true, weight initializer: He Uniform, bias initializer: Constant (value: 0))';

        self::assertSame($expected, (string) $this->layer);
    }
}
