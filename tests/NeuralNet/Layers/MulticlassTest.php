<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers;

use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;
use Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(Multiclass::class)]
class MulticlassTest extends TestCase
{
    protected NDArray $input;

    /**
     * @var string[]
     */
    protected array $labels;

    protected Optimizer $optimizer;

    protected Multiclass $layer;

    /**
     * @return array<string, array{0: int}>
     */
    public static function initializeProvider() : array
    {
        return [
            'fanInEqualsClasses' => [3],
        ];
    }

    /**
     * @return array<string, array{0: array<int, array<int, float>>}>
     */
    public static function forwardProvider() : array
    {
        return [
            'expectedForward' => [[
                [0.1719820, 0.0498033, 0.6219707],
                [0.7707700, 0.0450639, 0.0015386],
                [0.0572478, 0.9051328, 0.3764906],
            ]],
        ];
    }

    /**
     * @return array<string, array{0: array<int, array<int, float>>}>
     */
    public static function backProvider() : array
    {
        return [
            'expectedGradient' => [[
                [-0.0920019, 0.0055337, 0.0691078],
                [0.0856411, -0.1061040, 0.0001709],
                [0.0063608, 0.1005703, -0.0692788],
            ]],
        ];
    }

    /**
     * @return array<string, array{0: array<int, array<int, float>>}>
     */
    public static function inferProvider() : array
    {
        // Same expectations as forward
        return self::forwardProvider();
    }

    protected function setUp() : void
    {
        // Column layout [classes, batch] matching Dense / FeedForward.
        $this->input = NumPower::array([
            [1.0, 0.1, 0.002],
            [2.5, 0.0, -6.0],
            [-0.1, 3.0, -0.5],
        ]);

        $this->labels = ['hot', 'cold', 'ice cold'];

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Multiclass(
            classes: ['hot', 'cold', 'ice cold'],
            costFn: new MulticlassCrossEntropy()
        );
    }

    #[Test]
    #[TestDox('Constructor rejects invalid number of classes')]
    public function constructorRejectsInvalidClasses() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Multiclass(classes: ['only-one-class'], costFn: new MulticlassCrossEntropy());
    }

    #[Test]
    #[TestDox('Method width() returns number of classes')]
    public function widthReturnsNumberOfClasses() : void
    {
        self::assertSame(3, $this->layer->width());
    }

    #[Test]
    #[TestDox('Initializes and returns correct fan out')]
    #[DataProvider('initializeProvider')]
    public function initializeReturnsFanOut(int $fanIn) : void
    {
        $fanOut = $this->layer->initialize($fanIn, dataType: 'float32');

        self::assertSame($fanIn, $fanOut);
        self::assertSame(3, $this->layer->width());
    }

    #[Test]
    #[TestDox('Computes forward softmax probabilities')]
    #[DataProvider('forwardProvider')]
    public function forward(array $expected) : void
    {
        $this->layer->initialize(3, dataType: 'float32');

        self::assertEquals(3, $this->layer->width());

        $forward = $this->layer->forward($this->input);

        self::assertEqualsWithDelta($expected, $forward->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Backpropagates and returns output gradient')]
    #[DataProvider('backProvider')]
    public function back(array $expected) : void
    {
        $this->layer->initialize(3, dataType: 'float32');

        // Set internal caches
        $this->layer->forward($this->input);

        [$computation, $loss] = $this->layer->back(
            labels: $this->labels,
            optimizer: $this->optimizer
        );

        self::assertInstanceOf(Deferred::class, $computation);
        self::assertIsFloat($loss);

        $gradient = $computation->compute();

        self::assertInstanceOf(NDArray::class, $gradient);
        self::assertEqualsWithDelta($expected, $gradient->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Computes gradient for previous layer directly')]
    #[DataProvider('backProvider')]
    public function gradient(array $expectedGradient) : void
    {
        $this->layer->initialize(3, dataType: 'float32');

        // Forward pass to obtain output probabilities
        $output = $this->layer->forward($this->input);

        // Rebuild expected one-hot matrix the same way as Multiclass::back()
        $expected = [];

        foreach (['hot', 'cold', 'ice cold'] as $class) {
            $row = [];

            foreach ($this->labels as $label) {
                $row[] = $class === $label ? 1.0 : 0.0;
            }

            $expected[] = $row;
        }

        $expectedNd = NumPower::array($expected);

        $gradient = $this->layer->gradient($this->input, $output, $expectedNd);

        self::assertEqualsWithDelta($expectedGradient, $gradient->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Computes exact Softmax Jacobian-vector product for losses other than Cross Entropy')]
    public function gradientWithRelativeEntropy() : void
    {
        $expectedGradient = [
            [-0.0920019936, 0.0055337012, 0.0691078631],
            [0.0856411220, -0.1061040102, 0.0001709579],
            [0.0063608715, 0.1005703090, -0.0692788210],
        ];

        $layer = new Multiclass(
            classes: ['hot', 'cold', 'ice cold'],
            costFn: new RelativeEntropy()
        );

        $layer->initialize(3, dataType: 'float32');

        $output = $layer->forward($this->input);

        // Rebuild expected one-hot matrix the same way as Multiclass::back()
        $expected = [];

        foreach (['hot', 'cold', 'ice cold'] as $class) {
            $row = [];

            foreach ($this->labels as $label) {
                $row[] = $class === $label ? 1.0 : 0.0;
            }

            $expected[] = $row;
        }

        $gradient = $layer->gradient($this->input, $output, NumPower::array($expected));

        self::assertEquals($this->input->shape(), $gradient->shape());
        self::assertEqualsWithDelta($expectedGradient, $gradient->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Computes infer softmax probabilities')]
    #[DataProvider('inferProvider')]
    public function infer(array $expected) : void
    {
        $this->layer->initialize(3, dataType: 'float32');

        $infer = $this->layer->infer($this->input);

        self::assertEqualsWithDelta($expected, $infer->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('It returns correct string representation')]
    public function toStringReturnsCorrectValue() : void
    {
        $expected = 'Multiclass (cost function: Multiclass Cross Entropy)';

        self::assertSame($expected, (string) $this->layer);
    }
}
