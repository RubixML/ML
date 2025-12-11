<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers\Multiclass;

use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\NeuralNet\Optimizers\Base\Optimizer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Multiclass\Multiclass;
use Rubix\ML\NeuralNet\Optimizers\Stochastic\Stochastic;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy\CrossEntropy;
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
                [0.1719820, 0.7707700, 0.0572478],
                [0.0498033, 0.0450639, 0.9051327],
                [0.6219707, 0.0015385, 0.3764905],
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
                [-0.0920019, 0.0856411, 0.0063608],
                [0.0055337, -0.1061040, 0.1005703],
                [0.0691078, 0.00017093, -0.0692788],
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
        $this->input = NumPower::array([
            [1.0, 2.5, -0.1],
            [0.1, 0.0, 3.0],
            [0.002, -6.0, -0.5],
        ]);

        $this->labels = ['hot', 'cold', 'ice cold'];

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Multiclass(
            classes: ['hot', 'cold', 'ice cold'],
            costFn: new CrossEntropy()
        );
    }

    #[Test]
    #[TestDox('Constructor rejects invalid number of classes')]
    public function testConstructorRejectsInvalidClasses() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Multiclass(classes: ['only-one-class']);
    }

    #[Test]
    #[TestDox('Method width() returns number of classes')]
    public function testWidthReturnsNumberOfClasses() : void
    {
        self::assertSame(3, $this->layer->width());
    }

    #[Test]
    #[TestDox('Initializes and returns correct fan out')]
    #[DataProvider('initializeProvider')]
    public function testInitializeReturnsFanOut(int $fanIn) : void
    {
        $fanOut = $this->layer->initialize($fanIn);

        self::assertSame($fanIn, $fanOut);
        self::assertSame(3, $this->layer->width());
    }

    #[Test]
    #[TestDox('Computes forward softmax probabilities')]
    #[DataProvider('forwardProvider')]
    public function testForward(array $expected) : void
    {
        $this->layer->initialize(3);

        self::assertEquals(3, $this->layer->width());

        $forward = $this->layer->forward($this->input);

        self::assertEqualsWithDelta($expected, $forward->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Backpropagates and returns output gradient')]
    #[DataProvider('backProvider')]
    public function testBack(array $expected) : void
    {
        $this->layer->initialize(3);

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
    public function testGradient(array $expectedGradient) : void
    {
        $this->layer->initialize(3);

        // Forward pass to obtain output probabilities
        $output = $this->layer->forward($this->input);

        // Rebuild expected one-hot matrix the same way as Multiclass::back()
        $expected = [];

        foreach ($this->labels as $label) {
            $dist = [];

            foreach (['hot', 'cold', 'ice cold'] as $class) {
                $dist[] = $class === $label ? 1.0 : 0.0;
            }

            $expected[] = $dist;
        }

        $expectedNd = NumPower::array($expected);

        $gradient = $this->layer->gradient($this->input, $output, $expectedNd);

        self::assertEqualsWithDelta($expectedGradient, $gradient->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Computes infer softmax probabilities')]
    #[DataProvider('inferProvider')]
    public function testInfer(array $expected) : void
    {
        $this->layer->initialize(3);

        $infer = $this->layer->infer($this->input);

        self::assertEqualsWithDelta($expected, $infer->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('It returns correct string representation')]
    public function testToStringReturnsCorrectValue() : void
    {
        $expected = 'Multiclass (cost function: Cross Entropy)';

        self::assertSame($expected, (string) $this->layer);
    }
}
