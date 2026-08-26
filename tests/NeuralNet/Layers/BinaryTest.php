<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\Attributes\DataProvider;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use NDArray;
use NumPower;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Binary;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(Binary::class)]
class BinaryTest extends TestCase
{
    protected NDArray $input;

    /**
     * @var string[]
     */
    protected array $labels;

    protected Optimizer $optimizer;

    protected Binary $layer;

    /**
     * @return array<int, array{0:array}>
     */
    public static function forwardProvider() : array
    {
        return [
            [
                [
                    [0.7310585, 0.9241418, 0.4750207],
                ],
            ],
        ];
    }

    /**
     * @return array<int, array{0:array}>
     */
    public static function backProvider() : array
    {
        return [
            [
                [
                    [0.2436861, -0.0252860, 0.1583402],
                ],
            ],
        ];
    }

    /**
     * @return array<string, array{0: array<int, string>}>
     */
    public static function badClassesProvider() : array
    {
        return [
            'empty' => [[]],
            'single' => [['hot']],
            'duplicatesToOne' => [['hot', 'hot']],
            'threeUnique' => [['hot', 'cold', 'warm']],
        ];
    }

    protected function setUp() : void
    {
        $this->input = NumPower::array([
            [1.0, 2.5, -0.1],
        ]);

        $this->labels = ['hot', 'cold', 'hot'];

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Binary(classes: ['hot', 'cold'], costFn: new BinaryCrossEntropy());
    }

    #[Test]
    #[TestDox('Returns string representation')]
    public function testToString() : void
    {
        $this->layer->initialize(1);

        self::assertEquals('Binary (cost function: Binary Cross Entropy)', (string) $this->layer);
    }

    #[Test]
    #[TestDox('Initializes and reports width')]
    public function testInitializeWidth() : void
    {
        $this->layer->initialize(1);
        self::assertEquals(1, $this->layer->width());
    }

    #[Test]
    #[TestDox('Constructor rejects invalid classes arrays')]
    #[DataProvider('badClassesProvider')]
    public function testConstructorRejectsInvalidClasses(array $classes) : void
    {
        $this->expectException(InvalidArgumentException::class);
        new Binary(classes: $classes, costFn: new BinaryCrossEntropy());
    }

    #[Test]
    #[TestDox('Constructor accepts classes arrays that dedupe to exactly 2 labels')]
    public function testConstructorAcceptsDuplicateClassesThatDedupeToTwo() : void
    {
        $layer = new Binary(classes: ['hot', 'cold', 'hot'], costFn: new BinaryCrossEntropy());
        // Should initialize without throwing and report correct width
        $layer->initialize(1);
        self::assertEquals(1, $layer->width());
    }

    #[Test]
    #[TestDox('Computes forward pass')]
    #[DataProvider('forwardProvider')]
    public function testForward(array $expected) : void
    {
        $this->layer->initialize(1);

        $forward = $this->layer->forward($this->input);
        self::assertEqualsWithDelta($expected, $forward->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Backpropagates and returns gradient for previous layer')]
    #[DataProvider('backProvider')]
    public function testBack(array $expectedGradient) : void
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
    #[TestDox('Computes gradient directly given input, output and expected')]
    #[DataProvider('backProvider')]
    public function testGradient(array $expectedGradient) : void
    {
        $this->layer->initialize(1);

        $input = $this->input;
        $output = $this->layer->forward($input);

        // Build expected NDArray (1, batch) using the Binary classes mapping: hot=>0.0, cold=>1.0
        $expected = [];

        foreach ($this->labels as $label) {
            $expected[] = ($label === 'cold') ? 1.0 : 0.0;
        }
        $expected = NumPower::array([$expected]);

        $gradient = $this->layer->gradient($input, $output, $expected);

        self::assertInstanceOf(NDArray::class, $gradient);
        self::assertEqualsWithDelta($expectedGradient, $gradient->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Computes inference activations')]
    #[DataProvider('forwardProvider')]
    public function testInfer(array $expected) : void
    {
        $this->layer->initialize(1);

        $infer = $this->layer->infer($this->input);
        self::assertEqualsWithDelta($expected, $infer->toArray(), 1e-7);
    }
}
