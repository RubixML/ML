<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\ActivationFunctions\Softmax;

#[Group('ActivationFunctions')]
#[CoversClass(Softmax::class)]
class SoftmaxTest extends TestCase
{
    /**
     * @var Softmax
     */
    protected Softmax $activationFn;

    /**
     * @return Generator<array>
     */
    public static function computeProvider() : Generator
    {
        // Inputs use network layout [classes, batch].
        yield [
            NumPower::array([
                [2.0],
                [1.0],
                [-0.5],
                [0.0],
            ]),
            [
                [0.6307955],
                [0.2320567],
                [0.0517788],
                [0.0853688],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.99, 0.05],
                [0.31, 0.08, -0.52],
                [-0.49, -0.03, 0.54],
            ]),
            [
                [0.3097901, 0.5671766, 0.3127109],
                [0.4762272, 0.2283023, 0.1768459],
                [0.2139826, 0.2045210, 0.5104430],
            ],
        ];

        yield [
            NumPower::array([
                [0.0],
                [0.0],
                [0.0],
                [0.0],
            ]),
            [
                [0.25],
                [0.25],
                [0.25],
                [0.25],
            ],
        ];

        yield [
            NumPower::array([
                [1, 3],
                [2, 4],
            ]),
            [
                [0.2689414, 0.2689414],
                [0.7310585, 0.7310585],
            ],
        ];
    }

    /**
     * @return Generator<array>
     */
    public static function differentiateProvider() : Generator
    {
        yield [
            NumPower::array([
                [0.6],
                [0.4],
            ]),
            [
                [0.24],
                [0.24],
            ],
        ];

        yield [
            NumPower::array([
                [0.3],
                [0.5],
                [0.2],
            ]),
            [
                [0.21],
                [0.25],
                [0.16],
            ],
        ];

        yield [
            NumPower::array([
                [0.2689414],
                [0.7310585],
            ]),
            [
                [0.1966119],
                [0.1966120],
            ],
        ];

        // A batch of 3 samples must be differentiated independently per column.
        yield [
            NumPower::array([
                [0.3097901, 0.5671766, 0.3127109],
                [0.4762272, 0.2283023, 0.1768459],
                [0.2139826, 0.2045210, 0.5104430],
            ]),
            [
                [0.2138202, 0.2454873, 0.2149228],
                [0.2494349, 0.1761804, 0.1455714],
                [0.1681940, 0.1626922, 0.2498909],
            ],
        ];
    }

    /**
     * @return Generator<array>
     */
    public static function sumToOneProvider() : Generator
    {
        yield [
            NumPower::array([
                [10.0],
                [-5.0],
                [3.0],
                [2.0],
            ]),
        ];

        yield [
            NumPower::array([
                [-10.0],
                [-20.0],
                [-30.0],
            ]),
        ];

        yield [
            NumPower::array([
                [0.1, 5.0, -1.0],
                [0.2, 4.0, -2.0],
                [0.3, 3.0, -3.0],
                [0.4, 2.0, -4.0],
            ]),
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new Softmax();
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('Softmax', (string) $this->activationFn);
    }

    #[Test]
    #[TestDox('Correctly activates the input')]
    #[DataProvider('computeProvider')]
    public function testActivate(NDArray $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->toArray();

        static::assertEqualsWithDelta($expected, $activations, 1e-7);
    }

    #[Test]
    #[TestDox('Correctly differentiates the activation')]
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(NDArray $output, array $expected) : void
    {
        $input = NumPower::zeros($output->shape());
        $derivatives = $this->activationFn->differentiate($input, $output);

        static::assertEquals($output->shape(), $derivatives->shape());

        $this->assertEqualsWithDelta($expected, $derivatives->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Output values always sum to 1')]
    #[DataProvider('sumToOneProvider')]
    public function testSumToOne(NDArray $input) : void
    {
        $activations = $this->activationFn->activate($input)->toArray();

        $columns = count($activations[0]);

        for ($column = 0; $column < $columns; ++$column) {
            $sum = 0.0;

            foreach ($activations as $row) {
                $sum += $row[$column];
            }

            static::assertEqualsWithDelta(1.0, $sum, 1e-7);
        }
    }

    #[Test]
    #[TestDox('Output values are always between 0 and 1')]
    #[DataProvider('sumToOneProvider')]
    public function testOutputRange(NDArray $input) : void
    {
        $activations = $this->activationFn->activate($input)->toArray();

        foreach ($activations as $row) {
            foreach ($row as $value) {
                static::assertGreaterThanOrEqual(0.0, $value);
                static::assertLessThanOrEqual(1.0, $value);
            }
        }
    }
}
