<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\Softmax;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\ActivationFunctions\Softmax\Softmax;
use Tensor\Matrix;

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
        yield [
            NumPower::array([
                [2.0, 1.0, -0.5, 0.0],
            ]),
            [
                [0.6307954, 0.2320567, 0.0517789, 0.0853689],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.3097901, 0.4762271, 0.2139827],
                [0.5671765, 0.2283022, 0.2045210],
                [0.312711, 0.176846, 0.510443]
            ],
        ];

        // Test with zeros
        yield [
            NumPower::array([
                [0.0, 0.0, 0.0, 0.0],
            ]),
            [
                [0.25, 0.25, 0.25, 0.25],
            ],
        ];
    }

    /**
     * @return Generator<array>
     */
    public static function differentiateProvider() : Generator
    {
        // Test with simple values
        yield [
            NumPower::array([
                [0.6, 0.4],
            ]),
            [
                [0.24, -0.24],
                [-0.24, 0.24],
            ],
        ];

        // Test with more complex values
        yield [
            NumPower::array([
                [0.3, 0.5, 0.2],
            ]),
            [
                [0.21, -0.15, -0.06],
                [-0.15, 0.25, -0.10],
                [-0.06, -0.10, 0.16],
            ],
        ];
    }

    /**
     * @return Generator<array>
     */
    public static function sumToOneProvider() : Generator
    {
        // Test with various input values
        yield [
            NumPower::array([
                [10.0, -5.0, 3.0, 2.0],
            ]),
        ];

        yield [
            NumPower::array([
                [-10.0, -20.0, -30.0],
            ]),
        ];

        yield [
            NumPower::array([
                [0.1, 0.2, 0.3, 0.4],
                [5.0, 4.0, 3.0, 2.0],
                [-1.0, -2.0, -3.0, -4.0]
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
    #[TestDox('Correctly activates the input')]
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(NDArray $output, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($output)->toArray();

        $this->assertEqualsWithDelta($expected, $derivatives, 1e-7);
    }

    #[Test]
    #[TestDox('Output values always sum to 1')]
    #[DataProvider('sumToOneProvider')]
    public function testSumToOne(NDArray $input) : void
    {
        $activations = $this->activationFn->activate($input);

        // Convert to array for easier processing
        $activationsArray = $activations->toArray();

        // Check that each row sums to 1
        foreach ($activationsArray as $row) {
            $sum = array_sum($row);
            // Use a slightly larger delta to account for rounding errors
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
