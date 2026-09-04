<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\ThresholdedReLU;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('ActivationFunctions')]
#[CoversClass(ThresholdedReLU::class)]
class ThresholdedReLUTest extends TestCase
{
    /**
     * @var ThresholdedReLU
     */
    protected ThresholdedReLU $activationFn;

    /**
     * @return Generator<mixed[]>
     */
    public static function computeProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [1.0, 0.0, 0.0, 20.0, 0.0],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.0, 0.31, 0.0],
                [0.99, 0.0, 0.0],
                [0.0, 0.0, 0.54],
            ],
        ];
    }

    /**
     * @return Generator<mixed[]>
     */
    public static function differentiateProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            Matrix::quick([
                [1.0, 0.0, 0.0, 20.0, 0.0],
            ]),
            [
                [1, 0, 0, 1, 0],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            Matrix::quick([
                [0.0, 0.31, 0.0],
                [0.99, 0.0, 0.0],
                [0.0, 0.0, 0.54],
            ]),
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->activationFn = new ThresholdedReLU(0.1);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(ThresholdedReLU::class, $this->activationFn);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFn);
    }

    /**
     * @param Matrix $input
     * @param list<list<float>> $expected $expected
     */
    #[DataProvider('computeProvider')]
    #[Test]
    public function activate(Matrix $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->asArray();

        $this->assertEquals($expected, $activations);
    }

    /**
     * @param Matrix $input
     * @param Matrix $activations
     * @param list<list<float>> $expected $expected
     */
    #[DataProvider('differentiateProvider')]
    #[Test]
    public function differentiate(Matrix $input, Matrix $activations, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($input, $activations)->asArray();

        $this->assertEquals($expected, $derivatives);
    }
}
