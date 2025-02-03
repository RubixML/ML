<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('ActivationFunctions')]
#[CoversClass(ReLU::class)]
class ReLUTest extends TestCase
{
    protected ReLU $activationFn;

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
                [0.99, 0.08, 0.0],
                [0.05, 0.0, 0.54],
            ],
        ];
    }

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
                [0.99, 0.08, 0.0],
                [0.05, 0.0, 0.54],
            ]),
            [
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 1],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->activationFn = new ReLU();
    }

    /**
     * @param Matrix $input
     * @param list<list<float>> $expected
     */
    #[DataProvider('computeProvider')]
    public function testActivate(Matrix $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->asArray();

        $this->assertEquals($expected, $activations);
    }

    /**
     * @param Matrix $input
     * @param Matrix $activations
     * @param list<list<float>> $expected
     */
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(Matrix $input, Matrix $activations, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($input, $activations)->asArray();

        $this->assertEquals($expected, $derivatives);
    }
}
