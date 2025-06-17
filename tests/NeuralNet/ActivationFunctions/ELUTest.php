<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Generator;

#[Group('ActivationFunctions')]
#[CoversClass(ELU::class)]
class ELUTest extends TestCase
{
    protected ELU $activationFn;

    public static function computeProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [1.0, -0.3934693402873666, 0.0, 20.0, -0.9999546000702375],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.11307956328284252, 0.31, -0.3873736058155839],
                [0.99, 0.08, -0.029554466451491845],
                [0.05, -0.4054794520298056, 0.54],
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
                [1.0, -0.3934693402873666, 0.0, 20.0, -0.9999546000702375],
            ]),
            [
                [1.0, 0.6065306597126334, 1.0, 1.0, 4.539992976249074E-5],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            Matrix::quick([
                [-0.11307956328284252, 0.31, -0.3873736058155839],
                [0.99, 0.08, -0.029554466451491845],
                [0.05, -0.4054794520298056, 0.54],
            ]),
            [
                [0.8869204367171575, 1.0, 0.6126263941844161],
                [1.0, 1.0, 0.9704455335485082],
                [1.0, 0.5945205479701944, 1.0],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->activationFn = new ELU(1.0);
    }

    public function testBadAlpha() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new ELU(-346);
    }

    /**
     * @param Matrix $input
     * @param list<list<float>> $expected $expected
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
     * @param list<list<float>> $expected $expected
     */
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(Matrix $input, Matrix $activations, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($input, $activations)->asArray();

        $this->assertEquals($expected, $derivatives);
    }
}
