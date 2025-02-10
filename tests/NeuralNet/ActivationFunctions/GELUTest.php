<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\GELU;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('ActivationFunctions')]
#[CoversClass(GELU::class)]
class GELUTest extends TestCase
{
    protected GELU $activationFn;

    public static function computeProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [0.841191990607477, -0.15428599017516514, 0.0, 20.0, -0.0],
            ],
        ];

        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
                [2.0, 0.5, 0.00001, -20.0, 1.0],
            ]),
            [
                [0.841191990607477, -0.15428599017516514, 0.0, 20.0, -0.0],
                [1.9545976940871754, 0.34571400982483486, 5.0000398942280396E-6, 0.0, 0.841191990607477],
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
                [0.7310585786300049, -0.1887703343990727, 0.0, 19.999999958776925, -0.00045397868702434395],
            ]),
            [
                [1.082963928002244, 0.13263021771495387, 0.5, 1.0, 0.0],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->activationFn = new GELU();
    }

    /**
     * @param Matrix $input
     * @param array<list<float>> $expected
     */
    #[DataProvider('computeProvider')]
    public function testCompute(Matrix $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->asArray();

        $this->assertEqualsWithDelta($expected, $activations, 1e-8);
    }

    /**
     * @param Matrix $input
     * @param Matrix $activations
     * @param array<list<float>> $expected
     */
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(Matrix $input, Matrix $activations, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate(z: $input, computed: $activations)->asArray();

        $this->assertEqualsWithDelta($expected, $derivatives, 1e-8);
    }
}
