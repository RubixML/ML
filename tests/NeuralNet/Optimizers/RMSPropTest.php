<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Tensor\Tensor;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\RMSProp;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Optimizers')]
#[CoversClass(RMSProp::class)]
class RMSPropTest extends TestCase
{
    protected RMSProp $optimizer;

    public static function stepProvider() : Generator
    {
        yield [
            new Parameter(Matrix::quick([
                [0.1, 0.6, -0.4],
                [0.5, 0.6, -0.4],
                [0.1, 0.1, -0.7],
            ])),
            Matrix::quick([
                [0.01, 0.05, -0.02],
                [-0.01, 0.02, 0.03],
                [0.04, -0.01, -0.5],
            ]),
            [
                [0.0031622776601683794, 0.003162277660168379, -0.0031622776601683794],
                [-0.0031622776601683794, 0.0031622776601683794, 0.0031622776601683794],
                [0.0031622776601683794, -0.0031622776601683794, -0.0031622776601683794],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->optimizer = new RMSProp(rate: 0.001, decay: 0.1);
    }

    /**
     * @param Parameter $param
     * @param Tensor<int|float> $gradient
     * @param list<list<float>> $expected
     */
    #[DataProvider('stepProvider')]
    public function testStep(Parameter $param, Tensor $gradient, array $expected) : void
    {
        $this->optimizer->warm($param);

        $step = $this->optimizer->step(param: $param, gradient: $gradient);

        $this->assertEquals($expected, $step->asArray());
    }
}
