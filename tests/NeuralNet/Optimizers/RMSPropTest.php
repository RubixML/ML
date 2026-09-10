<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Tensor\Tensor;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\RMSProp;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\StepDecay;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Optimizers')]
#[CoversClass(RMSProp::class)]
class RMSPropTest extends TestCase
{
    /**
     * @var RMSProp
     */
    protected RMSProp $optimizer;

    /**
     * @return Generator<mixed[]>
     */
    public static function updateProvider() : Generator
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
        $this->optimizer = new RMSProp(new Constant(0.001), 0.1);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(RMSProp::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    #[Test]
    public function step() : void
    {
        $scheduler = new StepDecay(0.1, 2, 0.5);

        $optimizer = new RMSProp($scheduler);

        $initialRate = $scheduler->rate();

        $optimizer->scheduler()->tick();
        $optimizer->scheduler()->tick();

        $decreasedRate = $scheduler->rate();

        $this->assertLessThan($initialRate, $decreasedRate);
    }

    #[Test]
    public function badDecay() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->expectExceptionMessage('Decay must be between 0 and 1, 1.5 given.');

        new RMSProp(new Constant(0.001), 1.5);
    }

    /**
     * @param Parameter $param
     * @param Tensor<int|float> $gradient
     * @param list<list<float>> $expected
     */
    #[DataProvider('updateProvider')]
    #[Test]
    public function update(Parameter $param, Tensor $gradient, array $expected) : void
    {
        $this->optimizer->warm($param);

        $param->accumulateGradient($gradient);

        $step = $this->optimizer->update($param);

        $this->assertEquals($expected, $step->asArray());
    }

    #[Test]
    public function flush() : void
    {
        $param = new Parameter(Matrix::quick([[0.1, 0.2]]));

        $gradient = Matrix::quick([[0.01, -0.03]]);

        $this->optimizer->warm($param);

        $param->accumulateGradient($gradient);

        $this->optimizer->update($param);

        $param->resetGradient();

        $this->optimizer->flush();

        $this->optimizer->warm($param);

        $param->accumulateGradient($gradient);

        $step = $this->optimizer->update($param);

        $this->assertIsArray($step->asArray());
    }

    #[Test]
    public function stringRepresentation() : void
    {
        $this->assertEquals('RMS Prop (scheduler: Constant (rate: 0.001), decay: 0.1)', (string) $this->optimizer);
    }
}
