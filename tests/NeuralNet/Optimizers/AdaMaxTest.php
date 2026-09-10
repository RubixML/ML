<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Tensor\Tensor;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\AdaMax;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Adam;
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
#[CoversClass(AdaMax::class)]
class AdaMaxTest extends TestCase
{
    /**
     * @var AdaMax
     */
    protected AdaMax $optimizer;

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
                [0.0001, 0.0001, -0.0001],
                [-0.0001, 0.0001, 0.0001],
                [0.0001, -0.0001, -0.0001],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->optimizer = new AdaMax(new Constant(0.001), 0.1, 0.001);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(AdaMax::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    #[Test]
    public function step() : void
    {
        $scheduler = new StepDecay(0.1, 2, 0.5);

        $optimizer = new AdaMax($scheduler);

        $initialRate = $scheduler->rate();

        $optimizer->scheduler()->tick();
        $optimizer->scheduler()->tick();

        $decreasedRate = $scheduler->rate();

        $this->assertLessThan($initialRate, $decreasedRate);
    }

    #[Test]
    public function badMomentumDecay() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->expectExceptionMessage('Momentum decay must be between 0 and 1, 1.5 given.');

        new AdaMax(new Constant(0.001), 1.5);
    }

    #[Test]
    public function badNormDecay() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->expectExceptionMessage('Norm decay must be between 0 and 1, 1.5 given.');

        new AdaMax(new Constant(0.001), 0.1, 1.5);
    }

    #[Test]
    public function stepIsSmallerThanAdam() : void
    {
        $param = new Parameter(Matrix::quick([[0.01, 0.05, -0.02]]));

        $gradient = Matrix::quick([[0.01, 0.05, -0.02]]);

        $adam = new Adam(new Constant(1.0), 0.1, 0.001);

        $adamax = new AdaMax(new Constant(1.0), 0.1, 0.001);

        $adam->warm($param);

        $adamax->warm($param);

        $param->accumulateGradient($gradient);

        $adamStep = $adam->update($param)->asArray()[0];

        $param->resetGradient();

        $param->accumulateGradient($gradient);

        $adamaxStep = $adamax->update($param)->asArray()[0];

        foreach ($adamStep as $i => $adamValue) {
            $this->assertLessThan(abs($adamValue), abs($adamaxStep[$i]));
        }
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

        $this->assertEqualsWithDelta($expected, $step->asArray(), 1e-8);
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
        $this->assertEquals('AdaMax (scheduler: Constant (rate: 0.001), momentum_decay: 0.1, norm_decay: 0.001)', (string) $this->optimizer);
    }
}
