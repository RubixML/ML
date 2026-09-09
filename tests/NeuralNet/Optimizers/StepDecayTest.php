<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Tensor\Tensor;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\StepDecay;
use Rubix\ML\NeuralNet\Optimizers\Scheduler;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Optimizers')]
#[CoversClass(StepDecay::class)]
class StepDecayTest extends TestCase
{
    /**
     * @var StepDecay
     */
    protected StepDecay $optimizer;

    /**
     * @return Generator<mixed[]>
     */
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
                [1e-5, 5e-5, -2e-5],
                [-1e-5, 2e-5, 3e-5],
                [4e-5, -1e-5, -0.0005],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->optimizer = new StepDecay(0.001);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(StepDecay::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    /**
     * @param Parameter $param
     * @param Tensor<int|float> $gradient
     * @param list<list<float>> $expected
     */
    #[DataProvider('stepProvider')]
    #[Test]
    public function step(Parameter $param, Tensor $gradient, array $expected) : void
    {
        $step = $this->optimizer->step($param, $gradient);

        $this->assertEquals($expected, $step->asArray());
    }

    /**
     * The schedule's internal counter must advance by one per batch,
     * not one per parameter. A network with K trainable parameters
     * performs K `step()` calls per batch but only one `tick()`.
     *
     * This test would fail on the legacy implementation where `step()`
     * itself incremented `t`: the `step()` calls within a batch would
     * see different `t` values and return different rates, and after
     * B batches the internal `t` would be K * B instead of B.
     */
    #[Test]
    public function scheduleTicksPerBatchNotPerParameter() : void
    {
        $rate = 0.01;
        $losses = 1;
        $decay = 0.5;

        $optimizer = new StepDecay($rate, $losses, $decay);

        $K = 3;
        $B = 3;

        $parameters = [];

        for ($i = 0; $i < $K; ++$i) {
            $parameters[] = new Parameter(Matrix::quick([
                [1.0],
            ]));
        }

        $gradient = Matrix::quick([[1.0]]);

        $expectedBatch = [
            0.01,
            0.01 / (1 + 1 * $decay),
            0.01 / (1 + 2 * $decay),
        ];

        for ($batch = 0; $batch < $B; ++$batch) {
            $rates = [];

            foreach ($parameters as $parameter) {
                $step = $optimizer->step($parameter, $gradient);

                $rates[] = $step->asArray()[0][0];
            }

            $expected = $expectedBatch[$batch];

            foreach ($rates as $actual) {
                $this->assertEqualsWithDelta($expected, $actual, 1e-12);
            }

            if ($optimizer instanceof Scheduler) {
                $optimizer->tick();
            }
        }

        $probe = $optimizer->step($parameters[0], $gradient);
        $expectedProbe = 0.01 / (1 + $B * $decay);

        $this->assertEqualsWithDelta($expectedProbe, $probe->asArray()[0][0], 1e-12);
    }
}
