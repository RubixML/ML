<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\StepDecay;
use PHPUnit\Framework\TestCase;

#[Group('Optimizers')]
#[CoversClass(StepDecay::class)]
class StepDecayTest extends TestCase
{
    protected StepDecay $optimizer;

    public static function invalidConstructorProvider() : Generator
    {
        yield 'zero rate' => [0.0, 100, 0.001];
        yield 'negative rate' => [-0.001, 100, 0.001];
        yield 'zero losses' => [0.01, 0, 0.001];
        yield 'negative losses' => [0.01, -5, 0.001];
        yield 'negative decay' => [0.01, 100, -0.1];
    }

    public static function stepProvider() : Generator
    {
        yield [
            new Parameter(NumPower::array([
                [0.1, 0.6, -0.4],
                [0.5, 0.6, -0.4],
                [0.1, 0.1, -0.7],
            ])),
            NumPower::array([
                [0.01, 0.05, -0.02],
                [-0.01, 0.02, 0.03],
                [0.04, -0.01, -0.5],
            ]),
            [
                [0.00001, 0.00005, -0.00002],
                [-0.00001, 0.00002, 0.00003],
                [0.00004, -0.00001, -0.0005],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->optimizer = new StepDecay(rate: 0.001);
    }

    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        self::assertEquals('Step Decay (rate: 0.001, steps: 100, decay: 0.001)', (string) $this->optimizer);
    }

    /**
     * @param float $rate
     * @param int $losses
     * @param float $decay
     */
    #[Test]
    #[DataProvider('invalidConstructorProvider')]
    #[TestDox('Throws exception when constructed with invalid arguments')]
    public function invalidConstructorParams(float $rate, int $losses, float $decay) : void
    {
        $this->expectException(InvalidArgumentException::class);

        new StepDecay(rate: $rate, losses: $losses, decay: $decay);
    }

    /**
     * @param Parameter $param
     * @param NDArray $gradient
     * @param list<list<float>> $expected
     */
    #[Test]
    #[DataProvider('stepProvider')]
    #[TestDox('Can compute the step')]
    public function step(Parameter $param, NDArray $gradient, array $expected) : void
    {
        $step = $this->optimizer->step(param: $param, gradient: $gradient);

        self::assertEqualsWithDelta($expected, $step->toArray(), 1e-7);
    }
}
