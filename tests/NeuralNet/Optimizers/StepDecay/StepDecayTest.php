<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Optimizers\StepDecay;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\Optimizers\Stochastic\Stochastic;
use Rubix\ML\NeuralNet\Parameters\Parameter;
use Rubix\ML\NeuralNet\Optimizers\StepDecay\StepDecay;
use PHPUnit\Framework\TestCase;

#[Group('Optimizers')]
#[CoversClass(StepDecay::class)]
class StepDecayTest extends TestCase
{
    protected StepDecay $optimizer;

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

    #[Test]
    #[TestDox('Throws exception when constructed with invalid learning rate')]
    public function testConstructorWithInvalidRate() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new StepDecay(rate: 0.0);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with invalid losses')]
    public function testConstructorWithInvalidLosses() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new StepDecay(rate: 0.01, losses: 0);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with invalid decay')]
    public function testConstructorWithInvalidDecay() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new StepDecay(rate: 0.01, losses: 100, decay: -0.1);
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        self::assertEquals('Step Decay (rate: 0.001, steps: 100, decay: 0.001)', (string) $this->optimizer);
    }

    /**
     * @param Parameter $param
     * @param NDArray $gradient
     * @param list<list<float>> $expected
     */
    #[DataProvider('stepProvider')]
    public function testStep(Parameter $param, NDArray $gradient, array $expected) : void
    {
        $step = $this->optimizer->step(param: $param, gradient: $gradient);

        self::assertEqualsWithDelta($expected, $step->toArray(), 1e-7);
    }
}

