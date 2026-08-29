<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Generator;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;

#[Group('Optimizers')]
#[CoversClass(Stochastic::class)]
class StochasticTest extends TestCase
{
    protected Stochastic $optimizer;

    public static function invalidConstructorProvider() : Generator
    {
        yield 'zero rate' => [0.0];
        yield 'negative rate' => [-0.001];
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
        $this->optimizer = new Stochastic(0.001);
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        self::assertEquals('Stochastic (rate: 0.001)', (string) $this->optimizer);
    }

    /**
     * @param float $rate
     */
    #[Test]
    #[DataProvider('invalidConstructorProvider')]
    #[TestDox('Throws exception when constructed with invalid arguments')]
    public function testInvalidConstructorParams(float $rate) : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Stochastic($rate);
    }

    /**
     * @param Parameter $param
     * @param NDArray $gradient
     * @param list<list<float>> $expected
     */
    #[Test]
    #[DataProvider('stepProvider')]
    #[TestDox('Can compute the step')]
    public function testStep(Parameter $param, NDArray $gradient, array $expected) : void
    {
        $step = $this->optimizer->step(param: $param, gradient: $gradient);

        self::assertEqualsWithDelta($expected, $step->toArray(), 1e-7);
    }
}
