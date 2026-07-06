<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Optimizers\AdaMax;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use NDArray;
use NumPower;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\AdaMax;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Optimizers')]
#[CoversClass(AdaMax::class)]
class AdaMaxTest extends TestCase
{
    protected AdaMax $optimizer;

    public static function invalidConstructorProvider() : Generator
    {
        yield 'zero rate' => [0.0, 0.1, 0.001];
        yield 'negative rate' => [-0.001, 0.1, 0.001];
        yield 'zero momentum decay' => [0.001, 0.0, 0.001];
        yield 'momentum decay == 1' => [0.001, 1.0, 0.001];
        yield 'momentum decay > 1' => [0.001, 1.5, 0.001];
        yield 'negative momentum decay' => [0.001, -0.1, 0.001];
        yield 'zero norm decay' => [0.001, 0.1, 0.0];
        yield 'norm decay == 1' => [0.001, 0.1, 1.0];
        yield 'norm decay > 1' => [0.001, 0.1, 1.5];
        yield 'negative norm decay' => [0.001, 0.1, -0.1];
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
                [0.0001, 0.0001, -0.0001],
                [-0.0001, 0.0001, 0.0001],
                [0.0001, -0.0001, -0.0001],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->optimizer = new AdaMax(
            rate: 0.001,
            momentumDecay: 0.1,
            normDecay: 0.001
        );
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        self::assertEquals('AdaMax (rate: 0.001, momentum decay: 0.1, norm decay: 0.001)', (string) $this->optimizer);
    }

    /**
     * @param float $rate
     * @param float $momentumDecay
     * @param float $normDecay
     */
    #[Test]
    #[DataProvider('invalidConstructorProvider')]
    #[TestDox('Throws exception when constructed with invalid arguments')]
    public function testInvalidConstructorParams(float $rate, float $momentumDecay, float $normDecay) : void
    {
        $this->expectException(InvalidArgumentException::class);

        new AdaMax(rate: $rate, momentumDecay: $momentumDecay, normDecay: $normDecay);
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
        $this->optimizer->warm($param);

        $step = $this->optimizer->step(param: $param, gradient: $gradient);

        self::assertEqualsWithDelta($expected, $step->toArray(), 1e-7);
    }
}
