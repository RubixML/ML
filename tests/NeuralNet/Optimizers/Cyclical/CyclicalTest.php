<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Optimizers\Cyclical;

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
use Rubix\ML\NeuralNet\Optimizers\Cyclical\Cyclical;
use Rubix\ML\NeuralNet\Parameters\Parameter;

#[Group('Optimizers')]
#[CoversClass(Cyclical::class)]
class CyclicalTest extends TestCase
{
    protected Cyclical $optimizer;

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
            ]
        ];
    }

    protected function setUp() : void
    {
        $this->optimizer = new Cyclical(lower: 0.001, upper: 0.006, losses: 2000);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with zero lower bound')]
    public function testConstructorWithZeroLower() : void
    {
        $this->expectException(InvalidArgumentException::class);
        new Cyclical(lower: 0.0, upper: 0.006, losses: 2000);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with negative lower bound')]
    public function testConstructorWithNegativeLower() : void
    {
        $this->expectException(InvalidArgumentException::class);
        new Cyclical(lower: -0.001, upper: 0.006, losses: 2000);
    }

    #[Test]
    #[TestDox('Throws exception when lower bound is greater than upper bound')]
    public function testConstructorWithLowerGreaterThanUpper() : void
    {
        $this->expectException(InvalidArgumentException::class);
        new Cyclical(lower: 0.01, upper: 0.006, losses: 2000);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with zero steps per cycle')]
    public function testConstructorWithZeroSteps() : void
    {
        $this->expectException(InvalidArgumentException::class);
        new Cyclical(lower: 0.001, upper: 0.006, losses: 0);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with negative steps per cycle')]
    public function testConstructorWithNegativeSteps() : void
    {
        $this->expectException(InvalidArgumentException::class);
        new Cyclical(lower: 0.001, upper: 0.006, losses: -5);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with zero decay')]
    public function testConstructorWithZeroDecay() : void
    {
        $this->expectException(InvalidArgumentException::class);
        new Cyclical(lower: 0.001, upper: 0.006, losses: 2000, decay: 0.0);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with decay equal to 1')]
    public function testConstructorWithDecayEqualToOne() : void
    {
        $this->expectException(InvalidArgumentException::class);
        new Cyclical(lower: 0.001, upper: 0.006, losses: 2000, decay: 1.0);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with decay greater than 1')]
    public function testConstructorWithDecayGreaterThanOne() : void
    {
        $this->expectException(InvalidArgumentException::class);
        new Cyclical(lower: 0.001, upper: 0.006, losses: 2000, decay: 1.5);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with negative decay')]
    public function testConstructorWithNegativeDecay() : void
    {
        $this->expectException(InvalidArgumentException::class);
        new Cyclical(lower: 0.001, upper: 0.006, losses: 2000, decay: -0.1);
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        self::assertEquals('Cyclical (lower: 0.001, upper: 0.006, steps: 2000, decay: 0.99994)', (string) $this->optimizer);
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
