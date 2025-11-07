<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Optimizers\RMSProp;

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
use Rubix\ML\NeuralNet\Parameters\Parameter;
use Rubix\ML\NeuralNet\Optimizers\RMSProp\RMSProp;

#[Group('Optimizers')]
#[CoversClass(RMSProp::class)]
class RMSPropTest extends TestCase
{
    protected RMSProp $optimizer;

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
                [0.0031622, 0.0031622, -0.0031622],
                [-0.0031622, 0.0031622, 0.0031622],
                [0.0031622, -0.0031622, -0.0031622],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->optimizer = new RMSProp(rate: 0.001, decay: 0.1);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with zero rate')]
    public function testConstructorWithZeroRate() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RMSProp(rate: 0.0);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with negative rate')]
    public function testConstructorWithNegativeRate() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RMSProp(rate: -0.001);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with zero decay')]
    public function testConstructorWithZeroDecay() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RMSProp(rate: 0.001, decay: 0.0);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with decay equal to 1')]
    public function testConstructorWithDecayEqualToOne() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RMSProp(rate: 0.001, decay: 1.0);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with decay greater than 1')]
    public function testConstructorWithDecayGreaterThanOne() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RMSProp(rate: 0.001, decay: 1.5);
    }

    #[Test]
    #[TestDox('Throws exception when constructed with negative decay')]
    public function testConstructorWithNegativeDecay() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RMSProp(rate: 0.001, decay: -0.1);
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        self::assertEquals('RMS Prop (rate: 0.001, decay: 0.1)', (string) $this->optimizer);
    }

    /**
     * @param Parameter $param
     * @param NDArray $gradient
     * @param list<list<float>> $expected
     */
    #[DataProvider('stepProvider')]
    public function testStep(Parameter $param, NDArray $gradient, array $expected) : void
    {
        $this->optimizer->warm($param);

        $step = $this->optimizer->step(param: $param, gradient: $gradient);

        self::assertEqualsWithDelta($expected, $step->toArray(), 1e-7);
    }
}
