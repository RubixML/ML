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
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\RMSProp;

#[Group('Optimizers')]
#[CoversClass(RMSProp::class)]
class RMSPropTest extends TestCase
{
    protected RMSProp $optimizer;

    public static function invalidConstructorProvider() : Generator
    {
        yield 'zero rate' => [0.0, 0.1];
        yield 'negative rate' => [-0.001, 0.1];
        yield 'zero decay' => [0.001, 0.0];
        yield 'decay == 1' => [0.001, 1.0];
        yield 'decay > 1' => [0.001, 1.5];
        yield 'negative decay' => [0.001, -0.1];
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
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        self::assertEquals('RMS Prop (rate: 0.001, decay: 0.1)', (string) $this->optimizer);
    }

    #[Test]
    #[DataProvider('invalidConstructorProvider')]
    #[TestDox('Throws exception when constructed with invalid arguments')]
    public function testInvalidConstructorParams(float $rate, float $decay) : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RMSProp(rate: $rate, decay: $decay);
    }

    #[Test]
    #[TestDox('Warm initializes a zeroed velocity cache with the parameter\'s shape')]
    public function testWarmInitializesZeroedCache() : void
    {
        $param = new Parameter(NumPower::array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]));

        // Warm the optimizer for this parameter
        $this->optimizer->warm($param);

        // Use reflection to read the protected cache
        $ref = new \ReflectionClass($this->optimizer);
        $prop = $ref->getProperty('cache');
        $prop->setAccessible(true);
        $cache = $prop->getValue($this->optimizer);

        self::assertArrayHasKey($param->id(), $cache);

        $velocity = $cache[$param->id()];

        // Verify the velocity is an all-zeros tensor of the correct shape
        $zeros = NumPower::zeros($param->param()->shape());
        self::assertEqualsWithDelta($zeros->toArray(), $velocity->toArray(), 0.0);
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
