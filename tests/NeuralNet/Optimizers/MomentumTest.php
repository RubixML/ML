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
use PHPUnit\Framework\TestCase;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\Optimizers\Momentum;
use Rubix\ML\NeuralNet\Parameter;

#[Group('Optimizers')]
#[CoversClass(Momentum::class)]
class MomentumTest extends TestCase
{
    protected Momentum $optimizer;

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
                [0.00001, 0.00005, -0.00002],
                [-0.00001, 0.00002, 0.00003],
                [0.00004, -0.00001, -0.0005],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->optimizer = new Momentum(rate: 0.001, decay: 0.1, lookahead: false);
    }

    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        self::assertEquals('Momentum (rate: 0.001, decay: 0.1, lookahead: false)', (string) $this->optimizer);
    }

    /**
     * @param float $rate
     * @param float $decay
     */
    #[Test]
    #[DataProvider('invalidConstructorProvider')]
    #[TestDox('Throws exception when constructed with invalid arguments')]
    public function invalidConstructorParams(float $rate, float $decay) : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Momentum(rate: $rate, decay: $decay);
    }

    #[Test]
    #[TestDox('Warm initializes a zeroed velocity cache with the parameter\'s shape')]
    public function warmInitializesZeroedCache() : void
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

    #[Test]
    #[TestDox('Warms the cache with the parameter\'s data type')]
    public function warmWithFloat64Param() : void
    {
        $param = new Parameter(NumPower::array([1.0, 2.0], 'float64'));

        $this->optimizer->warm($param);

        $ref = new \ReflectionClass($this->optimizer);
        $prop = $ref->getProperty('cache');
        $prop->setAccessible(true);

        foreach ($prop->getValue($this->optimizer) as $entry) {
            self::assertInstanceOf(NDArray::class, $entry);
            self::assertSame('float64', $entry->dataType());
        }
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
        $this->optimizer->warm($param);

        $step = $this->optimizer->step(param: $param, gradient: $gradient);

        self::assertEqualsWithDelta($expected, $step->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Casts the cached NDArrays in place')]
    public function setCacheDataType() : void
    {
        $this->optimizer->warm(new Parameter(NumPower::array([1.0, 2.0])));

        $this->optimizer->setCacheDataType('float64');

        $ref = new \ReflectionClass($this->optimizer);
        $prop = $ref->getProperty('cache');
        $prop->setAccessible(true);

        foreach ($prop->getValue($this->optimizer) as $entry) {
            self::assertInstanceOf(NDArray::class, $entry);
            self::assertSame('float64', $entry->dataType());
        }
    }
}
