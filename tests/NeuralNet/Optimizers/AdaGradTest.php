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
use Rubix\ML\NeuralNet\Optimizers\AdaGrad;
use Rubix\ML\NeuralNet\Parameter;

#[Group('Optimizers')]
#[CoversClass(AdaGrad::class)]
class AdaGradTest extends TestCase
{
    protected AdaGrad $optimizer;

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
                [0.001, 0.001, -0.001],
                [-0.001, 0.001, 0.001],
                [0.001, -0.001, -0.001],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->optimizer = new AdaGrad(0.001);
    }

    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        self::assertSame('AdaGrad (rate: 0.01)', (string) (new AdaGrad()));
    }

    /**
     * @param float $rate
     */
    #[Test]
    #[DataProvider('invalidConstructorProvider')]
    #[TestDox('Throws exception when constructed with invalid arguments')]
    public function invalidConstructorParams(float $rate) : void
    {
        $this->expectException(InvalidArgumentException::class);

        new AdaGrad(rate: $rate);
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
