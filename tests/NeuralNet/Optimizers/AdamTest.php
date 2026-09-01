<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Adaptive;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Optimizers')]
#[CoversClass(Adaptive::class)]
class AdamTest extends TestCase
{
    protected Adam $optimizer;

    public static function invalidConstructorProvider() : Generator
    {
        // Invalid rates (<= 0)
        yield 'zero rate' => [0.0, 0.1, 0.001];
        yield 'negative rate' => [-0.5, 0.1, 0.001];

        // Invalid momentumDecay (<= 0 or >= 1)
        yield 'zero momentumDecay' => [0.001, 0.0, 0.001];
        yield 'negative momentumDecay' => [0.001, -0.1, 0.001];
        yield 'momentumDecay == 1' => [0.001, 1.0, 0.001];
        yield 'momentumDecay > 1' => [0.001, 1.1, 0.001];

        // Invalid normDecay (<= 0 or >= 1)
        yield 'zero normDecay' => [0.001, 0.1, 0.0];
        yield 'negative normDecay' => [0.001, 0.1, -0.1];
        yield 'normDecay == 1' => [0.001, 0.1, 1.0];
        yield 'normDecay > 1' => [0.001, 0.1, 1.1];
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
        $this->optimizer = new Adam(
            rate: 0.001,
            momentumDecay: 0.1,
            normDecay: 0.001
        );
    }

    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        $expected = 'Adam (rate: 0.001, momentum decay: 0.1, norm decay: 0.001)';
        self::assertSame($expected, (string) $this->optimizer);
    }

    #[Test]
    #[TestDox('Warm initializes zeroed velocity and norm caches with the parameter\'s shape')]
    public function warmInitializesZeroedCache() : void
    {
        $param = new Parameter(NumPower::array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]));

        // Warm the optimizer for this parameter
        $this->optimizer->warm($param);

        // Inspect protected cache via reflection
        $ref = new \ReflectionClass($this->optimizer);
        $prop = $ref->getProperty('cache');
        $prop->setAccessible(true);
        $cache = $prop->getValue($this->optimizer);

        self::assertArrayHasKey($param->id(), $cache);

        [$velocity, $norm] = $cache[$param->id()];

        $zeros = NumPower::zeros($param->param()->shape());
        self::assertEqualsWithDelta($zeros->toArray(), $velocity->toArray(), 0.0);
        self::assertEqualsWithDelta($zeros->toArray(), $norm->toArray(), 0.0);
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
            self::assertIsArray($entry);

            self::assertCount(2, $entry);

            foreach ($entry as $ndarray) {
                self::assertInstanceOf(NDArray::class, $ndarray);
                self::assertSame('float64', $ndarray->dataType());
            }
        }
    }

    /**
     * @param float $rate
     * @param float $momentumDecay
     * @param float $normDecay
     */
    #[Test]
    #[DataProvider('invalidConstructorProvider')]
    #[TestDox('Throws exception when constructed with invalid arguments')]
    public function invalidConstructorParams(float $rate, float $momentumDecay, float $normDecay) : void
    {
        $this->expectException(InvalidArgumentException::class);
        new Adam(rate: $rate, momentumDecay: $momentumDecay, normDecay: $normDecay);
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
    #[TestDox('Casts the cached velocity and norm NDArrays in place')]
    public function setCacheDataType() : void
    {
        $this->optimizer->warm(new Parameter(NumPower::array([1.0, 2.0])));

        $this->optimizer->setCacheDataType('float64');

        $ref = new \ReflectionClass($this->optimizer);
        $prop = $ref->getProperty('cache');
        $prop->setAccessible(true);

        foreach ($prop->getValue($this->optimizer) as $entry) {
            self::assertIsArray($entry);

            foreach ($entry as $ndarray) {
                self::assertInstanceOf(NDArray::class, $ndarray);
                self::assertSame('float64', $ndarray->dataType());
            }
        }
    }
}
