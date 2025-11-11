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

    public static function invalidConstructorProvider() : Generator
    {
        yield 'zero lower' => [0.0, 0.006, 2000, null];
        yield 'negative lower' => [-0.001, 0.006, 2000, null];
        yield 'lower > upper' => [0.01, 0.006, 2000, null];
        yield 'zero steps' => [0.001, 0.006, 0, null];
        yield 'negative steps' => [0.001, 0.006, -5, null];
        yield 'zero decay' => [0.001, 0.006, 2000, 0.0];
        yield 'decay == 1' => [0.001, 0.006, 2000, 1.0];
        yield 'decay > 1' => [0.001, 0.006, 2000, 1.5];
        yield 'negative decay' => [0.001, 0.006, 2000, -0.1];
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
            ]
        ];
    }

    protected function setUp() : void
    {
        $this->optimizer = new Cyclical(lower: 0.001, upper: 0.006, losses: 2000);
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        self::assertEquals('Cyclical (lower: 0.001, upper: 0.006, steps: 2000, decay: 0.99994)', (string) $this->optimizer);
    }

    /**
     * @param float $lower
     * @param float $upper
     * @param int $losses
     * @param float|null $decay
     * @return void
     */
    #[Test]
    #[DataProvider('invalidConstructorProvider')]
    #[TestDox('Throws exception when constructed with invalid arguments')]
    public function testConstructorInvalidArgs(float $lower, float $upper, int $losses, ?float $decay) : void
    {
        $this->expectException(InvalidArgumentException::class);

        if ($decay === null) {
            new Cyclical(lower: $lower, upper: $upper, losses: $losses);
        } else {
            new Cyclical(lower: $lower, upper: $upper, losses: $losses, decay: $decay);
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
    public function testStep(Parameter $param, NDArray $gradient, array $expected) : void
    {
        $step = $this->optimizer->step(param: $param, gradient: $gradient);

        self::assertEqualsWithDelta($expected, $step->toArray(), 1e-7);
    }
}
