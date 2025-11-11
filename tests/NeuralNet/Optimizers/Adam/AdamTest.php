<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Optimizers\Adam;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use NDArray;
use NumPower;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\Parameters\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Adam\Adam;
use Rubix\ML\NeuralNet\Optimizers\Base\Adaptive;
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
        yield [0.0, 0.1, 0.001];
        yield [-0.5, 0.1, 0.001];

        // Invalid momentumDecay (<= 0 or >= 1)
        yield [0.001, 0.0, 0.001];
        yield [0.001, -0.1, 0.001];
        yield [0.001, 1.0, 0.001];
        yield [0.001, 1.1, 0.001];

        // Invalid normDecay (<= 0 or >= 1)
        yield [0.001, 0.1, 0.0];
        yield [0.001, 0.1, -0.1];
        yield [0.001, 0.1, 1.0];
        yield [0.001, 0.1, 1.1];
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

    public function testToString() : void
    {
        $expected = 'Adam (rate: 0.001, momentum decay: 0.1, norm decay: 0.001)';
        self::assertSame($expected, (string) $this->optimizer);
    }

    #[DataProvider('invalidConstructorProvider')]
    public function testInvalidConstructorParams(float $rate, float $momentumDecay, float $normDecay) : void
    {
        $this->expectException(InvalidArgumentException::class);
        new Adam(rate: $rate, momentumDecay: $momentumDecay, normDecay: $normDecay);
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
