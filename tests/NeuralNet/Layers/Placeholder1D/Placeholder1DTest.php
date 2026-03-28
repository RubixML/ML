<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers\Placeholder1D;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use NDArray;
use NumPower;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\Layers\Placeholder1D\Placeholder1D;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(Placeholder1D::class)]
class Placeholder1DTest extends TestCase
{
    protected NDArray $input;

    protected Placeholder1D $layer;

    /**
     * @return array<int, array{NDArray,array<int, array<int, float>>}>
     */
    public static function inputProvider() : array
    {
        return [
            [
                NumPower::array([
                    [1.0, 2.5],
                    [0.1, 0.0],
                    [0.002, -6.0],
                ]),
                [
                    [1.0, 2.5],
                    [0.1, 0.0],
                    [0.002, -6.0],
                ],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->input = NumPower::array([
            [1.0, 2.5],
            [0.1, 0.0],
            [0.002, -6.0],
        ]);

        $this->layer = new Placeholder1D(3);
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        self::assertEquals('Placeholder 1D (inputs: 3)', (string) $this->layer);
    }

    #[Test]
    #[TestDox('Returns width equal to number of inputs')]
    public function testWidth() : void
    {
        self::assertEquals(3, $this->layer->width());
    }

    #[Test]
    #[TestDox('Constructor rejects invalid number of inputs')]
    public function testConstructorRejectsInvalidInputs() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Placeholder1D(0);
    }

    #[Test]
    #[TestDox('Initialize returns fan out equal to inputs without changing width')]
    public function testInitialize() : void
    {
        $fanOut = $this->layer->initialize(5);

        self::assertEquals(3, $fanOut);
        self::assertEquals(3, $this->layer->width());
    }

    #[Test]
    #[TestDox('Computes forward pass')]
    #[DataProvider('inputProvider')]
    public function testForward(NDArray $input, array $expected) : void
    {
        self::assertEquals(3, $this->layer->width());

        $forward = $this->layer->forward($input);

        self::assertEqualsWithDelta($expected, $forward->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Computes inference pass')]
    #[DataProvider('inputProvider')]
    public function testInfer(NDArray $input, array $expected) : void
    {
        self::assertEquals(3, $this->layer->width());

        $infer = $this->layer->infer($input);

        self::assertEqualsWithDelta($expected, $infer->toArray(), 1e-7);
    }
}
