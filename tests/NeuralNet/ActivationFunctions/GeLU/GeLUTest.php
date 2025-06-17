<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\GeLU;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\GeLU\GeLU;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group ActivationFunctions
 * @covers \Rubix\ML\NeuralNet\ActivationFunctions\GeLU\GeLU
 */
#[Group('ActivationFunctions')]
#[CoversClass(GeLU::class)]
class GeLUTest extends TestCase
{
    /**
     * @var GeLU
     */
    protected GeLU $activationFn;

    /**
     * @return Generator<array>
     */
    public static function computeProvider() : Generator
    {
        yield [
            NumPower::array([
                [2, 1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [1.9311704635620117, 0.8411920070648193, -0.15144622325897217, 0.0, 20.0, -0.0000005960464477539062],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.0540182888507843, 0.19418436288833618, -0.15014779567718506],
                [0.8305627107620239, 0.04266344755887985, -0.014624974690377712],
                [0.02604134939610958, -0.15386739373207092, 0.3839401602745056],
            ],
        ];
    }

    /**
     * @return Generator<array>
     */
    public static function differentiateProvider() : Generator
    {
        yield [
            NumPower::array([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [1.0839147567749023, 0.5289157032966614, 0.5, 1.0, 0.0],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.5218315124511719, 0.8476452827453613, 0.5337989926338196],
                [1.0788949728012085, 0.5460013151168823, 0.5148338675498962],
                [0.5230125784873962, 0.5249776244163513, 0.9520893096923828],
            ],
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new GeLU();
    }

    /**
     * @test
     */
    public function testToString() : void
    {
        static::assertEquals('GeLU', (string) $this->activationFn);
    }

    /**
     * @param NDArray $input
     * @param list<list<float>> $expected
     */
    #[DataProvider('computeProvider')]
    public function testActivate(NDArray $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->toArray();

        static::assertEquals($expected, $activations);
    }

    /**
     * @param NDArray $input
     * @param list<list<float>> $expected
     */
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(NDArray $input, array $expected) : void
    {
        static::markTestSkipped('Differentiation is not implemented very well');

        $derivatives = $this->activationFn->differentiate($input)->toArray();

        static::assertEquals($expected, $derivatives);
    }
}
