<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions\HyperbolicTangent;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\HyperbolicTangent\HyperbolicTangent;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('ActivationFunctions')]
#[CoversClass(HyperbolicTangent::class)]
class HyperbolicTangentTest extends TestCase
{
    /**
     * @var HyperbolicTangent
     */
    protected HyperbolicTangent $activationFn;

    /**
     * @return Generator<array>
     */
    public static function computeProvider() : Generator
    {
        yield [
            NumPower::array([
                [9.0, 2.5, 2.0, 1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [0.9999999403953552, 0.9866142868995667, 0.9640275835990906, 0.7615941762924194, -0.46211716532707214, 0.0, 1.0, -1.0],
            ],
        ];

        yield [
            NumPower::array([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.11942730098962784, 0.3004370927810669, -0.45421645045280457],
                [0.7573622465133667, 0.07982976734638214, -0.02999100275337696],
                [0.049958378076553345, -0.47769999504089355, 0.4929879903793335],
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
                [0.9640275835990906, 0.7615941762924194, -0.46211716532707214, 0.0, 1.0, -1.0],
            ]),
            [
                [0.07065081596374512, 0.41997432708740234, 0.7864477038383484, 1.0, 0.0, 0.0],
            ],
        ];

        yield [
            NumPower::array([
                [-0.11942730098962784, 0.3004370927810669, -0.45421645045280457],
                [0.7573623085022249, 0.07978830223560329, -0.029991223630861304],
                [0.049958395721942955, -0.4778087574005698, 0.4930591567725708],
            ]),
            [
                [0.985737144947052, 0.9097375273704529, 0.7936874032020569],
                [0.42640233039855957, 0.9936338067054749, 0.9991005063056946],
                [0.9975041747093201, 0.7716988325119019, 0.7568926811218262],
            ],
        ];
    }

    /**
     * Set up the test case.
     */
    protected function setUp() : void
    {
        parent::setUp();

        $this->activationFn = new HyperbolicTangent();
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        static::assertEquals('Hyperbolic Tangent', (string) $this->activationFn);
    }

    #[Test]
    #[TestDox('Correctly activates the input')]
    #[DataProvider('computeProvider')]
    public function testActivate(NDArray $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->toArray();

        static::assertEqualsWithDelta($expected, $activations, 1e-16);
    }

    #[Test]
    #[TestDox('Correctly differentiates the output')]
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(NDArray $output, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($output)->toArray();

        static::assertEqualsWithDelta($expected, $derivatives, 1e-16);
    }
}
