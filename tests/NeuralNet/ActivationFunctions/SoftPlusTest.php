<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\SoftPlus;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('ActivationFunctions')]
#[CoversClass(SoftPlus::class)]
class SoftPlusTest extends TestCase
{
    /**
     * @var SoftPlus
     */
    protected $activationFn;

    /**
     * @return Generator<mixed[]>
     */
    public static function computeProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [1.3132616875182228, 0.4740769841801067, 0.6931471805599453, 20.000000002061153, 4.539889921686465E-5],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [0.6349461015956135, 0.8601118864387145, 0.47786415060626164],
                [1.3059609474567209, 0.73394696731759, 0.6782596763414485],
                [0.7184596480132863, 0.46657309416461806, 0.9991627362708936],
            ],
        ];
    }

    /**
     * @return Generator<mixed[]>
     */
    public static function differentiateProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            Matrix::quick([
                [1.3132616875182228, 0.4740769841801067, 0.6931471805599453, 20.000000002061153, 4.5398899216870535E-5],
            ]),
            [
                [0.7310585786300049, 0.3775406687981454, 0.5, 0.9999999979388463, 4.5397868702434395E-5],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            Matrix::quick([
                [0.6349461015956135, 0.8601118864387145, 0.47786415060626164],
                [1.3059609474567209, 0.7339469673175899, 0.6782596763414485],
                [0.7184596480132864, 0.466573094164618, 0.9991627362708937],
            ]),
            [
                [0.4700359482354282, 0.5768852611320463, 0.3798935676569099],
                [0.7290879223493065, 0.5199893401555818, 0.4925005624493796],
                [0.5124973964842103, 0.3728522336868044, 0.6318124177361016],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->activationFn = new SoftPlus();
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(SoftPlus::class, $this->activationFn);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFn);
    }

    /**
     * @param Matrix $input
     * @param list<list<float>> $expected $expected
     */
    #[DataProvider('computeProvider')]
    #[Test]
    public function activate(Matrix $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->asArray();

        $this->assertEqualsWithDelta($expected, $activations, 1e-8);
    }

    /**
     * @param Matrix $input
     * @param Matrix $activations
     * @param list<list<float>> $expected $expected
     */
    #[DataProvider('differentiateProvider')]
    #[Test]
    public function differentiate(Matrix $input, Matrix $activations, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($input, $activations)->asArray();

        $this->assertEquals($expected, $derivatives);
    }
}
