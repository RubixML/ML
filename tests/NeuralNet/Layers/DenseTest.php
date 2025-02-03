<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Initializers\Constant;
use PHPUnit\Framework\TestCase;

#[Group('Layer')]
#[CoversClass(Dense::class)]
class DenseTest extends TestCase
{
    protected const int RANDOM_SEED = 0;

    /**
     * @var positive-int
     */
    protected int $fanIn;

    protected Matrix $input;

    protected Deferred $prevGrad;

    protected Optimizer $optimizer;

    protected Dense $layer;

    protected function setUp() : void
    {
        $this->fanIn = 3;

        $this->input = Matrix::quick([
            [1.0, 2.5, -0.1],
            [0.1, 0.0, 3.0],
            [0.002, -6.0, -0.5],
        ]);

        $this->prevGrad = new Deferred(fn: function () {
            return Matrix::quick([
                [0.50, 0.2, 0.01],
                [0.25, 0.1, 0.89],
            ]);
        });

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Dense(
            neurons: 2,
            l2Penalty: 0.0,
            bias: true,
            weightInitializer: new He(),
            biasInitializer: new Constant(0.0)
        );

        srand(self::RANDOM_SEED);
    }

    public function testInitializeForwardBackInfer() : void
    {
        $this->layer->initialize($this->fanIn);

        $this->assertEquals(2, $this->layer->width());

        $expected = [
            [0.1331636897703166, -2.659941938483866, 0.37781475642889195],
            [0.8082829632098398, -2.9282037817258764, 0.21589538926944302],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertEqualsWithDelta($expected, $forward->asArray(), 1e-8);

        $gradient = $this->layer->back(
            prevGradient: $this->prevGrad,
            optimizer: $this->optimizer
        )->compute();

        $expected = [
            [0.2513486032877107, 0.10053944131508427, 0.698223970571707],
            [0.16407184592276702, 0.0656287383691068, 0.2102008334557029],
            [0.44839890381544645, 0.1793595615261786, 0.7297101185916894],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEqualsWithDelta($expected, $gradient->asArray(), 1e-8);

        $expected = [
            [0.1314490977703166, -2.670373438483866, 0.376362656428892],
            [0.8063645522098398, -2.9367382817258765, 0.20608923926944314],
        ];

        $infer = $this->layer->infer($this->input);

        $this->assertEqualsWithDelta($expected, $infer->asArray(), 1e-8);
    }
}
