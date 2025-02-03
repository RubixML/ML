<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Initializers\Constant;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(BatchNorm::class)]
class BatchNormTest extends TestCase
{
    /**
     * @var positive-int
     */
    protected int $fanIn;

    protected Matrix $input;

    protected Deferred $prevGrad;

    protected Optimizer $optimizer;

    protected BatchNorm $layer;

    protected function setUp() : void
    {
        $this->fanIn = 3;

        $this->input = Matrix::quick([
            [1., 2.5, -0.1],
            [0.1, 0., 3.],
            [0.002, -6., -0.5],
        ]);

        $this->prevGrad = new Deferred(fn: function () {
            return Matrix::quick([
                [0.25, 0.7, 0.1],
                [0.50, 0.2, 0.01],
                [0.25, 0.1, 0.89],
            ]);
        });

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new BatchNorm(
            decay: 0.9,
            betaInitializer: new Constant(0.),
            gammaInitializer: new Constant(1.)
        );
    }

    public function testInitializeForwardBackInfer() : void
    {
        $this->layer->initialize($this->fanIn);

        $this->assertEquals($this->fanIn, $this->layer->width());

        $expected = [
            [-0.12512224941797084, 1.2825030565342015, -1.1573808071162308],
            [-0.6708631792558644, -0.7427413770332784, 1.4136045562891426],
            [0.7974157342978961, -1.4101900024437888, 0.6127742681458925],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertEqualsWithDelta($expected, $forward->asArray(), 1e-8);

        $gradient = $this->layer->back(
            prevGradient: $this->prevGrad,
            optimizer: $this->optimizer
        )->compute();

        $expected = [
            [-0.06445877134888621, 0.027271018647605647, 0.03718775270128047],
            [0.11375900761901864, -0.10996704069838469, -0.0037919669206339162],
            [-0.11909780311643131, -0.01087038130262698, 0.1299681844190583],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEqualsWithDelta($expected, $gradient->asArray(), 1e-8);

        $expected = [
            [-0.12607831595417437, 1.2804902385302876, -1.1575619225761131],
            [-0.6718883801743488, -0.7438003494787433, 1.4135587296530918],
            [0.7956943312039361, -1.4105786650534555, 0.6111643338495193],
        ];

        $infer = $this->layer->infer($this->input);

        $this->assertEqualsWithDelta($expected, $infer->asArray(), 1e-8);
    }
}
