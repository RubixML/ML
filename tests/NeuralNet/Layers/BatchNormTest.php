<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Initializers\Constant;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;

#[Group('Layers')]
#[CoversClass(BatchNorm::class)]
class BatchNormTest extends TestCase
{
    /**
     * @var positive-int
     */
    protected int $fanIn;

    /**
     * @var Matrix
     */
    protected Matrix $input;

    /**
     * @var Deferred
     */
    protected Deferred $prevGrad;

    /**
     * @var Optimizer
     */
    protected Optimizer $optimizer;

    /**
     * @var BatchNorm
     */
    protected BatchNorm $layer;

    protected function setUp() : void
    {
        $this->fanIn = 3;

        $this->input = Matrix::quick([
            [1., 2.5, -0.1],
            [0.1, 0., 3.],
            [0.002, -6., -0.5],
        ]);

        $this->prevGrad = new Deferred(function () {
            return Matrix::quick([
                [0.25, 0.7, 0.1],
                [0.50, 0.2, 0.01],
                [0.25, 0.1, 0.89],
            ]);
        });

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new BatchNorm(0.9, new Constant(0.), new Constant(1.));
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(BatchNorm::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);
    }

    #[Test]
    public function initializeForwardBackInfer() : void
    {
        $this->layer->initialize($this->fanIn);

        $this->assertEquals($this->fanIn, $this->layer->width());

        $expected = [
            [-0.12512224941797084, 1.2825030565342015, -1.1573808071162308],
            [-0.6708631792558644, -0.7427413770332784, 1.4136045562891426],
            [0.7974157342978961, -1.4101900024437888, 0.6127742681458925],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEqualsWithDelta($expected, $forward->asArray(), 1e-8);

        $gradient = $this->layer->back($this->prevGrad, $this->optimizer)->compute();

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

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEqualsWithDelta($expected, $infer->asArray(), 1e-8);
    }

    #[Test]
    public function normalizesOverBatchSize() : void
    {
        $fanIn = 3;

        $input = Matrix::quick([
            [1.0, 2.5, -0.1, 0.5],
            [0.1, 0.0, 3.0, -1.0],
            [0.002, -6.0, -0.5, 2.0],
        ]);

        $prevGrad = new Deferred(function () {
            return Matrix::quick([
                [0.25, 0.7, 0.1, 0.3],
                [0.50, 0.2, 0.01, -0.4],
                [0.25, 0.1, 0.89, 0.6],
            ]);
        });

        $optimizer = new Stochastic(0.001);

        $layer = new BatchNorm(0.9, new Constant(0.), new Constant(1.));

        $layer->initialize($fanIn);

        $expected = [
            [0.025967457200229, 1.584014889214, -1.1166006596098, -0.49338168680435],
            [-0.28480067232747, -0.35181259522806, 1.6585450917894, -1.0219318242339],
            [0.3797862163235, -1.643717441354, 0.21054282476168, 1.0533884002688],
        ];

        $forward = $layer->forward($input);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEqualsWithDelta($expected, $forward->asArray(), 1e-8);

        $expected = [
            [-0.096655673462753, 0.024584160424222, 0.0014008068617791, 0.070670706176752],
            [0.29326885034768, 0.094619781903042, -0.10430387932137, -0.28358475292935],
            [-0.094806324008858, -0.017466016738221, 0.13166046771019, -0.019388126963108],
        ];

        $gradient = $layer->back($prevGrad, $optimizer)->compute();

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEqualsWithDelta($expected, $gradient->asArray(), 1e-8);

        $expected = [
            [0.024595238724167, 1.5813095621742, -1.1169952651392, -0.49430953575917],
            [-0.28505012503587, -0.35204780151489, 1.6578824928559, -1.0220245663052],
            [0.37766138009295, -1.6443246681253, 0.20854491954554, 1.0507583684868],
        ];

        $infer = $layer->infer($input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEqualsWithDelta($expected, $infer->asArray(), 1e-8);
    }
}
