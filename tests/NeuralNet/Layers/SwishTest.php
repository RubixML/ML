<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Swish;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Initializers\Constant;
use PHPUnit\Framework\TestCase;

/**
 * @group Layers
 * @covers \Rubix\ML\NeuralNet\Layers\Swish
 */
class SwishTest extends TestCase
{
    protected const RANDOM_SEED = 0;

    /**
     * @var positive-int
     */
    protected $fanIn;

    /**
     * @var Matrix
     */
    protected $input;

    /**
     * @var Deferred
     */
    protected $prevGrad;

    /**
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * @var Swish
     */
    protected $layer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->fanIn = 3;

        $this->input = Matrix::quick([
            [1.0, 2.5, -0.1],
            [0.1, 0.1, 3.0],
            [0.002, -6.0, -0.5],
        ]);

        $this->prevGrad = new Deferred(function () {
            return Matrix::quick([
                [0.25, 0.7, 0.1],
                [0.50, 0.2, 0.01],
                [0.25, 0.1, 0.89],
            ]);
        });

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Swish(new Constant(1.0));

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Swish::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);
    }

    /**
     * @test
     */
    public function initializeForwardBackInfer() : void
    {
        $this->layer->initialize($this->fanIn);

        $this->assertEquals($this->fanIn, $this->layer->width());

        $forward = $this->layer->forward($this->input);

        $expected = [
            [0.7310585786300049, 2.3103545499468914, -0.047502081252106004],
            [0.052497918747894, 0.052497918747894, 2.8577223804673],
            [0.0010009999996666667, -0.014835738939808645, -0.1887703343990727],
        ];

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals($expected, $forward->asArray());

        $gradient = $this->layer->back($this->prevGrad, $this->optimizer)->compute();

        $expected = [
            [0.2319176279678717, 0.7695807779390686, 0.045008320850177086],
            [0.2749583957491146, 0.10998335829964585, 0.010881041060151694],
            [0.12524999983333343, -0.0012326432591525515, 0.2314345433006399],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEquals($expected, $gradient->asArray());

        $expected = [
            [0.73098855815682207, 2.3101984637539816, -0.047502969296234883],
            [0.052497904255326272, 0.052497904255326272, 2.8577200175383615],
            [0.0010009999384987184, -0.014841171304366609, -0.18877392808985191],
        ];

        $infer = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }

    /**
     * @test
     */
    public function betaGreaterThanOneForwardBackInfer() : void
    {
        $layer = new Swish(new Constant(2.0));

        $layer->initialize($this->fanIn);

        $forward = $layer->forward($this->input);

        $expected = [
            [0.88079707797788231, 2.4832678726892881, -0.045016600268752219],
            [0.054983399731247801, 0.054983399731247801, 2.9925821305300961],
            [0.0010019999973333376, -3.6865047613288306e-5, -0.13447071068499755],
        ];

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals($expected, $forward->asArray());

        $gradient = $layer->back($this->prevGrad, $this->optimizer)->compute();

        $expected = [
            [0.27269606219622389, 0.71858320270076581, 0.04006626881451502],
            [0.29966865592742498, 0.11986746237097, 0.01012326432591525],
            [0.12549999866666986, -6.7585467613783386e-6, 0.064373244434376795],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEquals($expected, $gradient->asArray());

        $expected = [
            [0.88079124218900928, 2.4832655631140583, -0.045016737841374749],
            [0.054983394893284243, 0.054983394893284243, 2.9925820871405762],
            [0.0010019999535649886, -3.6874729997420951e-5, -0.1344728620494075],
        ];

        $infer = $layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }
}
