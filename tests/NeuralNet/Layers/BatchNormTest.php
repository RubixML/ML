<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use PHPUnit\Framework\TestCase;

class BatchNormTest extends TestCase
{
    protected $fanIn;

    protected $input;

    protected $prevGrad;

    protected $optimizer;

    protected $layer;

    public function setUp()
    {
        $this->fanIn = 3;

        $this->input = Matrix::quick([
            [1., 2.5, -0.1],
            [0.1, 0., 3.],
            [0.002, -6., -0.5],
        ]);

        $this->prevGrad = function () {
            return Matrix::quick([
                [0.25, 0.7, 0.1],
                [0.50, 0.2, 0.01],
                [0.25, 0.1, 0.89],
            ]);
        };

        $this->optimizer = new Stochastic();

        $this->layer = new BatchNorm();

        $this->layer->initialize($this->fanIn);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(BatchNorm::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);
    }

    public function test_width()
    {
        $this->assertEquals($this->fanIn, $this->layer->width());
    }

    public function test_forward_back_infer()
    {
        $forward = $this->layer->forward($this->input);

        $output = [
            [-0.12512224941797084, 1.2825030565342015, -1.1573808071162308],
            [-0.6708631792558644, -0.7427413770332784, 1.4136045562891426],
            [0.7974157342978961, -1.4101900024437888, 0.6127742681458925],
        ];

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals([3, 3], $forward->shape());
        $this->assertEquals($output, $forward->asArray());

        $back = $this->layer->back($this->prevGrad, $this->optimizer);

        $this->assertInternalType('callable', $back);

        $back = $back();

        $dYdX = [
            [-0.0849052805222375, 0.4108673492513107, -0.2796836491513708],
            [0.21773431346266278, 0.012460969795807176, -0.4380486992723796],
            [-0.07399439144834394, -0.009086169905025584, 0.17130353331384188],
        ];

        $this->assertInstanceOf(Matrix::class, $back);
        $this->assertEquals([3, 3], $back->shape());
        $this->assertEquals($dYdX, $back->asArray());

        $output = [
            [-0.12607831595417437, 1.2804902385302876, -1.1575619225761133],
            [-0.6718883801743488, -0.7438003494787433, 1.413558729653092],
            [0.7956943312039362, -1.4105786650534555, 0.6111643338495193],
        ];

        $infer = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals([3, 3], $infer->shape());
        $this->assertEquals($output, $infer->asArray());
    }
}
