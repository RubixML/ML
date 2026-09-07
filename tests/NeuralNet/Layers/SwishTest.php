<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Layers\Swish;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Initializers\Constant;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;

#[Group('Layers')]
#[CoversClass(Swish::class)]
class SwishTest extends TestCase
{
    protected const RANDOM_SEED = 0;

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
     * @var Swish
     */
    protected Swish $layer;

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

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(Swish::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);
    }

    #[Test]
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
            [0.2749583957491146, 0.10998335829964585, 0.010881041060151695],
            [0.12524999983333343, -0.0012326432591525513, 0.2314345433006399],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEquals($expected, $gradient->asArray());

        $expected = [
            [0.7309885581568221, 2.3101984637539816, -0.04750296929623488],
            [0.05249790425532627, 0.05249790425532627, 2.8577200175383615],
            [0.0010009999384987184, -0.014841171304366609, -0.1887739280898519],
        ];

        $infer = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }

    #[Test]
    public function initializeForwardBackInferWithNonDefaultBeta() : void
    {
        $layer = new Swish(new Constant(0.75));

        $layer->initialize(3);

        $input = Matrix::quick([
            [1.5, 0.0, -2.0],
            [0.75, -0.25, 4.0],
            [0.0, -7.5, 0.001],
        ]);

        $prevGrad = new Deferred(function () {
            return Matrix::quick([
                [0.9, 0.33, 0.05],
                [0.61, 0.44, 0.02],
                [0.77, 0.08, 0.95],
            ]);
        });

        $forward = $layer->forward($input);

        $expected = [
            [1.1323724803014423, 0.0, -0.3648510476127127],
            [0.4777730958602874, -0.11331546200384654, 3.8102965072897335],
            [0.0, -0.026952019360650677, 0.0005001874999912109],
        ];

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals($expected, $forward->asArray());

        $gradient = $layer->back($prevGrad, $this->optimizer)->compute();

        $expected = [
            [0.8667545670195208, 0.165, -0.0020647077149571467],
            [0.46792702600108194, 0.1789904306519722, 0.02176208212030339],
            [0.385, -0.001323821664344498, 0.4753562499666015],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEquals($expected, $gradient->asArray());

        $expected = [
            [1.1322040679936571, 0.0, -0.36509242346948234],
            [0.477760010156692, -0.11331702029615623, 3.810223770680048],
            [0.0, -0.026955265002565797, 0.0005001874959628773],
        ];

        $infer = $layer->infer($input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }

    #[Test]
    public function parametersRestoreRoundTrip() : void
    {
        $this->layer->initialize($this->fanIn);

        $parameters = iterator_to_array($this->layer->parameters());

        $this->assertArrayHasKey('beta', $parameters);
        $this->assertCount(1, $parameters);
        $this->assertInstanceOf(Parameter::class, $parameters['beta']);

        $fresh = new Swish(new Constant(1.0));
        $fresh->initialize($this->fanIn);
        $fresh->restore(['beta' => $parameters['beta']]);

        $restored = iterator_to_array($fresh->parameters())['beta'];

        $this->assertSame(
            $parameters['beta']->param()->asArray(),
            $restored->param()->asArray()
        );
    }
}
