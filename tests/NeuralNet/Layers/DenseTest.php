<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use PHPUnit\Framework\TestCase;

class DenseTest extends TestCase
{
    protected const RANDOM_SEED = 0;

    /**
     * @var int
     */
    protected $fanIn;

    /**
     * @var \Tensor\Matrix
     */
    protected $input;

    /**
     * @var \Rubix\ML\Deferred
     */
    protected $prevGrad;

    /**
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * @var \Rubix\ML\NeuralNet\Layers\Dense
     */
    protected $layer;

    public function setUp() : void
    {
        $this->fanIn = 3;

        $this->input = Matrix::quick([
            [1., 2.5, -0.1],
            [0.1, 0., 3.],
            [0.002, -6., -0.5],
        ]);

        $this->prevGrad = new Deferred(function () {
            return Matrix::quick([
                [0.50, 0.2, 0.01],
                [0.25, 0.1, 0.89],
            ]);
        });

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Dense(2, new He());

        srand(self::RANDOM_SEED);
    }

    public function test_build_layer() : void
    {
        $this->assertInstanceOf(Dense::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);

        $this->layer->initialize($this->fanIn);

        $this->assertEquals(2, $this->layer->width());
    }

    public function test_forward_back_infer() : void
    {
        $this->layer->initialize($this->fanIn);

        $expected = [
            [0.16048798423102456, -1.1251921552787814, 1.3407869114604536],
            [0.08541202409567002, -1.7363342816159082, -0.6972828043325793],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals($expected, $forward->asArray());

        $gradient = $this->layer->back($this->prevGrad, $this->optimizer)->compute();

        $expected = [
            [0.08105979414456366, 0.032423917657825464, 0.09199581049540438],
            [0.20137427617064546, 0.08054971046825819, -0.14969755527424305],
            [0.19988818890079615, 0.07995527556031846, 0.29776315973824286],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEquals($expected, $gradient->asArray());

        $expected = [
            [0.1587733922310246, -1.1356236552787815, 1.3393348114604533],
            [0.08349361309567001, -1.7448687816159083, -0.7070889543325792],
        ];

        $infer = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }
}
