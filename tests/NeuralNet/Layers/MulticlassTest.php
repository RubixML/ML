<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;
use Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;

#[Group('Layers')]
#[CoversClass(Multiclass::class)]
class MulticlassTest extends TestCase
{
    protected const RANDOM_SEED = 0;

    /**
     * @var Matrix
     */
    protected Matrix $input;

    /**
     * @var string[]
     */
    protected array $labels;

    /**
     * @var Optimizer
     */
    protected Optimizer $optimizer;

    /**
     * @var Multiclass
     */
    protected Multiclass $layer;

    protected function setUp() : void
    {
        $this->input = Matrix::quick([
            [1.0, 2.5, -0.1],
            [0.1, 0.0, 3.0],
            [0.002, -6.0, -0.5],
        ]);

        $this->labels = ['hot', 'cold', 'ice cold'];

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Multiclass(['hot', 'cold', 'ice cold'], new MulticlassCrossEntropy());

        srand(self::RANDOM_SEED);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(Multiclass::class, $this->layer);
        $this->assertInstanceOf(Output::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
    }

    #[Test]
    public function initializeForwardBackInfer() : void
    {
        $this->layer->initialize(3);

        $this->assertEquals(3, $this->layer->width());

        $forward = $this->layer->forward($this->input);

        $expected = [
            [0.5633213801579335, 0.9239680829071899, 0.0418966244467313],
            [0.22902938185541574, 0.07584391881396309, 0.930019228325398],
            [0.2076492379866508, 0.0001879982788470176, 0.028084147227870816],
        ];

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEqualsWithDelta($expected, $forward->asArray(), 1e-8);

        [$computation, $loss] = $this->layer->back($this->labels, $this->optimizer);

        $this->assertInstanceOf(Deferred::class, $computation);
        $this->assertIsFloat($loss);

        $gradient = $computation->compute();

        $expected = [
            [-0.14555953994735552, 0.3079893609690633, 0.013965541482243765],
            [0.07634312728513858, -0.3080520270620123, 0.31000640944179936],
            [0.06921641266221694, 6.266609294900586E-5, -0.3239719509240431],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEqualsWithDelta($expected, $gradient->asArray(), 1e-8);

        $infer = $this->layer->infer($this->input);

        $expected = [
            [0.5633213801579335, 0.9239680829071899, 0.0418966244467313],
            [0.22902938185541574, 0.07584391881396309, 0.930019228325398],
            [0.2076492379866508, 0.0001879982788470176, 0.028084147227870816],
        ];

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEqualsWithDelta($expected, $infer->asArray(), 1e-8);
    }

    /**
     * The gradient with a non-cross-entropy loss exercises the Softmax Jacobian
     * path and its off-diagonal coupling.
     */
    #[Test]
    public function gradientWithSoftmaxJacobian() : void
    {
        $layer = new Multiclass(['hot', 'cold', 'ice cold'], new RelativeEntropy());

        $layer->initialize(3);

        $forward = $layer->forward($this->input);

        $expected = [
            [0.6, 0.1, 0.2],
            [0.3, 0.6, 0.1],
            [0.1, 0.3, 0.7],
        ];

        $gradient = $layer->gradient($this->input, $forward, Matrix::quick($expected));

        $expected = [
            [-0.012226206614022184, 0.27465602763572997, -0.0527011251844229],
            [-0.023656872714861417, -0.174718693728679, 0.276673076108466],
            [0.0358830793288836, -0.09993733390705099, -0.2239719509240431],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEqualsWithDelta($expected, $gradient->asArray(), 1e-8);
    }
}
