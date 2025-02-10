<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(Multiclass::class)]
class MulticlassTest extends TestCase
{
    protected const int RANDOM_SEED = 0;

    protected Matrix $input;

    /**
     * @var string[]
     */
    protected array $labels;

    protected Optimizer $optimizer;

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

        $this->layer = new Multiclass(
            classes: ['hot', 'cold', 'ice cold'],
            costFn: new CrossEntropy()
        );

        srand(self::RANDOM_SEED);
    }

    public function testInitializeForwardBackInfer() : void
    {
        $this->layer->initialize(3);

        $this->assertEquals(3, $this->layer->width());

        $forward = $this->layer->forward($this->input);

        $expected = [
            [0.5633213801579335, 0.9239680829071899, 0.0418966244467313],
            [0.22902938185541574, 0.07584391881396309, 0.930019228325398],
            [0.2076492379866508, 0.0001879982788470176, 0.028084147227870816],
        ];

        $this->assertEqualsWithDelta($expected, $forward->asArray(), 1e-8);

        [$computation, $loss] = $this->layer->back(
            labels: $this->labels,
            optimizer: $this->optimizer
        );

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

        $this->assertEqualsWithDelta($expected, $infer->asArray(), 1e-8);
    }
}
