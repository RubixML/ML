<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Layers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Continuous;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(Continuous::class)]
class ContinuousTest extends TestCase
{
    protected const int RANDOM_SEED = 0;

    protected Matrix $input;

    /**
     * @var (int|float)[]
     */
    protected array $labels;

    protected Optimizer $optimizer;

    protected Continuous $layer;

    protected function setUp() : void
    {
        $this->input = Matrix::quick([
            [2.5, 0.0, -6.0],
        ]);

        $this->labels = [0.0, -2.5, 90];

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Continuous(new LeastSquares());

        srand(self::RANDOM_SEED);
    }

    public function testInitializeForwardBackInfer() : void
    {
        $this->layer->initialize(1);

        $this->assertEquals(1, $this->layer->width());

        $expected = [
            [2.5, 0.0, -6.0],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertEqualsWithDelta($expected, $forward->asArray(), 1e-8);

        [$computation, $loss] = $this->layer->back(labels: $this->labels, optimizer: $this->optimizer);

        $this->assertInstanceOf(Deferred::class, $computation);
        $this->assertIsFloat($loss);

        $gradient = $computation->compute();

        $expected = [
            [0.8333333333333334, 0.8333333333333334, -32.0],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEqualsWithDelta($expected, $gradient->asArray(), 1e-8);

        $expected = [
            [2.5, 0.0, -6.0],
        ];

        $infer = $this->layer->infer($this->input);

        $this->assertEqualsWithDelta($expected, $infer->asArray(), 1e-8);
    }
}
