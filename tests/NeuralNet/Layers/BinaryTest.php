<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\NeuralNet\Layers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Binary;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(Binary::class)]
class BinaryTest extends TestCase
{
    protected const int RANDOM_SEED = 0;

    protected Matrix $input;

    /**
     * @var string[]
     */
    protected array $labels;

    protected Optimizer $optimizer;

    protected Binary $layer;

    protected function setUp() : void
    {
        $this->input = Matrix::quick([
            [1.0, 2.5, -0.1],
        ]);

        $this->labels = ['hot', 'cold', 'hot'];

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Binary(classes: ['hot', 'cold'], costFn: new CrossEntropy());

        srand(self::RANDOM_SEED);
    }

    public function testInitializeForwardBackInfer() : void
    {
        $this->layer->initialize(1);

        $this->assertEquals(1, $this->layer->width());

        $expected = [
            [0.7310585786300049, 0.9241418199787566, 0.47502081252106],
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertEqualsWithDelta($expected, $forward->asArray(), 1e-8);

        [$computation, $loss] = $this->layer->back(labels: $this->labels, optimizer: $this->optimizer);

        $this->assertInstanceOf(Deferred::class, $computation);
        $this->assertIsFloat($loss);

        $gradient = $computation->compute();

        $expected = [
            [0.2436861928766683, -0.02528606000708115, 0.15834027084035332],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEqualsWithDelta($expected, $gradient->asArray(), 1e-8);

        $expected = [
            [0.7310585786300049, 0.9241418199787566, 0.47502081252106],
        ];

        $infer = $this->layer->infer($this->input);

        $this->assertEqualsWithDelta($expected, $infer->asArray(), 1e-8);
    }
}
