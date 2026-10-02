<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Binary;
use Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(Binary::class)]
class BinaryTest extends TestCase
{
    protected const RANDOM_SEED = 0;

    /**
     * @var Matrix
     */
    protected Matrix $x;

    /**
     * @var list<list<int>>
     */
    protected array $indices;

    /**
     * @var Binary
     */
    protected Binary $layer;

    protected function setUp() : void
    {
        $this->x = Matrix::fromArray([
            [1.0, 2.5, -0.1],
        ], false);

        $this->indices = [[0, 1, 0]];

        $this->layer = new Binary(new BinaryCrossEntropy());

        srand(self::RANDOM_SEED);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(Binary::class, $this->layer);
        $this->assertInstanceOf(Output::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
    }

    #[Test]
    public function initializeForwardBackInfer() : void
    {
        $this->layer->initialize(1);

        $this->assertEquals(1, $this->layer->width());

        $expected = [
            [0.7310585786300049, 0.9241418199787566, 0.47502081252106],
        ];

        $forward = $this->layer->forward($this->x);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEqualsWithDelta($expected, $forward->asArray(), 1e-8);

        [$computation, $loss] = $this->layer->back(Matrix::fromArray($this->indices, false));

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

        $infer = $this->layer->infer($this->x);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEqualsWithDelta($expected, $infer->asArray(), 1e-8);
    }

    /**
     * The gradient handed back to the previous layer must be the derivative of the
     * loss that back() reports. Binary folds its Sigmoid into the fused
     * cross-entropy gradient, so the loss must be evaluated through the Sigmoid in
     * order to differentiate with respect to the input.
     *
     * Two output nodes are used here so that the divisor is the total number of
     * elements rather than the number of columns, which the single node fixture
     * above cannot distinguish.
     */
    #[Test]
    public function gradientIsDerivativeOfReportedLoss() : void
    {
        $x = Matrix::fromArray([
            [1.0, 2.5, -0.1],
            [0.4, -1.2, 2.1],
        ], false);

        $y = Matrix::fromArray([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ], false);

        $this->layer->forward($x);

        [$computation, $loss] = $this->layer->back($y);

        $gradient = $computation->compute()->asArray();

        $this->assertIsFloat($loss);

        $epsilon = 1e-6;

        foreach ($x->asArray() as $i => $row) {
            foreach ($row as $j => $_) {
                $plus = $x->asArray();
                $minus = $x->asArray();

                $plus[$i][$j] += $epsilon;
                $minus[$i][$j] -= $epsilon;

                $numeric = ($this->lossOf($plus, $y) - $this->lossOf($minus, $y)) / (2 * $epsilon);

                $this->assertEqualsWithDelta($numeric, $gradient[$i][$j], 1e-6);
            }
        }
    }

    /**
     * Evaluate BinaryCrossEntropy over a matrix of logits.
     *
     * @param array<list<float>> $logits
     * @param Matrix $y
     * @return float
     */
    private function lossOf(array $logits, Matrix $y) : float
    {
        $probabilities = [];

        foreach ($logits as $row) {
            $activated = [];

            foreach ($row as $logit) {
                $activated[] = 1.0 / (1.0 + exp(-$logit));
            }

            $probabilities[] = $activated;
        }

        return (new BinaryCrossEntropy())->compute(Matrix::fromArray($probabilities, false), $y);
    }
}
