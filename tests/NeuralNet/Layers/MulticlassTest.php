<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;
use Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(Multiclass::class)]
class MulticlassTest extends TestCase
{
    protected const RANDOM_SEED = 0;

    /**
     * @var Matrix
     */
    protected Matrix $x;

    /**
     * @var array<list<int>>
     */
    protected array $expected;

    /**
     * @var Multiclass
     */
    protected Multiclass $layer;

    protected function setUp() : void
    {
        $this->x = Matrix::fromArray([
            [1.0, 2.5, -0.1],
            [0.1, 0.0, 3.0],
            [0.002, -6.0, -0.5],
        ], false);

        $this->expected = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ];

        $this->layer = new Multiclass(3, new MulticlassCrossEntropy());

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

        $forward = $this->layer->forward($this->x);

        $expected = [
            [0.5633213801579335, 0.9239680829071899, 0.0418966244467313],
            [0.22902938185541574, 0.07584391881396309, 0.930019228325398],
            [0.2076492379866508, 0.0001879982788470176, 0.028084147227870816],
        ];

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEqualsWithDelta($expected, $forward->asArray(), 1e-8);

        [$computation, $loss] = $this->layer->back(Matrix::fromArray($this->expected, false));

        $this->assertInstanceOf(Deferred::class, $computation);
        $this->assertIsFloat($loss);

        $gradient = $computation->compute();

        $expected = [
            [-0.04851984664911851, 0.1026631203230211, 0.004655180494081254],
            [0.02544770909504619, -0.10268400902067076, 0.1033354698139331],
            [0.02307213755407231, 2.088869764966862E-5, -0.10799065030801436],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEqualsWithDelta($expected, $gradient->asArray(), 1e-8);

        $infer = $this->layer->infer($this->x);

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
        $layer = new Multiclass(3, new RelativeEntropy());

        $layer->initialize(3);

        $forward = $layer->forward($this->x);

        $expected = [
            [0.6, 0.1, 0.2],
            [0.3, 0.6, 0.1],
            [0.1, 0.3, 0.7],
        ];

        $gradient = $layer->gradient($forward, Matrix::fromArray($expected, false));

        $expected = [
            [-0.004075402204674061, 0.09155200921190998, -0.017567041728140966],
            [-0.00788562423828714, -0.058239564576226324, 0.09222435870282196],
            [0.011961026442961199, -0.03331244463568366, -0.074657316974681],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEqualsWithDelta($expected, $gradient->asArray(), 1e-8);
    }

    /**
     * The gradient handed back to the previous layer must be the derivative of the
     * loss that back() reports. Rows are classes and columns are samples, so the
     * Softmax normalizes each column and the loss is averaged over all elements.
     */
    #[Test]
    public function gradientIsDerivativeOfReportedLoss() : void
    {
        $this->layer->initialize(3);

        $this->layer->forward($this->x);

        $y = Matrix::fromArray($this->expected, false);

        [$computation, $loss] = $this->layer->back($y);

        $gradient = $computation->compute()->asArray();

        $this->assertIsFloat($loss);

        $epsilon = 1e-6;

        $logits = $this->x->asArray();

        foreach ($logits as $i => $row) {
            foreach ($row as $j => $_) {
                $plus = $logits;
                $minus = $logits;

                $plus[$i][$j] += $epsilon;
                $minus[$i][$j] -= $epsilon;

                $numeric = ($this->lossOf($plus, $y) - $this->lossOf($minus, $y)) / (2 * $epsilon);

                $this->assertEqualsWithDelta($numeric, $gradient[$i][$j], 1e-6);
            }
        }
    }

    /**
     * Evaluate MulticlassCrossEntropy over a matrix of logits.
     *
     * @param array<list<float>> $logits
     * @param Matrix $y
     * @return float
     */
    private function lossOf(array $logits, Matrix $y) : float
    {
        $classes = count($logits);
        $columns = count($logits[0]);

        $normalized = [];

        for ($j = 0; $j < $columns; ++$j) {
            $column = [];

            for ($i = 0; $i < $classes; ++$i) {
                $column[] = $logits[$i][$j];
            }

            $maximum = max($column);

            $exponentials = [];
            $sum = 0.0;

            foreach ($column as $logit) {
                $exponential = exp($logit - $maximum);

                $exponentials[] = $exponential;

                $sum += $exponential;
            }

            foreach ($exponentials as $exponential) {
                $normalized[] = $exponential / $sum;
            }
        }

        $probabilities = array_fill(0, $classes, array_fill(0, $columns, 0.0));

        foreach ($normalized as $index => $probability) {
            $probabilities[$index % $classes][intdiv($index, $classes)] = $probability;
        }

        return (new MulticlassCrossEntropy())->compute(Matrix::fromArray($probabilities, false), $y);
    }
}
