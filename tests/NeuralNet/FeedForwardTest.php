<?php

namespace Rubix\ML\Tests\NeuralNet;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Initializers\Constant as Initializer;

#[Group('NeuralNet')]
#[CoversClass(FeedForward::class)]
class FeedForwardTest extends TestCase
{
    /**
     * @var Labeled
     */
    protected Labeled $dataset;

    /**
     * @var FeedForward
     */
    protected FeedForward $network;

    /**
     * @var Input
     */
    protected Input $input;

    /**
     * @var \Rubix\ML\NeuralNet\Layers\Hidden[]
     */
    protected array $hidden;

    /**
     * @var Output
     */
    protected Output $output;

    protected function setUp() : void
    {
        $this->dataset = Labeled::quick([
            [1.0, 2.5],
            [0.1, 0.0],
            [0.002, -6.0],
        ], ['yes', 'no', 'maybe']);

        $this->input = new Placeholder1D(2);

        $this->hidden = [
            new Dense(10),
            new Activation(new ReLU()),
            new Dense(5),
            new Activation(new ReLU()),
            new Dense(3),
        ];

        $this->output = new Multiclass(['yes', 'no', 'maybe'], new MulticlassCrossEntropy());

        $this->network = new FeedForward($this->input, $this->hidden, $this->output, new Adam(new Constant(0.001)));
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(FeedForward::class, $this->network);
        $this->assertInstanceOf(Network::class, $this->network);
    }

    #[Test]
    public function layers() : void
    {
        $this->assertCount(5, iterator_to_array($this->network->layers()));
    }

    #[Test]
    public function input() : void
    {
        $this->assertInstanceOf(Placeholder1D::class, $this->network->input());
    }

    #[Test]
    public function hidden() : void
    {
        $this->assertCount(5, $this->network->hidden());
    }

    #[Test]
    public function testOutput() : void
    {
        $this->assertInstanceOf(Output::class, $this->network->output());
    }

    #[Test]
    public function numParams() : void
    {
        $this->network->initialize();

        $this->assertEquals(103, $this->network->numParams());
    }

    #[Test]
    public function roundtrip() : void
    {
        $this->network->initialize();

        $loss = $this->network->roundtrip($this->dataset);

        $this->assertIsFloat($loss);
    }

    #[Test]
    public function accumulatesGradients() : void
    {
        $accumulator = new FeedForward($this->input, $this->hidden, $this->output, new Adam(new Constant(0.001)), 2);

        $accumulator->initialize();

        $dense = $this->hidden[0];

        $initial = $dense->parameters()->current()->param()->asArray();

        $accumulator->roundtrip($this->dataset);

        $this->assertEquals($initial, $dense->parameters()->current()->param()->asArray());

        $accumulated = 0;

        foreach ($accumulator->layers() as $layer) {
            if ($layer instanceof Parametric) {
                foreach ($layer->parameters() as $param) {
                    if ($param->gradient()) {
                        ++$accumulated;
                    }
                }
            }
        }

        $this->assertGreaterThan(0, $accumulated);

        $accumulator->roundtrip($this->dataset);

        $this->assertNotEquals($initial, $dense->parameters()->current()->param()->asArray());

        $accumulated = 0;

        foreach ($accumulator->layers() as $layer) {
            if ($layer instanceof Parametric) {
                foreach ($layer->parameters() as $param) {
                    if ($param->gradient()) {
                        ++$accumulated;
                    }
                }
            }
        }

        $this->assertSame(0, $accumulated);
    }

    #[Test]
    public function effectiveBatchIsEquivalent() : void
    {
        $seed = 42;

        srand($seed);

        $dense = new Dense(2, 0.0, true, new Initializer(0.01));

        $full = new FeedForward(
            new Placeholder1D(2),
            [$dense],
            new Multiclass(['yes', 'no'], new MulticlassCrossEntropy()),
            new Adam(new Constant(0.001))
        );

        $full->initialize();

        $full->roundtrip(Labeled::quick([[1.0, 2.5], [0.1, 0.0]], ['yes', 'no']));

        $param = $dense->parameters()->current()->param()->asArray();

        srand($seed);

        $dense = new Dense(2, 0.0, true, new Initializer(0.01));

        $accumulated = new FeedForward(
            new Placeholder1D(2),
            [$dense],
            new Multiclass(['yes', 'no'], new MulticlassCrossEntropy()),
            new Adam(new Constant(0.001)),
            2
        );

        $accumulated->initialize();

        $accumulated->roundtrip(Labeled::quick([[1.0, 2.5]], ['yes']));

        $accumulated->roundtrip(Labeled::quick([[0.1, 0.0]], ['no']));

        $this->assertEquals($param, $dense->parameters()->current()->param()->asArray());
    }
}
