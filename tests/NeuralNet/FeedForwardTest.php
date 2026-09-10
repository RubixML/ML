<?php

namespace Rubix\ML\Tests\NeuralNet;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Parameter;

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

    /**
     * @return array{int, int}[]
     */
    public static function freezeProvider() : array
    {
        return [
            [1, 73],
            [2, 73],
            [3, 18],
            [4, 18],
            [5, 0],
        ];
    }

    /**
     * @return array{int}[]
     */
    public static function invalidKProvider() : array
    {
        return [[0], [-1], [6], [10]];
    }

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

        $this->network = new FeedForward($this->input, $this->hidden, $this->output);
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
        $network = new FeedForward($this->input, $this->hidden, $this->output);

        $network->initialize();

        $dense = $this->hidden[0];

        $initial = $dense->parameters()->current()->param()->asArray();

        $network->roundtrip($this->dataset);

        $this->assertEquals($initial, $dense->parameters()->current()->param()->asArray());

        $accumulated = 0;

        foreach ($network->layers() as $layer) {
            if ($layer instanceof Parametric) {
                foreach ($layer->parameters() as $param) {
                    if ($param->gradient()) {
                        ++$accumulated;
                    }
                }
            }
        }

        $this->assertGreaterThan(0, $accumulated);
    }

    #[Test]
    public function accumulatedGradientNormalizesToBatchAverage() : void
    {
        $dataset = Labeled::quick([
            [1.0, 2.5],
            [0.1, 0.0],
            [0.002, -6.0],
            [0.5, 1.0],
        ], ['yes', 'no', 'maybe', 'yes']);

        $batches = $dataset->batch(2);

        $combined = Labeled::quick($dataset->samples(), $dataset->labels());

        $network = new FeedForward($this->input, $this->hidden, $this->output);

        $network->initialize();

        /** @var list<Parameter> $params */
        $params = iterator_to_array($network->parameters());

        $initial = array_map(static fn (Parameter $param) => $param->param()->asArray(), $params);

        $network->roundtrip($batches[0]);
        $network->roundtrip($batches[1]);

        $summed = array_map(static fn (Parameter $param) => $param->gradient()->asArray(), $params);

        foreach ($params as $i => $param) {
            $this->assertEqualsWithDelta($initial[$i], $param->param()->asArray(), 1e-12);

            $param->resetGradient();
        }

        $network->roundtrip($combined);

        $single = array_map(static fn (Parameter $param) => $param->gradient()->asArray(), $params);

        foreach ($params as $i => $param) {
            $this->assertEqualsWithDelta($this->scaled($summed[$i], 0.5), $single[$i], 1e-9);

            $param->resetGradient();
        }

        $network->roundtrip($batches[0]);
        $network->roundtrip($batches[1]);

        foreach ($params as $param) {
            $param->scaleGradient(0.5);
        }

        foreach ($params as $i => $param) {
            $this->assertEqualsWithDelta($single[$i], $param->gradient()->asArray(), 1e-9);
        }
    }

    #[Test]
    public function trainableParametersBaseline() : void
    {
        $this->network->initialize();

        $this->assertEquals(103, $this->network->numParams());
        $this->assertEquals(103, $this->trainableCount());

        foreach ($this->network->parameters() as $param) {
            $this->assertFalse($param->frozen());
        }
    }

    #[Test]
    #[DataProvider('freezeProvider')]
    public function freezeFirstKLayers(int $k, int $expectedTrainable) : void
    {
        $this->network->initialize();

        $this->network->freezeFirstKLayers($k);

        $this->assertEquals($expectedTrainable, $this->trainableCount());
    }

    #[Test]
    public function freezeFirstKLayersFreezesLeadingLayers() : void
    {
        $this->network->initialize();

        $this->network->freezeFirstKLayers(3);

        $hidden = $this->network->hidden();

        foreach ([$hidden[0], $hidden[1], $hidden[2]] as $layer) {
            if ($layer instanceof Parametric) {
                foreach ($layer->parameters() as $param) {
                    $this->assertTrue($param->frozen());
                }
            }
        }

        foreach ($hidden[4]->parameters() as $param) {
            $this->assertFalse($param->frozen());
        }
    }

    #[Test]
    #[DataProvider('invalidKProvider')]
    public function freezeFirstKLayersThrowsForInvalidK(int $k) : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->network->freezeFirstKLayers($k);
    }

    #[Test]
    public function unfreezeMakesAllTrainable() : void
    {
        $this->network->initialize();

        $this->network->freezeFirstKLayers(2);
        $this->network->unfreeze();

        $this->assertEquals(103, $this->trainableCount());

        foreach ($this->network->parameters() as $param) {
            $this->assertFalse($param->frozen());
        }
    }

    /**
     * Return the number of trainable (unfrozen) parameter elements.
     *
     * @return int
     */
    private function trainableCount() : int
    {
        $count = 0;

        foreach ($this->network->trainableParameters() as $param) {
            $count += $param->param()->size();
        }

        return $count;
    }

    /**
     * Return a copy of a numeric nested array with every element multiplied by a scalar.
     *
     * @param list<mixed> $array
     * @param float $scale
     * @return list<mixed>
     */
    private function scaled(array $array, float $scale) : array
    {
        $scaled = [];

        foreach ($array as $element) {
            $scaled[] = is_array($element) ? $this->scaled($element, $scale) : $element * $scale;
        }

        return $scaled;
    }
}
