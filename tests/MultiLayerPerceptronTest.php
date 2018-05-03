<?php

use Rubix\Engine\Estimator;
use Rubix\Engine\Classifier;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\MultiLayerPerceptron;
use Rubix\Engine\Persisters\Persistable;
use Rubix\Engine\NeuralNet\Layers\Input;
use Rubix\Engine\NeuralNet\Layers\Hidden;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\NeuralNet\Layers\Multiclass;
use PHPUnit\Framework\TestCase;
use League\Csv\Reader;

class MultiLayerPerceptronTest extends TestCase
{
    protected $dataset;

    protected $estimator;

    public function setUp()
    {
        $this->dataset = Supervised::fromIterator(Reader::createFromPath(dirname(__DIR__) . '/datasets/iris.csv')
            ->setDelimiter(','));

        $this->estimator = new MultiLayerPerceptron(new Input(4), [new Hidden(5), new Hidden(5)], new Multiclass($this->dataset->labels()), 1, new Adam(0.01), 0.95, 3);
    }

    public function test_build_multi_layer_perceptron()
    {
        $this->assertInstanceOf(MultiLayerPerceptron::class, $this->estimator);
        $this->assertInstanceOf(Classifier::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_predict_sample()
    {
        $this->estimator->train($this->dataset);

        $prediction = $this->estimator->predict([4.3, 3.0, 1.1, 0.1]);

        $this->assertEquals('Iris-setosa', $prediction->outcome());
    }
}
