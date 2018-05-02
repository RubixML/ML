<?php

use Rubix\Engine\Estimator;
use Rubix\Engine\Classifier;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\MultiLayerPerceptron;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
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

        $this->estimator = new MultiLayerPerceptron(4, [5, 5], $this->dataset->labels(), 1, new Adam(0.01), 0.95, 3);
    }

    public function test_build_multi_layer_perceptron()
    {
        $this->assertInstanceOf(MultiLayerPerceptron::class, $this->estimator);
        $this->assertInstanceOf(Classifier::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_predict_sample()
    {
        $this->estimator->train($this->dataset);

        $prediction = $this->estimator->predict([4.3, 3.0, 1.1, 0.1]);

        $this->assertEquals('Iris-setosa', $prediction->outcome());
    }
}
