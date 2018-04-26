<?php

use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\MultiLayerPerceptron;
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

        $this->estimator = new MultiLayerPerceptron(4, [5, 5], $this->dataset->labels(), 5, 10);

        $this->estimator->train($this->dataset);
    }

    public function test_predict_sample()
    {
        $prediction = $this->estimator->predict([4.3, 3.0, 1.1, 0.1]);

        $this->assertEquals('Iris-setosa', $prediction->outcome());
    }
}
