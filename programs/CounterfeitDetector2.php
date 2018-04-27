<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\MultiLayerPerceptron;
use Rubix\Engine\Reports\ConfusionMatrix;
use Rubix\Engine\Transformers\L2Regularizer;
use Rubix\Engine\Reports\ClassificationReport;
use Rubix\Engine\NeuralNetwork\Optimizers\Adam;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Counterfeit Detector using Multi Layer Perceptron   ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/banknotes.csv')->setDelimiter(',')->getRecords();

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.8);

$estimator = new Prototype(new Pipeline(new MultiLayerPerceptron($dataset->columns(), [10, 10], $dataset->labels(), 3, 10, new Adam(0.01)), [
    new L2Regularizer(),
]), [
    new ConfusionMatrix($dataset->labels()),
    new ClassificationReport(),
]);

$estimator->train($training);

$estimator->test($testing);
