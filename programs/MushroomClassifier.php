<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Metrics\Accuracy;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\MultiLayerPerceptron;
use Rubix\Engine\Reports\ConfusionMatrix;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\Transformers\OneHotEncoder;
use Rubix\Engine\Reports\ClassificationReport;
use Rubix\Engine\NeuralNet\ActivationFunctions\PReLU;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Mushroom Classifier using Multi Layer Perceptron    ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/mushrooms.csv')->setDelimiter(',')->getRecords();

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.8);

$estimator = new Prototype(new Pipeline(new MultiLayerPerceptron(23, [[8, new PReLU()], [8, new PReLU()]], $dataset->labels(),
                    10, new Adam(0.001), 0.999, 3, new Accuracy(), 100), [
    new OneHotEncoder(),
]), [
    new ConfusionMatrix($dataset->labels()),
    new ClassificationReport(),
]);

$estimator->train($training);

var_dump($estimator->progress());

$estimator->test($testing);

var_dump($estimator->predict($dataset->sample(0)));
