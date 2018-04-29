<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\Metrics\Accuracy;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\MultiLayerPerceptron;
use Rubix\Engine\Persisters\Filesystem;
use Rubix\Engine\Reports\ConfusionMatrix;
use Rubix\Engine\Transformers\OneHotEncoder;
use Rubix\Engine\NeuralNet\LearningRates\Adam;
use Rubix\Engine\Reports\ClassificationReport;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Mushroom Classifier using Multi Layer Perceptron    ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/mushrooms.csv')->setDelimiter(',')->getRecords();

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.8);

$estimator = new Prototype(new Pipeline(new MultiLayerPerceptron(23, [6, 6], $dataset->labels(),
                    3, new Adam(0.01), 0.999, 2, new Accuracy(), 100), [
    new OneHotEncoder(),
]), [
    new ConfusionMatrix($dataset->labels()),
    new ClassificationReport(),
]);

$estimator->train($training);

var_dump($estimator->progress());

$estimator->test($testing);

$persister = new Filesystem(dirname(__DIR__) . '/models/sentiment.model');

$persister->save($estimator);
