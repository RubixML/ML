<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Prototype;
use Rubix\Engine\GridSearch;
use Rubix\Engine\Metrics\MCC;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Reports\ConfusionMatrix;
use Rubix\Engine\Reports\ClassificationReport;
use Rubix\Engine\Estimators\KNearestNeighbors;
use Rubix\Engine\Metrics\DistanceFunctions\Euclidean;
use Rubix\Engine\Metrics\DistanceFunctions\Manhattan;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Iris Classifier using K Nearest Neighbors           ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/iris.csv')->setDelimiter(',');

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->split(0.8);

$estimator = new Prototype(new GridSearch(KNearestNeighbors::class, [[1, 3, 5], [new Euclidean(), new Manhattan()]], new MCC()), [
    new ConfusionMatrix($dataset->labels()),
    new ClassificationReport(),
]);

$estimator->train($training);

$estimator->test($testing);
