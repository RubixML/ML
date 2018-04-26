<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Prototype;
use Rubix\Engine\GridSearch;
use Rubix\Engine\Metrics\Accuracy;
use Rubix\Engine\KNearestNeighbors;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Graph\DistanceFunctions\Euclidean;
use Rubix\Engine\Graph\DistanceFunctions\Manhattan;
use Rubix\Engine\Metrics\Reports\ClassificationReport;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Iris Classifier using K Nearest Neighbors           ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/iris.csv')->setDelimiter(',');

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.1);

$estimator = new Prototype(new GridSearch(KNearestNeighbors::class, [[1, 3, 5], [new Euclidean(), new Manhattan()]], new Accuracy(), 0.1), [
    new ClassificationReport(),
]);

$estimator->train($training);

$estimator->test($testing);
