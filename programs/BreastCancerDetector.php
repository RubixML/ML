<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Prototype;
use Rubix\Engine\GridSearch;
use Rubix\Engine\Metrics\Accuracy;
use Rubix\Engine\KNearestNeighbors;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Reports\ConfusionMatrix;
use Rubix\Engine\Reports\ClassificationReport;
use Rubix\Engine\Transformers\MinMaxNormalizer;
use Rubix\Engine\Transformers\MissingDataImputer;
use Rubix\Engine\Metrics\DistanceFunctions\Euclidean;
use Rubix\Engine\Metrics\DistanceFunctions\Manhattan;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Breast Cancer Detector using K Nearest Neighbors    ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/breast-cancer.csv')->setDelimiter(',');

$dataset = Supervised::fromIterator($dataset);

list($training, $testing) = $dataset->randomize()->split(0.8);

$estimator = new Prototype(new Pipeline(new GridSearch(KNearestNeighbors::class, [[1, 3, 5, 10], [new Euclidean(), new Manhattan()]], new Accuracy()), [
    new MissingDataImputer('?'),
    new MinMaxNormalizer(),
]), [
    new ConfusionMatrix($dataset->labels()),
    new ClassificationReport(),
]);

$estimator->train($training);

var_dump($estimator->trials());

$estimator->test($testing);
