<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\GridSearch;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\RandomParams;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Reports\ConfusionMatrix;
use Rubix\ML\Kernels\Distance\Diagonal;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\CrossValidation\Metrics\MCC;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\CrossValidation\Metrics\F1Score;
use Rubix\ML\Transformers\NumericStringConverter;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Iris Classifier using K Nearest Neighbors           ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo "\n";

$reader = Reader::createFromPath(dirname(__DIR__) . '/datasets/iris.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = iterator_to_array($reader->getRecords([
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
]));

$labels = iterator_to_array($reader->fetchColumn('class'));

$dataset = new Labeled($samples, $labels);

$dataset->apply(new NumericStringConverter());

$params = [
    RandomParams::ints(1, 10, 5), [new Euclidean(), new Diagonal()],
];

$estimator = new GridSearch(KNearestNeighbors::class, $params, new MCC(), new KFold(10));

$validator = new KFold(10);

$estimator->train($dataset);

var_dump($estimator->results());

var_dump($estimator->best());

var_dump($validator->test($estimator, $dataset, new MCC()));
