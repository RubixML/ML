<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\GridSearch;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Metrics\Validation\MCC;
use Rubix\ML\Metrics\Distance\Diagonal;
use Rubix\ML\Metrics\Distance\Euclidean;
use Rubix\ML\Metrics\Distance\Manhattan;
use Rubix\ML\Metrics\Validation\F1Score;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
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

$dataset->randomize();

$params = [
    [1, 3, 5, 7, 9], [new Euclidean(), new Diagonal(), new Manhattan()],
];

$estimator = new Pipeline(new GridSearch(KNearestNeighbors::class, $params, new MCC(), new KFold(10)), [
    new NumericStringConverter(),
]);

$validator = new KFold(10);

$estimator->train($dataset);

var_dump($estimator->results());

var_dump($estimator->best());

var_dump($validator->test($estimator, $dataset, new MCC()));
