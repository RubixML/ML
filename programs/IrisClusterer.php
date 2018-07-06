<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\GridSearch;
use Rubix\ML\RandomParams;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\MeanShift;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Reports\AggregateReport;
use Rubix\ML\Reports\PredictionSpeed;
use Rubix\ML\Reports\ContingencyTable;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Minkowski;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\Transformers\NumericStringConverter;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Iris Clusterer using Mean Shift                     ║' . "\n";
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

$params = [
    RandomParams::floats(0.5, 1.0, 10), [new Euclidean(), new Minkowski(3.0)],
];

$estimator = new Pipeline(new GridSearch(MeanShift::class, $params, new VMeasure(), new KFold(10)), [
    new NumericStringConverter(),
]);

$estimator->train($dataset);

var_dump($estimator->results());

var_dump($estimator->best());

var_dump($estimator->predict($dataset->randomize()->head(10)));
