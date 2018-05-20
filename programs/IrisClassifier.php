<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Estimators\GridSearch;
use Rubix\Engine\CrossValidation\KFold;
use Rubix\Engine\Metrics\Validation\MCC;
use Rubix\Engine\Metrics\Distance\Euclidean;
use Rubix\Engine\Metrics\Distance\Manhattan;
use Rubix\Engine\Estimators\KNearestNeighbors;
use Rubix\Engine\Estimators\Wrappers\Pipeline;
use Rubix\Engine\Transformers\NumericStringConverter;
use Rubix\Engine\CrossValidation\Reports\ConfusionMatrix;
use Rubix\Engine\CrossValidation\Reports\ClassificationReport;
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

$dataset = new Supervised($samples, $labels);

$estimator = new Pipeline(new GridSearch(KNearestNeighbors::class, [
    [1, 3, 5, 10], [new Euclidean(), new Manhattan()],
], new KFold(new MCC(), 10)), [
    new NumericStringConverter(),
]);

$estimator->train($dataset);

var_dump($estimator->results());
