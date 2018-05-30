<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\GridSearch;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\CrossValidation\KFold;
use Rubix\Engine\Metrics\Validation\MCC;
use Rubix\Engine\Metrics\Distance\Euclidean;
use Rubix\Engine\Metrics\Distance\Manhattan;
use Rubix\Engine\Classifiers\KNearestNeighbors;
use Rubix\Engine\CrossValidation\ReportGenerator;
use Rubix\Engine\Transformers\NumericStringConverter;
use Rubix\Engine\CrossValidation\Reports\AggregateReport;
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

$dataset = new Labeled($samples, $labels);

$estimator = new Pipeline(new GridSearch(KNearestNeighbors::class, [
    [1, 3, 5, 10], [new Euclidean(), new Manhattan()],
], new KFold(new MCC(), 10)), [
    new NumericStringConverter(),
]);

$report = new ReportGenerator(new AggregateReport([
    new ConfusionMatrix(),
    new ClassificationReport(),
]), 0.2);

var_dump($report->generate($estimator, $dataset));

$estimator->train($dataset->randomize()->leave(5));

var_dump($estimator->proba($dataset));
