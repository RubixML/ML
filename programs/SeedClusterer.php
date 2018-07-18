<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Clusterers\FuzzyCMeans;
use Rubix\ML\Reports\AggregateReport;
use Rubix\ML\Reports\PredictionSpeed;
use Rubix\ML\Reports\ContingencyTable;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Transformers\QuartileStandardizer;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\CrossValidation\Metrics\Concentration;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Seed Clusterer using Fuzzy C Means                  ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$reader = Reader::createFromPath(dirname(__DIR__) . '/datasets/seeds.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = iterator_to_array($reader->getRecords([
    'area', 'perimeter', 'compactness', 'length_of_kernel', 'width_of_kernel',
    'asymmetry_coefficient', 'length_of_kernel_grove',
]));

$labels = iterator_to_array($reader->fetchColumn('class'));

$dataset = new Labeled($samples, $labels);

$estimator = new Pipeline(new FuzzyCMeans(3, 1.1, new Euclidean(), 1e-4, 200), [
    new NumericStringConverter(),
    new QuartileStandardizer(),
]);

$validator = new KFold(10);

$report = new AggregateReport([
    new ContingencyTable(),
    new PredictionSpeed(),
]);

var_dump($validator->test($estimator, $dataset, new Concentration()));

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.8);

$estimator->train($training);

var_dump($estimator->progress());

var_dump($report->generate($estimator, $testing));

var_dump($estimator->proba($dataset->randomize()->head(3)));
