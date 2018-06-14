<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Datasets\Unlabeled;
use Rubix\Engine\CrossValidation\KFold;
use Rubix\Engine\Clusterers\FuzzyCMeans;
use Rubix\Engine\Metrics\Distance\Euclidean;
use Rubix\Engine\Metrics\Validation\VMeasure;
use Rubix\Engine\Transformers\ZScaleStandardizer;
use Rubix\Engine\Transformers\NumericStringConverter;
use Rubix\Engine\CrossValidation\Reports\ContingencyTable;
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

$estimator = new Pipeline(new FuzzyCMeans(3, 1.5, new Euclidean(), 1e-4), [
    new NumericStringConverter(),
    new ZScaleStandardizer(),
]);

$validator = new KFold(new VMeasure(), 10);

$report = new ContingencyTable();

var_dump($validator->score($estimator, $dataset));

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.8);

$estimator->train($training);

var_dump($report->generate($estimator, $testing));

var_dump($estimator->proba($dataset->randomize()->head(5)));
