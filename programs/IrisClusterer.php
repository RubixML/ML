<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Clusterers\MeanShift;
use Rubix\Engine\CrossValidation\KFold;
use Rubix\Engine\Metrics\Distance\Euclidean;
use Rubix\Engine\Metrics\Validation\VMeasure;
use Rubix\Engine\Transformers\ZScaleStandardizer;
use Rubix\Engine\Transformers\NumericStringConverter;
use Rubix\Engine\CrossValidation\Reports\ContingencyTable;
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

$estimator = new Pipeline(new MeanShift(1.3, new Euclidean(), 1e-8), [
    new NumericStringConverter(),
    new ZScaleStandardizer(),
]);

$validator = new KFold(new VMeasure(), 10);

$report = new ContingencyTable();

var_dump($validator->score($estimator, $dataset));

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.8);

$estimator->train($training);

var_dump($report->generate($estimator, $testing));

var_dump($estimator->predict($dataset->randomize()->head(5)));
