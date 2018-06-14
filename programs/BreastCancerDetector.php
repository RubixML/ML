<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Metrics\Validation\MCC;
use Rubix\Engine\CrossValidation\Holdout;
use Rubix\Engine\Classifiers\RandomForest;
use Rubix\Engine\Transformers\MissingDataImputer;
use Rubix\Engine\Transformers\SparseRandomProjector;
use Rubix\Engine\Transformers\NumericStringConverter;
use Rubix\Engine\CrossValidation\Reports\AggregateReport;
use Rubix\Engine\CrossValidation\Reports\ConfusionMatrix;
use Rubix\Engine\CrossValidation\Reports\ClassificationReport;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Breast Cancer Screener using Random Forest          ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

echo  "\n";

$reader = Reader::createFromPath(dirname(__DIR__) . '/datasets/breast-cancer.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = iterator_to_array($reader->getRecords([
    'radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness',
    'concavity', 'n_concave_points', 'symetry', 'fractal_dimension',
]));

$labels = iterator_to_array($reader->fetchColumn('diagnosis'));

$dataset = new Labeled($samples, $labels);

$estimator = new Pipeline(new RandomForest(50, 0.1, 10, 5), [
    new MissingDataImputer('?'),
    new NumericStringConverter(),
]);

$validator = new Holdout(new MCC(), 0.2);

$report = new AggregateReport([
    new ConfusionMatrix(),
    new ClassificationReport(),
]);

var_dump($validator->score($estimator, $dataset));

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.8);

$estimator->train($training);

var_dump($report->generate($estimator, $testing));

var_dump($estimator->proba($dataset->randomize()->head(5)));
