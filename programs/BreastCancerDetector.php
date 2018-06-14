<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Metrics\Validation\MCC;
use Rubix\ML\CrossValidation\Holdout;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\SparseRandomProjector;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\ClassificationReport;
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
