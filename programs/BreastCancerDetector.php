<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\CrossValidation\Holdout;
use Rubix\Engine\Metrics\Validation\MCC;
use Rubix\Engine\Estimators\RandomForest;
use Rubix\Engine\Estimators\Wrappers\Pipeline;
use Rubix\Engine\Transformers\MissingDataImputer;
use Rubix\Engine\Transformers\NumericStringConverter;
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

$dataset = new Supervised($samples, $labels);

$estimator = new Pipeline(new RandomForest(10, 0.3, 5, 10), [
    new MissingDataImputer('?'),
    new NumericStringConverter(),
]);

$validator = new Holdout(new MCC(), 0.2);

var_dump($validator->score($estimator, $dataset));
