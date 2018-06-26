<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\Metrics\Validation\MCC;
use Rubix\ML\Classifiers\DecisionTree;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Metrics\Distance\Euclidean;
use Rubix\ML\Classifiers\CommitteeMachine;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\MultiLayerPerceptron;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Breast Cancer Screener using a Committee Machine    ║' . "\n";
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

$hidden = [
    new Dense(10, new ELU()),
    new Dense(10, new ELU()),
    new Dense(10, new ELU()),
    new Dense(10, new ELU()),
];

$estimator = new Pipeline(new CommitteeMachine([
    new MultiLayerPerceptron($hidden, 50, new Adam(0.001), 1e-4, new MCC()),
    new RandomForest(100, 0.3, 50, 3, 1e-2),
    new DecisionTree(500, 1, 1e-4),
    new KNearestNeighbors(3, new Euclidean()),
]), [
    new NumericStringConverter(),
    new MissingDataImputer('?'),
    new ZScaleStandardizer(),
]);

$report = new AggregateReport([
    new ConfusionMatrix(),
    new MulticlassBreakdown(),
]);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.20);

$estimator->train($training);

var_dump($report->generate($estimator, $testing));

var_dump($estimator->proba($dataset->randomize()->head(3)));
