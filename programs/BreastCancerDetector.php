<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\CommitteeMachine;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\ExtraTreeClassifier;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\Reports\AggregateReport;
use Rubix\ML\Reports\ConfusionMatrix;
use Rubix\ML\Reports\PredictionSpeed;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Reports\MulticlassBreakdown;
use Rubix\ML\CrossValidation\Metrics\MCC;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\MultiLayerPerceptron;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\Transformers\NumericStringConverter;
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
    new Dense(20, new ELU()),
    new Dense(20, new ELU()),
    new Dense(20, new ELU()),
    new Dense(20, new ELU()),
];

$estimator = new Pipeline(new CommitteeMachine([
    new MultiLayerPerceptron($hidden, 50, new Adam(0.001), 1e-4, new MCC()),
    new RandomForest(500, 0.1, 10, 3, 3, 1e-4, ExtraTreeClassifier::class),
    new ClassificationTree(100, 3, 5, 1e-4),
    new GaussianNB(),
]), [
    new NumericStringConverter(),
    new MissingDataImputer('?'),
    new ZScaleStandardizer(),
]);

$report = new AggregateReport([
    new ConfusionMatrix(),
    new MulticlassBreakdown(),
    new PredictionSpeed(),
]);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.80);

$estimator->train($training);

var_dump($report->generate($estimator, $testing));

var_dump($estimator->predict($testing->randomize()->head(3)));
