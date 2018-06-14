<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\Metrics\Validation\MCC;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Classifiers\MultiLayerPerceptron;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\Transformers\SparseRandomProjector;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\ClassificationReport;
use League\Csv\Reader;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Mushroom Classifier using Multi Layer Perceptron    ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$reader = Reader::createFromPath(dirname(__DIR__) . '/datasets/mushrooms.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$header = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
    'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
    'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population',
    'habitat',
];

$samples = iterator_to_array($reader->getRecords($header));

$labels = iterator_to_array($reader->fetchColumn('class'));

$dataset = new Labeled($samples, $labels);

$hidden = [
    new Dense(10, new ELU()),
    new Dense(10, new ELU()),
    new Dense(10, new ELU()),
    new Dense(10, new ELU()),
];

$estimator = new Pipeline(new MultiLayerPerceptron($hidden, 50, new Adam(0.001),
    1e-4, new MCC(), 0.2, 3, 100), [
        new OneHotEncoder(),
        new SparseRandomProjector(30),
    ]);

$report = new AggregateReport([
    new ConfusionMatrix(),
    new ClassificationReport(),
]);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.8);

$estimator->train($training);

$model = new PersistentModel($estimator);

var_dump($report->generate($estimator, $testing));

var_dump($estimator->progress());

$model->save(dirname(__DIR__) . '/models/mushroom.model');

var_dump($model->proba($dataset->randomize()->head(5)));
