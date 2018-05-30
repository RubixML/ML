<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\Pipeline;
use Rubix\Engine\PersistentModel;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\NeuralNet\Layers\Dense;
use Rubix\Engine\Metrics\Validation\MCC;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\Transformers\OneHotEncoder;
use Rubix\Engine\CrossValidation\ReportGenerator;
use Rubix\Engine\Classifiers\MultiLayerPerceptron;
use Rubix\Engine\NeuralNet\ActivationFunctions\PReLU;
use Rubix\Engine\CrossValidation\Reports\AggregateReport;
use Rubix\Engine\CrossValidation\Reports\ConfusionMatrix;
use Rubix\Engine\CrossValidation\Reports\ClassificationReport;
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
    new Dense(10, new PReLU()),
    new Dense(10, new PReLU()),
    new Dense(10, new PReLU()),
];

$estimator = new Pipeline(new MultiLayerPerceptron($hidden, 10, new Adam(0.001),
    1e-4, new MCC(), 0.2, 3, 100), [
        new OneHotEncoder(),
    ]);

$report = new ReportGenerator(new AggregateReport([
    new ConfusionMatrix(),
    new ClassificationReport(),
]), 0.2);

$model = new PersistentModel($estimator);

var_dump($report->generate($estimator, $dataset));

var_dump($estimator->progress());

$model->save(dirname(__DIR__) . '/models/mushroom.model');

var_dump($model->proba($dataset->randomize()->head(5)));
